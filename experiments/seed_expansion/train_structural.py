#!/usr/bin/env python3
"""
v3: substrate-hash features + structural features (deps, complexity,
ast_size, ast_depth, token_count).

Hypothesis: structural features correlate with token co-occurrence
patterns. The model can interpolate over them on held-out inputs
because similar functions share dependency multisets and structural
shape — unlike raw hashes which are uncorrelated by design.

Test: 40 train / 10 held-out. Same architecture as v2, just richer
feature vector.
"""

import json, math, random
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim

torch.manual_seed(7)
random.seed(7)

ED = Path(__file__).parent
samples = [json.loads(l) for l in open(ED / "corpus_structural.jsonl") if l.strip()]
random.shuffle(samples)
TRAIN, TEST = samples[:40], samples[40:]

PAD, BOS, EOS, RESERVED = 0, 1, 2, 3

observed = set()
for s in TRAIN:
    observed.update(s["tokens"])
id_map = {old: new + RESERVED for new, old in enumerate(sorted(observed))}
vocab_size = RESERVED + len(observed)
print(f"vocab from TRAIN: {vocab_size}")

# Build dep vocab from TRAIN deps only.
dep_vocab = sorted({d for s in TRAIN for d in s["deps"]})
dep_idx = {d: i for i, d in enumerate(dep_vocab)}
N_DEPS = len(dep_vocab)
print(f"dep vocab from TRAIN: {N_DEPS}")

def remap(tokens):
    return [id_map.get(t, PAD) for t in tokens]

def features(s):
    """48-dim feature vector: 16 substrate + N_DEPS deps + 4 structural."""
    raw = s["raw"]; dist = s["distance"]; res = s["resonance"]
    f = []
    # Substrate hash features (16 dims, same as v2).
    for p in [3, 5, 7, 11, 13, 17, 19, 23]:
        f.append(((raw % p) / p) * 2 - 1)
    f.append(math.tanh(math.log10(abs(raw)+1)/20.0))
    f.append(math.tanh(math.log10(abs(dist)+1)/20.0))
    f.append(math.tanh(res * 1e10))
    f.append((raw % 1009) / 1009.0 * 2 - 1)
    chunk = raw & 0xFFFF
    f.append(((chunk >> 0) & 0xFF) / 255.0 * 2 - 1)
    f.append(((chunk >> 8) & 0xFF) / 255.0 * 2 - 1)
    f.append((dist & 0xFF) / 255.0 * 2 - 1)
    f.append(((dist >> 16) & 0xFF) / 255.0 * 2 - 1)
    # Dep presence indicators (N_DEPS dims, one-hot multiset).
    deps = set(s["deps"])
    for d in dep_vocab:
        f.append(1.0 if d in deps else 0.0)
    # Structural metrics, normalised.
    f.append(math.tanh(s["complexity"] / 5.0))
    f.append(math.tanh(s["ast_size"] / 30.0))
    f.append(math.tanh(s["ast_depth"] / 5.0))
    f.append(math.tanh(s["token_count"] / 30.0))
    return f

feat_dim = 16 + N_DEPS + 4
print(f"feature dim: {feat_dim}")

seqs_tr = [[BOS] + remap(s["tokens"]) + [EOS] for s in TRAIN]
seqs_te = [[BOS] + remap(s["tokens"]) + [EOS] for s in TEST]
feats_tr = [features(s) for s in TRAIN]
feats_te = [features(s) for s in TEST]
max_len = max(max(len(seq) for seq in seqs_tr), max(len(seq) for seq in seqs_te))

def pad(s, L): return s + [PAD] * (L - len(s))
x_tr = torch.tensor([pad(seq, max_len) for seq in seqs_tr], dtype=torch.long)
x_te = torch.tensor([pad(seq, max_len) for seq in seqs_te], dtype=torch.long)
F_tr = torch.tensor(feats_tr, dtype=torch.float32)
F_te = torch.tensor(feats_te, dtype=torch.float32)

class Expander(nn.Module):
    def __init__(self, feat_dim, vocab, hidden=128, embed=64, layers=2):
        super().__init__()
        self.cond = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden * layers), nn.Tanh(),
        )
        self.embed = nn.Embedding(vocab, embed)
        self.gru = nn.GRU(embed, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, vocab)
        self.hidden = hidden; self.layers = layers
    def forward(self, seed, inp):
        B = seed.size(0)
        h0 = self.cond(seed).view(B, self.layers, self.hidden).transpose(0, 1).contiguous()
        out, _ = self.gru(self.embed(inp), h0)
        return self.out(out)
    @torch.no_grad()
    def decode(self, seed, L):
        h = self.cond(seed).view(1, self.layers, self.hidden).transpose(0, 1).contiguous()
        toks = [BOS]
        for _ in range(L - 1):
            inp = torch.tensor([[toks[-1]]])
            out, h = self.gru(self.embed(inp), h)
            t = int(self.out(out[:, -1]).argmax(-1).item())
            toks.append(t)
            if t == EOS: break
        return toks

model = Expander(feat_dim, vocab_size)
opt = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1500)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
N, B = x_tr.size(0), 16
print(f"params: {sum(p.numel() for p in model.parameters()):,}")
print("training v3...")
for epoch in range(1500):
    model.train()
    perm = torch.randperm(N)
    total = 0.0
    for i in range(0, N, B):
        idx = perm[i:i+B]
        logits = model(F_tr[idx], x_tr[idx, :-1])
        loss = loss_fn(logits.reshape(-1, vocab_size), x_tr[idx, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * idx.size(0)
    sched.step()
    if (epoch + 1) % 200 == 0:
        print(f"  epoch {epoch+1:4d} loss {total/N:.4f}")

model.eval()
def eval_set(F, seqs, samples_list, name):
    exact, near, prefix_sum = 0, 0, 0.0
    misses = []
    for i in range(len(seqs)):
        decoded = model.decode(F[i:i+1], max_len)
        target = seqs[i]
        def strip(seq):
            out = []
            for t in seq[1:]:
                if t in (EOS, PAD): break
                out.append(t)
            return out
        d, t = strip(decoded), strip(target)
        is_exact = d == t
        n = min(len(d), len(t))
        p = 0
        while p < n and d[p] == t[p]: p += 1
        pr = p / max(1, len(t))
        if is_exact: exact += 1
        elif name == "HELD-OUT":
            misses.append((samples_list[i]["canonical"], pr))
        if pr > 0.8: near += 1
        prefix_sum += pr
    total = len(seqs)
    print(f"{name}: exact={exact}/{total} ({100*exact/total:.1f}%)  "
          f">=80%prefix={near}/{total} ({100*near/total:.1f}%)  "
          f"mean_prefix={prefix_sum/total:.3f}")
    return exact, near, prefix_sum / total, misses

print("\n=== Results ===")
tr_e, tr_n, tr_p, _ = eval_set(F_tr, seqs_tr, TRAIN, "TRAIN")
te_e, te_n, te_p, te_misses = eval_set(F_te, seqs_te, TEST, "HELD-OUT")

if te_misses:
    print("\nHeld-out samples that did NOT reconstruct exactly:")
    for src, pr in te_misses[:5]:
        print(f"  prefix={pr:.2f}  {src}")

out = {
    "feat_dim": feat_dim,
    "train_exact": tr_e, "train_n": x_tr.size(0),
    "test_exact": te_e, "test_n": x_te.size(0),
    "test_near": te_n,
    "test_mean_prefix": te_p,
    "n_deps": N_DEPS,
    "vocab_size": vocab_size,
    "params": sum(p.numel() for p in model.parameters()),
}
with open(ED / "results_structural.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nwrote {ED / 'results_structural.json'}")

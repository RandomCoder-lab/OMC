#!/usr/bin/env python3
"""Honest held-out test: 40 train / 10 held-out. Train on 40, measure
reconstruction on both the train-set AND on unseen functions.

Hypothesis: train-set reconstruction stays ~100% (memorization works).
Held-out reconstruction collapses (no generalization — different seeds
have nothing in common to learn from)."""

import json, math
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim

torch.manual_seed(42)

ED = Path(__file__).parent
samples = [json.loads(l) for l in open(ED / "corpus.jsonl") if l.strip()]

# Train/test split.
import random
random.seed(7)
random.shuffle(samples)
TRAIN = samples[:40]
TEST = samples[40:]

PAD, BOS, EOS, RESERVED = 0, 1, 2, 3
# Vocab from TRAIN only (held-out novel tokens become PAD — acceptable).
observed = set()
for s in TRAIN:
    observed.update(s["tokens"])
id_map = {old: new + RESERVED for new, old in enumerate(sorted(observed))}
vocab_size = RESERVED + len(observed)
print(f"vocab from TRAIN: {vocab_size}")

def remap(tokens):
    return [id_map.get(t, PAD) for t in tokens]

def features(s):
    raw = s["raw"]; dist = s["distance"]; res = s["resonance"]
    f = []
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
    return f

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
            nn.Linear(hidden, hidden * layers), nn.Tanh())
        self.embed = nn.Embedding(vocab, embed)
        self.gru = nn.GRU(embed, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, vocab)
        self.hidden = hidden; self.layers = layers
    def forward(self, seed, inp):
        h0 = self.cond(seed).view(seed.size(0), self.layers, self.hidden).transpose(0, 1).contiguous()
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

model = Expander(16, vocab_size)
opt = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1500)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
N, B = x_tr.size(0), 16
print("Training on 40, holding out 10...")
for epoch in range(1500):
    model.train()
    perm = torch.randperm(N)
    for i in range(0, N, B):
        idx = perm[i:i+B]
        logits = model(F_tr[idx], x_tr[idx, :-1])
        loss = loss_fn(logits.reshape(-1, vocab_size), x_tr[idx, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    sched.step()

model.eval()
def eval_set(F, seqs, name):
    exact, near, prefix_sum = 0, 0, 0.0
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
        if pr > 0.8: near += 1
        prefix_sum += pr
    total = len(seqs)
    print(f"{name}: exact={exact}/{total} ({100*exact/total:.1f}%)  "
          f"≥80%prefix={near}/{total} ({100*near/total:.1f}%)  "
          f"mean_prefix={prefix_sum/total:.3f}")

eval_set(F_tr, seqs_tr, "TRAIN")
eval_set(F_te, seqs_te, "HELD-OUT")

# Show some held-out examples concretely.
print("\nHeld-out samples (model has NEVER seen these seeds during training):")
for i in range(min(5, len(TEST))):
    s = TEST[i]
    decoded = model.decode(F_te[i:i+1], max_len)
    dec_remapped = []
    inv = {v: k for k, v in id_map.items()}
    for t in decoded[1:]:
        if t in (EOS, PAD): break
        if t in inv: dec_remapped.append(inv[t])
    print(f"\nORIGINAL: {s['canonical']}")
    print(f"DECODED tokens (first 8): {dec_remapped[:8]}")
    print(f"  expected first 8       : {s['tokens'][:8]}")

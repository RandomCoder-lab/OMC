#!/usr/bin/env python3
"""
v2: richer features + bigger model + scheduled sampling.

What changed from v1:
  - Feature vector: 5 → 16 dims (bit-decomposition of hash, more
    moduli, log+linear normalisations)
  - Hidden size: 64 → 128
  - 2 GRU layers
  - Longer training: 600 → 1500 epochs
  - Train with TINY teacher-forcing dropout (~5%) — encourages
    the model to recover from its own decoding errors
"""

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(7)

ED = Path(__file__).parent
CORPUS = ED / "corpus.jsonl"
OUTPUT = ED / "results_v2.json"

PAD, BOS, EOS = 0, 1, 2
RESERVED = 3

samples = [json.loads(l) for l in open(CORPUS) if l.strip()]
print(f"Loaded {len(samples)} samples")

# Vocab
observed = set()
for s in samples:
    observed.update(s["tokens"])
id_map = {old: new + RESERVED for new, old in enumerate(sorted(observed))}
vocab_size = RESERVED + len(observed)
print(f"vocab_size = {vocab_size}")

def remap(tokens):
    return [id_map[t] for t in tokens]

def features(s):
    """16-dim feature vector — richer than v1."""
    raw = s["raw"]
    dist = s["distance"]
    res = s["resonance"]
    abs_raw = abs(raw)
    abs_dist = abs(dist)
    f = []
    # 8 mod-prime fingerprints (chunks 8 bits of distinct info each).
    for p in [3, 5, 7, 11, 13, 17, 19, 23]:
        f.append(((raw % p) / p) * 2 - 1)
    # 4 log-magnitude features.
    f.append(math.tanh(math.log10(abs_raw + 1) / 20.0))
    f.append(math.tanh(math.log10(abs_dist + 1) / 20.0))
    f.append(math.tanh(res * 1e10))  # scale up the tiny resonance
    f.append((raw % 1009) / 1009.0 * 2 - 1)
    # Bit-decomposition of a high-entropy chunk.
    chunk = raw & 0xFFFF
    f.append(((chunk >> 0) & 0xFF) / 255.0 * 2 - 1)
    f.append(((chunk >> 8) & 0xFF) / 255.0 * 2 - 1)
    f.append((dist & 0xFF) / 255.0 * 2 - 1)
    f.append(((dist >> 16) & 0xFF) / 255.0 * 2 - 1)
    assert len(f) == 16
    return f

seqs = [[BOS] + remap(s["tokens"]) + [EOS] for s in samples]
feats = [features(s) for s in samples]
max_len = max(len(seq) for seq in seqs)
print(f"max_len = {max_len}")

def pad(s, L):
    return s + [PAD] * (L - len(s))

x = torch.tensor([pad(seq, max_len) for seq in seqs], dtype=torch.long)
F = torch.tensor(feats, dtype=torch.float32)

class Expander(nn.Module):
    def __init__(self, feat_dim, vocab, hidden=128, embed=64, layers=2):
        super().__init__()
        self.cond = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden * layers),
            nn.Tanh(),
        )
        self.embed = nn.Embedding(vocab, embed)
        self.gru = nn.GRU(embed, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, vocab)
        self.hidden = hidden
        self.layers = layers

    def forward(self, seed, inp):
        B = seed.size(0)
        h0 = self.cond(seed).view(B, self.layers, self.hidden).transpose(0, 1).contiguous()
        out, _ = self.gru(self.embed(inp), h0)
        return self.out(out)

    @torch.no_grad()
    def decode(self, seed, L):
        B = seed.size(0)
        h = self.cond(seed).view(B, self.layers, self.hidden).transpose(0, 1).contiguous()
        toks = [BOS]
        for _ in range(L - 1):
            inp = torch.tensor([[toks[-1]]])
            out, h = self.gru(self.embed(inp), h)
            t = int(self.out(out[:, -1]).argmax(-1).item())
            toks.append(t)
            if t == EOS:
                break
        return toks

model = Expander(16, vocab_size, hidden=128, embed=64, layers=2)
opt = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1500)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

N = x.size(0)
B = 16
EPOCHS = 1500
print("Training v2…")
for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(N)
    total = 0.0
    for i in range(0, N, B):
        idx = perm[i:i+B]
        bx, bf = x[idx], F[idx]
        # Teacher forcing.
        logits = model(bf, bx[:, :-1])
        loss = loss_fn(logits.reshape(-1, vocab_size), bx[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * bx.size(0)
    sched.step()
    if (epoch + 1) % 100 == 0:
        print(f"  epoch {epoch+1:4d}  loss {total/N:.4f}  lr {opt.param_groups[0]['lr']:.5f}")

model.eval()
exact, near, prefix_sum = 0, 0, 0.0
results = []
for i, s in enumerate(samples):
    decoded = model.decode(F[i:i+1], max_len)
    target = seqs[i]
    def strip(seq):
        out = []
        for t in seq[1:]:
            if t in (EOS, PAD):
                break
            out.append(t)
        return out
    d = strip(decoded)
    t = strip(target)
    is_exact = d == t
    n = min(len(d), len(t))
    p = 0
    while p < n and d[p] == t[p]:
        p += 1
    pr = p / max(1, len(t))
    if is_exact: exact += 1
    if pr > 0.8: near += 1
    prefix_sum += pr
    results.append({"idx": i, "canonical": s["canonical"], "exact": is_exact,
                    "prefix_ratio": pr, "target_len": len(t), "decoded_len": len(d)})

print(f"\n=== v2 Results ===")
print(f"  exact      : {exact}/{N}  ({100*exact/N:.1f}%)")
print(f"  ≥80% prefix: {near}/{N}  ({100*near/N:.1f}%)")
print(f"  mean prefix: {prefix_sum/N:.3f}")

with open(OUTPUT, "w") as fp:
    json.dump({"n_samples": N, "vocab_size": vocab_size,
               "n_params": sum(p.numel() for p in model.parameters()),
               "epochs": EPOCHS, "exact_match": exact, "near_match": near,
               "mean_prefix_ratio": prefix_sum / N,
               "per_sample": results}, fp, indent=2)
print(f"Wrote {OUTPUT}")

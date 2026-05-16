#!/usr/bin/env python3
"""
Seed expansion experiment — can a tiny PyTorch model reconstruct an
OMC token sequence from a substrate-derived seed?

Honest framing:
  - 50 training samples
  - Each has a 5-dim substrate seed (raw hash, distance, attractor,
    resonance, plus a derived hash-mod-prime fingerprint)
  - Target: variable-length token sequence
  - Tiny GRU decoder conditioned on seed
  - Measure exact-match reconstruction on the training set

What we're testing: with substrate conditioning, can a model serve as
a learned "expansion table" that decompresses single seeds to full
token sequences?

NOT testing: generalization to novel seeds (that's a separate
hypothesis that needs orders-of-magnitude more data).
"""

import json
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)

EXPERIMENT_DIR = Path(__file__).parent
CORPUS = EXPERIMENT_DIR / "corpus.jsonl"
OUTPUT = EXPERIMENT_DIR / "results.json"

# ---------------------------------------------------------------- data
samples = []
with open(CORPUS) as f:
    for line in f:
        line = line.strip()
        if line:
            samples.append(json.loads(line))
print(f"Loaded {len(samples)} samples")

# Build token vocab from observed tokens + reserved BOS/EOS/PAD.
PAD, BOS, EOS = 0, 1, 2
RESERVED = 3
observed = set()
for s in samples:
    observed.update(s["tokens"])
# Reindex: shift observed token IDs above RESERVED to avoid clash with
# our PAD/BOS/EOS. Old-ID → new-ID lookup.
id_map = {}
for new_id, old_id in enumerate(sorted(observed)):
    id_map[old_id] = new_id + RESERVED
inv_id_map = {v: k for k, v in id_map.items()}
vocab_size = RESERVED + len(observed)
print(f"Vocab size (with PAD/BOS/EOS): {vocab_size}")

def remap(tokens):
    return [id_map[t] for t in tokens]

def derive_features(s):
    """5-dim feature vector from a sample's substrate metadata.
    Normalised so all features are in ~[-1, 1]."""
    raw = s["raw"]
    dist = s["distance"]
    res = s["resonance"]
    # Hash-mod-prime fingerprint for additional bits of entropy.
    fp1 = (raw % 100003) / 100003.0
    fp2 = (raw % 7919) / 7919.0
    # Log-magnitude normalisations.
    raw_n = math.tanh(math.log10(abs(raw) + 1) / 20.0)
    dist_n = math.tanh(math.log10(abs(dist) + 1) / 20.0)
    return [raw_n, dist_n, fp1, fp2, res]

# Sequences for the decoder: BOS + tokens + EOS.
seqs = []
features = []
for s in samples:
    seqs.append([BOS] + remap(s["tokens"]) + [EOS])
    features.append(derive_features(s))

max_len = max(len(seq) for seq in seqs)
print(f"max sequence length: {max_len}")

# Pad sequences.
def pad(seq, L):
    return seq + [PAD] * (L - len(seq))

x = torch.tensor([pad(seq, max_len) for seq in seqs], dtype=torch.long)
feats = torch.tensor(features, dtype=torch.float32)

# ---------------------------------------------------------------- model
class SeedExpander(nn.Module):
    """Tiny conditional GRU. Seed features → initial hidden state
    via a 1-layer MLP. Then GRU decodes the token sequence."""

    def __init__(self, feat_dim, vocab_size, hidden=64, embed=32):
        super().__init__()
        self.cond = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.embed = nn.Embedding(vocab_size, embed)
        self.gru = nn.GRU(embed, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, seed_feats, input_tokens):
        # seed_feats: (B, F); input_tokens: (B, T)
        h0 = self.cond(seed_feats).unsqueeze(0)  # (1, B, hidden)
        emb = self.embed(input_tokens)            # (B, T, embed)
        out, _ = self.gru(emb, h0)
        return self.out(out)                      # (B, T, vocab)

    @torch.no_grad()
    def decode_greedy(self, seed_feats, max_len, bos=BOS, eos=EOS):
        h = self.cond(seed_feats).unsqueeze(0)
        device = seed_feats.device
        tokens = [bos]
        for _ in range(max_len - 1):
            inp = torch.tensor([[tokens[-1]]], device=device)
            emb = self.embed(inp)
            out, h = self.gru(emb, h)
            next_tok = int(self.out(out[:, -1]).argmax(-1).item())
            tokens.append(next_tok)
            if next_tok == eos:
                break
        return tokens

model = SeedExpander(5, vocab_size, hidden=64, embed=32)
opt = optim.Adam(model.parameters(), lr=3e-3)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params:,} params")

# ---------------------------------------------------------------- train
print("\nTraining…")
N = x.size(0)
B = 16
epochs = 600
for epoch in range(epochs):
    model.train()
    total = 0.0
    # Shuffle.
    perm = torch.randperm(N)
    for i in range(0, N, B):
        batch_idx = perm[i:i+B]
        bx = x[batch_idx]
        bf = feats[batch_idx]
        # Teacher forcing: input = seq[:-1], target = seq[1:]
        inp = bx[:, :-1]
        tgt = bx[:, 1:]
        logits = model(bf, inp)
        loss = loss_fn(logits.reshape(-1, vocab_size), tgt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * bx.size(0)
    if (epoch + 1) % 50 == 0:
        print(f"  epoch {epoch+1:4d}  avg loss {total/N:.4f}")

# ---------------------------------------------------------------- eval
print("\nEvaluating reconstruction on training set…")
model.eval()
exact_match = 0
near_match = 0
prefix_avg = 0.0
results = []
for i, s in enumerate(samples):
    feat = feats[i:i+1]
    decoded = model.decode_greedy(feat, max_len)
    target = seqs[i]
    # Compare ignoring trailing PAD; strip BOS, stop at EOS.
    def strip(seq):
        out = []
        for t in seq[1:]:  # drop BOS
            if t == EOS or t == PAD:
                break
            out.append(t)
        return out
    dec_stripped = strip(decoded)
    tgt_stripped = strip(target)
    is_exact = dec_stripped == tgt_stripped
    # Common prefix length.
    n = min(len(dec_stripped), len(tgt_stripped))
    p = 0
    while p < n and dec_stripped[p] == tgt_stripped[p]:
        p += 1
    prefix_ratio = p / max(1, len(tgt_stripped))
    if is_exact:
        exact_match += 1
    if prefix_ratio > 0.8:
        near_match += 1
    prefix_avg += prefix_ratio
    results.append({
        "idx": i,
        "canonical": s["canonical"],
        "exact": is_exact,
        "prefix_ratio": prefix_ratio,
        "target_len": len(tgt_stripped),
        "decoded_len": len(dec_stripped),
    })

print(f"\n=== Results ===")
print(f"  exact-match  : {exact_match}/{N}  ({100*exact_match/N:.1f}%)")
print(f"  ≥80% prefix  : {near_match}/{N}  ({100*near_match/N:.1f}%)")
print(f"  mean prefix  : {prefix_avg/N:.3f}")

out = {
    "n_samples": N,
    "vocab_size": vocab_size,
    "n_params": n_params,
    "epochs": epochs,
    "exact_match": exact_match,
    "near_match": near_match,
    "mean_prefix_ratio": prefix_avg / N,
    "per_sample": results,
}
with open(OUTPUT, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nWrote {OUTPUT}")

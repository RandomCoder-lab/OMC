#!/usr/bin/env python3
"""Sanity: actually decode some samples back to OMC source and print
the source-level comparison."""

import json
from pathlib import Path

# Quick: same model in-place.
import torch, torch.nn as nn, math
torch.manual_seed(7)

ED = Path(__file__).parent
samples = [json.loads(l) for l in open(ED / "corpus.jsonl") if l.strip()]

PAD, BOS, EOS = 0, 1, 2
RESERVED = 3
observed = set()
for s in samples:
    observed.update(s["tokens"])
id_map = {old: new + RESERVED for new, old in enumerate(sorted(observed))}
inv_id_map = {v: k for k, v in id_map.items()}
vocab_size = RESERVED + len(observed)

# Load token vocab from OMC binary for decoding.
import subprocess
def get_vocab():
    # Inline OMC: print token vocab as JSON.
    code = "print(json_stringify(omc_token_vocab()));"
    p = subprocess.run(
        ["./target/release/omnimcode-standalone", "/dev/stdin"],
        input=code, capture_output=True, text=True, cwd="/home/thearchitect/OMC",
        env={"PYO3_USE_ABI3_FORWARD_COMPATIBILITY": "1", "PATH": "/usr/bin"},
    )
    if p.returncode != 0:
        raise RuntimeError(p.stderr)
    return json.loads(p.stdout.strip())

token_vocab = get_vocab()
print(f"Token vocab size: {len(token_vocab)}")

# Decode an OMC token-id sequence back to source.
def decode_omc_tokens(ids):
    out = []
    i = 0
    while i < len(ids):
        t = ids[i]
        if t == 0 and i + 1 < len(ids):
            out.append(chr(ids[i+1] & 0xff))
            i += 2
        elif 0 <= t < len(token_vocab):
            out.append(token_vocab[t])
            i += 1
        else:
            i += 1
    return "".join(out)

# Reload the trained v2 model. Easiest: re-train (fast on CPU) and decode.
# Reproducing the architecture inline.
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

seqs = [[BOS] + [id_map[t] for t in s["tokens"]] + [EOS] for s in samples]
feats = [features(s) for s in samples]
max_len = max(len(seq) for seq in seqs)

def pad(s, L): return s + [PAD] * (L - len(s))
x = torch.tensor([pad(seq, max_len) for seq in seqs], dtype=torch.long)
F = torch.tensor(feats, dtype=torch.float32)

# Train (same hyperparams).
model = Expander(16, vocab_size)
import torch.optim as optim
opt = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1500)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
N, B = x.size(0), 16
print("Re-training (quick)…")
for epoch in range(1500):
    model.train()
    perm = torch.randperm(N)
    for i in range(0, N, B):
        idx = perm[i:i+B]
        logits = model(F[idx], x[idx, :-1])
        loss = loss_fn(logits.reshape(-1, vocab_size), x[idx, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    sched.step()

# Decode + show source.
print("\n=== SOURCE-LEVEL RECONSTRUCTION ===")
model.eval()
for i in [0, 5, 10, 15, 25, 30, 45]:
    s = samples[i]
    decoded_seq = model.decode(F[i:i+1], max_len)
    # Strip BOS/EOS/PAD, map back to OMC token IDs, then decode to text.
    dec_omc = []
    for t in decoded_seq[1:]:
        if t in (EOS, PAD): break
        if t in inv_id_map:
            dec_omc.append(inv_id_map[t])
    dec_text = decode_omc_tokens(dec_omc)
    print(f"\n--- sample {i} ---")
    print(f"ORIGINAL : {s['canonical']}")
    print(f"DECODED  : {dec_text}")
    print(f"MATCH    : {s['canonical'] == dec_text}")

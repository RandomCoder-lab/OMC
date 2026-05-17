#!/usr/bin/env python3
"""
v4: token-sampled seed — the user's reframing.

Instead of compressing a function into a 16-dim seed (no info), use
1/N of the canonical token stream as the seed. The model is given a
sparse partial sequence and asked to fill in the gaps.

This is essentially masked-sequence completion, which is a much
easier task than "generate from seed alone" because the model can
leverage local context (which is shared between train and held-out).

Compression: original tokens / sampled tokens = N. Combined with the
existing tokenizer's 2-3x, total is ~2N-3N from source.
"""

import json, math, random
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim

torch.manual_seed(11)
random.seed(11)

ED = Path(__file__).parent
samples = [json.loads(l) for l in open(ED / "corpus_structural.jsonl") if l.strip()]
random.shuffle(samples)
TRAIN, TEST = samples[:40], samples[40:]

PAD, BOS, EOS, MASK, RESERVED = 0, 1, 2, 3, 4

# Build full vocab from TRAIN.
observed = set()
for s in TRAIN:
    observed.update(s["tokens"])
id_map = {old: new + RESERVED for new, old in enumerate(sorted(observed))}
inv_id_map = {v: k for k, v in id_map.items()}
vocab_size = RESERVED + len(observed)
print(f"vocab from TRAIN: {vocab_size}")

def remap(tokens):
    """Remap with UNK→PAD for tokens not in vocab."""
    return [id_map.get(t, PAD) for t in tokens]

SAMPLE_N = 2  # keep every Nth token

def make_pair(s):
    """Return (input_with_masks, target). Input keeps every Nth token,
    rest are MASK. Target is full sequence."""
    full = remap(s["tokens"])
    inp = []
    for i, t in enumerate(full):
        if i % SAMPLE_N == 0:
            inp.append(t)
        else:
            inp.append(MASK)
    return inp, full

def make_seed(s):
    """The sparse seed: just the kept tokens (1/N of full)."""
    full = remap(s["tokens"])
    return [t for i, t in enumerate(full) if i % SAMPLE_N == 0]

inputs_tr = [make_pair(s)[0] for s in TRAIN]
targets_tr = [make_pair(s)[1] for s in TRAIN]
inputs_te = [make_pair(s)[0] for s in TEST]
targets_te = [make_pair(s)[1] for s in TEST]

# Stats on the compression ratio.
import statistics
orig_lens = [len(s["tokens"]) for s in samples]
sampled_lens = [len(make_seed(s)) for s in samples]
print(f"original tokens: mean={statistics.mean(orig_lens):.1f}, max={max(orig_lens)}")
print(f"sampled (1/{SAMPLE_N}): mean={statistics.mean(sampled_lens):.1f}, max={max(sampled_lens)}")
print(f"effective compression vs source bytes: ~{statistics.mean(orig_lens)/statistics.mean(sampled_lens) * 2.5:.1f}x")

max_len = max(max(len(seq) for seq in inputs_tr), max(len(seq) for seq in inputs_te))

def pad(s, L): return s + [PAD] * (L - len(s))
x_in_tr = torch.tensor([pad(seq, max_len) for seq in inputs_tr], dtype=torch.long)
x_tgt_tr = torch.tensor([pad(seq, max_len) for seq in targets_tr], dtype=torch.long)
x_in_te = torch.tensor([pad(seq, max_len) for seq in inputs_te], dtype=torch.long)
x_tgt_te = torch.tensor([pad(seq, max_len) for seq in targets_te], dtype=torch.long)

class MaskedSeq2Seq(nn.Module):
    """Encoder reads the partial sequence; decoder produces the full
    sequence. Both are GRUs over the same vocab."""
    def __init__(self, vocab, hidden=128, embed=64, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed)
        self.enc = nn.GRU(embed, hidden, num_layers=layers, batch_first=True, bidirectional=True)
        # Bridge bidirectional encoder -> unidirectional decoder.
        self.bridge = nn.Linear(hidden * 2 * layers, hidden * layers)
        self.dec = nn.GRU(embed, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, vocab)
        self.hidden = hidden; self.layers = layers
    def forward(self, inp_partial, inp_tgt):
        B = inp_partial.size(0)
        e1 = self.embed(inp_partial)
        _, h_enc = self.enc(e1)
        # h_enc: (layers*2, B, hidden) -> reshape -> bridge -> (layers, B, hidden)
        h_enc = h_enc.permute(1, 0, 2).reshape(B, -1)  # (B, layers*2*hidden)
        h_dec = self.bridge(h_enc).reshape(B, self.layers, self.hidden).transpose(0, 1).contiguous()
        e2 = self.embed(inp_tgt)
        out, _ = self.dec(e2, h_dec)
        return self.out(out)
    @torch.no_grad()
    def decode(self, inp_partial, L):
        B = inp_partial.size(0)
        e1 = self.embed(inp_partial)
        _, h_enc = self.enc(e1)
        h_enc = h_enc.permute(1, 0, 2).reshape(B, -1)
        h = self.bridge(h_enc).reshape(B, self.layers, self.hidden).transpose(0, 1).contiguous()
        toks = [BOS]
        for _ in range(L - 1):
            inp = torch.tensor([[toks[-1]]])
            out, h = self.dec(self.embed(inp), h)
            t = int(self.out(out[:, -1]).argmax(-1).item())
            toks.append(t)
            if t == EOS: break
        return toks

# Training: input partial, target is full sequence (with BOS shift).
def with_bos(seqs):
    return [[BOS] + s + [EOS] for s in seqs]

shifted_tr = with_bos([list(seq) for seq in targets_tr])
shifted_te = with_bos([list(seq) for seq in targets_te])
max_len_full = max(max(len(s) for s in shifted_tr), max(len(s) for s in shifted_te))

x_tgt_tr_shift = torch.tensor([pad(s, max_len_full) for s in shifted_tr], dtype=torch.long)
x_tgt_te_shift = torch.tensor([pad(s, max_len_full) for s in shifted_te], dtype=torch.long)

# Re-pad partial inputs to max_len_full too so encoder sees the same width.
x_in_tr = torch.tensor([pad(seq, max_len_full) for seq in inputs_tr], dtype=torch.long)
x_in_te = torch.tensor([pad(seq, max_len_full) for seq in inputs_te], dtype=torch.long)

model = MaskedSeq2Seq(vocab_size)
opt = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1500)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
N, B = x_in_tr.size(0), 16
print(f"params: {sum(p.numel() for p in model.parameters()):,}")
print("training v4 (token-sampled seq2seq)...")

for epoch in range(1500):
    model.train()
    perm = torch.randperm(N)
    total = 0.0
    for i in range(0, N, B):
        idx = perm[i:i+B]
        # Teacher forcing: decoder input = target[:-1], target = target[1:]
        logits = model(x_in_tr[idx], x_tgt_tr_shift[idx, :-1])
        loss = loss_fn(logits.reshape(-1, vocab_size), x_tgt_tr_shift[idx, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * idx.size(0)
    sched.step()
    if (epoch + 1) % 200 == 0:
        print(f"  epoch {epoch+1:4d} loss {total/N:.4f}")

model.eval()
def eval_set(in_x, tgt_seqs, samples_list, name):
    exact, near, prefix_sum = 0, 0, 0.0
    misses = []
    hits = []
    for i in range(len(tgt_seqs)):
        decoded = model.decode(in_x[i:i+1], max_len_full)
        target = tgt_seqs[i]
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
        if is_exact:
            exact += 1
            if name == "HELD-OUT":
                hits.append(samples_list[i]["canonical"])
        elif name == "HELD-OUT":
            misses.append((samples_list[i]["canonical"], pr))
        if pr > 0.8: near += 1
        prefix_sum += pr
    total = len(tgt_seqs)
    print(f"{name}: exact={exact}/{total} ({100*exact/total:.1f}%)  "
          f">=80%prefix={near}/{total} ({100*near/total:.1f}%)  "
          f"mean_prefix={prefix_sum/total:.3f}")
    return exact, near, prefix_sum / total, misses, hits

print("\n=== Results ===")
tr_e, tr_n, tr_p, _, _ = eval_set(x_in_tr, shifted_tr, TRAIN, "TRAIN")
te_e, te_n, te_p, misses, hits = eval_set(x_in_te, shifted_te, TEST, "HELD-OUT")

if hits:
    print("\nHELD-OUT exact reconstructions:")
    for src in hits:
        print(f"  ✓ {src}")
if misses:
    print("\nHELD-OUT misses (partial reconstructions):")
    for src, pr in misses[:5]:
        print(f"  {pr:.2f}  {src}")

out = {
    "approach": "token-sampled seq2seq (1/N sparse input)",
    "sample_n": SAMPLE_N,
    "train_exact": tr_e, "train_n": x_in_tr.size(0),
    "test_exact": te_e, "test_n": x_in_te.size(0),
    "test_near": te_n,
    "test_mean_prefix": te_p,
    "vocab_size": vocab_size,
    "params": sum(p.numel() for p in model.parameters()),
}
with open(ED / "results_token_sampled.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nwrote {ED / 'results_token_sampled.json'}")

"""S-MOD α sweep on L1 multi-head transformer at TinyShakespeare scale.

Yesterday's S-MOD result used α=0.5 (untuned). Sweep over a small range
to find a stronger setting before committing to it as the production default.

α candidates: 0.0 (no S-MOD, vanilla softmax baseline), 0.1, 0.3, 0.5,
              1.0, 2.0

Single seed per α (cheap exploration; if a clear winner emerges, follow up
with 3+ seeds on top picks).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_4way import lcg, make_matrix, crt_pe, build_vocab
from torch_substrate_softmax import (
    AttentionL1_MH_Sub, BlockSub, ModelSub,
    softmax_smod, attractor_distance,
)


def softmax_smod_alpha(scores, alpha):
    if alpha == 0.0:
        return F.softmax(scores, dim=-1)
    base = F.softmax(scores, dim=-1)
    mod = 1.0 / (1.0 + alpha * attractor_distance(scores))
    out = base * mod
    return out / (out.sum(dim=-1, keepdim=True) + 1e-9)


# Patch the AttentionL1_MH_Sub forward to use a configurable alpha.
class AttentionAlpha(nn.Module):
    def __init__(self, d_model, n_heads, seq_len, seed, alpha):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.alpha = alpha
        s = seed + 11
        W_q, s = make_matrix(d_model, d_model, 0.3, s)
        W_v, s = make_matrix(d_model, d_model, 0.3, s)
        W_o, s = make_matrix(d_model, d_model, 0.3, s)
        self.W_q = nn.Parameter(W_q)
        self.W_v = nn.Parameter(W_v)
        self.W_o = nn.Parameter(W_o)
        pe_full = crt_pe(seq_len, d_model)
        pe_per_head = pe_full.view(seq_len, n_heads,
                                    self.d_head).transpose(0, 1)
        self.register_buffer("K_const_mh", pe_per_head)
        self.rng_state = s

    def forward(self, x):
        T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = (x @ self.W_q).view(T, H, dh).transpose(0, 1)
        v = (x @ self.W_v).view(T, H, dh).transpose(0, 1)
        k = self.K_const_mh
        scores = (q @ k.transpose(-2, -1)) / (dh ** 0.5)
        attn = softmax_smod_alpha(scores, self.alpha)
        out = attn @ v
        out = out.transpose(0, 1).contiguous().view(T, D)
        return out @ self.W_o


class BlockAlpha(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, seq_len, seed, alpha):
        super().__init__()
        self.attn = AttentionAlpha(d_model, n_heads, seq_len, seed, alpha)
        s = self.attn.rng_state
        self.ln1_g = nn.Parameter(torch.ones(d_model))
        self.ln1_b = nn.Parameter(torch.zeros(d_model))
        W_up, s = make_matrix(d_model, ff_dim, 0.3, s + 13)
        W_down, s = make_matrix(ff_dim, d_model, 0.3, s)
        self.ff_up = nn.Parameter(W_up)
        self.ff_up_b = nn.Parameter(torch.zeros(ff_dim))
        self.ff_down = nn.Parameter(W_down)
        self.ff_down_b = nn.Parameter(torch.zeros(d_model))
        self.ln2_g = nn.Parameter(torch.ones(d_model))
        self.ln2_b = nn.Parameter(torch.zeros(d_model))
        self.rng_state = s

    def forward(self, x):
        attn_out = self.attn(x)
        x_post_attn = x + attn_out
        normed1 = F.layer_norm(x_post_attn, (x.size(-1),),
                               weight=self.ln1_g, bias=self.ln1_b)
        up = normed1 @ self.ff_up + self.ff_up_b
        activated = F.relu(up)
        down = activated @ self.ff_down + self.ff_down_b
        x_post_ff = x_post_attn + down
        return F.layer_norm(x_post_ff, (x.size(-1),),
                            weight=self.ln2_g, bias=self.ln2_b)


class ModelAlpha(nn.Module):
    def __init__(self, vocab, d_model, n_heads, ff_dim, seq_len, n_blocks, seed, alpha):
        super().__init__()
        s = seed
        E, s = make_matrix(vocab, d_model, 0.3, s)
        self.embedding = nn.Parameter(E)
        self.register_buffer("pe_table", crt_pe(seq_len, d_model))
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            b = BlockAlpha(d_model, n_heads, ff_dim, seq_len,
                            s + 100 * (i + 1), alpha)
            self.blocks.append(b)
            s = b.rng_state
        W_head, _ = make_matrix(d_model, vocab, 0.3, s + 17)
        self.head = nn.Parameter(W_head)
        self.head_b = nn.Parameter(torch.zeros(vocab))

    def forward(self, token_ids):
        x = self.embedding[token_ids] + self.pe_table[:token_ids.size(0)]
        for b in self.blocks:
            x = b(x)
        return x @ self.head + self.head_b


def train_one(alpha, train_ids, val_ids, vocab_size, args, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    model = ModelAlpha(vocab_size, args.d_model, args.n_heads, args.ff_dim,
                       args.seq_len, args.n_blocks, seed, alpha)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                             betas=(0.9, 0.999), eps=1e-8)
    n_train, n_val = len(train_ids), len(val_ids)
    train_t = torch.tensor(train_ids, dtype=torch.long)
    val_t = torch.tensor(val_ids, dtype=torch.long)
    for step in range(args.steps):
        start = random.randint(0, n_train - args.seq_len - 2)
        w = train_t[start:start + args.seq_len]
        t = train_t[start + 1:start + 1 + args.seq_len]
        loss = F.cross_entropy(model(w), t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    vls = []
    with torch.no_grad():
        for _ in range(30):
            vs = random.randint(0, n_val - args.seq_len - 2)
            vw = val_t[vs:vs + args.seq_len]
            vt = val_t[vs + 1:vs + 1 + args.seq_len]
            vls.append(F.cross_entropy(model(vw), vt).item())
    return sum(vls) / len(vls)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--alphas", type=str, default="0.0,0.1,0.3,0.5,1.0,2.0")
    parser.add_argument("--out", type=str, default="results_torch_smod_alpha_sweep.json")
    args = parser.parse_args()

    corpus = (Path(__file__).parent.parent / "transformerless_lm"
              / "tinyshakespeare.txt").read_text()
    chars, lookup = build_vocab(corpus)
    vocab_size = len(chars)
    ids = [lookup[c] for c in corpus]
    split = int(len(ids) * 0.9)
    train_ids, val_ids = ids[:split], ids[split:]
    seeds = [int(s) for s in args.seeds.split(",")]
    alphas = [float(a) for a in args.alphas.split(",")]

    print(f"=== S-MOD α sweep on L1 multi-head @ TinyShakespeare ===")
    print(f"corpus={len(corpus):,} steps={args.steps} seeds={seeds}")
    print(f"alphas={alphas}\n", flush=True)

    results = {}
    for alpha in alphas:
        vals = []
        for seed in seeds:
            vm = train_one(alpha, train_ids, val_ids, vocab_size, args, seed)
            vals.append(vm)
            print(f"  α={alpha:.1f}  seed={seed}  val={vm:.4f}", flush=True)
        results[f"alpha={alpha}"] = {
            "alpha": alpha, "vals": vals,
            "mean": sum(vals) / len(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        }
        print(f"[α={alpha:.1f}] mean val={results[f'alpha={alpha}']['mean']:.4f}\n", flush=True)

    print("=== Sweep summary ===")
    base = results[f"alpha={alphas[0]}"]["mean"]
    print(f"{'α':>6}  {'mean val':>10}  {'vs α=0':>10}")
    for a in alphas:
        m = results[f"alpha={a}"]["mean"]
        rel = (m - base) / base * 100
        marker = "—" if a == alphas[0] else f"{rel:+.2f}%"
        print(f"{a:>6.1f}  {m:>10.4f}  {marker:>10}")

    # Find best.
    best_alpha = min(alphas, key=lambda a: results[f"alpha={a}"]["mean"])
    print(f"\nBest α: {best_alpha}  (val={results[f'alpha={best_alpha}']['mean']:.4f})")

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump({"results": results, "config": vars(args),
                   "best_alpha": best_alpha}, f, indent=2, default=float)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

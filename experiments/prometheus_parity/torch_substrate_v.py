"""Does S-MOD softmax rescue substrate-V?

Yesterday's finding: pure substrate-K (L1) wins, and S-MOD softmax
(α=1.0) wins on top. Substrate-V (L4) lost on its own when paired
with vanilla softmax — but the loss was attributed to off-attractor
attention amplifying off-attractor V components.

Hypothesis: with S-MOD softmax suppressing off-attractor attention,
a substrate-modulated V might recover. If so, it's a third
substrate-component win on the attention block.

Architecture (winning L1 multi-head + S-MOD α=1.0):
  Q  = learned per-head projection
  K  = CRT-Fibonacci substrate (frozen)
  V  = learned per-head projection
  softmax = S-MOD α=1.0
  output = learned per-head projection

Three V-variants tested:
  V0 (baseline): v = x @ W_v                          (current production)
  V1 (resample): v = substrate_resample(x @ W_v)      (post-projection snap)
  V2 (modulate): v = (x @ W_v) * (1 + γ·near_attractor_signal(x))
                                                      (input-conditional)

3 seeds on TinyShakespeare with S-MOD α=1.0.
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
    attractor_distance, nearest_attractor, softmax_smod,
    BlockSub, ModelSub,
)


def substrate_resample(x: torch.Tensor, scale: float = 10.0) -> torch.Tensor:
    """Snap-modulate: each component pulled toward nearest Fibonacci
    attractor. Returns x * 1/(1 + d/scale) where d = attractor_distance
    of (x * scale). Identity when x is already on an attractor."""
    scaled = x * scale
    d = attractor_distance(scaled)
    modulation = 1.0 / (1.0 + d / scale)
    return x * modulation


def near_attractor_signal(x: torch.Tensor, scale: float = 10.0) -> torch.Tensor:
    """Returns 1 / (1 + attractor_distance(x*scale)), in [0, 1].
    Close to 1 when x is near a Fibonacci attractor; close to 0 when
    far. Used as a per-component multiplicative gate."""
    return 1.0 / (1.0 + attractor_distance(x * scale))


class AttentionL1V(nn.Module):
    """L1 multi-head + S-MOD softmax + pluggable V variant."""
    def __init__(self, d_model, n_heads, seq_len, seed,
                 v_variant="V0", alpha=1.0, gamma=0.2):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.v_variant = v_variant
        self.alpha = alpha
        self.gamma = gamma
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
        v_proj = x @ self.W_v
        if self.v_variant == "V0":
            v_full = v_proj
        elif self.v_variant == "V1":
            v_full = substrate_resample(v_proj)
        elif self.v_variant == "V2":
            gate = near_attractor_signal(x)              # shape [T, D]
            v_full = v_proj * (1.0 + self.gamma * gate)
        else:
            raise ValueError(self.v_variant)
        v = v_full.view(T, H, dh).transpose(0, 1)
        k = self.K_const_mh
        scores = (q @ k.transpose(-2, -1)) / (dh ** 0.5)
        attn = softmax_smod(scores, dim=-1, alpha=self.alpha)
        out = attn @ v
        out = out.transpose(0, 1).contiguous().view(T, D)
        return out @ self.W_o


class BlockV(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, seq_len, seed,
                 v_variant, alpha, gamma):
        super().__init__()
        self.attn = AttentionL1V(d_model, n_heads, seq_len, seed,
                                  v_variant, alpha, gamma)
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


class ModelV(nn.Module):
    def __init__(self, vocab, d_model, n_heads, ff_dim, seq_len, n_blocks,
                 seed, v_variant, alpha, gamma):
        super().__init__()
        s = seed
        E, s = make_matrix(vocab, d_model, 0.3, s)
        self.embedding = nn.Parameter(E)
        self.register_buffer("pe_table", crt_pe(seq_len, d_model))
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            b = BlockV(d_model, n_heads, ff_dim, seq_len,
                       s + 100 * (i + 1), v_variant, alpha, gamma)
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


def train_one(v_variant, train_ids, val_ids, vocab_size, args, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    model = ModelV(vocab_size, args.d_model, args.n_heads, args.ff_dim,
                   args.seq_len, args.n_blocks, seed, v_variant,
                   args.alpha, args.gamma)
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
    parser.add_argument("--seeds", type=str, default="42,7,123")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--variants", type=str, default="V0,V1,V2")
    parser.add_argument("--out", type=str,
                         default="results_torch_substrate_v.json")
    args = parser.parse_args()

    corpus = (Path(__file__).parent.parent / "transformerless_lm"
              / "tinyshakespeare.txt").read_text()
    chars, lookup = build_vocab(corpus)
    vocab_size = len(chars)
    ids = [lookup[c] for c in corpus]
    split = int(len(ids) * 0.9)
    train_ids, val_ids = ids[:split], ids[split:]
    seeds = [int(s) for s in args.seeds.split(",")]
    variants = args.variants.split(",")

    print("=== Substrate-V on L1-MH + S-MOD softmax (TinyShakespeare) ===")
    print(f"variants={variants} seeds={seeds} steps={args.steps} "
          f"α={args.alpha} γ={args.gamma}\n", flush=True)

    results = {}
    for v in variants:
        vals = []
        for seed in seeds:
            vm = train_one(v, train_ids, val_ids, vocab_size, args, seed)
            vals.append(vm)
            print(f"  {v}  seed={seed}  val={vm:.4f}", flush=True)
        results[v] = {
            "vals": vals,
            "mean": sum(vals) / len(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        }
        print(f"[{v}] mean val={results[v]['mean']:.4f}  "
              f"std={results[v]['std']:.4f}\n", flush=True)

    print("=== Summary ===")
    base = results[variants[0]]["mean"]
    print(f"{'variant':>8}  {'mean val':>10}  {'std':>7}  {'vs V0':>8}")
    for v in variants:
        m = results[v]["mean"]
        rel = (m - base) / base * 100
        marker = "—" if v == variants[0] else f"{rel:+.2f}%"
        print(f"{v:>8}  {m:>10.4f}  {results[v]['std']:>7.4f}  {marker:>8}")
    best = min(variants, key=lambda v: results[v]["mean"])
    print(f"\nBest: {best}  ({results[best]['mean']:.4f})")

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump({"results": results, "config": vars(args),
                    "best": best}, f, indent=2, default=float)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

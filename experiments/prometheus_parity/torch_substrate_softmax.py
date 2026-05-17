"""Substrate-aware normalization variants vs vanilla softmax.

Softmax: exp(s_i - max(s)) / Σ exp(s_j - max(s)). Differentiable;
no learnable params; the de-facto attention normalization.

The substrate-aware question: is there a substrate-flavored
normalization that beats softmax on the L1 (substrate-K) architecture?

Three candidates tested (all element-wise, no learnable params):

  S-RANK    Sort scores by attractor distance of their values; assign
            geometric weights by rank. Closer-to-attractor → higher
            weight. Differentiability via straight-through estimator.

  S-MOD     Standard softmax × harmonic modulation. Each post-softmax
            weight gets multiplied by 1/(1 + α·attractor_distance(s_i)),
            then renormalized. Substrate dampens off-attractor scores;
            softmax handles the heavy lifting.

  S-SNAP    Softmax with score values pulled toward nearest Fibonacci
            attractor before exp. scores → scores + β·(attractor−scores)
            then standard softmax. Substrate-biases score values toward
            harmonic alignment, preserves full differentiability.

Compared against vanilla softmax baseline. Architecture: L1 (substrate-K)
multi-head transformer at TinyShakespeare scale, 3 seeds. If any
substrate variant beats vanilla, we have a second substrate replacement
to add to the scoreboard. If they all lose, softmax is genuinely the
right normalization at this layer and the substrate stays out of it.

Hypothesis: S-MOD or S-SNAP might help slightly via additional
substrate regularization; S-RANK likely loses because rank-based weights
break smooth gradients.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_4way import lcg, make_matrix, crt_pe, build_vocab


# Fibonacci attractor table (matches OMC's phi_pi_fib).
FIBS = torch.tensor([1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377,
                     610, 987, 1597, 2584, 4181, 6765, 10946],
                    dtype=torch.float)


def attractor_distance(x):
    """Distance to nearest Fibonacci attractor. Shape preserved."""
    abs_x = x.abs()
    diffs = (abs_x.unsqueeze(-1) - FIBS.to(x.device)).abs()
    return diffs.min(dim=-1).values


def nearest_attractor(x):
    """Snap-to-nearest Fibonacci attractor (signed)."""
    abs_x = x.abs()
    diffs = (abs_x.unsqueeze(-1) - FIBS.to(x.device)).abs()
    idx = diffs.argmin(dim=-1)
    sign = x.sign()
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return sign * FIBS.to(x.device)[idx]


# ---- Normalizations ----


def softmax_standard(scores, dim=-1):
    return F.softmax(scores, dim=dim)


def softmax_smod(scores, dim=-1, alpha=0.5):
    """S-MOD: standard softmax × 1/(1 + α·attractor_distance(score)),
    then renormalize. Off-attractor positions get dampened."""
    base = F.softmax(scores, dim=dim)
    mod = 1.0 / (1.0 + alpha * attractor_distance(scores))
    out = base * mod
    return out / (out.sum(dim=dim, keepdim=True) + 1e-9)


def softmax_ssnap(scores, dim=-1, beta=0.1):
    """S-SNAP: pull scores toward nearest attractor by β, then softmax."""
    snapped = scores + beta * (nearest_attractor(scores) - scores)
    return F.softmax(snapped, dim=dim)


def softmax_srank(scores, dim=-1):
    """S-RANK: assign weights by rank of attractor-distance.
    Closer-to-attractor → smaller rank → larger weight.
    Uses softmax over (-rank * γ) for differentiability with
    straight-through estimator (rank gradient ≈ score gradient)."""
    d = attractor_distance(scores)
    # Geometric weights: weight = φ^(-rank). Approximate ranking
    # via -d * 5.0 so larger-distance positions get more negative
    # logit. φ ≈ 1.618 → log φ ≈ 0.481; scale d by 0.481 / typical_d
    # so the spread matches softmax's natural temperature.
    phi_log = math.log(1.618033988749895)
    logits = -d * phi_log * 5.0
    # Bridge to scores so the gradient flows through scores: add a
    # tiny copy of scores so backward isn't all-zero.
    return F.softmax(0.5 * scores + logits, dim=dim)


# ---- L1 multi-head attention with pluggable normalization ----


class AttentionL1_MH_Sub(nn.Module):
    """Multi-head substrate-K (L1) with a pluggable score normalization."""
    def __init__(self, d_model, n_heads, seq_len, seed, normalize="softmax"):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
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
        self.normalize = normalize
        self.rng_state = s

    def forward(self, x):
        T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = (x @ self.W_q).view(T, H, dh).transpose(0, 1)
        v = (x @ self.W_v).view(T, H, dh).transpose(0, 1)
        k = self.K_const_mh
        scores = (q @ k.transpose(-2, -1)) / (dh ** 0.5)
        if self.normalize == "softmax":
            attn = softmax_standard(scores, dim=-1)
        elif self.normalize == "smod":
            attn = softmax_smod(scores, dim=-1)
        elif self.normalize == "ssnap":
            attn = softmax_ssnap(scores, dim=-1)
        elif self.normalize == "srank":
            attn = softmax_srank(scores, dim=-1)
        else:
            raise ValueError(self.normalize)
        out = attn @ v
        out = out.transpose(0, 1).contiguous().view(T, D)
        return out @ self.W_o


# ---- Transformer block + model (same as torch_multihead) ----


class BlockSub(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, seq_len, seed, normalize):
        super().__init__()
        self.attn = AttentionL1_MH_Sub(d_model, n_heads, seq_len, seed, normalize)
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


class ModelSub(nn.Module):
    def __init__(self, vocab, d_model, n_heads, ff_dim, seq_len, n_blocks,
                 seed, normalize):
        super().__init__()
        s = seed
        E, s = make_matrix(vocab, d_model, 0.3, s)
        self.embedding = nn.Parameter(E)
        self.register_buffer("pe_table", crt_pe(seq_len, d_model))
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            b = BlockSub(d_model, n_heads, ff_dim, seq_len,
                         s + 100 * (i + 1), normalize)
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


def train_with_val(normalize, train_ids, val_ids, vocab_size, seq_len,
                   d_model, n_heads, ff_dim, n_blocks, lr, steps, seed,
                   val_every=200, n_val_batches=30):
    torch.manual_seed(seed)
    random.seed(seed)
    model = ModelSub(vocab_size, d_model, n_heads, ff_dim, seq_len,
                     n_blocks, seed, normalize)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   betas=(0.9, 0.999), eps=1e-8)
    n_train = len(train_ids)
    n_val = len(val_ids)
    train_tensor = torch.tensor(train_ids, dtype=torch.long)
    val_tensor = torch.tensor(val_ids, dtype=torch.long)
    val_history = []
    train_tail = []
    for step in range(steps):
        start = random.randint(0, n_train - seq_len - 2)
        window = train_tensor[start:start + seq_len]
        targets = train_tensor[start + 1:start + 1 + seq_len]
        logits = model(window)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step >= steps - 50:
            train_tail.append(loss.item())
        if (step + 1) % val_every == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                vls = []
                for _ in range(n_val_batches):
                    vs = random.randint(0, n_val - seq_len - 2)
                    vw = val_tensor[vs:vs + seq_len]
                    vt = val_tensor[vs + 1:vs + 1 + seq_len]
                    vls.append(F.cross_entropy(model(vw), vt).item())
                val_history.append(sum(vls) / len(vls))
            model.train()
    train_mean = sum(train_tail) / len(train_tail)
    val_mean = val_history[-1]
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return train_mean, val_mean, n_params


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
    parser.add_argument("--out", type=str,
                        default="results_torch_substrate_softmax.json")
    args = parser.parse_args()

    corpus = (Path(__file__).parent.parent
              / "transformerless_lm" / "tinyshakespeare.txt").read_text()
    chars, lookup = build_vocab(corpus)
    vocab_size = len(chars)
    ids = [lookup[c] for c in corpus]
    split = int(len(ids) * 0.9)
    train_ids, val_ids = ids[:split], ids[split:]
    seeds = [int(s) for s in args.seeds.split(",")]
    variants = ["softmax", "smod", "ssnap", "srank"]

    print(f"=== L1 multi-head attention × substrate-softmax A/B ===")
    print(f"corpus: {len(corpus):,} chars; train {len(train_ids):,}; "
          f"val {len(val_ids):,}")
    print(f"vocab={vocab_size} seq={args.seq_len} d_model={args.d_model} "
          f"heads={args.n_heads} blocks={args.n_blocks}")
    print(f"steps={args.steps} lr={args.lr} seeds={seeds}\n", flush=True)

    results = {}
    for v in variants:
        train_means, val_means = [], []
        n_params = 0
        for seed in seeds:
            tm, vm, n_params = train_with_val(
                v, train_ids, val_ids, vocab_size, args.seq_len,
                args.d_model, args.n_heads, args.ff_dim, args.n_blocks,
                args.lr, args.steps, seed,
            )
            train_means.append(tm)
            val_means.append(vm)
            print(f"  [{v}] seed={seed} train={tm:.4f} val={vm:.4f}", flush=True)
        results[v] = {
            "train": train_means, "val": val_means, "n_params": n_params,
            "train_mean": sum(train_means) / len(train_means),
            "val_mean": sum(val_means) / len(val_means),
            "val_std": statistics.stdev(val_means) if len(val_means) > 1 else 0.0,
        }
        print(f"[{v}] params={n_params}  "
              f"train={results[v]['train_mean']:.4f}  "
              f"val={results[v]['val_mean']:.4f} "
              f"(std={results[v]['val_std']:.4f})\n", flush=True)

    print("=== Substrate-softmax vs vanilla verdict ===")
    base = results["softmax"]
    for v in variants:
        r = results[v]
        rel = (r["val_mean"] - base["val_mean"]) / base["val_mean"] * 100
        wins = sum(1 for x, b in zip(r["val"], base["val"]) if x < b)
        marker = "—" if v == "softmax" else f"{rel:+.2f}%"
        print(f"  {v:<8} val={r['val_mean']:.4f}  vs softmax: {marker:>8}  "
              f"wins={wins}/{len(base['val'])}")

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump({"results": results, "config": vars(args)}, f, indent=2, default=float)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

"""Broader substrate-Q sweep — different phi_pi_fib primitives.

The narrow Q sweep (torch_substrate_q.py) tested the same operation
as V (substrate_resample = post-projection snap to nearest Fibonacci
attractor) and lost. The user's hypothesis: maybe Q's role calls for
a DIFFERENT substrate primitive, not the same modulation pattern.

Tracks tested here:

  Q0 (baseline):       q = x @ W_q                              — control
  Q3 (pre-snap):       q = substrate_resample(x) @ W_q          — snap input, then project
  Q4 (boost-not-damp): q = (x @ W_q) * (1 + α / (1 + d))        — substrate boosts on-attractor
  Q5 (signed-snap):    q = (x @ W_q) + β · nearest_attractor    — additive substrate bias
  Q6 (log-scale):      q = (x @ W_q) * exp(-γ · log_phi_pi(|q|))— log-distance modulation

The principle from substrate-V: substrate metric applied to quantities
with integer-coherent structure helps; replacing learned projections
hurts. The question for Q: which (if any) phi_pi_fib operation
preserves Q's role as the attention-steerer while still leveraging
substrate structure?

If any variant wins, that's the v0.6.1 substrate-Q chapter.
If they all lose, the v0.1 stack is the architectural ceiling for
attention substrate composition.
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
from torch_substrate_softmax import (
    attractor_distance, nearest_attractor, softmax_smod,
)
from torch_substrate_v import substrate_resample


def phi_pi_log_distance(x: torch.Tensor, scale: float = 10.0) -> torch.Tensor:
    """Approximate log_phi_pi_fibonacci(|x|): scaled log-distance to
    the substrate. Closer to a Fibonacci attractor → smaller value.
    Used by Q6 as a multiplicative modulation."""
    abs_x = (x * scale).abs() + 1.0
    return abs_x.log() / (math.pi * math.log(1.618033988749895))


class AttentionL1QBroader(nn.Module):
    """L1 multi-head + S-MOD + V1, varying the Q recipe across broader
    phi_pi_fib primitives.
    """
    def __init__(self, d_model, n_heads, seq_len, seed,
                 q_variant="Q0", alpha=1.0, beta=0.1, gamma=0.5):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.q_variant = q_variant
        self.alpha = alpha
        self.beta = beta
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
        if self.q_variant == "Q0":
            q_full = x @ self.W_q
        elif self.q_variant == "Q3":
            # Pre-projection snap: snap the input, then project.
            q_full = substrate_resample(x) @ self.W_q
        elif self.q_variant == "Q4":
            # Boost-not-dampen: substrate AMPLIFIES on-attractor
            # components instead of dampening off-attractor ones.
            # Preserves the learned projection's variance.
            q_proj = x @ self.W_q
            d = attractor_distance(q_proj * 10.0)
            boost = 1.0 + self.alpha / (1.0 + d)
            q_full = q_proj * boost
        elif self.q_variant == "Q5":
            # Signed additive snap: small substrate-bias on top of
            # learned Q. Adds, not multiplies, so doesn't kill variance.
            q_proj = x @ self.W_q
            snap = nearest_attractor(q_proj * 10.0) / 10.0
            q_full = q_proj + self.beta * snap
        elif self.q_variant == "Q6":
            # Log-distance scaling: phi_pi_fib log metric instead of
            # the linear attractor distance. Different metric, different
            # gradient landscape.
            q_proj = x @ self.W_q
            log_d = phi_pi_log_distance(q_proj)
            modulation = (-self.gamma * log_d).exp()
            q_full = q_proj * modulation
        else:
            raise ValueError(self.q_variant)
        v_full = substrate_resample(x @ self.W_v)
        q = q_full.view(T, H, dh).transpose(0, 1)
        v = v_full.view(T, H, dh).transpose(0, 1)
        k = self.K_const_mh
        scores = (q @ k.transpose(-2, -1)) / (dh ** 0.5)
        attn = softmax_smod(scores, dim=-1, alpha=1.0)
        out = attn @ v
        out = out.transpose(0, 1).contiguous().view(T, D)
        return out @ self.W_o


class BlockQB(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, seq_len, seed,
                 q_variant, alpha, beta, gamma):
        super().__init__()
        self.attn = AttentionL1QBroader(d_model, n_heads, seq_len, seed,
                                         q_variant, alpha, beta, gamma)
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


class ModelQB(nn.Module):
    def __init__(self, vocab, d_model, n_heads, ff_dim, seq_len, n_blocks,
                 seed, q_variant, alpha, beta, gamma):
        super().__init__()
        s = seed
        E, s = make_matrix(vocab, d_model, 0.3, s)
        self.embedding = nn.Parameter(E)
        self.register_buffer("pe_table", crt_pe(seq_len, d_model))
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            b = BlockQB(d_model, n_heads, ff_dim, seq_len,
                        s + 100 * (i + 1), q_variant, alpha, beta, gamma)
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


def train_one(q_variant, train_ids, val_ids, vocab_size, args, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    model = ModelQB(vocab_size, args.d_model, args.n_heads, args.ff_dim,
                    args.seq_len, args.n_blocks, seed, q_variant,
                    args.alpha, args.beta, args.gamma)
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
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--variants", type=str, default="Q0,Q3,Q4,Q5,Q6")
    parser.add_argument("--out", type=str,
                         default="results_torch_substrate_q_broader.json")
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

    print("=== Substrate-Q BROADER sweep — different phi_pi_fib primitives ===")
    print(f"variants={variants} seeds={seeds} steps={args.steps}\n", flush=True)

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
    print(f"{'variant':>8}  {'mean val':>10}  {'std':>7}  {'vs Q0':>8}")
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

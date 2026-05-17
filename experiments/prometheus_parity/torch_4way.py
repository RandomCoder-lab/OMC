"""4-way attention A/B in PyTorch.

Reproduce the substrate-attention experiment from
examples/prometheus_attention_4way.omc. Same architecture, same
task, same seed semantics (LCG-ported init for fair comparison).

If PyTorch shows the same monotonic substrate-ladder result (L3 >
L2 > L1 > L0), the win is cross-framework. If it doesn't, the OMC
result was specific to our implementation.

Variants:
  L0: standard QKV (learned matrices)
  L1: K = CRT-PE (substrate), Q + V learned
  L2: K, Q = CRT-PE; V learned
  L3: K, Q = CRT-PE; V = identity (parameter-free attention block)
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- Reproduce OMC's LCG init for fair comparison ----

def lcg(state: int) -> int:
    return (state * 1103515245 + 12345) % 2147483648


def make_matrix(rows: int, cols: int, bound: float, state: int):
    """Bit-identical port of _prom_random_matrix from prometheus.omc."""
    m = torch.empty(rows, cols)
    s = state
    for i in range(rows):
        for j in range(cols):
            s = lcg(s)
            r = s / 2147483648.0
            m[i, j] = (r * 2.0 - 1.0) * bound
    return m, s


# ---- CRT-Fibonacci positional encoding (same moduli as OMC) ----

FIB_MODULI = [5, 8, 13, 21, 34, 55, 89, 144]


def crt_pe(seq_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(seq_len, d_model)
    n_pairs = d_model // 2
    for i in range(n_pairs):
        m = FIB_MODULI[i % len(FIB_MODULI)]
        for pos in range(seq_len):
            residue = pos % m
            angle = 2.0 * math.pi * residue / m
            pe[pos, 2 * i] = math.sin(angle)
            pe[pos, 2 * i + 1] = math.cos(angle)
    return pe


# ---- Attention variants ----


class AttentionL0(nn.Module):
    """Standard QKV — learned matrices."""
    def __init__(self, d_model: int, seq_len: int, seed: int):
        super().__init__()
        W_q, s = make_matrix(d_model, d_model, 0.3, seed + 11)
        W_k, s = make_matrix(d_model, d_model, 0.3, s)
        W_v, s = make_matrix(d_model, d_model, 0.3, s)
        self.W_q = nn.Parameter(W_q)
        self.W_k = nn.Parameter(W_k)
        self.W_v = nn.Parameter(W_v)
        self.rng_state = s

    def forward(self, x):
        q = x @ self.W_q
        k = x @ self.W_k
        v = x @ self.W_v
        scores = q @ k.T
        attn = F.softmax(scores, dim=-1)
        return attn @ v


class AttentionL1(nn.Module):
    """K = CRT-PE; Q + V learned."""
    def __init__(self, d_model: int, seq_len: int, seed: int):
        super().__init__()
        W_q, s = make_matrix(d_model, d_model, 0.3, seed + 11)
        W_v, s = make_matrix(d_model, d_model, 0.3, s)
        self.W_q = nn.Parameter(W_q)
        self.W_v = nn.Parameter(W_v)
        self.register_buffer("K_const", crt_pe(seq_len, d_model))
        self.rng_state = s

    def forward(self, x):
        q = x @ self.W_q
        v = x @ self.W_v
        k = self.K_const
        scores = q @ k.T
        attn = F.softmax(scores, dim=-1)
        return attn @ v


class AttentionL2(nn.Module):
    """K, Q = CRT-PE; only V learned."""
    def __init__(self, d_model: int, seq_len: int, seed: int):
        super().__init__()
        W_v, s = make_matrix(d_model, d_model, 0.3, seed + 11)
        self.W_v = nn.Parameter(W_v)
        pe = crt_pe(seq_len, d_model)
        self.register_buffer("K_const", pe)
        self.register_buffer("Q_const", pe)
        self.rng_state = s

    def forward(self, x):
        v = x @ self.W_v
        scores = self.Q_const @ self.K_const.T
        attn = F.softmax(scores, dim=-1)
        return attn @ v


class AttentionL3(nn.Module):
    """K, Q = CRT-PE; V = identity (parameter-free)."""
    def __init__(self, d_model: int, seq_len: int, seed: int):
        super().__init__()
        pe = crt_pe(seq_len, d_model)
        self.register_buffer("K_const", pe)
        self.register_buffer("Q_const", pe)
        self.rng_state = seed + 11

    def forward(self, x):
        scores = self.Q_const @ self.K_const.T
        attn = F.softmax(scores, dim=-1)
        return attn @ x


# ---- Full transformer block (same for all variants) ----


class TransformerModel(nn.Module):
    def __init__(self, variant: str, vocab: int, d_model: int, ff_dim: int,
                 seq_len: int, seed: int):
        super().__init__()
        s = seed
        E, s = make_matrix(vocab, d_model, 0.3, s)
        self.embedding = nn.Parameter(E)

        attn_cls = {"L0": AttentionL0, "L1": AttentionL1,
                    "L2": AttentionL2, "L3": AttentionL3}[variant]
        self.attn = attn_cls(d_model, seq_len, s)
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

        W_head, _ = make_matrix(d_model, vocab, 0.3, s + 17)
        self.head = nn.Parameter(W_head)
        self.head_b = nn.Parameter(torch.zeros(vocab))

        # Precompute CRT-PE for the embed-side position add.
        self.register_buffer("pe_table", crt_pe(seq_len, d_model))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [N]
        x = self.embedding[token_ids]                           # [N, d]
        x = x + self.pe_table[:x.size(0)]                       # add CRT-PE
        attn_out = self.attn(x)                                  # [N, d]
        x_post_attn = x + attn_out
        normed1 = F.layer_norm(x_post_attn, (x.size(-1),),
                               weight=self.ln1_g, bias=self.ln1_b)
        up = normed1 @ self.ff_up + self.ff_up_b
        activated = F.relu(up)
        down = activated @ self.ff_down + self.ff_down_b
        x_post_ff = x_post_attn + down
        normed2 = F.layer_norm(x_post_ff, (x.size(-1),),
                               weight=self.ln2_g, bias=self.ln2_b)
        return normed2 @ self.head + self.head_b                # [N, vocab]


# ---- Training loop ----


def build_vocab(text: str):
    chars = []
    lookup = {}
    for ch in text:
        if ch not in lookup:
            lookup[ch] = len(chars)
            chars.append(ch)
    return chars, lookup


def train_arm(variant: str, ids: list, vocab_size: int, seq_len: int,
              d_model: int, ff_dim: int, lr: float, steps: int, seed: int):
    torch.manual_seed(seed)
    model = TransformerModel(variant, vocab_size, d_model, ff_dim, seq_len, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    n_windows = len(ids) - seq_len - 1
    ids_tensor = torch.tensor(ids, dtype=torch.long)
    tail_losses = []
    for step in range(steps):
        start = step % n_windows
        window = ids_tensor[start:start + seq_len]
        targets = ids_tensor[start + 1:start + 1 + seq_len]
        logits = model(window)
        loss = F.cross_entropy(logits, targets, reduction="mean")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step >= steps - 10:
            tail_losses.append(loss.item())
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(tail_losses) / len(tail_losses), n_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="42,7,123")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--out", type=str, default="results_torch_4way.json")
    args = parser.parse_args()

    text = "the quick brown fox jumps over the lazy dog and the dog sleeps in the sun"
    chars, lookup = build_vocab(text)
    vocab_size = len(chars)
    ids = [lookup[c] for c in text]
    seq_len = 8
    d_model = 16
    ff_dim = 32
    seeds = [int(s) for s in args.seeds.split(",")]
    variants = ["L0", "L1", "L2", "L3"]

    print("=== PyTorch 4-way attention A/B ===")
    print(f"setup: corpus={len(text)} vocab={vocab_size} seq={seq_len} "
          f"d={d_model} ff={ff_dim}")
    print(f"  steps={args.steps} lr={args.lr} seeds={seeds}\n", flush=True)

    results = {}
    for v in variants:
        losses = []
        for seed in seeds:
            loss, n_params = train_arm(v, ids, vocab_size, seq_len,
                                        d_model, ff_dim, args.lr, args.steps, seed)
            losses.append(loss)
        results[v] = {"losses": losses, "n_params": n_params,
                      "mean": sum(losses) / len(losses),
                      "std": statistics.stdev(losses) if len(losses) > 1 else 0.0}
        print(f"[{v}] params={n_params:4d}  mean={results[v]['mean']:.4f}  "
              f"std={results[v]['std']:.4f}  per-seed={[f'{x:.3f}' for x in losses]}",
              flush=True)

    print("\n=== Summary vs L0 ===")
    base_mean = results["L0"]["mean"]
    base_losses = results["L0"]["losses"]
    for v in variants:
        wins = sum(1 for x, b in zip(results[v]["losses"], base_losses) if x < b)
        rel = (results[v]["mean"] - base_mean) / base_mean * 100
        marker = "—" if v == "L0" else f"{rel:+.1f}%"
        print(f"  {v}: mean={results[v]['mean']:.4f}  vs L0: {marker:>8}  "
              f"wins={wins}/{len(base_losses)}")

    print("\n=== Cross-framework comparison ===")
    print("OMC result (from examples/prometheus_attention_4way.omc):")
    print("  L0=2.576  L1=2.506 (-2.7%)  L2=2.157 (-16.3%)  L3=2.023 (-21.5%)")
    print("PyTorch result (this run):")
    for v in variants:
        print(f"  {v}={results[v]['mean']:.3f}", end="  ")
    print()

    # Verdict
    l0 = results["L0"]["mean"]
    l3 = results["L3"]["mean"]
    if l3 < l0:
        delta_pct = (l3 - l0) / l0 * 100
        print(f"\n[CROSS-FRAMEWORK WIN] L3 beats L0 by {delta_pct:.1f}% in PyTorch too.")
        print("  Substrate-as-attention-replacement validated across runtimes.")
    else:
        delta_pct = (l3 - l0) / l0 * 100
        print(f"\n[OMC-SPECIFIC] L3 LOSES to L0 by {delta_pct:.1f}% in PyTorch.")
        print("  OMC result didn't replicate — investigate runtime-specific factors.")

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump({
            "results": {k: {"losses": v["losses"], "n_params": v["n_params"],
                            "mean": v["mean"], "std": v["std"]}
                       for k, v in results.items()},
            "config": {"seeds": seeds, "steps": args.steps, "lr": args.lr,
                       "vocab": vocab_size, "d_model": d_model, "ff_dim": ff_dim,
                       "seq_len": seq_len},
        }, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

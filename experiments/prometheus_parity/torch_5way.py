"""5-way A/B: add L4 with substrate-derived V.

L3 keeps V = identity (input x passes through unchanged). L4 derives
V from a substrate function of x. If L4 beats L3, going further
beyond identity-V helps. If L3 still wins, identity already
captures everything useful.

Substrate V options tried here:
  L4a: V = harmonic_resample(x)
       Project each row through the Fibonacci attractor table
       (snap each component to nearest attractor / attractor_distance).
  L4b: V = x * crt_pe (element-wise modulated)

We test L4a — the cleanest substrate transform of x.
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

from torch_4way import (
    lcg, make_matrix, crt_pe,
    AttentionL0, AttentionL1, AttentionL2, AttentionL3,
    TransformerModel,
    build_vocab,
)


# Fibonacci attractor table (matches OMC's phi_pi_fib).
FIBS = torch.tensor([1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377,
                     610, 987, 1597, 2584, 4181, 6765, 10946], dtype=torch.float)


def attractor_distance(x: torch.Tensor) -> torch.Tensor:
    """For each scalar in x, return distance to nearest Fibonacci
    attractor."""
    abs_x = x.abs()
    diffs = (abs_x.unsqueeze(-1) - FIBS.to(x.device)).abs()
    return diffs.min(dim=-1).values


def substrate_resample(x: torch.Tensor) -> torch.Tensor:
    """Substrate transform: x → x * (1 - attractor_distance(scaled_x))
    Pulls each component toward its nearest Fibonacci attractor.
    Scaling factor 10 maps small float values into a useful range."""
    scaled = x * 10.0
    d = attractor_distance(scaled)
    # Closer to attractor → higher modulation (close to 1.0).
    modulation = 1.0 / (1.0 + d / 10.0)
    return x * modulation


class AttentionL4(nn.Module):
    """K, Q = CRT-PE; V = substrate_resample(x)."""
    def __init__(self, d_model: int, seq_len: int, seed: int):
        super().__init__()
        pe = crt_pe(seq_len, d_model)
        self.register_buffer("K_const", pe)
        self.register_buffer("Q_const", pe)
        self.rng_state = seed + 11

    def forward(self, x):
        scores = self.Q_const @ self.K_const.T
        attn = F.softmax(scores, dim=-1)
        v = substrate_resample(x)
        return attn @ v


# Quick test: a TransformerModel that uses L4.
class TransformerModelL4(TransformerModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the attn block with L4.
        seq_len = args[4] if len(args) > 4 else kwargs.get("seq_len")
        d_model = args[2] if len(args) > 2 else kwargs.get("d_model")
        seed = args[5] if len(args) > 5 else kwargs.get("seed")
        self.attn = AttentionL4(d_model, seq_len, seed)


def build_model(variant: str, vocab: int, d_model: int, ff_dim: int,
                seq_len: int, seed: int):
    if variant == "L4":
        return TransformerModelL4(variant="L3", vocab=vocab, d_model=d_model,
                                   ff_dim=ff_dim, seq_len=seq_len, seed=seed)
    return TransformerModel(variant=variant, vocab=vocab, d_model=d_model,
                             ff_dim=ff_dim, seq_len=seq_len, seed=seed)


def train_arm(variant, ids, vocab_size, seq_len, d_model, ff_dim, lr, steps, seed):
    torch.manual_seed(seed)
    model = build_model(variant, vocab_size, d_model, ff_dim, seq_len, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   betas=(0.9, 0.999), eps=1e-8)
    n_windows = len(ids) - seq_len - 1
    ids_tensor = torch.tensor(ids, dtype=torch.long)
    tail_losses = []
    for step in range(steps):
        start = step % n_windows
        window = ids_tensor[start:start + seq_len]
        targets = ids_tensor[start + 1:start + 1 + seq_len]
        logits = model(window)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step >= steps - 10:
            tail_losses.append(loss.item())
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(tail_losses) / len(tail_losses), n_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="42,7,123,2026,1")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--out", type=str, default="results_torch_5way.json")
    args = parser.parse_args()

    text = "the quick brown fox jumps over the lazy dog and the dog sleeps in the sun"
    chars, lookup = build_vocab(text)
    vocab_size = len(chars)
    ids = [lookup[c] for c in text]
    seq_len = 8
    d_model = 16
    ff_dim = 32
    seeds = [int(s) for s in args.seeds.split(",")]
    variants = ["L0", "L3", "L4"]

    print("=== 5-way A/B: does substrate-V (L4) beat identity-V (L3)? ===")
    print(f"setup: corpus={len(text)} vocab={vocab_size} seq={seq_len} "
          f"d={d_model} ff={ff_dim}")
    print(f"  steps={args.steps} lr={args.lr} seeds={seeds}\n", flush=True)

    results = {}
    for v in variants:
        losses = []
        n_params = 0
        for seed in seeds:
            loss, n_params = train_arm(v, ids, vocab_size, seq_len, d_model,
                                        ff_dim, args.lr, args.steps, seed)
            losses.append(loss)
        results[v] = {"losses": losses, "n_params": n_params,
                      "mean": sum(losses) / len(losses),
                      "std": statistics.stdev(losses) if len(losses) > 1 else 0.0}
        print(f"[{v}] params={n_params:4d}  mean={results[v]['mean']:.4f}  "
              f"std={results[v]['std']:.4f}  per-seed={[f'{x:.3f}' for x in losses]}",
              flush=True)

    print("\n=== Summary ===")
    l3_mean = results["L3"]["mean"]
    l4_mean = results["L4"]["mean"]
    l3_losses = results["L3"]["losses"]
    l4_wins = sum(1 for x, b in zip(results["L4"]["losses"], l3_losses) if x < b)
    rel = (l4_mean - l3_mean) / l3_mean * 100
    print(f"  L3 (identity-V): mean={l3_mean:.4f}")
    print(f"  L4 (substrate-V): mean={l4_mean:.4f}")
    print(f"  L4 vs L3: {rel:+.1f}%   wins={l4_wins}/{len(l3_losses)}")
    if l4_mean < l3_mean:
        print(f"  [L4 BEATS L3] Substrate V helps further.")
    else:
        print(f"  [L3 BEATS L4] Identity V is already optimal.")

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump({"results": results, "config": vars(args)}, f,
                  indent=2, default=float)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

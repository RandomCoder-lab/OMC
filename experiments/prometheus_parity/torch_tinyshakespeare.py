"""TinyShakespeare 4-way A/B (PyTorch).

The scale test. If the substrate-attention ranking holds on
1.1MB of real English, it's a paper-grade result, not a tiny-toy
artifact.

Setup:
  - TinyShakespeare corpus (~1.1MB, vocab ~65)
  - Single-block transformer (the regime where substrate-L3 won
    most decisively at 73-char scale)
  - Random windows from the full corpus
  - Larger d_model (32) for real-vocab work
  - More steps (1000) for real training
  - 5 seeds for stat
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

from torch_4way import (
    lcg, make_matrix, crt_pe,
    AttentionL0, AttentionL1, AttentionL2, AttentionL3,
    TransformerModel,
)


def load_corpus():
    p = Path(__file__).parent.parent / "transformerless_lm" / "tinyshakespeare.txt"
    return p.read_text()


def build_vocab(text: str):
    chars = sorted(set(text))
    lookup = {c: i for i, c in enumerate(chars)}
    return chars, lookup


def train_arm(variant: str, ids: torch.Tensor, vocab_size: int, seq_len: int,
              d_model: int, ff_dim: int, lr: float, steps: int, seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    model = TransformerModel(variant, vocab_size, d_model, ff_dim, seq_len, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   betas=(0.9, 0.999), eps=1e-8)
    n = len(ids)
    tail_losses = []
    for step in range(steps):
        # Random window from anywhere in the corpus.
        start = random.randint(0, n - seq_len - 2)
        window = ids[start:start + seq_len]
        targets = ids[start + 1:start + 1 + seq_len]
        logits = model(window)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step >= steps - 50:
            tail_losses.append(loss.item())
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(tail_losses) / len(tail_losses), n_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="42,7,123,2026,1")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--ff-dim", type=int, default=64)
    parser.add_argument("--out", type=str, default="results_torch_tinyshakespeare.json")
    args = parser.parse_args()

    text = load_corpus()
    chars, lookup = build_vocab(text)
    vocab_size = len(chars)
    ids = torch.tensor([lookup[c] for c in text], dtype=torch.long)
    seeds = [int(s) for s in args.seeds.split(",")]
    variants = ["L0", "L1", "L2", "L3"]

    print(f"=== TinyShakespeare 4-way A/B (PyTorch) ===")
    print(f"corpus: {len(text):,} chars, vocab={vocab_size}")
    print(f"setup: seq={args.seq_len} d={args.d_model} ff={args.ff_dim}")
    print(f"  steps={args.steps} lr={args.lr} seeds={seeds}\n", flush=True)

    results = {}
    for v in variants:
        losses = []
        n_params = 0
        for seed in seeds:
            loss, n_params = train_arm(v, ids, vocab_size, args.seq_len,
                                        args.d_model, args.ff_dim, args.lr,
                                        args.steps, seed)
            losses.append(loss)
            print(f"  [{v}] seed={seed} loss={loss:.4f}", flush=True)
        results[v] = {"losses": losses, "n_params": n_params,
                      "mean": sum(losses) / len(losses),
                      "std": statistics.stdev(losses) if len(losses) > 1 else 0.0}
        print(f"[{v}] params={n_params}  mean={results[v]['mean']:.4f}  "
              f"std={results[v]['std']:.4f}\n", flush=True)

    print("\n=== Summary vs L0 ===")
    base_mean = results["L0"]["mean"]
    base_losses = results["L0"]["losses"]
    for v in variants:
        wins = sum(1 for x, b in zip(results[v]["losses"], base_losses) if x < b)
        rel = (results[v]["mean"] - base_mean) / base_mean * 100
        marker = "—" if v == "L0" else f"{rel:+.1f}%"
        print(f"  {v}: mean={results[v]['mean']:.4f}  vs L0: {marker:>8}  "
              f"wins={wins}/{len(base_losses)}")

    print()
    l3_mean = results["L3"]["mean"]
    delta = (l3_mean - base_mean) / base_mean * 100
    if l3_mean < base_mean:
        print(f"[TinyShakespeare-SCALE WIN] L3 beats L0 by {delta:.1f}% on 1.1MB corpus.")
        print("  Substrate-as-attention-replacement holds at real-corpus scale.")
    else:
        print(f"[SCALE LIMIT FOUND] L3 LOSES to L0 by {delta:.1f}% at this scale.")
        print("  Substrate advantage is scale-bounded; investigate.")

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump({"results": results, "config": vars(args)}, f,
                  indent=2, default=float)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

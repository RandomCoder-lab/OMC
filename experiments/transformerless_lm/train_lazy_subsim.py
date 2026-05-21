"""Lazy training applied to FibGen seed components.

Two substrate-aligned variants tested:

  (1) LAZY_DROPOUT: Bernoulli mask on each FibGen seed component.
      keep_prob = 1/sqrt(tier) so low-tier (small Fibonacci index)
      components active near-always, high-tier components active
      stochastically. Eval rescales by keep_prob to match expected
      training magnitudes. This is "lazy loading at the seed level":
      each step uses only a substrate-defined subset of components.

  (2) TIER_LR_SCALE: keep all components active in the forward, but
      scale each component's GRADIENT by 1/sqrt(tier) before
      optimizer.step(). Low-tier components learn fast (full LR),
      high-tier learn slowly. Over training, low-tier components
      accumulate more signal. Deterministic, no train/eval mismatch.

Both share the substrate intent ("fold to respected tier") but
differ in implementation. We also include the pure-baseline Subsim
for direct comparison.

The deployment payoff (orthogonal to which training scheme wins):
post-training, prune high-tier components and measure perplexity
loss. The lazy-trained model should prune more gracefully because
high-tier components were either inactive (variant 1) or had small
learned magnitudes (variant 2).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from corpus import make_dataset
from models import make_model
from models_subsim import SubsimLM
from models_fibgen import FibGenLinear
from train_distractor_mix import build_distractor_stream
from lazy_data import fib_positions_in_window, get_fib_strided_batch


def evaluate(model, val_split, batch_size, window, fib_positions, generator,
              n_batches=16):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = get_fib_strided_batch(val_split, batch_size, window,
                                           fib_positions, generator)
            logits = model(x)
            losses.append(F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item())
    model.train()
    return sum(losses) / len(losses)


def apply_tier_lr_scale(model: torch.nn.Module):
    """For each FibGenLinear, multiply seed.grad by tier_lr_scale.
    Tier-1 components get full grad; tier-k get grad * 1/sqrt(k)."""
    for m in model.modules():
        if isinstance(m, FibGenLinear) and m.seed.grad is not None:
            m.seed.grad.mul_(m.tier_lr_scale)


def train_one(name, model, train_split, val_split, args, fib_positions,
               apply_lr_scale: bool = False):
    torch.manual_seed(args.seed)
    gen = torch.Generator(); gen.manual_seed(args.seed + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[train {name}] params={n_params:,}  "
          f"apply_lr_scale={apply_lr_scale}", flush=True)
    t0 = time.time()
    best_val = float("inf")
    best_step = -1
    eval_every = 200
    val_hist = []
    for step in range(args.steps):
        x, y = get_fib_strided_batch(train_split, args.batch_size, args.seq_len,
                                       fib_positions, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad(); loss.backward()
        if apply_lr_scale:
            apply_tier_lr_scale(model)
        optimizer.step()
        if step % eval_every == 0 or step == args.steps - 1:
            vl = evaluate(model, val_split, args.batch_size, args.seq_len,
                          fib_positions, gen)
            val_hist.append((step, vl, time.time() - t0))
            marker = ""
            if vl < best_val:
                best_val = vl
                best_step = step
                marker = " ← BEST"
            print(f"  step {step:5d}  val={vl:.4f}  ({time.time()-t0:.1f}s){marker}",
                  flush=True)
    return {"name": name, "n_params": n_params, "best_val": best_val,
             "best_step": best_step, "wall_time": time.time() - t0,
             "val_history": val_hist}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distractor-frac", type=float, default=0.20)
    parser.add_argument("--out", type=str, default="results_lazy_subsim.json")
    args = parser.parse_args()

    chars, stoi, itos, encoded = make_dataset(seq_len=args.seq_len,
                                                 source="tinyshakespeare")
    vocab_size = len(chars)
    train_split, val_split = build_distractor_stream(
        encoded, args.distractor_frac, args.seq_len, args.seed,
    )
    fib_positions = fib_positions_in_window(args.seq_len)

    results = {}

    # 1. Baseline Subsim (no lazy)
    m = SubsimLM(vocab_size=vocab_size, d_model=args.d_model,
                  n_blocks=args.n_blocks, seq_len=args.seq_len,
                  K=32, fibgen_K=32, mode="cross",
                  lazy_tier_dropout=False)
    results["subsim_baseline"] = train_one(
        "subsim_baseline", m, train_split, val_split, args, fib_positions,
    )

    # 2. Subsim + lazy seed dropout
    m = SubsimLM(vocab_size=vocab_size, d_model=args.d_model,
                  n_blocks=args.n_blocks, seq_len=args.seq_len,
                  K=32, fibgen_K=32, mode="cross",
                  lazy_tier_dropout=True)
    results["subsim_lazy_dropout"] = train_one(
        "subsim_lazy_dropout", m, train_split, val_split, args, fib_positions,
    )

    # 3. Subsim + tier-weighted gradient scaling
    m = SubsimLM(vocab_size=vocab_size, d_model=args.d_model,
                  n_blocks=args.n_blocks, seq_len=args.seq_len,
                  K=32, fibgen_K=32, mode="cross",
                  lazy_tier_dropout=False)
    results["subsim_tier_lr"] = train_one(
        "subsim_tier_lr", m, train_split, val_split, args, fib_positions,
        apply_lr_scale=True,
    )

    # Summary
    print()
    print("=" * 84)
    print(f"{'config':<24} {'params':>10} {'best_val':>10} {'best_step':>10} "
          f"{'wall':>10}")
    print("-" * 84)
    for name, r in results.items():
        print(f"{name:<24} {r['n_params']:>10,} {r['best_val']:>10.4f} "
              f"{r['best_step']:>10} {r['wall_time']:>9.1f}s")

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

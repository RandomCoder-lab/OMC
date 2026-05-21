"""Composed-fast-substrate bench: significantly faster training at d=128.

  baseline_dense       : dense_crt with lazy-loading data
  subsim_lazy_data     : Subsim (L1-dist attn + FibGen weights) with lazy data
  subsim_stofib_depth  : Subsim + Stochastic Fibonacci block depth (the
                          composed-fast variant — block i active with
                          probability 1/F(i+1) per step)

All three trained 2500 steps on the same data. Reports best-val
checkpoint, total wall time, and speedup vs dense_crt.

The user's "should be significantly faster" requirement: the
substrate-composed variant must beat dense in wall-clock on the
same hardware, not just match compute-FLOPs in theory.
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


def train_one(name, model, train_split, val_split, args, fib_positions):
    torch.manual_seed(args.seed)
    gen = torch.Generator(); gen.manual_seed(args.seed + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[train {name}] params={n_params:,}", flush=True)
    t0 = time.time()
    best_val = float("inf")
    best_step = -1
    val_hist = []
    for step in range(args.steps):
        x, y = get_fib_strided_batch(train_split, args.batch_size, args.seq_len,
                                       fib_positions, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                y.reshape(-1))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 250 == 0 or step == args.steps - 1:
            vl = evaluate(model, val_split, args.batch_size, args.seq_len,
                          fib_positions, gen)
            val_hist.append((step, vl, time.time() - t0))
            marker = ""
            if vl < best_val:
                best_val = vl; best_step = step
                marker = " ← BEST"
            print(f"  step {step:5d}  val={vl:.4f}  ({time.time()-t0:.1f}s){marker}",
                  flush=True)
    return {"name": name, "n_params": n_params, "best_val": best_val,
             "best_step": best_step, "wall": time.time() - t0,
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
    parser.add_argument("--out", type=str, default="results_fast_substrate.json")
    args = parser.parse_args()

    chars, stoi, itos, encoded = make_dataset(seq_len=args.seq_len,
                                                 source="tinyshakespeare")
    vocab_size = len(chars)
    train_split, val_split = build_distractor_stream(
        encoded, args.distractor_frac, args.seq_len, args.seed,
    )
    fib_positions = fib_positions_in_window(args.seq_len)

    results = {}

    results["baseline_dense"] = train_one(
        "baseline_dense",
        make_model("crt_only", vocab_size=vocab_size, seq_len=args.seq_len,
                    d_model=args.d_model, n_blocks=args.n_blocks),
        train_split, val_split, args, fib_positions,
    )

    results["subsim_lazy_data"] = train_one(
        "subsim_lazy_data",
        SubsimLM(vocab_size=vocab_size, d_model=args.d_model,
                  n_blocks=args.n_blocks, seq_len=args.seq_len,
                  K=32, fibgen_K=32, mode="cross"),
        train_split, val_split, args, fib_positions,
    )

    results["subsim_stofib_depth"] = train_one(
        "subsim_stofib_depth",
        SubsimLM(vocab_size=vocab_size, d_model=args.d_model,
                  n_blocks=args.n_blocks, seq_len=args.seq_len,
                  K=32, fibgen_K=32, mode="cross",
                  stochastic_fib_depth=True),
        train_split, val_split, args, fib_positions,
    )

    # Summary
    base = results["baseline_dense"]
    print()
    print("=" * 96)
    print(f"{'arch':<26} {'params':>10} {'best_val':>10} {'wall':>10} "
          f"{'speedup':>10}")
    print("-" * 96)
    for name, r in results.items():
        speedup = base["wall"] / r["wall"]
        print(f"{name:<26} {r['n_params']:>10,} {r['best_val']:>10.4f} "
              f"{r['wall']:>9.1f}s {speedup:>9.2f}x")

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

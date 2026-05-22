"""Progressive Fibonacci-K growth — substrate-aligned lazy training.

Start training with very few active Fibonacci frequencies per axis
(K_active = 3 or 4). Periodically expand K_active via Fibonacci
stepping (3 → 5 → 8 → 13 → 21 → 32) so the model's expressive
capacity grows over training.

Why this should give a real speedup that random K-subsampling didn't:
  - DETERMINISTIC schedule: each K-stage trains long enough to
    converge on its subset before expansion
  - PREFIX schedule: always activate the FIRST K_active indices —
    the smallest Fibonacci frequencies (lowest-tier in the substrate
    sense). Each expansion ADDS higher-tier components on top of a
    learned base
  - Per-stage compute is K²-quadratic in K_active for the inner mix;
    at K_active=4 the inner cost is 16/1024 = ~64x cheaper than full K
  - Outer projections shrink linearly with K_active

Bench:
  baseline_full     : K=32 from step 0 (~standard FibGen training)
  progressive_K     : Fibonacci-stepped K_active across stages
                       3 → 5 → 8 → 13 → 21 → 32

Both run for the same total step count. Reports wall-clock and best-
val. The substrate-lazy hypothesis: progressive matches or beats
baseline_full on val while running significantly faster.
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
from models_fibgen import FibGenLinear, FibGenLM
from train_distractor_mix import build_distractor_stream
from lazy_data import fib_positions_in_window, get_fib_strided_batch


def set_K_active_recursive(model: torch.nn.Module, K_active: int):
    """Walk the model and set K_active on every FibGenLinear."""
    for m in model.modules():
        if isinstance(m, FibGenLinear):
            m.set_K_active(K_active)


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


def train_progressive(name, model, schedule, train_split, val_split, args,
                       fib_positions):
    """schedule: list of (start_step, K_active). At each transition,
    set_K_active is called. End K_active = K_full means full capacity."""
    torch.manual_seed(args.seed)
    gen = torch.Generator(); gen.manual_seed(args.seed + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[train {name}] params={n_params:,}", flush=True)
    print(f"  K-schedule: {schedule}", flush=True)

    t0 = time.time()
    best_val = float("inf")
    best_step = -1
    val_hist = []
    cur_K = None
    sched_iter = iter(schedule)
    next_change = next(sched_iter, (args.steps + 1, None))
    for step in range(args.steps):
        # Advance schedule
        while step >= next_change[0]:
            new_K = next_change[1]
            if new_K != cur_K:
                set_K_active_recursive(model, new_K)
                cur_K = new_K
                print(f"  [step {step}] K_active -> {new_K}", flush=True)
            next_change = next(sched_iter, (args.steps + 1, None))

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
            print(f"    step {step:5d}  val={vl:.4f}  (K_active={cur_K})  "
                  f"({time.time()-t0:.1f}s){marker}", flush=True)
    return {"name": name, "n_params": n_params, "best_val": best_val,
             "best_step": best_step, "wall": time.time() - t0,
             "val_history": val_hist, "schedule": schedule}


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
    parser.add_argument("--K-full", type=int, default=32)
    parser.add_argument("--out", type=str, default="results_progressive_K.json")
    args = parser.parse_args()

    chars, stoi, itos, encoded = make_dataset(seq_len=args.seq_len,
                                                 source="tinyshakespeare")
    vocab_size = len(chars)
    train_split, val_split = build_distractor_stream(
        encoded, args.distractor_frac, args.seq_len, args.seed,
    )
    fib_positions = fib_positions_in_window(args.seq_len)

    # Use SubsimLM since it's the validated substrate operator
    def make_subsim():
        return SubsimLM(vocab_size=vocab_size, d_model=args.d_model,
                         n_blocks=args.n_blocks, seq_len=args.seq_len,
                         K=args.K_full, fibgen_K=args.K_full, mode="cross")

    results = {}

    # 1. Baseline: K_full from step 0 (effectively progressive at K_full only)
    full_schedule = [(0, args.K_full)]
    results["baseline_K32_full"] = train_progressive(
        "baseline_K32_full", make_subsim(), full_schedule,
        train_split, val_split, args, fib_positions,
    )

    # 2. Progressive Fibonacci K-stepping: 3 -> 5 -> 8 -> 13 -> 21 -> 32
    stages_K = [3, 5, 8, 13, 21, args.K_full]
    steps_per_stage = args.steps // len(stages_K)
    progressive_schedule = [(i * steps_per_stage, K)
                              for i, K in enumerate(stages_K)]
    results["progressive_fib"] = train_progressive(
        "progressive_fib", make_subsim(), progressive_schedule,
        train_split, val_split, args, fib_positions,
    )

    # 3. Reverse-progressive (sanity check: start big, shrink) — should
    #    LOSE to progressive if substrate-fold-to-tier-1 is the right intuition
    reverse_K = list(reversed(stages_K))
    reverse_schedule = [(i * steps_per_stage, K)
                          for i, K in enumerate(reverse_K)]
    results["reverse_progressive"] = train_progressive(
        "reverse_progressive", make_subsim(), reverse_schedule,
        train_split, val_split, args, fib_positions,
    )

    # Summary
    print()
    print("=" * 92)
    base_wall = results["baseline_K32_full"]["wall"]
    base_val = results["baseline_K32_full"]["best_val"]
    print(f"{'arch':<26} {'params':>10} {'best_val':>10} {'wall':>10} "
          f"{'speedup':>10} {'Δ val':>10}")
    print("-" * 92)
    for name, r in results.items():
        speedup = base_wall / r["wall"]
        dval = r["best_val"] - base_val
        print(f"{name:<26} {r['n_params']:>10,} {r['best_val']:>10.4f} "
              f"{r['wall']:>9.1f}s {speedup:>9.2f}x {dval:>+10.4f}")

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

"""K-shrink schedule — HIERARCHICAL substrate compression over training.

Per the user: large corpus -> model learns granular pieces at K=large
-> K shrinks, model picks best words -> K shrinks more, picks best
sentences -> K shrinks more, picks best paragraphs. Each K-tier
represents a level of linguistic abstraction; shrinking K FORCES
promotion to a more compressed representational tier.

Substrate-canonical mapping (for d=128 OMC, 4-block FibRecLM):
  K=89: granular char patterns / subword fragments
  K=55: word-level patterns
  K=34: phrase patterns
  K=21: sentence patterns
  K=13: paragraph patterns
  K=8:  discourse structure
  K=5:  high-level semantic skeleton

Substrate-canonical decay formula:
    K(t) = nearest_Fibonacci(K_init · φ^(−π · t / T_max))

For K_init=89, T_max=10000:
    step    0 →  K=89  (full capacity)
    step 2500 →  K=34
    step 5000 →  K=21
    step 7500 →  K=8
    step 10000 → K=5  (extreme compression)

The schedule walks through Fibonacci values, modulated by φ^π
(the substrate's canonical contraction ratio).

Bench:
  static_K5     : K=5 static throughout (the deployment target)
  static_K89    : K=89 static (reference: max capacity used during training)
  shrink_K      : K shrinks from 89 to 5 via φ^π schedule
                  Final K = 5, same deployment storage as static_K5

If shrink_K beats static_K5 in val loss at the same final K, the
substrate-auto-compression idea is validated: bigger temporary K
discovers structure that smaller fixed K can't find on its own.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from corpus import make_dataset
from models_fibrec import FibRecLM
from models_fibgen import FibGenLinear, FIBONACCI
from optimizers_fib import FibonacciAdamW
from train_distractor_mix import build_distractor_stream
from lazy_data import fib_positions_in_window, get_fib_strided_batch


PHI = (1 + math.sqrt(5)) / 2
PHI_PI = PHI ** math.pi


def K_schedule_substrate(step: int, max_steps: int,
                          K_init: int = 89, K_min: int = 3) -> int:
    """Substrate-canonical K decay.
        K(t) = nearest_Fibonacci(K_init · φ^(-π · t / max_steps))
    Snapped to the largest Fibonacci value <= raw K, with floor at K_min.
    """
    raw_K = K_init * (PHI ** (-math.pi * step / max_steps))
    # Find largest Fibonacci <= raw_K (so K only decreases)
    for k in reversed(FIBONACCI):
        if k <= raw_K and k >= K_min:
            return k
    return K_min


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


def set_K_active_recursive(model, K_a: int):
    for m in model.modules():
        if isinstance(m, FibGenLinear):
            m.set_K_active(K_a)


def train(name, model, optimizer, train_split, val_split, args, fib_positions,
           K_schedule_fn=None):
    """K_schedule_fn(step, max_steps) -> K_active (or None for static)."""
    torch.manual_seed(args.seed)
    gen = torch.Generator(); gen.manual_seed(args.seed + 1)
    n_params = sum(p.numel() for p in model.parameters())
    compr = model.storage_summary()["compression"]
    print(f"\n[train {name}] params={n_params:,}  compression={compr:.1f}x",
          flush=True)
    t0 = time.time()
    best_val = float("inf"); best_step = -1
    val_hist = []
    K_history = []
    eval_every = max(args.steps // 15, 250)
    cur_K = None
    for step in range(args.steps):
        if K_schedule_fn is not None:
            new_K = K_schedule_fn(step, args.steps)
            if new_K != cur_K:
                set_K_active_recursive(model, new_K)
                cur_K = new_K
                K_history.append((step, new_K))
        x, y = get_fib_strided_batch(train_split, args.batch_size, args.seq_len,
                                       fib_positions, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                y.reshape(-1))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % eval_every == 0 or step == args.steps - 1:
            vl = evaluate(model, val_split, args.batch_size, args.seq_len,
                          fib_positions, gen)
            val_hist.append((step, vl, time.time() - t0))
            marker = ""
            if vl < best_val:
                best_val = vl; best_step = step
                marker = " ← BEST"
            ktag = f" K={cur_K}" if cur_K is not None else ""
            print(f"  step {step:5d}  val={vl:.4f}{ktag}  ({time.time()-t0:.1f}s){marker}",
                  flush=True)
    # Final eval
    final_val = evaluate(model, val_split, args.batch_size, args.seq_len,
                          fib_positions, gen, n_batches=32)
    return {"name": name, "n_params": n_params, "compression": compr,
             "best_val": best_val, "best_step": best_step,
             "final_val": final_val, "wall": time.time() - t0,
             "K_history": K_history}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distractor-frac", type=float, default=0.20)
    parser.add_argument("--K-init", type=int, default=89)
    parser.add_argument("--K-min", type=int, default=5)
    parser.add_argument("--corpus", type=str, default="omc",
                        choices=["omc", "tinyshakespeare"])
    parser.add_argument("--out", type=str, default="results_K_shrink.json")
    args = parser.parse_args()

    chars, stoi, itos, encoded = make_dataset(seq_len=args.seq_len,
                                                 source=args.corpus)
    vocab_size = len(chars)
    train_split, val_split = build_distractor_stream(
        encoded, args.distractor_frac, args.seq_len, args.seed,
    )
    fib_positions = fib_positions_in_window(args.seq_len)

    print(f"K-shrink bench on OMC, d={args.d_model}, {args.steps} steps")
    print(f"K_init={args.K_init}, K_min={args.K_min}", flush=True)

    # Preview the K schedule
    print("\nK schedule preview:")
    for frac in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
        step = int(args.steps * frac)
        K = K_schedule_substrate(step, args.steps,
                                   K_init=args.K_init, K_min=args.K_min)
        print(f"  step {step:>5} ({frac*100:.0f}%): K={K}")

    results = {}

    # 1. Static K=89 (max capacity, reference)
    print("\n" + "=" * 60)
    print(f"static K={args.K_init} (max capacity reference)")
    print("=" * 60)
    m = FibRecLM(vocab_size=vocab_size, d_model=args.d_model,
                  n_blocks=args.n_blocks, seq_len=args.seq_len,
                  K=args.K_init, mode="cross")
    opt = FibonacciAdamW(m.parameters(), lr=args.lr)
    results[f"static_K{args.K_init}"] = train(
        f"static_K{args.K_init}", m, opt, train_split, val_split,
        args, fib_positions, K_schedule_fn=None)

    # 2. Static K=K_min (deployment target compression)
    print("\n" + "=" * 60)
    print(f"static K={args.K_min} (deployment target)")
    print("=" * 60)
    m = FibRecLM(vocab_size=vocab_size, d_model=args.d_model,
                  n_blocks=args.n_blocks, seq_len=args.seq_len,
                  K=args.K_min, mode="cross")
    opt = FibonacciAdamW(m.parameters(), lr=args.lr)
    results[f"static_K{args.K_min}"] = train(
        f"static_K{args.K_min}", m, opt, train_split, val_split,
        args, fib_positions, K_schedule_fn=None)

    # 3. Shrinking K (89 -> 5 via phi^pi schedule)
    print("\n" + "=" * 60)
    print(f"shrink K={args.K_init} -> {args.K_min} via phi^pi schedule")
    print("=" * 60)
    m = FibRecLM(vocab_size=vocab_size, d_model=args.d_model,
                  n_blocks=args.n_blocks, seq_len=args.seq_len,
                  K=args.K_init, mode="cross")  # init at K_init capacity
    opt = FibonacciAdamW(m.parameters(), lr=args.lr)
    sched = lambda s, T: K_schedule_substrate(s, T,
                                                 K_init=args.K_init,
                                                 K_min=args.K_min)
    results["shrink"] = train(
        f"shrink_K{args.K_init}_to_K{args.K_min}", m, opt, train_split,
        val_split, args, fib_positions, K_schedule_fn=sched)

    # Summary — reference dense baselines depend on corpus
    DENSE_REF = {"omc": 2.3586, "tinyshakespeare": 2.4396}
    DENSE_VAL = DENSE_REF.get(args.corpus, 2.4)
    print()
    print("=" * 84)
    print(f"Reference: dense_crt at d={args.d_model} {args.corpus} = val {DENSE_VAL}")
    print('-' * 84)
    print(f"{'config':<26} {'params':>10} {'best_val':>10} {'final_val':>10} "
          f"{'gap %':>10}")
    print('-' * 84)
    for name, r in results.items():
        gap = (r["best_val"] - DENSE_VAL) / DENSE_VAL * 100
        print(f"{name:<26} {r['n_params']:>10,} {r['best_val']:>10.4f} "
              f"{r['final_val']:>10.4f} {gap:>+9.1f}%")

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

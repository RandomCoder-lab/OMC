"""K-sweep on OMC at d=128 with 15K steps — does scaling K close the gap?

The 20K-step OMC bench showed substrate at K=32 plateaus at val 2.58
while dense reaches 2.36 (+9.4% gap). The hypothesis: K=32 has fixed
capacity (K²=1024 effective rank per layer) and that's insufficient
for the OMC corpus. If K scales WITH corpus complexity, the gap should
close.

Bench: FibRecLM + FibAdamW at K ∈ {32, 48, 64} on OMC at d=128.
15K steps each (long enough for substrate to plateau).
Plus reuse the dense baseline (best_val 2.36 at step 14K) for comparison.

Storage scaling at K (FibRecLM, d=128):
  K=32: seed ~50K + embed ~27K = 77K params (11.6x compression)
  K=48: seed ~110K + embed ~27K = 137K params (6.5x compression)
  K=64: seed ~195K + embed ~27K = 222K params (4.0x compression)

If gap shrinks with K, the K-scaling-with-d hypothesis is validated
and the substrate's path to LLM scale becomes "K grows as ~sqrt(d)
or similar." If gap stays at +9% regardless of K, the bottleneck is
elsewhere.
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
from models_fibrec import FibRecLM
from optimizers_fib import FibonacciAdamW
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


def train_one(name, model, optimizer, train_split, val_split, args,
               fib_positions):
    torch.manual_seed(args.seed)
    gen = torch.Generator(); gen.manual_seed(args.seed + 1)
    n_params = sum(p.numel() for p in model.parameters())
    compr = model.storage_summary()["compression"]
    print(f"\n[train {name}] params={n_params:,}  compression={compr:.1f}x",
          flush=True)
    t0 = time.time()
    best_val = float("inf"); best_step = -1
    val_hist = []
    eval_every = max(args.steps // 15, 250)
    for step in range(args.steps):
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
            print(f"  step {step:5d}  val={vl:.4f}  ({time.time()-t0:.1f}s){marker}",
                  flush=True)
    return {"name": name, "n_params": n_params, "compression": compr,
             "best_val": best_val, "best_step": best_step,
             "wall": time.time() - t0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distractor-frac", type=float, default=0.20)
    parser.add_argument("--K-values", type=str, default="32,48,64")
    parser.add_argument("--out", type=str, default="results_K_sweep.json")
    args = parser.parse_args()

    chars, stoi, itos, encoded = make_dataset(seq_len=args.seq_len, source="omc")
    vocab_size = len(chars)
    train_split, val_split = build_distractor_stream(
        encoded, args.distractor_frac, args.seq_len, args.seed,
    )
    fib_positions = fib_positions_in_window(args.seq_len)

    print(f"K-sweep on OMC, d={args.d_model}, {args.steps} steps")
    print(f"K values: {args.K_values}", flush=True)

    K_values = [int(x) for x in args.K_values.split(",")]
    results = {}

    for K in K_values:
        m = FibRecLM(vocab_size=vocab_size, d_model=args.d_model,
                      n_blocks=args.n_blocks, seq_len=args.seq_len,
                      K=K, mode="cross")
        opt = FibonacciAdamW(m.parameters(), lr=args.lr)
        results[f"K{K}"] = train_one(
            f"K{K}", m, opt, train_split, val_split, args, fib_positions)

    # Summary
    DENSE_VAL = 2.3586   # from previous 20K-step OMC bench
    print()
    print("=" * 84)
    print(f"Reference: dense_crt at d=128 OMC = val {DENSE_VAL} (step 14000)")
    print('-' * 84)
    print(f"{'K':<6} {'params':>10} {'compression':>12} {'best_val':>10} "
          f"{'gap %':>10}")
    print('-' * 84)
    for K in K_values:
        r = results[f"K{K}"]
        gap = (r["best_val"] - DENSE_VAL) / DENSE_VAL * 100
        print(f"{K:<6} {r['n_params']:>10,} {r['compression']:>11.1f}x "
              f"{r['best_val']:>10.4f} {gap:>+9.1f}%")

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

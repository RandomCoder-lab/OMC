"""Follow-up training bench for the three open questions after the
Pareto result.

  (A) Does the composed transformerless arch keep closing the gap with
      more training? At step 1500 it's at +5.6%. Run 4500 steps and
      check if the trajectory continues downward.

  (B) Does K need to scale with d? At d=256 K=32 lost +29.8%. Test
      K=48 (~sqrt(2)·32) and K=64 (=2·32) to see if higher K rescues
      the scale.

  (C) Does the composed arch keep its win at d=256? Run the
      FibGenTransformerless at d=256 and compare to dense_crt_d256.

Lazy-loading data by default.
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
from models_fibgen import FibGenLM, FibGenTransformerless
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


def train_one(name, make_fn, steps, vocab_size, train_split, val_split, args,
               fib_positions):
    torch.manual_seed(args.seed)
    gen = torch.Generator(); gen.manual_seed(args.seed + 1)
    model = make_fn()
    n_params = sum(p.numel() for p in model.parameters())
    ss = model.storage_summary() if hasattr(model, "storage_summary") else None
    compr_tag = f"  compression={ss['compression']:.1f}x" if ss else ""
    print(f"\n[{name}] params={n_params:,}{compr_tag}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    t0 = time.time()
    val_hist = []
    eval_every = max(steps // 10, 100)
    for step in range(steps):
        x, y = get_fib_strided_batch(train_split, args.batch_size, args.seq_len,
                                       fib_positions, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % eval_every == 0 or step == steps - 1:
            vl = evaluate(model, val_split, args.batch_size, args.seq_len,
                          fib_positions, gen)
            val_hist.append((step, vl, time.time() - t0))
            print(f"    step {step:5d}  val={vl:.4f}  ({time.time()-t0:.1f}s)",
                  flush=True)
    final = evaluate(model, val_split, args.batch_size, args.seq_len,
                      fib_positions, gen, n_batches=32)
    return {"name": name, "steps": steps, "n_params": n_params,
             "final_val": final, "wall_time": time.time() - t0,
             "val_history": val_hist}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distractor-frac", type=float, default=0.20)
    parser.add_argument("--out", type=str, default="results_followups.json")
    args = parser.parse_args()

    chars, stoi, itos, encoded = make_dataset(seq_len=args.seq_len,
                                                 source="tinyshakespeare")
    vocab_size = len(chars)
    train_split, val_split = build_distractor_stream(
        encoded, args.distractor_frac, args.seq_len, args.seed,
    )
    fib_positions = fib_positions_in_window(args.seq_len)

    results = {}

    # (A) Composed transformerless @ d=128, 4500 steps
    print("\n" + "=" * 70)
    print("(A) Composed transformerless @ d=128 — extended training (4500 steps)")
    print("=" * 70)
    results["composed_d128_4500steps"] = train_one(
        "composed_d128_4500steps",
        lambda: FibGenTransformerless(
            vocab_size=vocab_size, d_model=128, n_blocks=args.n_blocks,
            seq_len=args.seq_len, K=32, mode="cross", n_specialists=5,
        ),
        steps=4500, vocab_size=vocab_size, train_split=train_split,
        val_split=val_split, args=args, fib_positions=fib_positions,
    )

    # (B) K scaling at d=256
    print("\n" + "=" * 70)
    print("(B) K scaling at d=256 — does K=48 or K=64 rescue the scale gap?")
    print("=" * 70)
    for K in [48, 64]:
        results[f"fibgen_K{K}_cross_d256"] = train_one(
            f"fibgen_K{K}_cross_d256",
            lambda K=K: FibGenLM(vocab_size=vocab_size, d_model=256,
                                   n_blocks=args.n_blocks,
                                   seq_len=args.seq_len, K=K, mode="cross"),
            steps=1500, vocab_size=vocab_size, train_split=train_split,
            val_split=val_split, args=args, fib_positions=fib_positions,
        )

    # (C) Composed transformerless @ d=256
    print("\n" + "=" * 70)
    print("(C) Composed transformerless @ d=256 — does the win hold at scale?")
    print("=" * 70)
    results["composed_d256_1500steps"] = train_one(
        "composed_d256_1500steps",
        lambda: FibGenTransformerless(
            vocab_size=vocab_size, d_model=256, n_blocks=args.n_blocks,
            seq_len=args.seq_len, K=32, mode="cross", n_specialists=5,
        ),
        steps=1500, vocab_size=vocab_size, train_split=train_split,
        val_split=val_split, args=args, fib_positions=fib_positions,
    )

    # Summary
    print()
    print("=" * 92)
    print(f"{'config':<32} {'steps':>6} {'params':>10} {'val':>10} {'wall':>10}")
    print("-" * 92)
    for name, r in results.items():
        print(f"{name:<32} {r['steps']:>6} {r['n_params']:>10,} "
              f"{r['final_val']:>10.4f} {r['wall_time']:>9.1f}s")

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

"""FibGen Pareto exploration — three directions tested in one bench.

  (1) K-extension: cross-mode at K in {32, 48, 64} to test whether
      higher K closes the +6.3% gap further.
  (2) Scale test: d_model=256 to verify the Pareto holds at 4x scale.
      At d=4096 (LLM scale) the compression ratio grows as d^2/K^2,
      so if the loss penalty stays in single digits the substrate-
      generated weight basis scales positively.
  (3) Composed transformerless: FibGen weights + Fibonacci-offset
      attention + Zeckendorf-routed FFN. Tests whether stacking all
      the validated substrate primitives compounds or interferes.

Uses lazy-loading by default. dense_crt baselines at both d=128 and
d=256 for fair anchoring.
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
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            )
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train_one(name, make_fn, vocab_size, train_split, val_split, args,
               fib_positions):
    torch.manual_seed(args.seed)
    gen = torch.Generator(); gen.manual_seed(args.seed + 1)
    model = make_fn()
    n_params = sum(p.numel() for p in model.parameters())
    compr_tag = ""
    if hasattr(model, "storage_summary"):
        ss = model.storage_summary()
        compr_tag = f"  compression={ss['compression']:.1f}x"
    print(f"\n[{name}] params={n_params:,}{compr_tag}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    t0 = time.time()
    val_hist = []
    for step in range(args.steps):
        x, y = get_fib_strided_batch(train_split, args.batch_size, args.seq_len,
                                       fib_positions, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % args.eval_every == 0 or step == args.steps - 1:
            vl = evaluate(model, val_split, args.batch_size, args.seq_len,
                          fib_positions, gen)
            val_hist.append((step, vl, time.time() - t0))
            print(f"    step {step:5d}  val={vl:.4f}  ({time.time()-t0:.1f}s)",
                  flush=True)
    final = evaluate(model, val_split, args.batch_size, args.seq_len,
                      fib_positions, gen, n_batches=32)
    return {"name": name, "n_params": n_params, "final_val": final,
             "wall_time": time.time() - t0, "val_history": val_hist}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distractor-frac", type=float, default=0.20)
    parser.add_argument("--out", type=str, default="results_fibgen_pareto.json")
    args = parser.parse_args()

    chars, stoi, itos, encoded = make_dataset(seq_len=args.seq_len,
                                                 source="tinyshakespeare")
    vocab_size = len(chars)
    train_split, val_split = build_distractor_stream(
        encoded, args.distractor_frac, args.seq_len, args.seed,
    )
    fib_positions = fib_positions_in_window(args.seq_len)

    print(f"FibGen Pareto bench")
    print(f"Corpus: TinyShakespeare ({encoded.numel():,} chars, vocab {vocab_size})")
    print(f"Lazy-loading: {len(fib_positions)} positions / sequence", flush=True)

    results = {}

    # ============================================================
    # BASELINES + K EXTENSION at d=128
    # ============================================================
    print("\n" + "=" * 70)
    print("BLOCK 1: K-extension at d=128 (test whether higher K closes gap)")
    print("=" * 70)
    d = 128
    results[f"dense_crt_d{d}"] = train_one(
        f"dense_crt_d{d}",
        lambda d=d: make_model("crt_only", vocab_size=vocab_size,
                                seq_len=args.seq_len, d_model=d,
                                n_blocks=args.n_blocks),
        vocab_size, train_split, val_split, args, fib_positions,
    )
    for K in [32, 48, 64]:
        results[f"fibgen_K{K}_cross_d{d}"] = train_one(
            f"fibgen_K{K}_cross_d{d}",
            lambda K=K, d=d: FibGenLM(vocab_size=vocab_size, d_model=d,
                                         n_blocks=args.n_blocks,
                                         seq_len=args.seq_len, K=K, mode="cross"),
            vocab_size, train_split, val_split, args, fib_positions,
        )

    # ============================================================
    # SCALE TEST at d=256
    # ============================================================
    print("\n" + "=" * 70)
    print("BLOCK 2: scale test at d=256")
    print("=" * 70)
    d = 256
    results[f"dense_crt_d{d}"] = train_one(
        f"dense_crt_d{d}",
        lambda d=d: make_model("crt_only", vocab_size=vocab_size,
                                seq_len=args.seq_len, d_model=d,
                                n_blocks=args.n_blocks),
        vocab_size, train_split, val_split, args, fib_positions,
    )
    results[f"fibgen_K32_cross_d{d}"] = train_one(
        f"fibgen_K32_cross_d{d}",
        lambda d=d: FibGenLM(vocab_size=vocab_size, d_model=d,
                              n_blocks=args.n_blocks, seq_len=args.seq_len,
                              K=32, mode="cross"),
        vocab_size, train_split, val_split, args, fib_positions,
    )

    # ============================================================
    # COMPOSED transformerless candidate at d=128
    # ============================================================
    print("\n" + "=" * 70)
    print("BLOCK 3: composed transformerless candidate at d=128")
    print("=" * 70)
    results["transformerless_K32_cross"] = train_one(
        "transformerless_K32_cross",
        lambda: FibGenTransformerless(
            vocab_size=vocab_size, d_model=128, n_blocks=args.n_blocks,
            seq_len=args.seq_len, K=32, mode="cross", n_specialists=5,
        ),
        vocab_size, train_split, val_split, args, fib_positions,
    )

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 92)
    print(f"{'config':<32} {'params':>10} {'val':>10} {'wall':>10} "
          f"{'vs dense (same d)':>20}")
    print("-" * 92)
    # Dense baselines for comparison
    dense_vals = {128: results.get("dense_crt_d128", {}).get("final_val"),
                  256: results.get("dense_crt_d256", {}).get("final_val")}
    for name, r in results.items():
        d = 256 if "d256" in name else 128
        base = dense_vals.get(d) or 1.0
        gap = (r["final_val"] - base) / base * 100 if r["final_val"] > 0 else 0
        print(f"{name:<32} {r['n_params']:>10,} {r['final_val']:>10.4f} "
              f"{r['wall_time']:>9.1f}s {gap:>+18.1f}%")

    # Verdict for each block
    print()
    print("VERDICT:")
    print("\n  Block 1 — K-extension at d=128 (cross mode):")
    base = results["dense_crt_d128"]["final_val"]
    for K in [32, 48, 64]:
        r = results[f"fibgen_K{K}_cross_d128"]
        gap = (r["final_val"] - base) / base * 100
        print(f"    K={K:>3}: val={r['final_val']:.4f}  gap_vs_dense={gap:+5.1f}%")

    print("\n  Block 2 — scale to d=256:")
    base = results["dense_crt_d256"]["final_val"]
    r = results["fibgen_K32_cross_d256"]
    gap = (r["final_val"] - base) / base * 100
    print(f"    dense_crt_d256: val={base:.4f}")
    print(f"    fibgen_K32_cross_d256: val={r['final_val']:.4f}  gap={gap:+5.1f}%")

    print("\n  Block 3 — composed transformerless:")
    base = results["dense_crt_d128"]["final_val"]
    r = results["transformerless_K32_cross"]
    gap = (r["final_val"] - base) / base * 100
    print(f"    transformerless_K32_cross: val={r['final_val']:.4f}  gap={gap:+5.1f}%")

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

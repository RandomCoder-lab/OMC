"""FibGen training bench — can a 100x-compressed model learn anything?

The single question: does a model whose entire weight space is generated
from a small Fibonacci seed (8,064 params total vs 800K dense) train to
non-trivial loss? log(65) ≈ 4.17 is the uniform-random floor; anything
below that means the substrate basis is rich enough to capture some
structure.

Uses lazy_data.get_fib_strided_batch as the default loader (5.6x training
speedup, per the lazy-loading bench).

Comparisons:
  dense_crt   : standard crt_only baseline (~800K params, val ≈ 2.44)
  fibgen_K16  : 8K params, K=16 Fibonacci components per layer
  fibgen_K32  : 16K params, K=32 components (more capacity)
  fibgen_K8   : 4K params, K=8  (less capacity)

The K sweep tests how compression-vs-quality trades off.
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
from models_fibgen import FibGenLM, FibGenLinear
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


def train_one(arch_name, vocab_size, train_split, val_split, args, fib_positions,
               make_model_fn):
    torch.manual_seed(args.seed)
    gen = torch.Generator(); gen.manual_seed(args.seed + 1)
    model = make_model_fn()
    n_params = sum(p.numel() for p in model.parameters())
    if hasattr(model, "storage_summary"):
        ss = model.storage_summary()
        print(f"\n[arch={arch_name}] params={n_params:,}  "
              f"compression={ss['compression']:.1f}x  "
              f"(dense_equivalent={ss['dense_equivalent']:,})", flush=True)
    else:
        print(f"\n[arch={arch_name}] params={n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    t0 = time.time()
    val_hist = []
    P = len(fib_positions)
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
            print(f"    step {step:5d}  train={loss.item():.4f}  val={vl:.4f}  "
                  f"({time.time()-t0:.1f}s)", flush=True)
    final = evaluate(model, val_split, args.batch_size, args.seq_len,
                      fib_positions, gen, n_batches=32)
    return {
        "arch": arch_name,
        "n_params": n_params,
        "final_val": final,
        "wall_time": time.time() - t0,
        "val_history": val_hist,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distractor-frac", type=float, default=0.20)
    parser.add_argument("--K-sweep", type=str, default="8,16,32",
                        help="Comma-separated K values for FibGen.")
    parser.add_argument("--out", type=str, default="results_fibgen.json")
    args = parser.parse_args()

    chars, stoi, itos, encoded = make_dataset(
        seq_len=args.seq_len, source="tinyshakespeare",
    )
    vocab_size = len(chars)
    train_split, val_split = build_distractor_stream(
        encoded, args.distractor_frac, args.seq_len, args.seed,
    )

    fib_positions = fib_positions_in_window(args.seq_len)
    print(f"FibGen training bench (lazy-loading: {len(fib_positions)} tokens/seq)")
    print(f"Corpus: TinyShakespeare ({encoded.numel():,} chars, vocab {vocab_size})")
    print(f"Model: d={args.d_model}, n_blocks={args.n_blocks}, seq_len={args.seq_len}",
          flush=True)
    print(f"Random baseline (uniform over vocab): val = ln({vocab_size}) = "
          f"{torch.log(torch.tensor(float(vocab_size))).item():.4f}", flush=True)

    results = {}

    # 1. dense_crt baseline (with lazy loading too, for fair comparison)
    def make_crt():
        return make_model("crt_only", vocab_size=vocab_size,
                          seq_len=args.seq_len, d_model=args.d_model,
                          n_blocks=args.n_blocks)
    results["dense_crt"] = train_one("dense_crt", vocab_size, train_split,
                                       val_split, args, fib_positions, make_crt)

    # 2. FibGen at each K
    K_values = [int(k) for k in args.K_sweep.split(",")]
    for K in K_values:
        def make_fibgen(K=K):
            return FibGenLM(vocab_size=vocab_size, d_model=args.d_model,
                             n_blocks=args.n_blocks, seq_len=args.seq_len, K=K)
        results[f"fibgen_K{K}"] = train_one(
            f"fibgen_K{K}", vocab_size, train_split, val_split, args,
            fib_positions, make_fibgen,
        )

    # Summary
    print()
    print("=" * 90)
    print(f"{'arch':<14} {'params':>10} {'compr':>8} {'val':>10} {'wall':>10} "
          f"{'vs uniform':>12}")
    print("-" * 90)
    uniform_floor = torch.log(torch.tensor(float(vocab_size))).item()
    for name, r in results.items():
        if "compression" in r:
            compr = f"{r['compression']:.1f}x"
        else:
            # Compute live for the fibgen models
            compr = "—"
        vs_uniform = (uniform_floor - r["final_val"]) / uniform_floor * 100
        print(f"{name:<14} {r['n_params']:>10,} {compr:>8} {r['final_val']:>10.4f} "
              f"{r['wall_time']:>9.1f}s {vs_uniform:>+11.1f}%")
    print()

    # Verdict
    base_val = results["dense_crt"]["final_val"]
    print(f"VERDICT (uniform-random floor: {uniform_floor:.4f}, dense_crt: {base_val:.4f}):")
    for K in K_values:
        r = results[f"fibgen_K{K}"]
        if r["final_val"] < uniform_floor * 0.85:
            tag = "LEARNED (≤85% of uniform floor)"
        elif r["final_val"] < uniform_floor * 0.95:
            tag = "WEAK LEARNING"
        else:
            tag = "FAILED (near uniform-random)"
        # Compute compression
        dense_eq = 0
        stored = 0
        m = FibGenLM(vocab_size=vocab_size, d_model=args.d_model,
                      n_blocks=args.n_blocks, seq_len=args.seq_len, K=K)
        ss = m.storage_summary()
        compr = ss["compression"]
        print(f"  K={K:>3}: val={r['final_val']:.4f}  "
              f"compression={compr:.1f}x  → {tag}")

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

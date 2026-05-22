"""Bench the recursive-self-improvement ideas at small scale.

Tests:
  baseline_fibgen          : SubsimLM (substrate operator, validated baseline)
  fibrec_lm                : Inter-layer Fibonacci recurrence on FibGen seeds
                              (depth ~free in storage)
  fibrec_lm_deep           : Same but at n_blocks=8 — should still fit
                              in similar storage as n_blocks=4
  baseline_adamw_phi       : SubsimLM with FibonacciAdamW (β1=1/φ, β2=1/φ²)
                              instead of standard AdamW

Reports: stored params, compression, best val, wall time. The
substrate-recursive primitives are validated if (a) they train to
comparable quality and (b) they unlock something dense couldn't —
free depth or principled optimizer dynamics.
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
from models_fibgen import FibGenLM
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
    compr_tag = ""
    if hasattr(model, "storage_summary"):
        ss = model.storage_summary()
        compr_tag = f"  compression={ss['compression']:.1f}x"
    print(f"\n[train {name}] params={n_params:,}{compr_tag}", flush=True)
    t0 = time.time()
    best_val = float("inf"); best_step = -1
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
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distractor-frac", type=float, default=0.20)
    parser.add_argument("--out", type=str, default="results_recursive.json")
    args = parser.parse_args()

    chars, stoi, itos, encoded = make_dataset(seq_len=args.seq_len,
                                                 source="tinyshakespeare")
    vocab_size = len(chars)
    train_split, val_split = build_distractor_stream(
        encoded, args.distractor_frac, args.seq_len, args.seed,
    )
    fib_positions = fib_positions_in_window(args.seq_len)

    print(f"Recursive substrate bench")
    print(f"Lazy data: P={len(fib_positions)} tokens/seq", flush=True)

    results = {}

    # 1. Baseline Subsim, 4 blocks, AdamW
    m = SubsimLM(vocab_size=vocab_size, d_model=args.d_model, n_blocks=4,
                  seq_len=args.seq_len, K=32, fibgen_K=32, mode="cross")
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr)
    results["subsim_baseline"] = train_one(
        "subsim_baseline", m, opt, train_split, val_split, args, fib_positions)

    # 2. FibRecLM at n_blocks=4 (apples-to-apples vs baseline)
    m = FibRecLM(vocab_size=vocab_size, d_model=args.d_model, n_blocks=4,
                  seq_len=args.seq_len, K=32, mode="cross")
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr)
    results["fibrec_n4"] = train_one(
        "fibrec_n4", m, opt, train_split, val_split, args, fib_positions)

    # 3. FibRecLM at n_blocks=8 — twice the depth, ~same storage
    m = FibRecLM(vocab_size=vocab_size, d_model=args.d_model, n_blocks=8,
                  seq_len=args.seq_len, K=32, mode="cross")
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr)
    results["fibrec_n8"] = train_one(
        "fibrec_n8", m, opt, train_split, val_split, args, fib_positions)

    # 4. Subsim with FibonacciAdamW
    m = SubsimLM(vocab_size=vocab_size, d_model=args.d_model, n_blocks=4,
                  seq_len=args.seq_len, K=32, fibgen_K=32, mode="cross")
    opt = FibonacciAdamW(m.parameters(), lr=args.lr)
    results["subsim_fibadamw"] = train_one(
        "subsim_fibadamw", m, opt, train_split, val_split, args, fib_positions)

    # 5. FibRecLM with FibonacciAdamW (composed substrate-recursive)
    m = FibRecLM(vocab_size=vocab_size, d_model=args.d_model, n_blocks=4,
                  seq_len=args.seq_len, K=32, mode="cross")
    opt = FibonacciAdamW(m.parameters(), lr=args.lr)
    results["fibrec_fibadamw"] = train_one(
        "fibrec_fibadamw", m, opt, train_split, val_split, args, fib_positions)

    # Summary
    print()
    print("=" * 96)
    print(f"{'arch':<22} {'params':>10} {'best_val':>10} {'wall':>10} "
          f"{'compression':>12}")
    print("-" * 96)
    for name, r in results.items():
        # Try to compute compression
        compr = ""
        if "fibrec" in name:
            # FibRec compression varies by depth
            compr = "see model"
        print(f"{name:<22} {r['n_params']:>10,} {r['best_val']:>10.4f} "
              f"{r['wall']:>9.1f}s {compr:>12}")

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

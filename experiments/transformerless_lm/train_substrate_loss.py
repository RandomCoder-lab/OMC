"""A/B bench for substrate-aware loss vs standard cross-entropy.

Trains identical FibRecLM + FibAdamW + K-shrink setups on TinyShakespeare,
varying ONLY the loss function:

  CE_baseline:        standard cross-entropy
  CE + attractor:     CE + λ · attractor_distance(softmax(logits))
  CE + fib_fft:       CE + λ · Fibonacci-frequency-mismatch
  attractor_only:     pure substrate distance, no CE (sanity check)

Same architecture, same data, same optimizer, same K-schedule, same
seed. The ONLY variable is the loss function — so any difference
attributes directly to the substrate-aware loss term.

The hypothesis: substrate-aware loss gives the model an incentive to
produce SUBSTRATE-SHAPED outputs, not just probability-mass-on-target
outputs. If true, the CE+substrate variants reach lower val (or same
val but with structurally better outputs).
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
from train_K_shrink import K_schedule_tier_walk, set_K_active_recursive
from losses_substrate import (substrate_aware_loss, substrate_only_loss,
                                substrate_fft_loss)


def evaluate(model, val_split, batch_size, window, fib_positions, generator,
              n_batches=16):
    """Eval always uses standard CE so val numbers are comparable."""
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


def train_with_loss(name, model, optimizer, loss_fn, train_split, val_split,
                     vocab_size, args, fib_positions, K_schedule_fn):
    torch.manual_seed(args.seed)
    gen = torch.Generator(); gen.manual_seed(args.seed + 1)
    print(f"\n[train {name}]  loss={loss_fn.__name__}", flush=True)
    t0 = time.time()
    best_val = float("inf"); best_step = -1
    cur_K = None
    val_hist = []
    eval_every = max(args.steps // 15, 250)
    for step in range(args.steps):
        if K_schedule_fn is not None:
            new_K = K_schedule_fn(step, args.steps)
            if new_K != cur_K:
                set_K_active_recursive(model, new_K)
                cur_K = new_K
        x, y = get_fib_strided_batch(train_split, args.batch_size, args.seq_len,
                                       fib_positions, gen)
        logits = model(x)
        loss = loss_fn(logits, y, vocab_size)
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
    return {"name": name, "best_val": best_val, "best_step": best_step,
             "wall": time.time() - t0}


def make_loss_fn(kind: str, lambda_sub: float):
    """Return a (logits, targets, vocab_size) -> scalar loss closure."""
    if kind == "ce":
        return lambda logits, targets, V: F.cross_entropy(
            logits.reshape(-1, V), targets.reshape(-1))
    if kind == "ce_attractor":
        return lambda logits, targets, V: substrate_aware_loss(
            logits, targets, V, lambda_substrate=lambda_sub)
    if kind == "ce_fft":
        return lambda logits, targets, V: substrate_fft_loss(
            logits, targets, V, lambda_substrate=lambda_sub)
    if kind == "attractor_only":
        return lambda logits, targets, V: substrate_only_loss(
            logits, targets, V)
    raise ValueError(kind)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distractor-frac", type=float, default=0.20)
    parser.add_argument("--K-init", type=int, default=89)
    parser.add_argument("--K-min", type=int, default=13)
    parser.add_argument("--lambda-sub", type=float, default=0.01)
    parser.add_argument("--losses", type=str, default="ce,ce_attractor,ce_fft")
    parser.add_argument("--out", type=str, default="results_substrate_loss.json")
    args = parser.parse_args()

    chars, stoi, itos, encoded = make_dataset(seq_len=args.seq_len,
                                                 source="tinyshakespeare")
    vocab_size = len(chars)
    train_split, val_split = build_distractor_stream(
        encoded, args.distractor_frac, args.seq_len, args.seed,
    )
    fib_positions = fib_positions_in_window(args.seq_len)

    print(f"Substrate-loss A/B on TinyShakespeare")
    print(f"d={args.d_model}, n_blocks={args.n_blocks}, K_init={args.K_init} "
          f"K_min={args.K_min}, λ_sub={args.lambda_sub}", flush=True)

    losses = [s.strip() for s in args.losses.split(",")]
    results = {}

    for kind in losses:
        m = FibRecLM(vocab_size=vocab_size, d_model=args.d_model,
                      n_blocks=args.n_blocks, seq_len=args.seq_len,
                      K=args.K_init, mode="cross")
        opt = FibonacciAdamW(m.parameters(), lr=args.lr)
        sched = lambda s, T: K_schedule_tier_walk(s, T, K_init=args.K_init,
                                                     K_min=args.K_min)
        loss_fn = make_loss_fn(kind, args.lambda_sub)
        results[kind] = train_with_loss(kind, m, opt, loss_fn,
                                          train_split, val_split, vocab_size,
                                          args, fib_positions, sched)

    print()
    print("=" * 84)
    print(f"{'loss':<24} {'best_val':>10} {'step':>8} {'wall':>10}")
    print('-' * 84)
    base = results.get("ce", {"best_val": None})
    for kind, r in results.items():
        delta = ""
        if base["best_val"] is not None and kind != "ce":
            d = (r["best_val"] - base["best_val"]) / base["best_val"] * 100
            delta = f"  ({d:+.2f}% vs ce)"
        print(f"{kind:<24} {r['best_val']:>10.4f} {r['best_step']:>8} "
              f"{r['wall']:>9.1f}s{delta}")

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

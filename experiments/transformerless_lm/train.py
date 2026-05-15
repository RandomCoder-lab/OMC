"""Training driver for the transformerless-LM bench. Trains all three
architectures from the same seed with the same hyperparameters,
plots the loss curves, and prints the final validation losses.

Usage:
    python3 train.py [--steps 1000] [--seed 42]

Output:
    Per-step training loss for each arch
    Final validation loss summary
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Make the experiment dir importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).parent))
from corpus import CORPUS, get_batch, make_dataset
from models import make_model


def evaluate(model, encoded, batch_size: int, seq_len: int, n_batches: int, generator):
    """Mean cross-entropy loss over n_batches random samples."""
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = get_batch(encoded, batch_size, seq_len, generator)
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train_one(arch: str, encoded, vocab_size: int, args, seed: int):
    """Train one architecture from scratch with a fixed seed.
    Returns dict of metrics."""
    torch.manual_seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed + 1)

    model = make_model(arch, vocab_size=vocab_size, seq_len=args.seq_len)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_losses = []
    val_losses = []

    print(f"\n[arch={arch}] params={n_params:,}")
    t0 = time.time()
    for step in range(args.steps):
        x, y = get_batch(encoded, args.batch_size, args.seq_len, gen)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps - 1:
            tl = loss.item()
            vl = evaluate(model, encoded, args.batch_size, args.seq_len, n_batches=8, generator=gen)
            train_losses.append((step, tl))
            val_losses.append((step, vl))
            elapsed = time.time() - t0
            print(f"  step {step:5d}  train={tl:.4f}  val={vl:.4f}  ({elapsed:.1f}s)")

    final_val = val_losses[-1][1]
    return dict(
        arch=arch,
        n_params=n_params,
        train_losses=train_losses,
        val_losses=val_losses,
        final_val=final_val,
        time=time.time() - t0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    chars, stoi, itos, encoded = make_dataset(seq_len=args.seq_len)
    vocab_size = len(chars)
    n_chars = encoded.numel()
    print(f"Corpus: {n_chars} chars, vocab size {vocab_size}")
    print(f"Training: steps={args.steps}, batch={args.batch_size}, seq_len={args.seq_len}, lr={args.lr}, seed={args.seed}")
    print(f"Note: tiny corpus + tiny model — purpose is to compare LOSS CURVES, not produce a useful LM.")

    results = []
    for arch in ["standard", "crt_only", "hybrid"]:
        r = train_one(arch, encoded, vocab_size, args, args.seed)
        results.append(r)

    print()
    print("=" * 70)
    print(f"{'arch':<12} {'params':>10} {'final_val_loss':>16} {'time_s':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['arch']:<12} {r['n_params']:>10,} {r['final_val']:>16.4f} {r['time']:>8.1f}")
    print()
    base = next(r for r in results if r["arch"] == "standard")
    for r in results:
        if r["arch"] == "standard":
            continue
        delta = r["final_val"] - base["final_val"]
        rel = (delta / base["final_val"]) * 100
        verdict = "WORSE" if delta > 0 else "BETTER"
        print(f"  {r['arch']:<12} vs standard: {delta:+.4f} ({rel:+.1f}%) — {verdict}")


if __name__ == "__main__":
    main()

"""Scale experiment: train standard / crt_only / hybrid on TinyShakespeare
with a larger model. Proper train/val split so we measure generalization,
not just memorization.

Default config: d_model=128, n_blocks=4, seq_len=128, batch=32, 2000 steps.
That's ~5 minutes per arch on CPU, ~15 min total per seed.

Splits the 1.1MB corpus 90/10 train/val. Validation loss is on the
held-out 10% so the win (or loss) reflects actual generalization.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from corpus import make_dataset
from models import make_model


def get_batch_split(encoded_split, batch_size: int, seq_len: int, generator):
    n = encoded_split.numel()
    ix = torch.randint(0, n - seq_len - 1, (batch_size,), generator=generator)
    x = torch.stack([encoded_split[i:i + seq_len] for i in ix])
    y = torch.stack([encoded_split[i + 1:i + seq_len + 1] for i in ix])
    return x, y


def evaluate(model, val_split, batch_size, seq_len, n_batches, generator):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = get_batch_split(val_split, batch_size, seq_len, generator)
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train_one(arch, train_split, val_split, vocab_size, args, seed):
    torch.manual_seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed + 1)

    model = make_model(
        arch,
        vocab_size=vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_blocks=args.n_blocks,
    )
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"\n[arch={arch}] params={n_params:,}", flush=True)
    t0 = time.time()
    val_history = []
    for step in range(args.steps):
        x, y = get_batch_split(train_split, args.batch_size, args.seq_len, gen)
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
            vl = evaluate(model, val_split, args.batch_size, args.seq_len, n_batches=16, generator=gen)
            val_history.append((step, vl))
            elapsed = time.time() - t0
            print(f"  step {step:5d}  train={tl:.4f}  val={vl:.4f}  ({elapsed:.1f}s)", flush=True)

    # Average the LAST few evaluation points for a more stable final number
    last_few = val_history[-3:]
    final_val = sum(v for _, v in last_few) / len(last_few)
    return dict(
        arch=arch,
        n_params=n_params,
        val_history=val_history,
        final_val=final_val,
        time=time.time() - t0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seeds", type=str, default="42")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    chars, stoi, itos, encoded = make_dataset(seq_len=args.seq_len, source="tinyshakespeare")
    vocab_size = len(chars)
    n = encoded.numel()
    n_train = int(n * 0.9)
    train_split = encoded[:n_train]
    val_split = encoded[n_train:]
    print(f"Corpus: TinyShakespeare ({n:,} chars, vocab {vocab_size})")
    print(f"Split: {n_train:,} train / {n - n_train:,} val")
    print(f"Model: d_model={args.d_model}, n_blocks={args.n_blocks}, seq_len={args.seq_len}")
    print(f"Training: steps={args.steps}, batch={args.batch_size}, lr={args.lr}, seeds={seeds}", flush=True)

    all_results = {arch: [] for arch in ["standard", "crt_only", "hybrid"]}
    for seed in seeds:
        print(f"\n=========== seed {seed} ===========")
        for arch in ["standard", "crt_only", "hybrid"]:
            r = train_one(arch, train_split, val_split, vocab_size, args, seed)
            all_results[arch].append(r["final_val"])
            print(f"  [seed {seed}] {arch}: final_val={r['final_val']:.4f}", flush=True)

    print()
    print("=" * 70)
    print(f"{'arch':<12} {'mean_final_val':>16} {'std':>10} {'win_rate':>12}")
    print("-" * 70)
    import statistics
    base = all_results["standard"]
    for arch in ["standard", "crt_only", "hybrid"]:
        vals = all_results[arch]
        mean = sum(vals) / len(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        if arch == "standard":
            wr = "—"
        else:
            wins = sum(1 for v, b in zip(vals, base) if v < b)
            wr = f"{wins}/{len(vals)}"
        print(f"{arch:<12} {mean:>16.4f} {std:>10.4f} {wr:>12}")
    print()
    base_mean = sum(base) / len(base)
    for arch in ["crt_only", "hybrid"]:
        vals = all_results[arch]
        mean = sum(vals) / len(vals)
        rel = (mean - base_mean) / base_mean * 100
        verdict = "BETTER" if mean < base_mean else "WORSE"
        print(f"  {arch:<12} vs standard: {mean - base_mean:+.4f} ({rel:+.1f}%) — {verdict}")


if __name__ == "__main__":
    main()

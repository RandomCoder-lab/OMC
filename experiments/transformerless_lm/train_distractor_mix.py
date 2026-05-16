"""Adversarial-mix scaling test for the CRT-PE + HBit-gate stack.

The README's transformerless-LM section explicitly predicts that the
`hybrid` arch (CRT-PE + HBit-tension gate) loses to `crt_only` on
clean training data because the gate has nothing useful to gate
against. The architectural prescription:

    "OR train with mixed-clean-and-distractor batches so the gate
     has something to gate against."

This file builds the distractor-mix corpus and re-runs the three
architectures on it. If the README's prediction is correct, `hybrid`
should now beat `crt_only` on validation loss against the on-distribution
held-out set (because the gate learns to attend to real-text patterns
and skip the distractor patterns during training).

CONSTRUCTION:
    - Take TinyShakespeare as the on-distribution corpus
    - Build distractors by char-shuffling random windows of the same
      corpus (same char distribution, no structural patterns)
    - Mix into the training stream at distractor_frac (default 20%)
    - Validate on PURE shakespeare (the actual task) so we measure
      "does the model learn shakespeare *despite* the noise?"

Hypothesis: `hybrid` wins this regime because the gate's down-
weighting of off-manifold keys helps the model ignore the noise
chunks. If `hybrid` ties or loses, the README's architectural
hypothesis is falsified at this scale.
"""

import argparse
import sys
import time
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from corpus import make_dataset
from models import make_model


def build_distractor_stream(
    encoded: torch.Tensor,
    distractor_frac: float,
    seq_len: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a training stream where `distractor_frac` of seq_len-sized
    chunks are char-shuffled versions of randomly-drawn windows from
    the same corpus. Same char distribution as the original (so the
    softmax baseline can't exploit a vocabulary shift); structural
    patterns destroyed.

    Returns (train_stream, on_dist_val) where:
        train_stream is a 1-D tensor with mixed clean + distractor chunks
        on_dist_val is the unchanged tail of the input for held-out eval
    """
    g = torch.Generator()
    g.manual_seed(seed)
    n = encoded.numel()
    n_train_total = int(n * 0.9)
    n_val = n - n_train_total
    val_split = encoded[n_train_total:]   # PURE shakespeare; not touched

    # Build the mixed training stream chunk by chunk.
    n_chunks = n_train_total // seq_len
    chunks = []
    for i in range(n_chunks):
        if torch.rand(1, generator=g).item() < distractor_frac:
            # Distractor: take a random window, shuffle its chars in-place.
            start = torch.randint(0, n_train_total - seq_len, (1,), generator=g).item()
            window = encoded[start:start + seq_len].clone()
            perm = torch.randperm(seq_len, generator=g)
            chunks.append(window[perm])
        else:
            # Clean: contiguous shakespeare slice.
            start = torch.randint(0, n_train_total - seq_len, (1,), generator=g).item()
            chunks.append(encoded[start:start + seq_len].clone())
    train_stream = torch.cat(chunks)
    print(f"Mixed-stream: {len(chunks)} chunks ({seq_len} chars each), "
          f"distractor_frac={distractor_frac:.2f}; val on {n_val:,} clean chars")
    return train_stream, val_split


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
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seeds", type=str, default="42,7,123")
    parser.add_argument("--distractor-frac", type=float, default=0.20,
                        help="Fraction of training chunks that are char-shuffled.")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    chars, stoi, itos, encoded = make_dataset(seq_len=args.seq_len, source="tinyshakespeare")
    vocab_size = len(chars)

    print(f"Corpus: TinyShakespeare ({encoded.numel():,} chars, vocab {vocab_size})")
    print(f"Adversarial-mix test: distractor_frac={args.distractor_frac:.2f}")
    print(f"Model: d_model={args.d_model}, n_blocks={args.n_blocks}, seq_len={args.seq_len}")
    print(f"Training: steps={args.steps}, batch={args.batch_size}, lr={args.lr}, seeds={seeds}", flush=True)

    all_results = {arch: [] for arch in ["standard", "crt_only", "hybrid"]}
    for seed in seeds:
        print(f"\n=========== seed {seed} ===========")
        # Build the mixed stream FRESH per seed so seeds are honest.
        train_split, val_split = build_distractor_stream(
            encoded, args.distractor_frac, args.seq_len, seed,
        )
        for arch in ["standard", "crt_only", "hybrid"]:
            r = train_one(arch, train_split, val_split, vocab_size, args, seed)
            all_results[arch].append(r["final_val"])
            print(f"  [seed {seed}] {arch}: final_val={r['final_val']:.4f}", flush=True)

    print()
    print("=" * 70)
    print(f"{'arch':<12} {'mean_final_val':>16} {'std':>10} {'win_rate':>12}")
    print("-" * 70)
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
    # Also compare hybrid vs crt_only directly — this is the key question.
    hyb_mean = sum(all_results["hybrid"]) / len(all_results["hybrid"])
    crt_mean = sum(all_results["crt_only"]) / len(all_results["crt_only"])
    rel = (hyb_mean - crt_mean) / crt_mean * 100
    crt_better = hyb_mean < crt_mean
    print(f"  hybrid    vs crt_only: {hyb_mean - crt_mean:+.4f} ({rel:+.1f}%) — "
          f"{'GATE EARNS KEEP' if crt_better else 'GATE STILL COSTS'}")


if __name__ == "__main__":
    main()

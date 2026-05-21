"""Lazy-loading test: Fibonacci-strided data ingestion for faster training.

The user's idea: use log_phi_pi_fib(T) to input training data faster --
instead of loading all T tokens of a sequence, load only those at
Fibonacci offsets {0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, ...} ∩ [0, T).
For T=128 that's 10 tokens per sequence instead of 128 — a ~13x reduction
in IO and per-step compute.

Substrate alignment: the data sparsity matches the FibonacciOffsetAttention
sparsity. The model never looks at the gap tokens during attention; lazy
loading means we never read them from disk either. Composed, training
IO + attention FLOPs both drop to O(T · log_phi_pi T).

This bench measures whether the data sparsity catastrophically hurts loss
relative to the throughput it saves:

  dense       : standard contiguous batches, dense_crt model
  fib_strided : Fibonacci-strided batches (10 tokens per "sequence" at
                 effective T=128), dense_crt model
                 → same model, sparser data: tests whether sparse data
                 covers the corpus enough to learn

Wall-clock per step is the primary metric (the lazy-loading thesis is
about throughput). Final val loss is the floor: if sparse training
matches dense val within ~10%, the substrate-aligned sparsity is "free"
IO savings.

If sparse loses badly (>2x val), we learn that pure-position Fibonacci
striding is too aggressive at char level — the gaps carry essential
context. That tells us the next experiment is chunk-level striding,
not position-level.
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
from models import make_model
from train_distractor_mix import build_distractor_stream, evaluate


FIBONACCI = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]


def fib_positions_in_window(window: int) -> list[int]:
    """Substrate-aligned positions in [0, window): {0} ∪ {Fibonacci ≤ window-1}.

    For window=128: [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89] = 11 positions
                   = ~log_phi_pi(128) ≈ 3.0  (count scales as log_phi_pi N).
    """
    pos = sorted(set([0] + [f for f in FIBONACCI if f < window]))
    return pos


def get_dense_batch(encoded, batch_size, seq_len, generator):
    """Standard contiguous-sequence batch."""
    n = encoded.numel()
    ix = torch.randint(0, n - seq_len - 1, (batch_size,), generator=generator)
    x = torch.stack([encoded[i:i + seq_len] for i in ix])
    y = torch.stack([encoded[i + 1:i + seq_len + 1] for i in ix])
    return x, y


def get_fib_strided_batch(encoded, batch_size, window, fib_positions,
                            generator):
    """Fibonacci-strided batch: each sequence picks a random start in the
    corpus and returns tokens at start + fib_positions. The "effective"
    window is `window`, but only len(fib_positions) tokens are actually
    read (and predicted).

    Returns (x, y) where x is [B, P] and y is [B, P] with P=len(fib_positions).
    Target y[t] is the NEXT token after the position x[t] in the corpus.
    """
    n = encoded.numel()
    P = len(fib_positions)
    fib_t = torch.tensor(fib_positions, dtype=torch.long)
    # Start indices that leave room for the largest offset + 1 (for next-tok target)
    max_off = fib_positions[-1] + 1
    ix = torch.randint(0, n - max_off - 1, (batch_size,), generator=generator)
    x = torch.stack([encoded[i + fib_t] for i in ix])
    y = torch.stack([encoded[i + fib_t + 1] for i in ix])
    return x, y


def measure_throughput(get_batch_fn, encoded, batch_size, n_steps_warmup,
                        n_steps_measure, generator):
    """Just IO + tensor-construction overhead, NO model. Measures the
    pure data-pipeline cost."""
    for _ in range(n_steps_warmup):
        get_batch_fn(encoded, batch_size, generator)
    t0 = time.time()
    total_tokens = 0
    for _ in range(n_steps_measure):
        x, y = get_batch_fn(encoded, batch_size, generator)
        total_tokens += x.numel()
    dt = time.time() - t0
    return total_tokens, dt, total_tokens / dt


def train_dense(model, train_split, val_split, args, gen):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    t0 = time.time()
    val_hist = []
    for step in range(args.steps):
        x, y = get_dense_batch(train_split, args.batch_size, args.seq_len, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % args.eval_every == 0 or step == args.steps - 1:
            vl = evaluate(model, val_split, args.batch_size, args.seq_len,
                          n_batches=16, generator=gen)
            val_hist.append((step, vl, time.time() - t0))
            print(f"    step {step:5d}  train={loss.item():.4f}  val={vl:.4f}  "
                  f"({time.time() - t0:.1f}s)", flush=True)
    final = evaluate(model, val_split, args.batch_size, args.seq_len,
                      n_batches=32, generator=gen)
    return val_hist, final, time.time() - t0


def train_fib_strided(model, train_split, val_split, args, gen, fib_positions):
    """Train model on Fibonacci-strided data.

    The model is still a standard crt_only with seq_len=args.seq_len
    (its PE / mask cover the full window). We pass in a shorter sequence
    of length P=len(fib_positions). The model sees these as the FIRST P
    positions in its window — that loses the absolute-position signal of
    the original strided positions, but is the simplest implementation.

    A cleaner version would inject the actual absolute positions into
    the PE, but for the throughput question, this loose coupling is
    enough to measure whether sparse data learns at all.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    t0 = time.time()
    val_hist = []
    P = len(fib_positions)
    for step in range(args.steps):
        x, y = get_fib_strided_batch(train_split, args.batch_size,
                                       args.seq_len, fib_positions, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % args.eval_every == 0 or step == args.steps - 1:
            # Evaluate on DENSE val data so the loss is comparable to dense.
            vl = evaluate(model, val_split, args.batch_size, args.seq_len,
                          n_batches=16, generator=gen)
            val_hist.append((step, vl, time.time() - t0))
            print(f"    step {step:5d}  train={loss.item():.4f}  "
                  f"val_dense={vl:.4f}  ({time.time() - t0:.1f}s)", flush=True)
    final = evaluate(model, val_split, args.batch_size, args.seq_len,
                      n_batches=32, generator=gen)
    return val_hist, final, time.time() - t0


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
    parser.add_argument("--out", type=str, default="results_lazy_loading.json")
    args = parser.parse_args()

    chars, stoi, itos, encoded = make_dataset(
        seq_len=args.seq_len, source="tinyshakespeare",
    )
    vocab_size = len(chars)
    train_split, val_split = build_distractor_stream(
        encoded, args.distractor_frac, args.seq_len, args.seed,
    )

    fib_positions = fib_positions_in_window(args.seq_len)
    P = len(fib_positions)
    print(f"Lazy-loading bench")
    print(f"Corpus: TinyShakespeare ({encoded.numel():,} chars)")
    print(f"Window: seq_len={args.seq_len}  Fibonacci positions: {fib_positions} (P={P})")
    print(f"Data density ratio: {args.seq_len / P:.1f}x reduction"
          f" ({P} sparse vs {args.seq_len} dense tokens per sequence)", flush=True)

    # ---- 1. Pure IO throughput comparison ----
    print(f"\n--- IO throughput (no model, 100 warmup + 200 measure batches) ---")
    gen_io = torch.Generator(); gen_io.manual_seed(args.seed)
    tot_d, dt_d, tps_d = measure_throughput(
        lambda enc, b, g: get_dense_batch(enc, b, args.seq_len, g),
        train_split, args.batch_size, 100, 200, gen_io,
    )
    print(f"  dense       : {tot_d:>9,} tokens in {dt_d:.2f}s = {tps_d/1e6:.1f}M tok/s")
    gen_io.manual_seed(args.seed)
    tot_s, dt_s, tps_s = measure_throughput(
        lambda enc, b, g: get_fib_strided_batch(enc, b, args.seq_len, fib_positions, g),
        train_split, args.batch_size, 100, 200, gen_io,
    )
    print(f"  fib_strided : {tot_s:>9,} tokens in {dt_s:.2f}s = {tps_s/1e6:.1f}M tok/s")
    io_speedup = dt_d / dt_s
    tok_ratio = tot_d / tot_s
    print(f"  IO speedup (same n_steps): {io_speedup:.2f}x")
    print(f"  Tokens-per-step ratio: {tok_ratio:.2f}x (sparse loads {1/tok_ratio:.1%} of dense)")

    # ---- 2. Train both configurations ----
    results = {
        "fib_positions": fib_positions,
        "io": {"dense_tps": tps_d, "fib_strided_tps": tps_s,
                "io_speedup": io_speedup, "tokens_per_step_ratio": tok_ratio},
        "training": {},
    }

    # Dense baseline
    print(f"\n--- Dense training (seq_len={args.seq_len}) ---")
    torch.manual_seed(args.seed)
    gen = torch.Generator(); gen.manual_seed(args.seed + 1)
    model_d = make_model("crt_only", vocab_size=vocab_size, seq_len=args.seq_len,
                          d_model=args.d_model, n_blocks=args.n_blocks)
    n_params = sum(p.numel() for p in model_d.parameters())
    print(f"  model params: {n_params:,}", flush=True)
    hist_d, final_d, time_d = train_dense(model_d, train_split, val_split, args, gen)
    results["training"]["dense"] = {
        "final_val": final_d, "wall_time": time_d,
        "val_history": [(s, v, t) for s, v, t in hist_d],
        "n_params": n_params,
    }
    print(f"  ✓ dense: final_val={final_d:.4f}, wall={time_d:.1f}s, "
          f"steps/sec={args.steps/time_d:.1f}")

    # Fibonacci-strided
    print(f"\n--- Fibonacci-strided training (P={P} tokens / step / seq) ---")
    torch.manual_seed(args.seed)
    gen = torch.Generator(); gen.manual_seed(args.seed + 1)
    # Model has its own seq_len matching the SPARSE length P, since we feed
    # length-P sequences. This is the apples-to-apples version (a model
    # actually sized for the sparse data).
    model_s = make_model("crt_only", vocab_size=vocab_size, seq_len=P,
                          d_model=args.d_model, n_blocks=args.n_blocks)
    n_params_s = sum(p.numel() for p in model_s.parameters())
    print(f"  model params: {n_params_s:,}  (model seq_len={P})", flush=True)

    # The val evaluation needs to handle the sparse model on dense val data.
    # The model has seq_len=P, so we eval on length-P batches (still dense
    # tokens, just shorter sequences). This is a fair next-token comparison
    # to the dense P=seq_len case at the same effective sequence length.
    args_eval = argparse.Namespace(**vars(args))
    args_eval.seq_len = P  # eval uses length-P windows

    hist_s = []
    optimizer = torch.optim.AdamW(model_s.parameters(), lr=args.lr)
    t0 = time.time()
    for step in range(args.steps):
        x, y = get_fib_strided_batch(train_split, args.batch_size,
                                       args.seq_len, fib_positions, gen)
        logits = model_s(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % args.eval_every == 0 or step == args.steps - 1:
            vl = evaluate(model_s, val_split, args.batch_size, P,
                          n_batches=16, generator=gen)
            hist_s.append((step, vl, time.time() - t0))
            print(f"    step {step:5d}  train={loss.item():.4f}  "
                  f"val_dense_lenP={vl:.4f}  ({time.time() - t0:.1f}s)", flush=True)
    final_s = evaluate(model_s, val_split, args.batch_size, P,
                        n_batches=32, generator=gen)
    time_s = time.time() - t0
    results["training"]["fib_strided"] = {
        "final_val": final_s, "wall_time": time_s,
        "val_history": [(s, v, t) for s, v, t in hist_s],
        "n_params": n_params_s,
    }
    print(f"  ✓ fib_strided: final_val={final_s:.4f}, wall={time_s:.1f}s, "
          f"steps/sec={args.steps/time_s:.1f}")

    # ---- 3. Summary ----
    print(f"\n{'=' * 84}")
    print(f"SUMMARY")
    print(f"{'-' * 84}")
    print(f"{'config':<14} {'val':>10} {'wall':>10} {'steps/s':>10} {'tok/step':>10}")
    print(f"{'dense':<14} {final_d:>10.4f} {time_d:>9.1f}s {args.steps/time_d:>10.1f} "
          f"{args.batch_size*args.seq_len:>10,}")
    print(f"{'fib_strided':<14} {final_s:>10.4f} {time_s:>9.1f}s {args.steps/time_s:>10.1f} "
          f"{args.batch_size*P:>10,}")
    print()
    print(f"  Sparse loss delta: {final_s - final_d:+.4f} ({(final_s-final_d)/final_d*100:+.1f}%)")
    print(f"  Wall-clock speedup: {time_d/time_s:.2f}x")
    print(f"  Throughput (val/sec at end of training):")
    print(f"    dense: {-final_d/time_d * 1000:.4f} (negative val per ms)")
    print(f"    sparse: {-final_s/time_s * 1000:.4f}")

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

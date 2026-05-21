"""LM-head Zeckendorf-rank compression test.

The architectural question: is language low-Zeckendorf-rank?

If YES, the substrate's compression primitive is the right axis for
building inference-cheap LLMs (the inference-first re-derivation in
INFERENCE_FIRST_DERIVATION.md). If NO, we need a different basis.

Test design:

  1. Train a `crt_only` baseline on TinyShakespeare (validated arch
     from the prior bench, ~800K params, mean val 2.46).
  2. Extract its LM head W ∈ R^[vocab, d_model]. Compute the full SVD
     W = U Σ V^T.
  3. Build three rank-K approximations Ŵ at varying K, all using the
     SAME total memory K·(vocab + d_model):
        - top_k:   first K singular components (Eckart-Young optimal).
        - fib_k:   singular components at Fibonacci indices ≤ K.
        - rand_k:  uniformly-random K indices from [0, min_dim).
  4. For each Ŵ, swap into the model and measure val perplexity.

Hypothesis: if Fibonacci-indexed singular components carry
disproportionately more language structure than random ones, then
fib_k > rand_k (closer to top_k) at matched K. If fib_k ≈ rand_k,
language is NOT preferentially low-Zeckendorf-rank and the substrate
compression story has no foothold at the LM head layer.

The result is a yes/no signal for the broader inference-first thesis.
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
from train_distractor_mix import (
    build_distractor_stream,
    get_batch_split,
    evaluate,
)

# Canonical Fibonacci table from omnimcode-core/src/phi_pi_fib.rs.
FIBONACCI = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987,
              1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368]

PHI = (1 + 5 ** 0.5) / 2          # golden ratio
PI = math.pi
PHI_PI = PHI ** PI                # ≈ 36.46, the substrate exponent base

# ---- Scheme 1: pure Fibonacci ----
# Indices = {0} ∪ {unique positive Fibonacci numbers}.
FIB_PURE_INDICES = sorted(set([0] + FIBONACCI))


# ---- Scheme 2: π/φ-modulated Fibonacci ----
# F(k) · π / φ. Pushes Fibonacci values outward by ~1.94×. The user
# observation: if φ is the derivation and Fibonacci is the basis, then
# the natural cross-multiplication with π is the next substrate term.
FIB_PHI_PI_INDICES = sorted(set([0] + [
    int(f * PI / PHI) for f in FIBONACCI
] + [int(f * PI / PHI) for f in FIBONACCI if int(f * PI / PHI) > 0]))


def phi_pi_canonical_indices(n_components: int, n_terms: int = 24) -> list[int]:
    """Substrate-canonical split-point offsets, scaled to the SVD rank range.

    Mirrors the formula in PHI_PI_FIB_ALGORITHM.md:
        offset(k) = n · F(k) / φ^(π·k)
    These cluster near 0 with rapidly diminishing reach — the same
    probe pattern phi_pi_fib_search_v2 uses on a sorted array.

    Returns sorted unique indices in [0, n_components).
    """
    offs = set([0])
    for k in range(1, n_terms + 1):
        Fk = FIBONACCI[k - 1] if k - 1 < len(FIBONACCI) else FIBONACCI[-1]
        idx = int(n_components * Fk / (PHI ** (PI * k)))
        if 0 <= idx < n_components:
            offs.add(idx)
    return sorted(offs)


def compress_lm_head(W: torch.Tensor, n_keep: int, scheme: str,
                     rng: torch.Generator) -> tuple[torch.Tensor, list[int]]:
    """Build an approximation of W keeping `n_keep` SVD components selected
    by the chosen scheme. Returns (Ŵ, indices_kept).

    All three schemes use the SAME n_keep, so memory footprint is
    identical: n_keep · (W.shape[0] + W.shape[1]) floats.
    """
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    n_components = S.numel()
    def _fill(candidates: list[int]) -> list[int]:
        """Take first n_keep candidates; pad with dense indices if short."""
        idx = [c for c in candidates if 0 <= c < n_components][:n_keep]
        if len(idx) < n_keep:
            for i in range(n_components):
                if i not in idx:
                    idx.append(i)
                if len(idx) >= n_keep:
                    break
        return sorted(idx)

    if scheme == "top_k":
        idx = list(range(min(n_keep, n_components)))
    elif scheme == "fib_pure":
        idx = _fill(FIB_PURE_INDICES)
    elif scheme == "fib_phi_pi":
        idx = _fill(FIB_PHI_PI_INDICES)
    elif scheme == "phi_pi_canonical":
        idx = _fill(phi_pi_canonical_indices(n_components))
    elif scheme == "rand_k":
        perm = torch.randperm(n_components, generator=rng).tolist()
        idx = sorted(perm[:n_keep])
    else:
        raise ValueError(scheme)

    idx_t = torch.tensor(idx, dtype=torch.long)
    U_k = U[:, idx_t]
    S_k = S[idx_t]
    Vh_k = Vh[idx_t, :]
    W_approx = (U_k * S_k) @ Vh_k
    return W_approx, idx


def measure_val_perplexity(model, val_split, batch_size, seq_len,
                            n_batches=32, generator=None):
    losses = []
    model.eval()
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


def train_baseline(args, vocab_size, train_split, val_split):
    """Train a fresh crt_only baseline and return the model."""
    torch.manual_seed(args.seed)
    gen = torch.Generator()
    gen.manual_seed(args.seed + 1)
    model = make_model(
        "crt_only", vocab_size=vocab_size, seq_len=args.seq_len,
        d_model=args.d_model, n_blocks=args.n_blocks,
    )
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"\n[baseline crt_only] params={n_params:,}", flush=True)
    t0 = time.time()
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
            vl = measure_val_perplexity(model, val_split, args.batch_size,
                                         args.seq_len, n_batches=16, generator=gen)
            elapsed = time.time() - t0
            print(f"  step {step:5d}  train={loss.item():.4f}  val={vl:.4f}  ({elapsed:.1f}s)",
                  flush=True)
    return model


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
    parser.add_argument("--n-rand-trials", type=int, default=5,
                        help="Random rank-K runs to average for the rand_k baseline.")
    parser.add_argument("--distractor-frac", type=float, default=0.20)
    parser.add_argument("--out", type=str, default="results_lm_head_compression.json")
    args = parser.parse_args()

    chars, stoi, itos, encoded = make_dataset(
        seq_len=args.seq_len, source="tinyshakespeare",
    )
    vocab_size = len(chars)

    print(f"LM-head Zeckendorf-rank compression test")
    print(f"Corpus: TinyShakespeare ({encoded.numel():,} chars, vocab {vocab_size})")
    print(f"Model: d_model={args.d_model}, n_blocks={args.n_blocks}, "
          f"seq_len={args.seq_len}", flush=True)

    train_split, val_split = build_distractor_stream(
        encoded, args.distractor_frac, args.seq_len, args.seed,
    )

    # ---- 1. Train baseline ----
    model = train_baseline(args, vocab_size, train_split, val_split)
    gen = torch.Generator()
    gen.manual_seed(args.seed + 1)
    baseline_val = measure_val_perplexity(
        model, val_split, args.batch_size, args.seq_len, n_batches=32, generator=gen,
    )
    print(f"\nBaseline val loss (full LM head): {baseline_val:.4f}")

    # ---- 2. Extract LM head ----
    # The model ties head.weight to embed.weight, so we work on a copy.
    W_orig = model.head.weight.detach().clone()    # [vocab, d_model]
    print(f"\nLM head shape: {tuple(W_orig.shape)}, total params: {W_orig.numel():,}")
    print(f"Full-rank memory: {W_orig.numel() * 4:,} bytes (fp32)")

    # ---- 3. Sweep K, compare schemes ----
    # K values to test. We use {1, 2, 3, 5, 8, 13, 21, 34, 55} (Fibonacci) +
    # interpolating dense values so every K is comparable.
    min_dim = min(W_orig.shape)
    # n_keep values where Fibonacci has a "natural" footprint. Including
    # in-between values lets us see whether the substrate ordering is
    # better than top-rank or just lucky at specific points.
    K_values = sorted(set([2, 3, 4, 5, 6, 8, 10, 13, 16, 21, 28, 34, 45, 55]))
    K_values = [k for k in K_values if k < min_dim]

    rng = torch.Generator()
    rng.manual_seed(args.seed + 100)

    results = []
    for K in K_values:
        compression_ratio = W_orig.numel() / (K * (W_orig.shape[0] + W_orig.shape[1]))
        print(f"\n--- K={K}  (compression ratio: {compression_ratio:.2f}x) ---")
        row = {"K": K, "compression": compression_ratio, "baseline_val": baseline_val}

        for scheme in ["top_k", "fib_pure", "fib_phi_pi", "phi_pi_canonical"]:
            W_approx, idx = compress_lm_head(W_orig, K, scheme, rng)
            with torch.no_grad():
                model.head.weight.copy_(W_approx)
                # Embedding is tied — copy through.
                model.embed.weight.copy_(W_approx)
            val = measure_val_perplexity(
                model, val_split, args.batch_size, args.seq_len,
                n_batches=32, generator=gen,
            )
            row[scheme] = {"val": val, "indices": idx}
            print(f"  {scheme:<8} val={val:.4f}  Δ={val - baseline_val:+.4f}  "
                  f"indices={idx[:6]}{'...' if len(idx) > 6 else ''}")

        # rand_k: average over multiple trials
        rand_vals = []
        rand_idx_samples = []
        for trial in range(args.n_rand_trials):
            W_approx, idx = compress_lm_head(W_orig, K, "rand_k", rng)
            with torch.no_grad():
                model.head.weight.copy_(W_approx)
                model.embed.weight.copy_(W_approx)
            val = measure_val_perplexity(
                model, val_split, args.batch_size, args.seq_len,
                n_batches=16, generator=gen,
            )
            rand_vals.append(val)
            rand_idx_samples.append(idx[:6])
        row["rand_k"] = {
            "val_mean": sum(rand_vals)/len(rand_vals),
            "val_std": (sum((v - sum(rand_vals)/len(rand_vals))**2 for v in rand_vals) / len(rand_vals))**0.5,
            "vals": rand_vals,
        }
        print(f"  {'rand_k':<8} val={row['rand_k']['val_mean']:.4f} "
              f"(std {row['rand_k']['val_std']:.4f}, n={args.n_rand_trials})  "
              f"Δ={row['rand_k']['val_mean'] - baseline_val:+.4f}")

        results.append(row)

    # Restore full-rank head before returning (so subsequent code can use the model).
    with torch.no_grad():
        model.head.weight.copy_(W_orig)
        model.embed.weight.copy_(W_orig)

    # ---- 4. Summary ----
    print()
    print("=" * 110)
    schemes = ["top_k", "fib_pure", "fib_phi_pi", "phi_pi_canonical"]
    print(f"{'K':>4} {'compress':>10} " + " ".join(f"{s:>15}" for s in schemes)
          + f" {'rand_k':>16}")
    print("-" * 110)
    for row in results:
        rand = row["rand_k"]
        rs = f"{rand['val_mean']:.4f}±{rand['val_std']:.3f}"
        cells = " ".join(f"{row[s]['val']:>15.4f}" for s in schemes)
        print(f"{row['K']:>4} {row['compression']:>9.2f}x {cells} {rs:>16}")

    print()
    print("Interpretation:")
    for s in ("fib_pure", "fib_phi_pi", "phi_pi_canonical"):
        better = sum(1 for r in results if r[s]["val"] < r["rand_k"]["val_mean"])
        gap_top = sum(r[s]["val"] - r["top_k"]["val"] for r in results) / len(results)
        gap_rand = sum(r[s]["val"] - r["rand_k"]["val_mean"] for r in results) / len(results)
        print(f"  {s:<18}  beats rand at {better}/{len(results)} Ks  "
              f"mean Δ vs top_k:{gap_top:+.4f}  mean Δ vs rand:{gap_rand:+.4f}")

    # Save
    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump({
            "baseline_val": baseline_val,
            "W_shape": list(W_orig.shape),
            "K_values": K_values,
            "results": results,
        }, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

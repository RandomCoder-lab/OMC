"""Scalability test for the substrate fast-init claim.

TRANSFORMERLESS_RESULT.md showed that token-CRT gives a measurable
early-phase convergence speedup (−2.8% at step 100, ~30% step-saving
in the warmup phase) but loses on final accuracy (+4.1% at step 1500)
because the fixed additive prior becomes interference once the learned
embedding is doing real work. The refined rule predicts a learnable β
fixes this — and the user's compute-efficiency framing predicts the
early-phase advantage should hold or grow with scale.

This bench tests both:

  1. Does adding learnable β rescue token-CRT's late-phase loss
     while keeping the early-phase win? (architectural attenuability)
  2. Does the early-phase token-CRT advantage hold/grow/shrink at
     d_model=256 (2x the previously-validated scale)? (compute-
     efficiency scaling)

Archs:
  crt_only         : baseline
  token_crt        : fixed token-substrate (the falsified variant)
  token_crt_beta   : token-substrate scaled by learnable β
  transformerless_v2: crt_only + token_crt_beta + geodesic

Two scales: d_model=128 (replicates prior bench) and d_model=256
(scaling test). 2 seeds each. 1500 steps each.

The scalability question is answered by comparing the early-phase
delta (val@100, val@300) of token_crt_beta vs crt_only between the
two scales:
  - growing delta → substrate scales positively, compute-efficiency
    claim strengthens at scale
  - flat delta → substrate is a constant-factor warmup, scales neutrally
  - shrinking delta → substrate is a small-model artifact, falsified at scale
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from corpus import make_dataset
from train_distractor_mix import build_distractor_stream, train_one


ARCHS = ["crt_only", "token_crt", "token_crt_beta", "transformerless_v2"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--scales", type=str, default="128,256",
                        help="Comma-separated d_model values to sweep.")
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seeds", type=str, default="42,7")
    parser.add_argument("--distractor-frac", type=float, default=0.20)
    parser.add_argument("--out", type=str, default="results_scaling.json")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    scales = [int(s) for s in args.scales.split(",")]

    chars, stoi, itos, encoded = make_dataset(
        seq_len=args.seq_len, source="tinyshakespeare",
    )
    vocab_size = len(chars)

    print(f"Scaling test — distractor_frac={args.distractor_frac:.2f}")
    print(f"Archs: {ARCHS}")
    print(f"Scales: d_model ∈ {scales}")
    print(f"Corpus: TinyShakespeare ({encoded.numel():,} chars, vocab {vocab_size})")
    print(f"Training: steps={args.steps}, batch={args.batch_size}, lr={args.lr}, "
          f"seeds={seeds}", flush=True)

    # results[d_model][arch] = list of (final_val, val_history)
    results = {d: {a: [] for a in ARCHS} for d in scales}
    per_run_logs = []

    for d_model in scales:
        scale_args = argparse.Namespace(**vars(args))
        scale_args.d_model = d_model
        print(f"\n############ d_model={d_model} ############", flush=True)
        for seed in seeds:
            print(f"\n=========== d={d_model} seed={seed} ===========", flush=True)
            train_split, val_split = build_distractor_stream(
                encoded, args.distractor_frac, args.seq_len, seed,
            )
            for arch in ARCHS:
                r = train_one(arch, train_split, val_split, vocab_size,
                              scale_args, seed)
                results[d_model][arch].append({
                    "seed": seed,
                    "final_val": r["final_val"],
                    "n_params": r["n_params"],
                    "time": r["time"],
                    "val_history": r["val_history"],
                })
                per_run_logs.append({
                    "d_model": d_model, "seed": seed, "arch": arch,
                    "final_val": r["final_val"],
                    "n_params": r["n_params"],
                    "time": r["time"],
                })
                print(f"  [d={d_model} seed={seed}] {arch}: "
                      f"final_val={r['final_val']:.4f} "
                      f"(n_params={r['n_params']:,}, {r['time']:.1f}s)",
                      flush=True)

    # ----- Summary tables -----
    print()
    print("=" * 80)
    print("FINAL ACCURACY (mean across seeds)")
    print("=" * 80)
    header = f"{'arch':<22} " + "  ".join(f"d={d:<5}" for d in scales)
    print(header)
    print("-" * len(header))
    for arch in ARCHS:
        cells = []
        for d in scales:
            vals = [r["final_val"] for r in results[d][arch]]
            cells.append(f"{sum(vals)/len(vals):.4f}")
        print(f"{arch:<22} " + "  ".join(f"{c:<7}" for c in cells))

    # Early-phase delta at step 100, 300, 500 (the scalability signal)
    print()
    print("=" * 80)
    print("EARLY-PHASE VAL LOSS (mean across seeds, by step budget)")
    print("=" * 80)
    for step_target in [100, 300, 500, 1000, 1500]:
        print(f"\n  step {step_target}:")
        print(f"  {'arch':<22} " + "  ".join(f"d={d:<6}" for d in scales))
        for arch in ARCHS:
            cells = []
            for d in scales:
                vals = []
                for r in results[d][arch]:
                    best = None
                    for step, val in r["val_history"]:
                        if step <= step_target:
                            best = val
                        else:
                            break
                    if best is not None:
                        vals.append(best)
                if vals:
                    cells.append(f"{sum(vals)/len(vals):.4f}")
                else:
                    cells.append("  --  ")
            print(f"  {arch:<22} " + "  ".join(f"{c:<8}" for c in cells))

    # Scalability verdict: token_crt_beta early-phase delta growth
    print()
    print("=" * 80)
    print("SCALABILITY: token_crt_beta vs crt_only early-phase delta")
    print("=" * 80)
    print(f"  {'step':<6} " + "  ".join(f"d={d:<10}" for d in scales))
    deltas = {d: [] for d in scales}
    for step_target in [100, 300, 500, 1000, 1500]:
        cells = []
        for d in scales:
            base = []
            beta = []
            for r in results[d]["crt_only"]:
                best = None
                for step, val in r["val_history"]:
                    if step <= step_target: best = val
                    else: break
                if best is not None: base.append(best)
            for r in results[d]["token_crt_beta"]:
                best = None
                for step, val in r["val_history"]:
                    if step <= step_target: best = val
                    else: break
                if best is not None: beta.append(best)
            if base and beta:
                bm = sum(base)/len(base)
                tm = sum(beta)/len(beta)
                rel = (tm - bm) / bm * 100
                deltas[d].append((step_target, rel))
                cells.append(f"{rel:+5.1f}%")
            else:
                cells.append("  --  ")
        print(f"  {step_target:<6} " + "  ".join(f"{c:<12}" for c in cells))

    # Final verdict
    if len(scales) >= 2:
        d_small, d_large = scales[0], scales[-1]
        early_small = deltas[d_small][0][1] if deltas[d_small] else 0
        early_large = deltas[d_large][0][1] if deltas[d_large] else 0
        late_large = deltas[d_large][-1][1] if deltas[d_large] else 0
        print()
        print("Scalability verdict:")
        print(f"  early-phase (step 100) delta: d={d_small}: {early_small:+.1f}%, "
              f"d={d_large}: {early_large:+.1f}%")
        if early_large <= early_small:
            print(f"  → SUBSTRATE FAST-INIT HOLDS OR GROWS WITH SCALE")
        else:
            print(f"  → SUBSTRATE FAST-INIT SHRINKS WITH SCALE")
        print(f"  late-phase (step {deltas[d_large][-1][0]}) delta at d={d_large}: "
              f"{late_large:+.1f}%")
        if late_large < 1.0:
            print(f"  → β SUCCESSFULLY ATTENUATES — no late-phase loss")
        else:
            print(f"  → β FAILS TO ATTENUATE — fixed-prior interference persists")

    # Save
    out_path = Path(__file__).parent / args.out
    summary = {
        "scales": scales,
        "seeds": seeds,
        "steps": args.steps,
        "archs": ARCHS,
        "runs": per_run_logs,
        "results": {
            str(d): {
                a: [
                    {"seed": r["seed"], "final_val": r["final_val"],
                     "n_params": r["n_params"], "time": r["time"],
                     "val_history": r["val_history"]}
                    for r in results[d][a]
                ]
                for a in ARCHS
            }
            for d in scales
        },
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

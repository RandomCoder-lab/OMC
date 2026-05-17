"""Gate reformulation: SCORE-level and LEARNED-threshold variants
vs. the validated `crt_only` baseline on the distractor mix.

Context — see distractor_mix_README.md "Implication" section. The
original `hybrid` (KEY-magnitude gate) was falsified at distractor
fraction 0.20 (worse than crt_only on 3/3 seeds). The README proposed
two follow-on architectures both keeping CRT-PE and only changing the
gate:

  1. SCORE-level gate: gate the raw attention scores BEFORE softmax,
     not the post-projection key magnitudes. The argument: softmax
     normalizes natively, so additive log-gates compose cleanly.

  2. LEARNED-threshold gate: replace fixed `1/(1+d)` with
     sigmoid(W*d + b) where W, b are trained scalars. Initialized
     to approximate the original gate but free to discover its own
     threshold and slope from loss signal.

This script trains: crt_only (reference), hybrid_score, hybrid_learned
× 3 seeds × 1500 steps on the 20%-distractor TinyShakespeare mix.

Same corpus / model / optimizer as train_distractor_mix.py — the only
variable is the gate definition.
"""

import argparse
import json
import sys
import time
import statistics
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
    train_one,
)


ARCHS = ["crt_only", "hybrid_score", "hybrid_learned"]


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
    parser.add_argument("--distractor-frac", type=float, default=0.20)
    parser.add_argument("--out", type=str, default="results_gate_reformulation.json")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    chars, stoi, itos, encoded = make_dataset(
        seq_len=args.seq_len, source="tinyshakespeare",
    )
    vocab_size = len(chars)

    print(f"Gate reformulation — distractor_frac={args.distractor_frac:.2f}")
    print(f"Archs: {ARCHS}")
    print(f"Corpus: TinyShakespeare ({encoded.numel():,} chars, vocab {vocab_size})")
    print(f"Model: d_model={args.d_model}, n_blocks={args.n_blocks}, seq_len={args.seq_len}")
    print(f"Training: steps={args.steps}, batch={args.batch_size}, lr={args.lr}, seeds={seeds}",
          flush=True)

    all_results = {arch: [] for arch in ARCHS}
    per_seed_logs = []
    for seed in seeds:
        print(f"\n=========== seed {seed} ===========", flush=True)
        train_split, val_split = build_distractor_stream(
            encoded, args.distractor_frac, args.seq_len, seed,
        )
        seed_record = {"seed": seed, "archs": {}}
        for arch in ARCHS:
            r = train_one(arch, train_split, val_split, vocab_size, args, seed)
            all_results[arch].append(r["final_val"])
            seed_record["archs"][arch] = {
                "final_val": r["final_val"],
                "n_params": r["n_params"],
                "time": r["time"],
            }
            print(f"  [seed {seed}] {arch}: final_val={r['final_val']:.4f}", flush=True)
        per_seed_logs.append(seed_record)

    print()
    print("=" * 70)
    print(f"{'arch':<18} {'mean_final_val':>16} {'std':>10} {'vs crt_only':>14}")
    print("-" * 70)
    base = all_results["crt_only"]
    base_mean = sum(base) / len(base)
    summary = {"distractor_frac": args.distractor_frac, "steps": args.steps,
               "seeds": seeds, "per_seed": per_seed_logs, "summary": {}}
    for arch in ARCHS:
        vals = all_results[arch]
        mean = sum(vals) / len(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        if arch == "crt_only":
            tag = "—"
        else:
            wins = sum(1 for v, b in zip(vals, base) if v < b)
            rel = (mean - base_mean) / base_mean * 100
            tag = f"{rel:+.1f}% ({wins}/{len(vals)})"
        print(f"{arch:<18} {mean:>16.4f} {std:>10.4f} {tag:>14}")
        summary["summary"][arch] = {"mean": mean, "std": std,
                                     "vals": vals}

    print()
    print("Interpretation:")
    for arch in ["hybrid_score", "hybrid_learned"]:
        m = sum(all_results[arch]) / len(all_results[arch])
        rel = (m - base_mean) / base_mean * 100
        verdict = "GATE EARNS KEEP" if m < base_mean else "GATE STILL COSTS"
        wins = sum(1 for v, b in zip(all_results[arch], base) if v < b)
        print(f"  {arch:<18}: {rel:+.1f}% vs crt_only, wins {wins}/{len(base)} — {verdict}")

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

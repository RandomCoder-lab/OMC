"""Geodesic attention vs crt_only on distractor-mix TinyShakespeare.

The LAST attempt at substrate-as-attention-modulator. See
GEODESIC_ATTENTION_DERIVATION.md for the derivation.

The change vs the three previously falsified gates: substrate metric
is applied to POSITION INDICES (integer, native to the substrate's
basis), not to learned float activations. Implemented as an
ALiBi-style additive pre-softmax bias:

    scores[i, j] = (q_i · k_j) / √d − α · geodesic(i, j)

where geodesic(i, j) is the CRT-Fibonacci geodesic distance using
the SAME moduli as CRT-PE (5, 8, 13, 21, 34, 55, 89, 144). The
table is precomputed at construction; α is one learnable scalar
per block, initialized to 0 (model has to discover the bias is
useful from loss gradient alone).
"""

import argparse
import json
import sys
import time
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from corpus import make_dataset
from models import make_model
from train_distractor_mix import (
    build_distractor_stream,
    train_one,
)


ARCHS = ["crt_only", "hybrid_geodesic"]


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
    parser.add_argument("--out", type=str, default="results_geodesic_attention.json")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    chars, stoi, itos, encoded = make_dataset(
        seq_len=args.seq_len, source="tinyshakespeare",
    )
    vocab_size = len(chars)

    print(f"Geodesic attention — distractor_frac={args.distractor_frac:.2f}")
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
        summary["summary"][arch] = {"mean": mean, "std": std, "vals": vals}

    print()
    print("Interpretation:")
    m_geo = sum(all_results["hybrid_geodesic"]) / len(all_results["hybrid_geodesic"])
    rel = (m_geo - base_mean) / base_mean * 100
    wins = sum(1 for v, b in zip(all_results["hybrid_geodesic"], base) if v < b)
    if m_geo < base_mean:
        verdict = "GEODESIC EARNS KEEP — substrate works on positions, not activations"
    else:
        verdict = "GEODESIC ALSO FAILS — substrate is exhausted as attention modulator"
    print(f"  hybrid_geodesic vs crt_only: {rel:+.1f}%, wins {wins}/{len(base)}")
    print(f"  → {verdict}")

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

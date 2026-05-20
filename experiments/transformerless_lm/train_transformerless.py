"""First end-to-end transformerless-LM candidate.

Combines the three validated in-loop substrate primitives:

  1. CRT-Fibonacci positional encoding (validated: -5.4% vs sinusoidal,
     TinyShakespeare scale, 3/3 seeds).
  2. CRT-Fibonacci token-ID encoding added to embeddings (NEW — the
     missing third primitive called out in GEODESIC_RESULT.md
     "What's next" item 3).
  3. Geodesic attention bias on integer position pairs (validated:
     -0.4% vs crt_only, distractor mix, 3/3 seeds).

All three respect the architectural rule derived in GEODESIC_RESULT.md:

    SUBSTRATE METRIC APPLIES TO INTEGER QUANTITIES.

Positions, token IDs, and position-pairs are all intrinsically integer-
valued — the rule says substrate is a fair modulation signal there,
and not on continuous learned activations.

Bench design (ablation across the three primitives, same setup as
the geodesic experiment so deltas compose):

  crt_only         : CRT-PE only                  (baseline)
  token_crt        : CRT-PE + token-CRT           (isolates token-substrate)
  hybrid_geodesic  : CRT-PE + geodesic            (re-verifies geodesic win)
  transformerless  : CRT-PE + token-CRT + geodesic (the headline)

Distractor-mix TinyShakespeare, d_model=128, n_blocks=4, seq_len=128,
1500 steps, 3 seeds. Same regime as train_geodesic_attention.py.
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


ARCHS = ["crt_only", "token_crt", "hybrid_geodesic", "transformerless"]


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
    parser.add_argument("--out", type=str, default="results_transformerless.json")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    chars, stoi, itos, encoded = make_dataset(
        seq_len=args.seq_len, source="tinyshakespeare",
    )
    vocab_size = len(chars)

    print(f"Transformerless candidate — distractor_frac={args.distractor_frac:.2f}")
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
    print("=" * 78)
    print(f"{'arch':<18} {'mean_final_val':>16} {'std':>10} {'vs crt_only':>16}")
    print("-" * 78)
    base = all_results["crt_only"]
    base_mean = sum(base) / len(base)
    summary = {
        "distractor_frac": args.distractor_frac,
        "steps": args.steps,
        "seeds": seeds,
        "per_seed": per_seed_logs,
        "summary": {},
    }
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
        print(f"{arch:<18} {mean:>16.4f} {std:>10.4f} {tag:>16}")
        summary["summary"][arch] = {"mean": mean, "std": std, "vals": vals}

    print()
    print("Interpretation:")
    for arch in ("token_crt", "hybrid_geodesic", "transformerless"):
        m = sum(all_results[arch]) / len(all_results[arch])
        rel = (m - base_mean) / base_mean * 100
        wins = sum(1 for v, b in zip(all_results[arch], base) if v < b)
        print(f"  {arch:<18} vs crt_only: {rel:+.1f}%, wins {wins}/{len(base)}")

    # Stacking question: does the transformerless arch beat the better
    # of (token_crt, hybrid_geodesic), or just match the best of the two?
    m_tok = sum(all_results["token_crt"]) / len(all_results["token_crt"])
    m_geo = sum(all_results["hybrid_geodesic"]) / len(all_results["hybrid_geodesic"])
    m_all = sum(all_results["transformerless"]) / len(all_results["transformerless"])
    best_single = min(m_tok, m_geo)
    stack_delta = (m_all - best_single) / best_single * 100
    if m_all < best_single:
        verdict = "PRIMITIVES STACK — substrate components combine additively"
    elif m_all > base_mean:
        verdict = "PRIMITIVES INTERFERE — combined worse than baseline"
    else:
        verdict = "PRIMITIVES SATURATE — combined ≈ best individual"
    print(f"  transformerless vs best-of-two: {stack_delta:+.1f}% → {verdict}")

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

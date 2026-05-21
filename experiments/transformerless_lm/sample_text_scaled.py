"""Scaled-up text sampling — d=384, n_blocks=6, longer training.

At d=128 / 4 blocks / 2500 steps, even dense produces gibberish, so the
"is FibGen output usable?" question couldn't be answered. This script
trains at GPT-2-tiny-class parameters (d=384, n_blocks=6) for enough
steps to push dense into "barely-coherent Shakespeare" territory, then
compares FibGen and composed at that scale.

Wall-time budget (rough CPU estimates):
  dense_crt        d=384 6blk 6000 steps:  ~20 min
  fibgen_K32_cross d=384 6blk 6000 steps:  ~50 min
  composed         d=384 6blk 6000 steps:  ~80 min
Total: ~2.5 hours.

Prints best-val checkpoints + generated text for each arch.
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
from models_fibgen import FibGenLM, FibGenTransformerless
from train_distractor_mix import build_distractor_stream
from lazy_data import fib_positions_in_window, get_fib_strided_batch
from sample_text import evaluate, train, generate_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-blocks", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distractor-frac", type=float, default=0.20)
    parser.add_argument("--prompt", type=str,
                        default="ROMEO:\nWhat light through")
    parser.add_argument("--n-new", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", type=str, default="results_samples_scaled.txt")
    parser.add_argument("--archs", type=str,
                        default="dense_crt,fibgen_K32_cross,composed_transformerless")
    args = parser.parse_args()

    chars, stoi, itos, encoded = make_dataset(seq_len=args.seq_len,
                                                 source="tinyshakespeare")
    vocab_size = len(chars)
    train_split, val_split = build_distractor_stream(
        encoded, args.distractor_frac, args.seq_len, args.seed,
    )
    fib_positions = fib_positions_in_window(args.seq_len)

    arch_factories = {
        "dense_crt": lambda: make_model(
            "crt_only", vocab_size=vocab_size, seq_len=args.seq_len,
            d_model=args.d_model, n_blocks=args.n_blocks,
        ),
        "fibgen_K32_cross": lambda: FibGenLM(
            vocab_size=vocab_size, d_model=args.d_model,
            n_blocks=args.n_blocks, seq_len=args.seq_len, K=32, mode="cross",
        ),
        "composed_transformerless": lambda: FibGenTransformerless(
            vocab_size=vocab_size, d_model=args.d_model, n_blocks=args.n_blocks,
            seq_len=args.seq_len, K=32, mode="cross", n_specialists=5,
        ),
    }

    selected_archs = [a.strip() for a in args.archs.split(",")]

    space_id = stoi.get(" ", 0)
    prompt_ids = torch.tensor(
        [[stoi.get(c, space_id) for c in args.prompt]], dtype=torch.long,
    )

    print(f"Scaled-up sampling: d={args.d_model}, n_blocks={args.n_blocks}, "
          f"steps={args.steps}", flush=True)
    print(f"Archs: {selected_archs}", flush=True)

    samples = {}
    meta = {}
    for name in selected_archs:
        if name not in arch_factories:
            print(f"  skipping unknown arch: {name}", flush=True)
            continue
        t_arch = time.time()
        model = arch_factories[name]()
        model, best_val, best_step = train(name, model, train_split, val_split,
                                              args, fib_positions)
        wall = time.time() - t_arch
        meta[name] = {"best_val": best_val, "best_step": best_step,
                      "n_params": sum(p.numel() for p in model.parameters()),
                      "wall_seconds": wall}
        out_ids = generate_text(model, prompt_ids, args.n_new, args.seq_len,
                                  itos, temperature=args.temperature,
                                  top_k=args.top_k)
        text = "".join(itos[int(i)] for i in out_ids[0].tolist())
        samples[name] = text
        print(f"\n{'=' * 70}")
        print(f"SAMPLE from {name}  best_val={best_val:.4f} @ step {best_step}  "
              f"wall={wall:.0f}s")
        print('=' * 70)
        print(text)
        print('=' * 70, flush=True)

        # Save partial result after each arch so we have results even if a later one crashes.
        out_path = Path(__file__).parent / args.out
        with open(out_path, "w") as f:
            f.write(f"# Scaled-up samples (d={args.d_model}, n_blocks={args.n_blocks}, "
                    f"steps={args.steps}, temperature={args.temperature}, "
                    f"top_k={args.top_k})\n")
            f.write(f"# Prompt: {args.prompt!r}\n\n")
            for n, s in samples.items():
                m = meta[n]
                f.write(f"\n{'=' * 70}\n{n}  best_val={m['best_val']:.4f} "
                        f"@ step {m['best_step']}  params={m['n_params']:,}  "
                        f"wall={m['wall_seconds']:.0f}s\n"
                        f"{'=' * 70}\n{s}\n")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

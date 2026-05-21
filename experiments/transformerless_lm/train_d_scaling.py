"""d-scale ablation: does the substrate-recursive stack hold quality as d grows?

The single most important question before scaling further: at d=128
the gap to dense is small (FibGen +13%, FibRecLM+FibAdamW -1.9%). At
d=256 the FibGen gap GREW to +30%. If the gap keeps growing with d
the substrate basis doesn't scale and we need a new mechanism.

Bench: dense_crt baseline (standard AdamW) vs FibRecLM + FibonacciAdamW
(the validated substrate-recursive composition), at d in {64, 128, 256, 384}.

For each d we report:
  - best_val for each arch
  - gap = (substrate_val - dense_val) / dense_val * 100
  - storage compression of substrate vs dense

If gap stays bounded (say < 10%) across all d, the substrate is
scale-stable and we can confidently extrapolate to LLM scale.
If gap grows monotonically with d, the basis doesn't scale and we
need to redesign K(d) relationship or pick a different generator.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from corpus import make_dataset
from models import make_model
from models_fibrec import FibRecLM
from optimizers_fib import FibonacciAdamW
from train_distractor_mix import build_distractor_stream
from lazy_data import fib_positions_in_window, get_fib_strided_batch


def evaluate(model, val_split, batch_size, window, fib_positions, generator,
              n_batches=16):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = get_fib_strided_batch(val_split, batch_size, window,
                                           fib_positions, generator)
            logits = model(x)
            losses.append(F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item())
    model.train()
    return sum(losses) / len(losses)


def train_one(name, model, optimizer, train_split, val_split, args,
               fib_positions):
    torch.manual_seed(args.seed)
    gen = torch.Generator(); gen.manual_seed(args.seed + 1)
    n_params = sum(p.numel() for p in model.parameters())
    compr = None
    if hasattr(model, "storage_summary"):
        compr = model.storage_summary()["compression"]
    print(f"\n[train {name}] params={n_params:,}" +
          (f"  compression={compr:.1f}x" if compr else ""), flush=True)
    t0 = time.time()
    best_val = float("inf"); best_step = -1
    val_hist = []
    eval_every = max(args.steps // 8, 100)
    for step in range(args.steps):
        x, y = get_fib_strided_batch(train_split, args.batch_size, args.seq_len,
                                       fib_positions, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                y.reshape(-1))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % eval_every == 0 or step == args.steps - 1:
            vl = evaluate(model, val_split, args.batch_size, args.seq_len,
                          fib_positions, gen)
            val_hist.append((step, vl, time.time() - t0))
            marker = ""
            if vl < best_val:
                best_val = vl; best_step = step
                marker = " ← BEST"
            print(f"  step {step:5d}  val={vl:.4f}  ({time.time()-t0:.1f}s){marker}",
                  flush=True)
    return {"name": name, "n_params": n_params, "compression": compr,
             "best_val": best_val, "best_step": best_step,
             "wall": time.time() - t0, "val_history": val_hist}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distractor-frac", type=float, default=0.20)
    parser.add_argument("--d-models", type=str, default="64,128,256,384")
    parser.add_argument("--out", type=str, default="results_d_scaling.json")
    args = parser.parse_args()

    chars, stoi, itos, encoded = make_dataset(seq_len=args.seq_len,
                                                 source="tinyshakespeare")
    vocab_size = len(chars)
    train_split, val_split = build_distractor_stream(
        encoded, args.distractor_frac, args.seq_len, args.seed,
    )
    fib_positions = fib_positions_in_window(args.seq_len)

    print(f"d-scale ablation: d_models = {args.d_models}")
    print(f"Lazy data: P={len(fib_positions)} tokens/seq", flush=True)

    d_values = [int(x) for x in args.d_models.split(",")]
    results = []

    for d in d_values:
        print(f"\n{'='*60}")
        print(f"d_model = {d}")
        print('='*60)

        # Dense baseline at this d
        m = make_model("crt_only", vocab_size=vocab_size,
                        seq_len=args.seq_len, d_model=d, n_blocks=4)
        opt = torch.optim.AdamW(m.parameters(), lr=args.lr)
        r_dense = train_one(f"dense_d{d}", m, opt, train_split, val_split,
                              args, fib_positions)
        r_dense["d_model"] = d
        results.append(r_dense)

        # FibRecLM + FibAdamW (the composed substrate-recursive stack)
        m = FibRecLM(vocab_size=vocab_size, d_model=d, n_blocks=4,
                      seq_len=args.seq_len, K=32, mode="cross")
        opt = FibonacciAdamW(m.parameters(), lr=args.lr)
        r_substrate = train_one(f"fibrec_fibadamw_d{d}", m, opt, train_split,
                                  val_split, args, fib_positions)
        r_substrate["d_model"] = d
        results.append(r_substrate)

    # Summary table
    print()
    print("=" * 92)
    print(f"{'d_model':>8} {'arch':<24} {'params':>12} {'compr':>8} "
          f"{'best_val':>10} {'gap %':>8}")
    print("-" * 92)
    by_d = {}
    for r in results:
        by_d.setdefault(r["d_model"], {})[r["name"].split("_d")[0]] = r
    for d, pair in by_d.items():
        d_r = pair["dense"]
        s_r = pair["fibrec_fibadamw"]
        gap = (s_r["best_val"] - d_r["best_val"]) / d_r["best_val"] * 100
        c_dense = "1.0x"
        c_sub = f"{s_r['compression']:.1f}x" if s_r["compression"] else "?"
        print(f"{d:>8} {d_r['name']:<24} {d_r['n_params']:>12,} {c_dense:>8} "
              f"{d_r['best_val']:>10.4f} {'-':>8}")
        print(f"{d:>8} {s_r['name']:<24} {s_r['n_params']:>12,} {c_sub:>8} "
              f"{s_r['best_val']:>10.4f} {gap:>+7.1f}%")

    print()
    print("VERDICT (gap as a function of d):")
    for d, pair in sorted(by_d.items()):
        d_r = pair["dense"]; s_r = pair["fibrec_fibadamw"]
        gap = (s_r["best_val"] - d_r["best_val"]) / d_r["best_val"] * 100
        print(f"  d={d:>4}: dense val={d_r['best_val']:.4f}, "
              f"substrate val={s_r['best_val']:.4f}, gap={gap:+.1f}%, "
              f"compression={s_r['compression']:.1f}x")

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

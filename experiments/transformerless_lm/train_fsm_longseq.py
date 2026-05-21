"""FSM at long-T: does the asymptotic O(T*d^2) win over attention's O(T^2*d)?

NO lazy-data this time — we want attention to pay its full quadratic
cost so FSM's linear cost can demonstrate the asymptotic win.

Bench design:
  T=128:  dense_crt   vs FSMLM   — expected DRAW (attention is cheap at small T)
  T=512:  dense_crt   vs FSMLM   — expected FSM WINS (T^2 quadrupled, linear flat)

If FSM is ~2x faster at T=512, the substrate-recurrence operator is
empirically validated as the right way to scale to long context.
If FSM is still slower at T=512, the Python-loop overhead eats the
asymptotic win and we need parallel scan / kernel work to realize it.
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
from models_fsm import FSMLM
from train_distractor_mix import build_distractor_stream


def get_dense_batch(encoded, batch_size, seq_len, generator):
    """Standard contiguous batches — NOT Fibonacci-strided."""
    n = encoded.numel()
    ix = torch.randint(0, n - seq_len - 1, (batch_size,), generator=generator)
    x = torch.stack([encoded[i:i + seq_len] for i in ix])
    y = torch.stack([encoded[i + 1:i + seq_len + 1] for i in ix])
    return x, y


def evaluate(model, val_split, batch_size, seq_len, generator, n_batches=16):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = get_dense_batch(val_split, batch_size, seq_len, generator)
            logits = model(x)
            losses.append(F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item())
    model.train()
    return sum(losses) / len(losses)


def train_one(name, model, train_split, val_split, args):
    torch.manual_seed(args.seed)
    gen = torch.Generator(); gen.manual_seed(args.seed + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[train {name}] params={n_params:,}", flush=True)
    t0 = time.time()
    best_val = float("inf")
    best_step = -1
    val_hist = []
    for step in range(args.steps):
        x, y = get_dense_batch(train_split, args.batch_size, args.seq_len, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                y.reshape(-1))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            vl = evaluate(model, val_split, args.batch_size, args.seq_len, gen)
            val_hist.append((step, vl, time.time() - t0))
            marker = ""
            if vl < best_val:
                best_val = vl; best_step = step
                marker = " ← BEST"
            print(f"  step {step:5d}  val={vl:.4f}  ({time.time()-t0:.1f}s){marker}",
                  flush=True)
    return {"name": name, "n_params": n_params, "best_val": best_val,
             "best_step": best_step, "wall": time.time() - t0,
             "val_history": val_hist, "seq_len": args.seq_len}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distractor-frac", type=float, default=0.20)
    parser.add_argument("--seq-lens", type=str, default="128,512")
    parser.add_argument("--out", type=str, default="results_fsm_longseq.json")
    args = parser.parse_args()

    seq_lens = [int(s) for s in args.seq_lens.split(",")]
    chars, stoi, itos, encoded = make_dataset(seq_len=max(seq_lens),
                                                 source="tinyshakespeare")
    vocab_size = len(chars)
    print(f"Corpus: TinyShakespeare ({encoded.numel():,} chars)")
    print(f"Seq lens to test: {seq_lens}")
    print(f"Training: steps={args.steps}, batch={args.batch_size}, lr={args.lr}",
          flush=True)

    results = []
    for T in seq_lens:
        # Build splits per-T (build_distractor_stream depends on seq_len)
        train_split, val_split = build_distractor_stream(
            encoded, args.distractor_frac, T, args.seed,
        )
        args_T = argparse.Namespace(**vars(args))
        args_T.seq_len = T

        # Dense baseline
        m = make_model("crt_only", vocab_size=vocab_size, seq_len=T,
                        d_model=args.d_model, n_blocks=args.n_blocks)
        results.append(train_one(f"dense_crt_T{T}", m, train_split, val_split, args_T))

        # FSM
        m = FSMLM(vocab_size=vocab_size, d_model=args.d_model,
                   n_blocks=args.n_blocks, seq_len=T, K=32, mode="cross")
        results.append(train_one(f"fsm_T{T}", m, train_split, val_split, args_T))

    print()
    print("=" * 92)
    print(f"{'config':<22} {'seq_len':>8} {'params':>10} {'best_val':>10} {'wall':>10}")
    print("-" * 92)
    for r in results:
        print(f"{r['name']:<22} {r['seq_len']:>8} {r['n_params']:>10,} "
              f"{r['best_val']:>10.4f} {r['wall']:>9.1f}s")

    # Speedup crossover table
    print()
    print("FSM vs DENSE speed at each T:")
    by_seq = {}
    for r in results:
        by_seq.setdefault(r["seq_len"], {})[r["name"].split("_T")[0]] = r
    for T, pair in by_seq.items():
        if "dense_crt" in pair and "fsm" in pair:
            d = pair["dense_crt"]; f = pair["fsm"]
            speedup = d["wall"] / f["wall"]
            qual_delta = (f["best_val"] - d["best_val"]) / d["best_val"] * 100
            print(f"  T={T:>4}: FSM is {speedup:.2f}x dense wall-clock; "
                  f"val delta {qual_delta:+.1f}%")

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

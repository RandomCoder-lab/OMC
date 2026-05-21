"""Substrate-native architecture bench.

Three architectures, all at matched parameter count (~801K) and same
training regime as the prior benches:

  dense_crt    : crt_only (the strongest prior baseline; dense matmul attn,
                  dense FFN at expansion=4)
  fib_offset   : Fibonacci-offset sparse attention + Zeckendorf-routed FFN
  crt_bucket   : CRT-bucket attention + Zeckendorf-routed FFN

The scalability claim being tested:

  Substrate-native attention is O(T · log_phi_pi(T) · d), substrate-native
  FFN is O(d²/K) per token. At fixed param count, the substrate variant
  performs strictly fewer FLOPs than the dense baseline, and the gap
  grows with sequence length.

What this bench measures:

  - Effective FLOPs (the architectural claim, kernel-independent)
  - Wall-clock per step (the implementation cost — currently a tax)
  - Val loss at fixed step budgets (does the architecture train?)

Wall-clock parity is a kernel question (custom sparse/grouped matmul).
This bench separates that from the architectural question.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from corpus import make_dataset
from models import make_model
from models_substrate import SubstrateLM
from train_distractor_mix import (
    build_distractor_stream,
    get_batch_split,
    evaluate,
)


def build_arch(arch: str, vocab_size: int, seq_len: int,
               d_model: int, n_blocks: int, K_specialists: int = 5,
               bucket_modulus: int = 13):
    if arch == "dense_crt":
        return make_model(
            "crt_only", vocab_size=vocab_size, seq_len=seq_len,
            d_model=d_model, n_blocks=n_blocks,
        ), None  # no effective_flops accessor
    if arch == "fib_offset":
        m = SubstrateLM(vocab_size=vocab_size, d_model=d_model,
                         n_blocks=n_blocks, seq_len=seq_len,
                         attn_kind="fib", K_specialists=K_specialists)
        return m, m
    if arch == "crt_bucket":
        m = SubstrateLM(vocab_size=vocab_size, d_model=d_model,
                         n_blocks=n_blocks, seq_len=seq_len,
                         attn_kind="bucket", K_specialists=K_specialists,
                         bucket_modulus=bucket_modulus)
        return m, m
    raise ValueError(f"unknown arch: {arch}")


def dense_attn_flops(T: int, d: int, n_blocks: int) -> int:
    # Dense causal attention: Q·K^T over T(T+1)/2 causal pairs, then ·V same.
    return n_blocks * 2 * 2 * (T * (T + 1) // 2) * d


def dense_ffn_flops_per_token(d: int, n_blocks: int, expansion: int = 4) -> int:
    return n_blocks * 2 * d * (expansion * d) * 2


def train_one(arch, train_split, val_split, vocab_size, args, seed):
    torch.manual_seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed + 1)

    model, substrate_handle = build_arch(
        arch, vocab_size, args.seq_len, args.d_model, args.n_blocks,
        K_specialists=args.K_specialists, bucket_modulus=args.bucket_modulus,
    )
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # FLOP accounting
    if substrate_handle is None:
        attn_flops = dense_attn_flops(args.seq_len, args.d_model, args.n_blocks)
        ffn_flops_per_tok = dense_ffn_flops_per_token(args.d_model, args.n_blocks)
    else:
        attn_flops = substrate_handle.effective_attention_flops()
        ffn_flops_per_tok = substrate_handle.effective_ffn_flops_per_token()
    # Per-forward total FLOPs (rough):  attn + T·ffn_per_tok
    fwd_flops = attn_flops + args.seq_len * ffn_flops_per_tok

    print(f"\n[arch={arch}] params={n_params:,}", flush=True)
    print(f"  attn_flops/fwd={attn_flops:,}  "
          f"ffn_flops/token={ffn_flops_per_tok:,}  "
          f"total_fwd_flops≈{fwd_flops:,}", flush=True)

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
            vl = evaluate(model, val_split, args.batch_size, args.seq_len,
                          n_batches=16, generator=gen)
            val_history.append((step, vl))
            elapsed = time.time() - t0
            print(f"  step {step:5d}  train={tl:.4f}  val={vl:.4f}  ({elapsed:.1f}s)",
                  flush=True)

    last_few = val_history[-3:]
    final_val = sum(v for _, v in last_few) / len(last_few)
    return dict(
        arch=arch,
        n_params=n_params,
        attn_flops=attn_flops,
        ffn_flops_per_token=ffn_flops_per_tok,
        fwd_flops=fwd_flops,
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
    parser.add_argument("--seeds", type=str, default="42,7")
    parser.add_argument("--distractor-frac", type=float, default=0.20)
    parser.add_argument("--K-specialists", type=int, default=5)
    parser.add_argument("--bucket-modulus", type=int, default=13)
    parser.add_argument("--out", type=str, default="results_substrate_native.json")
    parser.add_argument(
        "--archs", type=str, default="dense_crt,fib_offset,crt_bucket",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    archs = args.archs.split(",")

    chars, stoi, itos, encoded = make_dataset(
        seq_len=args.seq_len, source="tinyshakespeare",
    )
    vocab_size = len(chars)

    print(f"Substrate-native bench")
    print(f"Archs: {archs}")
    print(f"Corpus: TinyShakespeare ({encoded.numel():,} chars, vocab {vocab_size})")
    print(f"Model: d_model={args.d_model}, n_blocks={args.n_blocks}, "
          f"seq_len={args.seq_len}, K={args.K_specialists}")
    print(f"Training: steps={args.steps}, batch={args.batch_size}, lr={args.lr}, "
          f"seeds={seeds}", flush=True)

    all_results = {arch: [] for arch in archs}
    per_run_logs = []

    for seed in seeds:
        print(f"\n=========== seed {seed} ===========", flush=True)
        train_split, val_split = build_distractor_stream(
            encoded, args.distractor_frac, args.seq_len, seed,
        )
        for arch in archs:
            r = train_one(arch, train_split, val_split, vocab_size, args, seed)
            all_results[arch].append(r)
            per_run_logs.append({**r, "seed": seed})
            print(f"  [seed {seed}] {arch}: final_val={r['final_val']:.4f} "
                  f"(time={r['time']:.1f}s)", flush=True)

    # Summary
    print()
    print("=" * 84)
    print(f"{'arch':<14} {'params':>8} {'attn_flops':>14} {'ffn_flops':>14} "
          f"{'val(mean)':>11} {'time(s)':>9}")
    print("-" * 84)
    for arch in archs:
        runs = all_results[arch]
        vals = [r["final_val"] for r in runs]
        times = [r["time"] for r in runs]
        mean_v = sum(vals)/len(vals)
        mean_t = sum(times)/len(times)
        attn_f = runs[0]["attn_flops"]
        ffn_f = runs[0]["ffn_flops_per_token"]
        n_p = runs[0]["n_params"]
        print(f"{arch:<14} {n_p:>8,} {attn_f:>14,} {ffn_f:>14,} "
              f"{mean_v:>11.4f} {mean_t:>9.1f}")

    # FLOP ratios vs dense
    print()
    print("FLOP REDUCTION vs dense_crt baseline:")
    base = all_results.get("dense_crt", [None])[0]
    if base is not None:
        for arch in archs:
            if arch == "dense_crt":
                continue
            ar = all_results[arch][0]
            attn_ratio = base["attn_flops"] / max(ar["attn_flops"], 1)
            ffn_ratio = base["ffn_flops_per_token"] / max(ar["ffn_flops_per_token"], 1)
            fwd_ratio = base["fwd_flops"] / max(ar["fwd_flops"], 1)
            print(f"  {arch:<14} attn:{attn_ratio:5.1f}x  ffn:{ffn_ratio:5.1f}x  "
                  f"total_fwd:{fwd_ratio:5.1f}x")

    # Val loss at fixed step budgets
    print()
    print("VAL LOSS @ FIXED STEP BUDGET (mean across seeds):")
    print(f"  {'step':<6} " + "  ".join(f"{a:<14}" for a in archs))
    for step_target in [100, 300, 500, 1000, 1500]:
        cells = []
        for arch in archs:
            vals = []
            for r in all_results[arch]:
                best = None
                for step, val in r["val_history"]:
                    if step <= step_target: best = val
                    else: break
                if best is not None: vals.append(best)
            if vals:
                cells.append(f"{sum(vals)/len(vals):<14.4f}")
            else:
                cells.append(f"{'-':<14}")
        print(f"  {step_target:<6} " + "  ".join(cells))

    # Save
    out_path = Path(__file__).parent / args.out
    summary = {
        "archs": archs,
        "seeds": seeds,
        "steps": args.steps,
        "seq_len": args.seq_len,
        "d_model": args.d_model,
        "n_blocks": args.n_blocks,
        "K_specialists": args.K_specialists,
        "runs": per_run_logs,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

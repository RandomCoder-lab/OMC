"""Weight-substrate reformulation bench (the user's two principles).

Tests Principle A (tied QKV via substrate channel permutation) and
Principle B (Fibonacci-tier weight quantization) separately and combined.

Archs:
  dense_crt        : standard crt_only baseline (independent Q,K,V,out)
  tied_substrate   : ONE shared W; K and V are channel-rotations of W·x
                     by Fibonacci strides F_K=13, F_V=55. Output proj
                     is independent. Attention params: 2d² (vs 4d²).
  + fib_tier_quant : applied post-training to either of the above.

For each (arch, n_tiers) combination we report:
  - n_attention_params
  - val loss after training
  - val loss after Fibonacci-tier quantization
  - per-tier-value unique count (does the quantizer use all tiers?)

The hypotheses:
  A: tied_substrate trains to val loss within ~5% of dense_crt at
     ~half the attention params.
  B: post-hoc Fibonacci-tier quantization at n_tiers=8 (4-bit equiv.)
     loses < 0.1 nats of val loss vs the trained fp32 model.
  A+B: both principles compose; combined model trains AND quantizes
        cleanly.

If A or B fails: we learn which substrate orientation needs revisiting.
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from corpus import make_dataset
from models import make_model
from models_substrate import (
    SubstrateLM,
    fibonacci_quantize_model,
    fibonacci_tier_values,
)
from train_distractor_mix import (
    build_distractor_stream,
    get_batch_split,
    evaluate,
)


def train_arch(arch: str, train_split, val_split, vocab_size, args, seed: int):
    """Train one architecture; returns the trained model + final val loss."""
    torch.manual_seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed + 1)
    if arch == "dense_crt":
        model = make_model(
            "crt_only", vocab_size=vocab_size, seq_len=args.seq_len,
            d_model=args.d_model, n_blocks=args.n_blocks,
        )
    elif arch == "tied_substrate":
        model = SubstrateLM(
            vocab_size=vocab_size, d_model=args.d_model,
            n_blocks=args.n_blocks, seq_len=args.seq_len,
            attn_kind="tied", K_specialists=args.K_specialists,
            tied_F_K=args.tied_F_K, tied_F_V=args.tied_F_V,
        )
    else:
        raise ValueError(arch)

    n_params = sum(p.numel() for p in model.parameters())
    n_attn_params = sum(
        p.numel() for n, p in model.named_parameters()
        if any(s in n for s in ("attn", "qkv", ".W."))
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"\n[arch={arch}] total params={n_params:,}, "
          f"attn params={n_attn_params:,}", flush=True)
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
            vl = evaluate(model, val_split, args.batch_size, args.seq_len,
                          n_batches=16, generator=gen)
            elapsed = time.time() - t0
            print(f"  step {step:5d}  train={loss.item():.4f}  val={vl:.4f}  "
                  f"({elapsed:.1f}s)", flush=True)

    final_val = evaluate(model, val_split, args.batch_size, args.seq_len,
                          n_batches=32, generator=gen)
    return model, final_val, n_params, n_attn_params


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
    parser.add_argument("--K-specialists", type=int, default=5)
    parser.add_argument("--tied-F-K", type=int, default=13)
    parser.add_argument("--tied-F-V", type=int, default=55)
    parser.add_argument("--tier-sweep", type=str, default="4,8,16,32",
                        help="Comma-separated n_tiers values for the "
                             "quantization sweep.")
    parser.add_argument("--out", type=str, default="results_weight_substrate.json")
    args = parser.parse_args()

    tier_sweep = [int(t) for t in args.tier_sweep.split(",")]

    chars, stoi, itos, encoded = make_dataset(
        seq_len=args.seq_len, source="tinyshakespeare",
    )
    vocab_size = len(chars)
    train_split, val_split = build_distractor_stream(
        encoded, args.distractor_frac, args.seq_len, args.seed,
    )

    print(f"Weight-substrate reformulation bench")
    print(f"Corpus: TinyShakespeare ({encoded.numel():,} chars, vocab {vocab_size})")
    print(f"Model: d={args.d_model}, n_blocks={args.n_blocks}, "
          f"seq_len={args.seq_len}, F_K={args.tied_F_K}, F_V={args.tied_F_V}")
    print(f"Tier sweep: {tier_sweep}", flush=True)

    results = {"archs": {}}
    eval_gen = torch.Generator()

    for arch in ["dense_crt", "tied_substrate"]:
        model, final_val, n_params, n_attn = train_arch(
            arch, train_split, val_split, vocab_size, args, args.seed,
        )
        arch_record = {
            "n_params": n_params,
            "n_attn_params": n_attn,
            "val_fp32": final_val,
            "quantized": {},
        }
        print(f"\n  ✓ [arch={arch}] fp32 final_val = {final_val:.4f}")

        # ---- Principle B: post-hoc Fibonacci-tier quantization sweep ----
        state_dict_orig = {k: v.clone() for k, v in model.state_dict().items()}
        configs = []
        # Fibonacci basis: full cross product (already studied in v2)
        for reciprocals in [False, True]:
            for scale in ["per_tensor", "per_row"]:
                for n_tiers in tier_sweep:
                    configs.append((n_tiers, reciprocals, scale, "fibonacci"))
        # phi_power basis: only with per_row (the winning scale from v2);
        # reciprocals flag has no meaning for phi_power so leave False.
        for scale in ["per_tensor", "per_row"]:
            for n_tiers in tier_sweep:
                configs.append((n_tiers, False, scale, "phi_power"))
        for n_tiers, reciprocals, scale, tier_basis in configs:
            model.load_state_dict(state_dict_orig)
            stats = fibonacci_quantize_model(
                model, n_tiers=n_tiers,
                reciprocals=reciprocals, scale=scale, tier_basis=tier_basis,
            )
            eval_gen.manual_seed(args.seed + 1000)
            vq = evaluate(model, val_split, args.batch_size, args.seq_len,
                          n_batches=32, generator=eval_gen)
            n_unique_total = sum(s["n_unique_tier_values"]
                                  for s in stats["per_tensor"].values())
            n_tensors = stats["tensors_quantized"]
            avg_unique = n_unique_total / max(n_tensors, 1)
            basis_tag = "phi" if tier_basis == "phi_power" else (
                "frec" if reciprocals else "fnor")
            key = f"n{n_tiers}_{basis_tag}_{scale}"
            print(f"    {key:<24} → val={vq:.4f}  Δ={vq - final_val:+.4f}  "
                  f"avg_unique={avg_unique:.1f}", flush=True)
            arch_record["quantized"][key] = {
                "n_tiers": n_tiers,
                "reciprocals": reciprocals,
                "scale": scale,
                "tier_basis": tier_basis,
                "val": vq,
                "delta": vq - final_val,
                "params_quantized": stats["params_quantized"],
                "avg_unique_tier_values": avg_unique,
            }
        model.load_state_dict(state_dict_orig)
        results["archs"][arch] = arch_record

    # ---- Summary tables ----
    print()
    print("=" * 110)
    print("FP32 BASELINES")
    print("-" * 110)
    print(f"{'arch':<18} {'attn_params':>12} {'total':>10} {'fp32_val':>10}")
    for arch in ["dense_crt", "tied_substrate"]:
        r = results["archs"][arch]
        print(f"{arch:<18} {r['n_attn_params']:>12,} {r['n_params']:>10,} "
              f"{r['val_fp32']:>10.4f}")

    print()
    print("=" * 110)
    print("QUANTIZATION SWEEP — Δ vs fp32 for each arch")
    print("(rec=with reciprocal Fibonacci tiers; nor=Fibonacci only)")
    print("-" * 110)
    for arch in ["dense_crt", "tied_substrate"]:
        r = results["archs"][arch]
        print(f"\n  {arch}  (fp32 val = {r['val_fp32']:.4f}):")
        print(f"    {'n_tiers':>8} {'basis':>10} {'tier_set':>10} {'scale':>12} "
              f"{'val':>10} {'Δ':>10} {'unique':>10}")
        for key, q in r["quantized"].items():
            basis = q.get("tier_basis", "fibonacci")
            tag = "—" if basis == "phi_power" else (
                'rec' if q['reciprocals'] else 'nor')
            print(f"    {q['n_tiers']:>8} {basis:>10} {tag:>10} "
                  f"{q['scale']:>12} {q['val']:>10.4f} {q['delta']:>+10.4f} "
                  f"{q['avg_unique_tier_values']:>10.1f}")

    # ---- Interpretation ----
    print()
    print("=" * 110)
    print("INTERPRETATION")
    print("-" * 110)
    a_fp32 = results["archs"]["dense_crt"]["val_fp32"]
    t_fp32 = results["archs"]["tied_substrate"]["val_fp32"]
    a_attn = results["archs"]["dense_crt"]["n_attn_params"]
    t_attn = results["archs"]["tied_substrate"]["n_attn_params"]
    rel = (t_fp32 - a_fp32) / a_fp32 * 100
    print(f"\nPRINCIPLE A (tied substrate vs dense_crt):")
    print(f"  val_fp32 delta: {t_fp32 - a_fp32:+.4f} ({rel:+.1f}%)")
    print(f"  attn param reduction: {a_attn / t_attn:.2f}x")
    verdict_a = ("VALIDATED" if abs(rel) < 5 else
                  ("BEAT BASELINE" if rel < 0 else "NOT VALIDATED"))
    print(f"  → PRINCIPLE A: {verdict_a}")

    print(f"\nPRINCIPLE B (best quantizer per arch, vs ≤0.10 nat threshold):")
    for arch in ["dense_crt", "tied_substrate"]:
        r = results["archs"][arch]
        best_key = min(r["quantized"], key=lambda k: r["quantized"][k]["delta"])
        best = r["quantized"][best_key]
        verdict_b = "VALIDATED" if best["delta"] < 0.10 else (
            "USABLE" if best["delta"] < 0.30 else "BROKEN")
        print(f"  {arch}:")
        print(f"    best = {best_key}  (Δ={best['delta']:+.4f}, "
              f"n_tiers={best['n_tiers']}, rec={best['reciprocals']}, "
              f"scale={best['scale']})  → {verdict_b}")

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

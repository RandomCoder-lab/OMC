"""A/B bench for substrate-aware activation.

Patches FibRecLM's FFN to use SubstrateGELU instead of F.gelu.
Everything else identical: FibAdamW, ce_fft loss, K-shrink schedule,
TinyShakespeare data, lazy loading. Only the activation function
differs.

Three arms:
  ce_fft + gelu (baseline — the previous best)
  ce_fft + substrate_gelu     (hard attractor snap with STE)
  ce_fft + substrate_gelu_soft (blendable substrate coupling)
"""

import argparse
import json
import sys
import time
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from corpus import make_dataset
from models_fibrec import FibRecLM, stateless_fibgen_forward, make_fib_basis
from models_fibgen import FibGenLinear, FIBONACCI
from optimizers_fib import FibonacciAdamW
from train_distractor_mix import build_distractor_stream
from lazy_data import fib_positions_in_window, get_fib_strided_batch
from train_K_shrink import K_schedule_tier_walk, set_K_active_recursive
from losses_substrate import substrate_fft_loss
from activations_substrate import (SubstrateGELU, SubstrateGELUSoft,
                                      SubstrateGELUInverse, PhiPiFibActivation,
                                      BinetFibActivation, SubstrateNegAsymmetric,
                                      SubstrateNegAsymmetricMulti,
                                      SubstrateNegMultiRefined,
                                      SubstrateNegMultiAdvanced)


class FibRecLMWithActivation(FibRecLM):
    """FibRecLM but with a swappable activation in the FFN.

    Adds one activation module per block. Overrides _layer_forward to
    use that activation instead of F.gelu.
    """
    def __init__(self, *args, activation_cls=None, **kwargs):
        super().__init__(*args, **kwargs)
        if activation_cls is None:
            self.activations = None  # use F.gelu (baseline)
        else:
            self.activations = nn.ModuleList(
                [activation_cls() for _ in range(self.n_blocks)]
            )

    def _layer_forward(self, x, mask, n, seeds_n):
        qkv_s, out_s, w1_s, w2_s = seeds_n
        x_norm = self.ln1s[n](x)
        qkv_basis = {"cos_i": self.qkv_cos_i, "sin_i": self.qkv_sin_i,
                      "cos_j": self.qkv_cos_j, "sin_j": self.qkv_sin_j}
        qkv = stateless_fibgen_forward(x_norm, qkv_s, qkv_basis, self.K)
        q, k, v = qkv.chunk(3, dim=-1)
        scale = 1.0 / math.sqrt(self.d_model)
        scores = (q @ k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out_basis = {"cos_i": self.out_cos_i, "sin_i": self.out_sin_i,
                      "cos_j": self.out_cos_j, "sin_j": self.out_sin_j}
        x = x + stateless_fibgen_forward(attn @ v, out_s, out_basis, self.K)
        # FFN with swappable activation
        x_norm2 = self.ln2s[n](x)
        w1_basis = {"cos_i": self.w1_cos_i, "sin_i": self.w1_sin_i,
                      "cos_j": self.w1_cos_j, "sin_j": self.w1_sin_j}
        w2_basis = {"cos_i": self.w2_cos_i, "sin_i": self.w2_sin_i,
                      "cos_j": self.w2_cos_j, "sin_j": self.w2_sin_j}
        h = stateless_fibgen_forward(x_norm2, w1_s, w1_basis, self.K)
        if self.activations is not None:
            h = self.activations[n](h)
        else:
            h = F.gelu(h)
        x = x + stateless_fibgen_forward(h, w2_s, w2_basis, self.K)
        return x


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


def train_one(name, activation_cls, train_split, val_split, vocab_size,
               args, fib_positions):
    torch.manual_seed(args.seed)
    gen = torch.Generator(); gen.manual_seed(args.seed + 1)
    model = FibRecLMWithActivation(
        vocab_size=vocab_size, d_model=args.d_model, n_blocks=args.n_blocks,
        seq_len=args.seq_len, K=args.K_init, mode="cross",
        activation_cls=activation_cls,
    )
    optimizer = FibonacciAdamW(model.parameters(), lr=args.lr)
    sched = lambda s, T: K_schedule_tier_walk(s, T, K_init=args.K_init,
                                                 K_min=args.K_min)
    n_params = sum(p.numel() for p in model.parameters())
    act_name = activation_cls.__name__ if activation_cls else "GELU"
    print(f"\n[train {name}]  activation={act_name}  params={n_params:,}",
          flush=True)
    t0 = time.time()
    best_val = float("inf"); best_step = -1
    cur_K = None
    val_hist = []
    eval_every = max(args.steps // 15, 250)
    for step in range(args.steps):
        new_K = sched(step, args.steps)
        if new_K != cur_K:
            set_K_active_recursive(model, new_K)
            cur_K = new_K
        x, y = get_fib_strided_batch(train_split, args.batch_size, args.seq_len,
                                       fib_positions, gen)
        logits = model(x)
        loss = substrate_fft_loss(logits, y, vocab_size,
                                    lambda_substrate=args.lambda_sub)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % eval_every == 0 or step == args.steps - 1:
            vl = evaluate(model, val_split, args.batch_size, args.seq_len,
                          fib_positions, gen)
            val_hist.append((step, vl, time.time() - t0))
            marker = ""
            if vl < best_val:
                best_val = vl; best_step = step
                marker = " ← BEST"
            print(f"  step {step:5d}  val={vl:.4f}  K={cur_K}  "
                  f"({time.time()-t0:.1f}s){marker}", flush=True)
    return {"name": name, "n_params": n_params, "best_val": best_val,
             "best_step": best_step, "wall": time.time() - t0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distractor-frac", type=float, default=0.20)
    parser.add_argument("--K-init", type=int, default=89)
    parser.add_argument("--K-min", type=int, default=13)
    parser.add_argument("--lambda-sub", type=float, default=0.01)
    parser.add_argument("--out", type=str, default="results_substrate_activation.json")
    args = parser.parse_args()

    chars, stoi, itos, encoded = make_dataset(seq_len=args.seq_len,
                                                 source="tinyshakespeare")
    vocab_size = len(chars)
    train_split, val_split = build_distractor_stream(
        encoded, args.distractor_frac, args.seq_len, args.seed,
    )
    fib_positions = fib_positions_in_window(args.seq_len)

    results = {}
    # Baseline GELU val on this config is 2.5920 (from prior bench).
    # Skipping its re-run to save compute.
    for name, cls in [
        ("substrate_neg_multi_advanced", SubstrateNegMultiAdvanced),  # R2+R4+R5+R6
    ]:
        results[name] = train_one(name, cls, train_split, val_split,
                                    vocab_size, args, fib_positions)

    GELU_BASELINE_REF = 2.5920   # from prior identical-config bench
    print()
    print("=" * 84)
    print(f"{'arch':<26} {'params':>10} {'best_val':>10} {'wall':>10}")
    print('-' * 84)
    print(f"{'(gelu baseline ref)':<26} {'':>10} {GELU_BASELINE_REF:>10.4f} {'-':>10}")
    print('-' * 84)
    for name, r in results.items():
        d = (r["best_val"] - GELU_BASELINE_REF) / GELU_BASELINE_REF * 100
        delta = f"  ({d:+.2f}% vs gelu)"
        print(f"{name:<26} {r['n_params']:>10,} {r['best_val']:>10.4f} "
              f"{r['wall']:>9.1f}s{delta}")

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

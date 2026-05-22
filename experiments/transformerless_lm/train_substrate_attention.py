"""Substrate-similarity attention bench.

Replaces the Q·K^T dot product in FibRecLM's attention with substrate
L1 distance in a K_sig-dim signature space, while keeping softmax as
the probability normalizer (the substrate change is in score
computation, not in normalization).

Score formula:
    sig_q = q[..., :K_sig]                       # first K_sig dims of Q
    sig_k = k[..., :K_sig]                       # first K_sig dims of K
    dist[i,j] = sum |sig_q[i] - sig_k[j]|_1      # L1 attractor distance
    score[i,j] = -dist[i,j] / sqrt(K_sig)        # negate so close => high
    attn = softmax(score)
    out = attn @ v

LN, softmax, FibGen projections, and the V2 substrate activation all
stay -- the only change is the score function. This isolates the
substrate-attention hypothesis: does L1-distance-in-signature-space
produce a better attention pattern than dot-product attention?
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
from models_fibrec import FibRecLM, stateless_fibgen_forward
from optimizers_fib import FibonacciAdamW
from train_distractor_mix import build_distractor_stream
from lazy_data import fib_positions_in_window, get_fib_strided_batch
from train_K_shrink import K_schedule_tier_walk, set_K_active_recursive
from losses_substrate import substrate_fft_loss
from activations_substrate import SubstrateNegMultiAdvancedV2


class FibRecLMSubsim(FibRecLM):
    """FibRecLM with substrate-similarity attention (L1 distance) instead
    of Q·K^T dot product. Uses the first K_sig dims of Q and K as
    substrate signatures. V2 substrate activation in the FFN.
    """

    def __init__(self, *args, K_sig: int = 32, **kwargs):
        super().__init__(*args, **kwargs)
        self.K_sig = K_sig
        self.activations = nn.ModuleList(
            [SubstrateNegMultiAdvancedV2() for _ in range(self.n_blocks)]
        )

    def _layer_forward(self, x, mask, n, seeds_n):
        qkv_s, out_s, w1_s, w2_s = seeds_n
        x_norm = self.ln1s[n](x)
        qkv_basis = {"cos_i": self.qkv_cos_i, "sin_i": self.qkv_sin_i,
                      "cos_j": self.qkv_cos_j, "sin_j": self.qkv_sin_j}
        qkv = stateless_fibgen_forward(x_norm, qkv_s, qkv_basis, self.K)
        q, k, v = qkv.chunk(3, dim=-1)
        # Substrate-similarity attention: L1 distance on first K_sig dims
        # of Q and K as substrate signatures.
        sig_q = q[..., :self.K_sig]                          # [B, T, K_sig]
        sig_k = k[..., :self.K_sig]
        diff = sig_q.unsqueeze(2) - sig_k.unsqueeze(1)        # [B, T, T, K_sig]
        dist = diff.abs().sum(dim=-1)                          # [B, T, T]
        scores = -dist / math.sqrt(self.K_sig)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out_basis = {"cos_i": self.out_cos_i, "sin_i": self.out_sin_i,
                      "cos_j": self.out_cos_j, "sin_j": self.out_sin_j}
        x = x + stateless_fibgen_forward(attn @ v, out_s, out_basis, self.K)
        # FFN with substrate activation
        x_norm2 = self.ln2s[n](x)
        w1_basis = {"cos_i": self.w1_cos_i, "sin_i": self.w1_sin_i,
                      "cos_j": self.w1_cos_j, "sin_j": self.w1_sin_j}
        w2_basis = {"cos_i": self.w2_cos_i, "sin_i": self.w2_sin_i,
                      "cos_j": self.w2_cos_j, "sin_j": self.w2_sin_j}
        h = stateless_fibgen_forward(x_norm2, w1_s, w1_basis, self.K)
        h = self.activations[n](h)
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


def train_one(name, K_sig, train_split, val_split, vocab_size, args,
               fib_positions):
    torch.manual_seed(args.seed)
    gen = torch.Generator(); gen.manual_seed(args.seed + 1)
    model = FibRecLMSubsim(
        vocab_size=vocab_size, d_model=args.d_model, n_blocks=args.n_blocks,
        seq_len=args.seq_len, K=args.K_init, mode="cross", K_sig=K_sig,
    )
    optimizer = FibonacciAdamW(model.parameters(), lr=args.lr)
    sched = lambda s, T: K_schedule_tier_walk(s, T, K_init=args.K_init,
                                                 K_min=args.K_min)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[train {name}]  K_sig={K_sig}  params={n_params:,}",
          flush=True)
    t0 = time.time()
    best_val = float("inf"); best_step = -1
    cur_K = None
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
    parser.add_argument("--K-sig", type=int, default=32)
    parser.add_argument("--out", type=str,
                          default="results_substrate_attention.json")
    args = parser.parse_args()

    chars, stoi, itos, encoded = make_dataset(seq_len=args.seq_len,
                                                 source="tinyshakespeare")
    vocab_size = len(chars)
    train_split, val_split = build_distractor_stream(
        encoded, args.distractor_frac, args.seq_len, args.seed,
    )
    fib_positions = fib_positions_in_window(args.seq_len)

    # Skip baseline rerun (val=2.5889 locked) -- just bench subsim.
    results = {}
    results["subsim_attn"] = train_one("subsim_attn", args.K_sig,
                                          train_split, val_split,
                                          vocab_size, args, fib_positions)

    REFINED_REF = 2.5871
    V2_REF = 2.5889
    GELU_REF = 2.5920
    print()
    print("=" * 92)
    print(f"{'arm':<22} {'params':>10} {'best_val':>10} {'wall':>10}  vs_v2_act  vs_gelu")
    print('-' * 92)
    print(f"{'(refined act ref)':<22} {'':>10} {REFINED_REF:>10.4f} {'-':>10}  {'-':>10}  {'-':>7}")
    print(f"{'(v2 act ref)':<22} {'':>10} {V2_REF:>10.4f} {'-':>10}  {'-':>10}  {'-':>7}")
    print(f"{'(gelu ref)':<22} {'':>10} {GELU_REF:>10.4f} {'-':>10}  {'-':>10}  {'-':>7}")
    print('-' * 92)
    for name, r in results.items():
        d_v2 = (r["best_val"] - V2_REF) / V2_REF * 100
        d_gelu = (r["best_val"] - GELU_REF) / GELU_REF * 100
        print(f"{name:<22} {r['n_params']:>10,} {r['best_val']:>10.4f} "
              f"{r['wall']:>9.1f}s  {d_v2:+8.2f}%  {d_gelu:+6.2f}%")

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

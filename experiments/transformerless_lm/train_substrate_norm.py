"""A/B bench for substrate LayerNorm and Softmax.

Stacks on top of SubstrateNegMultiAdvancedV2 (R4+R5 substrate-canonical
reformulation, val=2.5889 on its own). Hypothesis: V2's extra capacity
(per-tier asymmetry + frequency rescaling) loses signal through
substrate-blind LN/softmax — pairing with SubstrateL1LN and
substrate_softmax (base phi^pi) may unlock it.

Four arms (all with V2 activation):
  baseline_v2               standard LN + standard softmax (= 2.5889 ref)
  + l1_ln                   SubstrateL1LN  + standard softmax
  + phi_pi_softmax          standard LN    + substrate_softmax
  + both                    SubstrateL1LN  + substrate_softmax
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
from layernorm_substrate import SubstrateL1LN, substrate_softmax


class FibRecLMSubstrateNorm(FibRecLM):
    """FibRecLM with refined activation and swappable LN + softmax.

    Always uses SubstrateNegMultiRefined for the FFN activation.
    LN class and softmax fn are configurable per instance.
    """
    def __init__(self, *args, ln_cls=nn.LayerNorm, softmax_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace LN modules with the chosen class.
        if ln_cls is not nn.LayerNorm:
            self.ln1s = nn.ModuleList(
                [ln_cls(self.d_model) for _ in range(self.n_blocks)]
            )
            self.ln2s = nn.ModuleList(
                [ln_cls(self.d_model) for _ in range(self.n_blocks)]
            )
            self.ln_f = ln_cls(self.d_model)
        self.activations = nn.ModuleList(
            [SubstrateNegMultiAdvancedV2() for _ in range(self.n_blocks)]
        )
        self._softmax_fn = softmax_fn if softmax_fn is not None else \
            (lambda x, dim=-1: F.softmax(x, dim=dim))

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
        attn = self._softmax_fn(scores, dim=-1)
        out_basis = {"cos_i": self.out_cos_i, "sin_i": self.out_sin_i,
                      "cos_j": self.out_cos_j, "sin_j": self.out_sin_j}
        x = x + stateless_fibgen_forward(attn @ v, out_s, out_basis, self.K)
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


def train_one(name, ln_cls, softmax_fn, train_split, val_split, vocab_size,
               args, fib_positions):
    torch.manual_seed(args.seed)
    gen = torch.Generator(); gen.manual_seed(args.seed + 1)
    model = FibRecLMSubstrateNorm(
        vocab_size=vocab_size, d_model=args.d_model, n_blocks=args.n_blocks,
        seq_len=args.seq_len, K=args.K_init, mode="cross",
        ln_cls=ln_cls, softmax_fn=softmax_fn,
    )
    optimizer = FibonacciAdamW(model.parameters(), lr=args.lr)
    sched = lambda s, T: K_schedule_tier_walk(s, T, K_init=args.K_init,
                                                 K_min=args.K_min)
    n_params = sum(p.numel() for p in model.parameters())
    ln_name = ln_cls.__name__
    sm_name = "substrate_softmax" if softmax_fn is substrate_softmax else "F.softmax"
    print(f"\n[train {name}]  ln={ln_name}  softmax={sm_name}  "
          f"params={n_params:,}", flush=True)
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
    parser.add_argument("--out", type=str, default="results_substrate_norm.json")
    args = parser.parse_args()

    chars, stoi, itos, encoded = make_dataset(seq_len=args.seq_len,
                                                 source="tinyshakespeare")
    vocab_size = len(chars)
    train_split, val_split = build_distractor_stream(
        encoded, args.distractor_frac, args.seq_len, args.seed,
    )
    fib_positions = fib_positions_in_window(args.seq_len)

    arms = [
        ("baseline_v2",       nn.LayerNorm,    None),
        ("l1_ln",             SubstrateL1LN,   None),
        ("phi_pi_softmax",    nn.LayerNorm,    substrate_softmax),
        ("both",              SubstrateL1LN,   substrate_softmax),
    ]
    results = {}
    for name, ln_cls, sm_fn in arms:
        results[name] = train_one(name, ln_cls, sm_fn, train_split, val_split,
                                    vocab_size, args, fib_positions)

    REFINED_REF = 2.5871   # refined activation, standard LN+SM
    GELU_REF = 2.5920
    print()
    print("=" * 92)
    print(f"{'arm':<22} {'params':>10} {'best_val':>10} {'wall':>10}  vs_refined  vs_gelu")
    print('-' * 92)
    print(f"{'(refined ref)':<22} {'':>10} {REFINED_REF:>10.4f} {'-':>10}  {'-':>10}  {'-':>7}")
    print(f"{'(gelu ref)':<22} {'':>10} {GELU_REF:>10.4f} {'-':>10}  {'-':>10}  {'-':>7}")
    print('-' * 92)
    for name, r in results.items():
        d_ref = (r["best_val"] - REFINED_REF) / REFINED_REF * 100
        d_gelu = (r["best_val"] - GELU_REF) / GELU_REF * 100
        print(f"{name:<22} {r['n_params']:>10,} {r['best_val']:>10.4f} "
              f"{r['wall']:>9.1f}s  {d_ref:+8.2f}%  {d_gelu:+6.2f}%")

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

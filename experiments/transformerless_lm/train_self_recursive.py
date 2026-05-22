"""Phase 2: substrate self-recursion training.

Builds on the locked substrate stack (Subsim attn + V2 activation +
FibGen weights + FibAdamW + ce_fft + K-shrink + FibRecLM depth). Adds
a SELF-HARMONY loss that requires no target -- the substrate's own
canonical Fibonacci-frequency decay pattern serves as the prior.

Three arms (all use the substrate stack, train on a TINY ~1k-char
Shakespeare seed):

  tiny_baseline       CE + ce_fft only (no self-recursion)
  tiny_with_harmony   CE + ce_fft + lambda * substrate_harmony_loss
  tiny_self_recursive Interleave supervised (CE on seed) with
                       self-generation + harmony scoring on model's
                       own output. Model generates, scores its own
                       harmony, backprops -- no external label needed.

Hypothesis: with tiny data, the substrate prior (Fibonacci-tier
decay) fills in what the data can't teach. Harmony regularizer
should reduce held-out val. Self-recursion should match or beat
even the harmony regularizer.

Compare against tiny_baseline_gelu (vanilla transformer block with
same data budget) to measure substrate's data-efficiency gain.
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
from lazy_data import fib_positions_in_window, get_fib_strided_batch
from train_K_shrink import K_schedule_tier_walk, set_K_active_recursive
from losses_substrate import substrate_fft_loss, substrate_harmony_loss
from activations_substrate import SubstrateNegMultiAdvancedV2
from train_substrate_attention import FibRecLMSubsim


def take_tiny_seed(encoded: torch.Tensor, n_chars: int,
                    seed: int = 42) -> torch.Tensor:
    """Slice an n-char window from encoded data at a deterministic offset."""
    g = torch.Generator(); g.manual_seed(seed)
    max_start = encoded.numel() - n_chars
    start = torch.randint(0, max_start, (1,), generator=g).item()
    return encoded[start: start + n_chars].clone()


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


def sample_tiny_batch(seed: torch.Tensor, batch_size: int, window: int,
                       gen: torch.Generator):
    """Random-stride batch from the tiny seed (cycled if needed)."""
    n = seed.numel()
    if n <= window + 1:
        # Pad by wrapping
        seed = seed.repeat((window + 2) // n + 1)
        n = seed.numel()
    starts = torch.randint(0, n - window - 1, (batch_size,), generator=gen)
    xs = torch.stack([seed[s: s + window] for s in starts])
    ys = torch.stack([seed[s + 1: s + window + 1] for s in starts])
    return xs, ys


def autoregressive_generate(model, prompt: torch.Tensor, n_new: int,
                              vocab_size: int, temperature: float = 1.0):
    """Sample n_new tokens autoregressively from prompt. Returns the full
    sequence (prompt + generated). No gradient tracked."""
    model.eval()
    with torch.no_grad():
        seq = prompt.clone()
        for _ in range(n_new):
            T = seq.shape[1]
            ctx = seq if T <= model.seq_len else seq[:, -model.seq_len:]
            logits = model(ctx)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            seq = torch.cat([seq, next_tok], dim=1)
    model.train()
    return seq


def train_arm(name, mode, train_seed, val_split, vocab_size, args,
               fib_positions):
    """mode in {'baseline', 'with_harmony', 'self_recursive'}."""
    torch.manual_seed(args.seed)
    gen = torch.Generator(); gen.manual_seed(args.seed + 1)
    model = FibRecLMSubsim(
        vocab_size=vocab_size, d_model=args.d_model, n_blocks=args.n_blocks,
        seq_len=args.seq_len, K=args.K_init, mode="cross", K_sig=args.K_sig,
    )
    optimizer = FibonacciAdamW(model.parameters(), lr=args.lr)
    sched = lambda s, T: K_schedule_tier_walk(s, T, K_init=args.K_init,
                                                 K_min=args.K_min)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[train {name}]  mode={mode}  tiny_seed_chars={train_seed.numel()}  "
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

        if mode == "self_recursive" and step > 0 and step % 5 == 0:
            # Self-recursion step: generate from a tiny prompt, score harmony.
            prompt_len = 16
            prompt = train_seed[:prompt_len].unsqueeze(0).repeat(
                args.batch_size, 1)
            seq = autoregressive_generate(model, prompt,
                                            n_new=args.seq_len - prompt_len,
                                            vocab_size=vocab_size)
            x = seq[:, :-1]; y = seq[:, 1:]
            logits = model(x)
            ce = F.cross_entropy(logits.reshape(-1, vocab_size),
                                   y.reshape(-1))
            harmony = substrate_harmony_loss(logits, vocab_size)
            loss = ce + args.lambda_harmony * harmony
        else:
            # Supervised step on tiny seed.
            x, y = sample_tiny_batch(train_seed, args.batch_size, args.seq_len,
                                       gen)
            logits = model(x)
            loss = substrate_fft_loss(logits, y, vocab_size,
                                        lambda_substrate=args.lambda_sub)
            if mode in ("with_harmony", "self_recursive"):
                harmony = substrate_harmony_loss(logits, vocab_size)
                loss = loss + args.lambda_harmony * harmony

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
    return {"name": name, "mode": mode, "n_params": n_params,
             "best_val": best_val, "best_step": best_step,
             "wall": time.time() - t0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--K-init", type=int, default=89)
    parser.add_argument("--K-min", type=int, default=13)
    parser.add_argument("--K-sig", type=int, default=32)
    parser.add_argument("--lambda-sub", type=float, default=0.01)
    parser.add_argument("--lambda-harmony", type=float, default=0.05)
    parser.add_argument("--tiny-chars", type=int, default=1024,
                          help="Size of the tiny training seed in chars")
    parser.add_argument("--out", type=str,
                          default="results_self_recursive.json")
    args = parser.parse_args()

    chars, stoi, itos, encoded = make_dataset(seq_len=args.seq_len,
                                                 source="tinyshakespeare")
    vocab_size = len(chars)
    # Tiny train seed; full val for evaluation
    train_seed = take_tiny_seed(encoded, args.tiny_chars, seed=args.seed)
    val_start = encoded.numel() // 10 * 9
    val_split = encoded[val_start:].clone()
    fib_positions = fib_positions_in_window(args.seq_len)

    print(f"Tiny training seed: {train_seed.numel()} chars; "
          f"val on {val_split.numel()} chars")

    arms = [
        ("tiny_baseline",         "baseline"),
        ("tiny_with_harmony",     "with_harmony"),
        ("tiny_self_recursive",   "self_recursive"),
    ]
    results = {}
    for name, mode in arms:
        results[name] = train_arm(name, mode, train_seed, val_split,
                                    vocab_size, args, fib_positions)

    print()
    print("=" * 92)
    print(f"{'arm':<24} {'params':>10} {'best_val':>10} {'wall':>10}")
    print('-' * 92)
    for name, r in results.items():
        print(f"{name:<24} {r['n_params']:>10,} {r['best_val']:>10.4f} "
              f"{r['wall']:>9.1f}s")

    # Compute deltas
    if "tiny_baseline" in results:
        base = results["tiny_baseline"]["best_val"]
        print()
        print(f"vs tiny_baseline ({base:.4f}):")
        for name, r in results.items():
            if name == "tiny_baseline":
                continue
            d = (r["best_val"] - base) / base * 100
            print(f"  {name:<24} {d:+.2f}%")

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

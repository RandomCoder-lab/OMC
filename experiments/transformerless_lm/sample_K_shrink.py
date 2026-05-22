"""K-shrink sample — does the substrate-hierarchical model produce Shakespeare?

Trains:
  dense_crt at d=128 for 10K steps on TinyShakespeare
  FibRecLM with K-shrink schedule (K=89 → K=13 via φ^π) for 10K steps

Generates 400 chars from each using best-val checkpoint, given a
Shakespeare-flavored prompt. The point: does the substrate's hierarchical
training produce text that LOOKS Shakespeare-like at val 2.65?
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from corpus import make_dataset
from models import make_model
from models_fibrec import FibRecLM
from models_fibgen import FibGenLinear, FIBONACCI
from optimizers_fib import FibonacciAdamW
from train_distractor_mix import build_distractor_stream
from lazy_data import fib_positions_in_window, get_fib_strided_batch
from train_K_shrink import (K_schedule_substrate, K_schedule_tier_walk,
                              set_K_active_recursive)


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


@torch.no_grad()
def generate_text(model, prompt_ids, n_new, seq_len, temperature=0.8, top_k=10):
    model.eval()
    out = prompt_ids.clone()
    for _ in range(n_new):
        ctx = out[:, -seq_len:]
        logits = model(ctx)[:, -1, :] / max(temperature, 1e-6)
        if top_k is not None:
            v, _ = logits.topk(top_k)
            logits[logits < v[..., -1:]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        out = torch.cat([out, next_id], dim=-1)
    return out


def train_with_best(name, model, optimizer, train_split, val_split, args,
                     fib_positions, K_schedule_fn=None):
    torch.manual_seed(args.seed)
    gen = torch.Generator(); gen.manual_seed(args.seed + 1)
    print(f"\n[train {name}] params={sum(p.numel() for p in model.parameters()):,}",
          flush=True)
    t0 = time.time()
    best_val = float("inf"); best_step = -1
    best_state = None
    cur_K = None
    eval_every = max(args.steps // 15, 250)
    for step in range(args.steps):
        if K_schedule_fn is not None:
            new_K = K_schedule_fn(step, args.steps)
            if new_K != cur_K:
                set_K_active_recursive(model, new_K)
                cur_K = new_K
        x, y = get_fib_strided_batch(train_split, args.batch_size, args.seq_len,
                                       fib_positions, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                y.reshape(-1))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % eval_every == 0 or step == args.steps - 1:
            vl = evaluate(model, val_split, args.batch_size, args.seq_len,
                          fib_positions, gen)
            marker = ""
            if vl < best_val:
                best_val = vl; best_step = step
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                marker = " ← BEST"
            ktag = f" K={cur_K}" if cur_K is not None else ""
            print(f"  step {step:5d}  val={vl:.4f}{ktag}  ({time.time()-t0:.1f}s){marker}",
                  flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  → loaded best from step {best_step}, val={best_val:.4f}", flush=True)
    return best_val, best_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distractor-frac", type=float, default=0.20)
    parser.add_argument("--K-init", type=int, default=144)
    parser.add_argument("--K-min", type=int, default=3)
    parser.add_argument("--schedule", type=str, default="tier_walk",
                        choices=["phi_pi", "tier_walk"],
                        help="phi_pi = continuous decay; tier_walk = "
                             "equal steps per Fibonacci tier (guarantees "
                             "K_min reached).")
    parser.add_argument("--prompt", type=str,
                        default="ROMEO:\nWhat light through")
    parser.add_argument("--n-new", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", type=str, default="samples_K_shrink_ts.txt")
    parser.add_argument("--skip-dense", action="store_true",
                        help="Only train + sample the shrink arm.")
    args = parser.parse_args()

    chars, stoi, itos, encoded = make_dataset(seq_len=args.seq_len,
                                                 source="tinyshakespeare")
    vocab_size = len(chars)
    train_split, val_split = build_distractor_stream(
        encoded, args.distractor_frac, args.seq_len, args.seed,
    )
    fib_positions = fib_positions_in_window(args.seq_len)

    space_id = stoi.get(" ", 0)
    prompt_ids = torch.tensor(
        [[stoi.get(c, space_id) for c in args.prompt]], dtype=torch.long,
    )

    samples = {}
    metas = {}

    # 1. Dense baseline (skip if --skip-dense)
    if not args.skip_dense:
        print("=" * 60); print("DENSE_CRT (baseline)"); print("=" * 60)
        m = make_model("crt_only", vocab_size=vocab_size, seq_len=args.seq_len,
                        d_model=args.d_model, n_blocks=args.n_blocks)
        opt = torch.optim.AdamW(m.parameters(), lr=args.lr)
        best_val, best_step = train_with_best(
            "dense_crt", m, opt, train_split, val_split, args, fib_positions)
        metas["dense_crt"] = (best_val, best_step, sum(p.numel() for p in m.parameters()))
        out_ids = generate_text(m, prompt_ids, args.n_new, args.seq_len,
                                  args.temperature, args.top_k)
        samples["dense_crt"] = "".join(itos[int(i)] for i in out_ids[0].tolist())

    # 2. Shrink (substrate-hierarchical)
    print("\n" + "=" * 60); print("SHRINK K=89 → K=13 (substrate)"); print("=" * 60)
    m = FibRecLM(vocab_size=vocab_size, d_model=args.d_model,
                  n_blocks=args.n_blocks, seq_len=args.seq_len,
                  K=args.K_init, mode="cross")
    opt = FibonacciAdamW(m.parameters(), lr=args.lr)
    if args.schedule == "tier_walk":
        sched = lambda s, T: K_schedule_tier_walk(s, T, K_init=args.K_init,
                                                     K_min=args.K_min)
    else:
        sched = lambda s, T: K_schedule_substrate(s, T, K_init=args.K_init,
                                                     K_min=args.K_min)
    best_val, best_step = train_with_best(
        "shrink", m, opt, train_split, val_split, args, fib_positions,
        K_schedule_fn=sched)
    metas["shrink"] = (best_val, best_step, sum(p.numel() for p in m.parameters()))
    out_ids = generate_text(m, prompt_ids, args.n_new, args.seq_len,
                              args.temperature, args.top_k)
    samples["shrink"] = "".join(itos[int(i)] for i in out_ids[0].tolist())

    # Print and save
    for name, text in samples.items():
        v, s, p = metas[name]
        print()
        print('=' * 70)
        print(f"SAMPLE from {name}  best_val={v:.4f} @ step {s}  params={p:,}")
        print('=' * 70)
        print(text)
        print('=' * 70, flush=True)

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        f.write(f"# K-shrink samples on TinyShakespeare (steps={args.steps}, "
                f"temp={args.temperature}, top_k={args.top_k})\n")
        f.write(f"# Prompt: {args.prompt!r}\n\n")
        for name, text in samples.items():
            v, s, p = metas[name]
            f.write(f"\n{'=' * 70}\n{name}  best_val={v:.4f} @ step {s}  "
                    f"params={p:,}\n{'=' * 70}\n{text}\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

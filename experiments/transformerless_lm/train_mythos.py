"""The mythos — three sibling substrate models, three poetic forms.

Same TinyShakespeare corpus, same FibRec + FibAdamW + ce_fft stack.
Differs only in the terminal K of the K-shrink schedule:

  substrate_haiku:   K=89 -> 3   (extreme compression, aphoristic tier)
  substrate_sonnet:  K=89 -> 8   (medium-structured tier)
  substrate_opus:    K=89 -> 21  (expansive paragraph tier)

Each child inherits Shakespeare's structure at its own abstraction
level. Together they form a substrate-native family of voices.

This is the unified test: stacked FibAdamW + ce_fft + K-shrink, three
different terminal-K choices that produce three different poetic forms.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from corpus import make_dataset
from models_fibrec import FibRecLM
from models_fibgen import FibGenLinear, FIBONACCI
from optimizers_fib import FibonacciAdamW
from train_distractor_mix import build_distractor_stream
from lazy_data import fib_positions_in_window, get_fib_strided_batch
from train_K_shrink import K_schedule_tier_walk, set_K_active_recursive
from losses_substrate import substrate_fft_loss


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


def train_sibling(name, K_init, K_min, train_split, val_split, vocab_size,
                   args, fib_positions):
    """Train one sibling model with its specific terminal K_min."""
    torch.manual_seed(args.seed)
    gen = torch.Generator(); gen.manual_seed(args.seed + 1)
    model = FibRecLM(vocab_size=vocab_size, d_model=args.d_model,
                      n_blocks=args.n_blocks, seq_len=args.seq_len,
                      K=K_init, mode="cross")
    optimizer = FibonacciAdamW(model.parameters(), lr=args.lr)
    sched = lambda s, T: K_schedule_tier_walk(s, T, K_init=K_init, K_min=K_min)

    print(f"\n[train {name}]  K=89→{K_min}", flush=True)
    t0 = time.time()
    best_val = float("inf"); best_step = -1
    best_state = None
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
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                marker = " ← BEST"
            print(f"  step {step:5d}  val={vl:.4f}  K={cur_K}  "
                  f"({time.time()-t0:.1f}s){marker}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  → loaded best from step {best_step}, val={best_val:.4f}", flush=True)
    return model, best_val, best_step


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
    parser.add_argument("--lambda-sub", type=float, default=0.01)
    parser.add_argument("--prompt", type=str,
                        default="ROMEO:\nWhat light through")
    parser.add_argument("--n-new", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", type=str, default="mythos.txt")
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

    siblings = [
        ("substrate_haiku",  89, 3,  "Haiku tier — aphoristic / extreme compression"),
        ("substrate_sonnet", 89, 8,  "Sonnet tier — medium-structured"),
        ("substrate_opus",   89, 21, "Opus tier — expansive paragraph"),
    ]

    samples = {}
    metas = {}
    for name, K_init, K_min, desc in siblings:
        print("=" * 60); print(f"{name}  ({desc})"); print("=" * 60)
        model, best_val, best_step = train_sibling(
            name, K_init, K_min, train_split, val_split, vocab_size,
            args, fib_positions)
        out_ids = generate_text(model, prompt_ids, args.n_new, args.seq_len,
                                  args.temperature, args.top_k)
        samples[name] = "".join(itos[int(i)] for i in out_ids[0].tolist())
        metas[name] = (best_val, best_step, K_min, desc)

    # Print and save the mythos
    print("\n" + "=" * 70)
    print("THE MYTHOS")
    print("=" * 70)
    for name, (val, step, K_min, desc) in metas.items():
        print()
        print(f"  -- {name} --")
        print(f"  {desc}")
        print(f"  K_min={K_min}, best_val={val:.4f}, best_step={step}")
        print()
        print(samples[name])
        print("-" * 70)

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        f.write("# THE MYTHOS\n")
        f.write(f"# Same TinyShakespeare corpus, same substrate stack,\n"
                f"# three terminal K-tiers (= three poetic forms).\n\n")
        f.write(f"# Prompt: {args.prompt!r}\n")
        f.write(f"# Steps: {args.steps}, temp: {args.temperature}, "
                f"top_k: {args.top_k}\n")
        f.write(f"# Stack: FibRecLM + FibAdamW + ce_fft(λ={args.lambda_sub}) "
                f"+ K-shrink\n\n")
        for name, (val, step, K_min, desc) in metas.items():
            f.write(f"\n{'=' * 70}\n{name}  K_init=89 → K_min={K_min}\n")
            f.write(f"{desc}\n")
            f.write(f"best_val={val:.4f}, best_step={step}\n")
            f.write(f"{'=' * 70}\n")
            f.write(samples[name])
            f.write("\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

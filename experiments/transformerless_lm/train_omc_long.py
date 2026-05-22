"""Long-steps OMC bench with text sampling — the capacity test.

Trains dense_crt and FibRecLM + FibAdamW on the OMC codebase corpus
for 20,000 steps each at d=128. Tracks best-val and generates 400-char
samples from each arch's best-val checkpoint.

The hypothesis being tested:
  - At 1500 steps both archs are undertrained at d=128 on OMC
  - With 20K steps both reach their natural quality limits
  - If the substrate gap STAYS BOUNDED or NARROWS as steps grow,
    the substrate basis has enough capacity for this corpus
  - If the gap GROWS with more steps, K=32 caps out and we need
    more substrate capacity

The text samples answer a separate question: at the substrate's
quality target, does it produce structurally plausible Python/Rust/MD
output? Or is it gibberish at the char level despite low val loss?
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


def train(name, model, optimizer, train_split, val_split, args, fib_positions):
    """Train, tracking best-val state for sampling."""
    torch.manual_seed(args.seed)
    gen = torch.Generator(); gen.manual_seed(args.seed + 1)
    n_params = sum(p.numel() for p in model.parameters())
    compr = None
    if hasattr(model, "storage_summary"):
        compr = model.storage_summary()["compression"]
    print(f"\n[train {name}] params={n_params:,}" +
          (f"  compression={compr:.1f}x" if compr else ""), flush=True)
    t0 = time.time()
    best_val = float("inf")
    best_step = -1
    best_state = None
    val_hist = []
    eval_every = max(args.steps // 20, 250)
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
                best_val = vl
                best_step = step
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                marker = " ← BEST"
            print(f"  step {step:5d}  val={vl:.4f}  ({time.time()-t0:.1f}s){marker}",
                  flush=True)
    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  → loaded best from step {best_step}, val={best_val:.4f}", flush=True)
    return {"name": name, "n_params": n_params, "compression": compr,
             "best_val": best_val, "best_step": best_step,
             "wall": time.time() - t0, "val_history": val_hist}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distractor-frac", type=float, default=0.20)
    parser.add_argument("--prompt", type=str,
                        default="def fibonacci(n):\n    ")
    parser.add_argument("--n-new", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", type=str, default="results_omc_long.json")
    parser.add_argument("--samples-out", type=str, default="results_omc_samples.txt")
    args = parser.parse_args()

    chars, stoi, itos, encoded = make_dataset(seq_len=args.seq_len, source="omc")
    vocab_size = len(chars)
    train_split, val_split = build_distractor_stream(
        encoded, args.distractor_frac, args.seq_len, args.seed,
    )
    fib_positions = fib_positions_in_window(args.seq_len)

    print(f"OMC long-steps bench")
    print(f"Corpus: OMC ({encoded.numel():,} chars, vocab {vocab_size})")
    print(f"Steps: {args.steps}, lazy data P={len(fib_positions)}", flush=True)

    # Encode prompt
    space_id = stoi.get(" ", 0)
    prompt_ids = torch.tensor(
        [[stoi.get(c, space_id) for c in args.prompt]], dtype=torch.long,
    )

    results = {}
    samples = {}

    # 1. Dense baseline
    m = make_model("crt_only", vocab_size=vocab_size, seq_len=args.seq_len,
                    d_model=args.d_model, n_blocks=args.n_blocks)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr)
    results["dense_crt"] = train(
        "dense_crt", m, opt, train_split, val_split, args, fib_positions)
    out_ids = generate_text(m, prompt_ids, args.n_new, args.seq_len,
                              temperature=args.temperature, top_k=args.top_k)
    samples["dense_crt"] = "".join(itos[int(i)] for i in out_ids[0].tolist())

    # 2. Substrate-recursive composed (FibRec + FibAdamW)
    m = FibRecLM(vocab_size=vocab_size, d_model=args.d_model,
                  n_blocks=args.n_blocks, seq_len=args.seq_len, K=32, mode="cross")
    opt = FibonacciAdamW(m.parameters(), lr=args.lr)
    results["fibrec_fibadamw"] = train(
        "fibrec_fibadamw", m, opt, train_split, val_split, args, fib_positions)
    out_ids = generate_text(m, prompt_ids, args.n_new, args.seq_len,
                              temperature=args.temperature, top_k=args.top_k)
    samples["fibrec_fibadamw"] = "".join(itos[int(i)] for i in out_ids[0].tolist())

    # Print samples
    for name, text in samples.items():
        print()
        print('=' * 70)
        print(f"SAMPLE from {name}  best_val={results[name]['best_val']:.4f}")
        print('=' * 70)
        print(text)
        print('=' * 70, flush=True)

    # Save
    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    sample_path = Path(__file__).parent / args.samples_out
    with open(sample_path, "w") as f:
        f.write(f"# OMC corpus samples (steps={args.steps}, "
                f"temp={args.temperature}, top_k={args.top_k})\n")
        f.write(f"# Prompt: {args.prompt!r}\n\n")
        for name, text in samples.items():
            r = results[name]
            f.write(f"\n{'=' * 70}\n{name}  best_val={r['best_val']:.4f} "
                    f"@ step {r['best_step']}  params={r['n_params']:,}\n"
                    f"{'=' * 70}\n{text}\n")
    print(f"\nWrote {out_path}")
    print(f"Wrote {sample_path}")


if __name__ == "__main__":
    main()

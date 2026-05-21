"""Sample text generation from trained models.

Loss numbers are abstract. Actual generated text is the deployment-meaningful
quality signal: does a +5-7% val-loss penalty translate to barely-perceptible
output or to broken text?

Trains dense_crt vs fibgen_K32_cross vs composed_transformerless on
TinyShakespeare with lazy-loading, then generates a sample from a fixed
prompt for each. Greedy decoding by default; temperature sampling
optional. Output is human-readable so you can eyeball it.

If the FibGen output is coherent and stylistically Shakespeare-ish, the
inference-economics result (90% throughput, 37x less memory) translates
into a deployable model.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from corpus import make_dataset
from models import make_model
from models_fibgen import FibGenLM, FibGenTransformerless
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


def train(name, model, train_split, val_split, args, fib_positions):
    """Train and return BEST-VAL checkpoint. Substrate models jump between
    Fibonacci-attractor configurations during training, so the best val
    is rarely at the final step — sample from the best attractor."""
    torch.manual_seed(args.seed)
    gen = torch.Generator(); gen.manual_seed(args.seed + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    t0 = time.time()
    eval_every = 200
    print(f"\n[train {name}] params={sum(p.numel() for p in model.parameters()):,}",
          flush=True)
    best_val = float("inf")
    best_state = None
    best_step = -1
    for step in range(args.steps):
        x, y = get_fib_strided_batch(train_split, args.batch_size, args.seq_len,
                                       fib_positions, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % eval_every == 0 or step == args.steps - 1:
            vl = evaluate(model, val_split, args.batch_size, args.seq_len,
                          fib_positions, gen)
            marker = ""
            if vl < best_val:
                best_val = vl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_step = step
                marker = " ← BEST"
            print(f"  step {step:5d}  train={loss.item():.4f}  val={vl:.4f}"
                  f"  ({time.time()-t0:.1f}s){marker}", flush=True)
    # Load best
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  → using best checkpoint from step {best_step}  val={best_val:.4f}",
          flush=True)
    return model, best_val, best_step


@torch.no_grad()
def generate_text(model, prompt_ids, n_new, seq_len, itos,
                   temperature: float = 1.0, top_k: int = None):
    model.eval()
    out = prompt_ids.clone()
    for _ in range(n_new):
        ctx = out[:, -seq_len:]
        logits = model(ctx)[:, -1, :] / max(temperature, 1e-6)
        if top_k is not None:
            v, _ = logits.topk(top_k)
            logits[logits < v[..., -1:]] = float("-inf")
        if temperature <= 1e-3:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        out = torch.cat([out, next_id], dim=-1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distractor-frac", type=float, default=0.20)
    parser.add_argument("--prompt", type=str,
                        default="ROMEO:\nWhat light through")
    parser.add_argument("--n-new", type=int, default=400,
                        help="Number of new characters to generate.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", type=str, default="results_samples.txt")
    args = parser.parse_args()

    chars, stoi, itos, encoded = make_dataset(seq_len=args.seq_len,
                                                 source="tinyshakespeare")
    vocab_size = len(chars)
    train_split, val_split = build_distractor_stream(
        encoded, args.distractor_frac, args.seq_len, args.seed,
    )
    fib_positions = fib_positions_in_window(args.seq_len)

    # Build the three archs
    archs = {
        "dense_crt": lambda: make_model(
            "crt_only", vocab_size=vocab_size, seq_len=args.seq_len,
            d_model=args.d_model, n_blocks=args.n_blocks,
        ),
        "fibgen_K32_cross": lambda: FibGenLM(
            vocab_size=vocab_size, d_model=args.d_model,
            n_blocks=args.n_blocks, seq_len=args.seq_len, K=32, mode="cross",
        ),
        "composed_transformerless": lambda: FibGenTransformerless(
            vocab_size=vocab_size, d_model=args.d_model, n_blocks=args.n_blocks,
            seq_len=args.seq_len, K=32, mode="cross", n_specialists=5,
        ),
    }

    # Encode prompt (handle unknown chars by mapping to space)
    space_id = stoi.get(" ", 0)
    prompt_ids = torch.tensor(
        [[stoi.get(c, space_id) for c in args.prompt]], dtype=torch.long,
    )

    samples = {}
    meta = {}
    for name, make_fn in archs.items():
        model = make_fn()
        model, best_val, best_step = train(name, model, train_split, val_split,
                                              args, fib_positions)
        meta[name] = {"best_val": best_val, "best_step": best_step,
                      "n_params": sum(p.numel() for p in model.parameters())}
        out_ids = generate_text(model, prompt_ids, args.n_new, args.seq_len,
                                  itos, temperature=args.temperature,
                                  top_k=args.top_k)
        text = "".join(itos[int(i)] for i in out_ids[0].tolist())
        samples[name] = text
        print(f"\n{'=' * 70}")
        print(f"SAMPLE from {name}  (best_val={best_val:.4f} @ step {best_step})")
        print('=' * 70)
        print(text)
        print('=' * 70, flush=True)

    # Write to file
    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        f.write(f"# Samples (steps={args.steps}, temperature={args.temperature}, "
                f"top_k={args.top_k})\n")
        f.write(f"# Prompt: {args.prompt!r}\n\n")
        for name, text in samples.items():
            m = meta[name]
            f.write(f"\n{'=' * 70}\n{name}  best_val={m['best_val']:.4f} @ step "
                    f"{m['best_step']}  params={m['n_params']:,}\n"
                    f"{'=' * 70}\n{text}\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

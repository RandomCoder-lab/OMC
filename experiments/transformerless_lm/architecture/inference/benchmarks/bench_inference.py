"""Inference-speed bench — autoregressive token generation throughput.

For the user's "fast inference / low hardware cost" target, we need to
measure DEPLOYMENT-time speed not training throughput. This bench:

  - Initializes each arch with random weights (we are NOT testing
    output quality here, just speed and memory).
  - Generates N=256 tokens autoregressively at batch=1 (a single user
    session).
  - Reports:
      tokens/sec
      ms per token
      weight-memory footprint (in MB)
      FibGen weight-cache savings (cache-once vs regenerate-per-token)

The interesting comparison: a FibGen model at deployment with a
one-time weight-cache (compute the dense W tensor once, reuse it for
all tokens) has IDENTICAL per-token forward cost to dense, but
dramatically lower persistent storage. That is the substrate's
inference win.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from models import make_model
from models_fibgen import FibGenLM, FibGenTransformerless, FibGenLinear


@torch.no_grad()
def autoregressive_generate(model, prompt_tokens: torch.Tensor,
                              n_new_tokens: int, seq_len: int) -> torch.Tensor:
    """Greedy autoregressive generation. prompt_tokens: [1, P]."""
    model.eval()
    out = prompt_tokens.clone()
    for _ in range(n_new_tokens):
        # take the last seq_len tokens as context
        ctx = out[:, -seq_len:]
        logits = model(ctx)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out = torch.cat([out, next_id], dim=-1)
    return out


def measure_inference(name: str, model: torch.nn.Module, n_tokens: int,
                       seq_len: int, vocab_size: int, n_warmup: int = 10):
    """Returns dict with tokens/sec, ms/tok, weight_mb."""
    prompt = torch.randint(0, vocab_size, (1, 10))   # 10-token prompt
    # Warmup
    _ = autoregressive_generate(model, prompt, n_warmup, seq_len)
    # Measure
    t0 = time.time()
    _ = autoregressive_generate(model, prompt, n_tokens, seq_len)
    dt = time.time() - t0
    weight_bytes = sum(p.numel() * p.element_size()
                        for p in model.parameters())
    return {
        "name": name,
        "tokens_generated": n_tokens,
        "wall_seconds": dt,
        "tokens_per_sec": n_tokens / dt,
        "ms_per_token": 1000 * dt / n_tokens,
        "weight_mb": weight_bytes / (1024 ** 2),
        "n_params": sum(p.numel() for p in model.parameters()),
    }


def fibgen_cache_weights(model: torch.nn.Module) -> torch.nn.Module:
    """Trigger weight-caching on every FibGenLinear in the model. After
    this each layer's forward returns its cached W (no on-the-fly
    generation). Same inference compute as a stored model, just derived
    once from the FibGen seed."""
    for m in model.modules():
        if isinstance(m, FibGenLinear):
            m.cache_weight()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tokens", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=65)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--out", type=str, default="results_inference.json")
    args = parser.parse_args()

    configs = []

    # d=128 archs
    configs.append(("dense_crt_d128",
                     lambda: make_model("crt_only", vocab_size=args.vocab_size,
                                          seq_len=args.seq_len, d_model=128,
                                          n_blocks=args.n_blocks)))
    configs.append(("fibgen_K32_cross_d128",
                     lambda: FibGenLM(vocab_size=args.vocab_size,
                                       d_model=128, n_blocks=args.n_blocks,
                                       seq_len=args.seq_len, K=32, mode="cross")))
    configs.append(("composed_transformerless_d128",
                     lambda: FibGenTransformerless(
                         vocab_size=args.vocab_size, d_model=128,
                         n_blocks=args.n_blocks, seq_len=args.seq_len,
                         K=32, mode="cross", n_specialists=5)))
    # d=256 archs
    configs.append(("dense_crt_d256",
                     lambda: make_model("crt_only", vocab_size=args.vocab_size,
                                          seq_len=args.seq_len, d_model=256,
                                          n_blocks=args.n_blocks)))
    configs.append(("fibgen_K32_cross_d256",
                     lambda: FibGenLM(vocab_size=args.vocab_size,
                                       d_model=256, n_blocks=args.n_blocks,
                                       seq_len=args.seq_len, K=32, mode="cross")))
    configs.append(("composed_transformerless_d256",
                     lambda: FibGenTransformerless(
                         vocab_size=args.vocab_size, d_model=256,
                         n_blocks=args.n_blocks, seq_len=args.seq_len,
                         K=32, mode="cross", n_specialists=5)))

    print(f"Inference bench")
    print(f"  generating {args.n_tokens} tokens autoregressively per config")
    print(f"  context window: {args.seq_len}")
    print(f"  vocab_size: {args.vocab_size}", flush=True)

    results = []
    for name, make_fn in configs:
        # First: naive inference (FibGen regenerates weights every forward)
        torch.manual_seed(42)
        model = make_fn()
        r_naive = measure_inference(f"{name}_naive", model, args.n_tokens,
                                      args.seq_len, args.vocab_size)
        print(f"\n  {r_naive['name']:<36}  params={r_naive['n_params']:>8,}  "
              f"weight_mb={r_naive['weight_mb']:>6.2f}  "
              f"tok/s={r_naive['tokens_per_sec']:>6.1f}  "
              f"ms/tok={r_naive['ms_per_token']:>5.1f}", flush=True)
        results.append(r_naive)

        # If the model has any FibGenLinear, also measure with weight cache.
        has_fibgen = any(isinstance(m, FibGenLinear) for m in model.modules())
        if has_fibgen:
            torch.manual_seed(42)
            model_cached = make_fn()
            model_cached = fibgen_cache_weights(model_cached)
            r_cached = measure_inference(f"{name}_cached", model_cached,
                                          args.n_tokens, args.seq_len,
                                          args.vocab_size)
            speedup = r_naive["ms_per_token"] / r_cached["ms_per_token"]
            print(f"  {r_cached['name']:<36}  params={r_cached['n_params']:>8,}  "
                  f"weight_mb={r_cached['weight_mb']:>6.2f}  "
                  f"tok/s={r_cached['tokens_per_sec']:>6.1f}  "
                  f"ms/tok={r_cached['ms_per_token']:>5.1f}  "
                  f"(cache speedup vs naive: {speedup:.2f}x)", flush=True)
            results.append(r_cached)

    # Compare across configs
    print()
    print("=" * 92)
    print(f"{'config':<38} {'params':>10} {'weight_MB':>10} {'tok/s':>10} "
          f"{'ms/tok':>10}")
    print("-" * 92)
    for r in results:
        print(f"{r['name']:<38} {r['n_params']:>10,} {r['weight_mb']:>10.2f} "
              f"{r['tokens_per_sec']:>10.1f} {r['ms_per_token']:>10.1f}")

    # Save
    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

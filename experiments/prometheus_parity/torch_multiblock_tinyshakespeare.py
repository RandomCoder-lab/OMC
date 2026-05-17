"""L1 at multi-block + TinyShakespeare scale (PyTorch).

The combined test: does substrate-K win when BOTH depth AND scale
are real?

Setup:
  - TinyShakespeare corpus (1.1MB, vocab=65), 90/10 train/val split
  - 4-block transformer (each block: Attn + LN + FFN + LN + residuals)
  - 5 seeds × 1500 steps, AdamW lr=0.005
  - d_model=32, seq_len=32, ff=64

Two variants:
  L0: standard QKV (4 attention layers, all with learned Q, K, V)
  L1: substrate-K (4 attention layers, all with CRT-PE as K + learned Q, V)

If L1 wins at multi-block + TinyShakespeare, that's the production
recommendation: substrate-K is the architectural default at every
scale + depth combination tested.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_4way import lcg, make_matrix, crt_pe, AttentionL0, AttentionL1, build_vocab
from torch_multiblock import TransformerBlock, MultiBlockTransformer


def train_with_val(variant, train_ids, val_ids, vocab_size, seq_len, d_model,
                   ff_dim, n_blocks, lr, steps, seed,
                   val_every=200, n_val_batches=30):
    torch.manual_seed(seed)
    random.seed(seed)
    model = MultiBlockTransformer(variant, vocab_size, d_model, ff_dim,
                                   seq_len, n_blocks, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   betas=(0.9, 0.999), eps=1e-8)
    n_train = len(train_ids)
    n_val = len(val_ids)
    train_tensor = torch.tensor(train_ids, dtype=torch.long)
    val_tensor = torch.tensor(val_ids, dtype=torch.long)
    val_history = []
    train_tail = []
    for step in range(steps):
        start = random.randint(0, n_train - seq_len - 2)
        window = train_tensor[start:start + seq_len]
        targets = train_tensor[start + 1:start + 1 + seq_len]
        logits = model(window)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step >= steps - 50:
            train_tail.append(loss.item())
        if (step + 1) % val_every == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(n_val_batches):
                    vs = random.randint(0, n_val - seq_len - 2)
                    vw = val_tensor[vs:vs + seq_len]
                    vt = val_tensor[vs + 1:vs + 1 + seq_len]
                    vl = F.cross_entropy(model(vw), vt)
                    val_losses.append(vl.item())
                val_history.append((step + 1, sum(val_losses) / len(val_losses)))
            model.train()
    train_mean = sum(train_tail) / len(train_tail)
    val_mean = val_history[-1][1] if val_history else float("nan")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return train_mean, val_mean, n_params, val_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="42,7,123,2026,1")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--ff-dim", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--out", type=str, default="results_torch_multiblock_tinyshakespeare.json")
    args = parser.parse_args()

    corpus_path = Path(__file__).parent.parent / "transformerless_lm" / "tinyshakespeare.txt"
    text = corpus_path.read_text()
    chars, lookup = build_vocab(text)
    vocab_size = len(chars)
    ids = [lookup[c] for c in text]

    split = int(len(ids) * 0.9)
    train_ids = ids[:split]
    val_ids = ids[split:]

    seeds = [int(s) for s in args.seeds.split(",")]
    variants = ["L0", "L1"]

    print(f"=== Multi-block ({args.n_blocks} layers) + TinyShakespeare ===")
    print(f"corpus: {len(text):,} chars; train {len(train_ids):,}; val {len(val_ids):,}")
    print(f"vocab={vocab_size} seq={args.seq_len} d={args.d_model} ff={args.ff_dim}")
    print(f"steps={args.steps} lr={args.lr} seeds={seeds}\n", flush=True)

    results = {}
    for v in variants:
        train_means, val_means = [], []
        n_params = 0
        for seed in seeds:
            tm, vm, n_params, _ = train_with_val(
                v, train_ids, val_ids, vocab_size, args.seq_len,
                args.d_model, args.ff_dim, args.n_blocks, args.lr,
                args.steps, seed,
            )
            train_means.append(tm)
            val_means.append(vm)
            print(f"  [{v}] seed={seed} train={tm:.4f} val={vm:.4f}", flush=True)
        results[v] = {
            "train": train_means, "val": val_means, "n_params": n_params,
            "train_mean": sum(train_means) / len(train_means),
            "val_mean": sum(val_means) / len(val_means),
            "val_std": statistics.stdev(val_means) if len(val_means) > 1 else 0.0,
        }
        print(f"[{v}] params={n_params:6d}  "
              f"train={results[v]['train_mean']:.4f}  "
              f"val={results[v]['val_mean']:.4f} (std={results[v]['val_std']:.4f})\n",
              flush=True)

    print("\n=== Multi-block + TinyShakespeare verdict ===")
    l0 = results["L0"]
    l1 = results["L1"]
    delta_train = l1["train_mean"] - l0["train_mean"]
    delta_val = l1["val_mean"] - l0["val_mean"]
    rel_val = delta_val / l0["val_mean"] * 100
    wins = sum(1 for x, b in zip(l1["val"], l0["val"]) if x < b)
    print(f"L0 params={l0['n_params']}  train={l0['train_mean']:.4f}  val={l0['val_mean']:.4f}")
    print(f"L1 params={l1['n_params']}  train={l1['train_mean']:.4f}  val={l1['val_mean']:.4f}")
    print(f"L1 vs L0 (val): {rel_val:+.1f}%  wins={wins}/{len(l0['val'])}")
    print(f"Param savings: {(l0['n_params'] - l1['n_params']) / l0['n_params'] * 100:.1f}%")
    if l1["val_mean"] < l0["val_mean"]:
        print(f"\n[L1 WINS] Substrate-K holds at depth=4 + TinyShakespeare scale.")
        print(f"  The architectural recommendation generalizes across all regimes.")
    else:
        print(f"\n[L0 wins at depth+scale combined] — investigate.")

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump({"results": results, "config": vars(args)}, f,
                  indent=2, default=float)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

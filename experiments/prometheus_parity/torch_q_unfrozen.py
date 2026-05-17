"""Unfreeze-Q variants + train/val split TinyShakespeare run.

The Scale Boundary writeup hypothesized that L2/L3's failure at
TinyShakespeare is specifically because Q is frozen. Test:

  L5: substrate K + LEARNED Q + identity V
      (the minimal Q-unfreeze; keeps K substrate, V identity)

  L6: substrate K + substrate-biased learned Q + identity V
      Q = x @ W_Q + alpha * CRT_PE — Q learns from content
      but starts with a substrate prior; alpha is a learnable scalar.

Both run on:
  (a) Tiny scale (73 chars, training-loss only) — should match
      original 4-way ranking
  (b) TinyShakespeare with TRAIN/VAL SPLIT — the honest test
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_4way import (
    lcg, make_matrix, crt_pe, AttentionL0, AttentionL3,
    TransformerModel, build_vocab,
)


class AttentionL5(nn.Module):
    """K = CRT-PE (substrate), Q = learned, V = identity."""
    def __init__(self, d_model: int, seq_len: int, seed: int):
        super().__init__()
        W_q, s = make_matrix(d_model, d_model, 0.3, seed + 11)
        self.W_q = nn.Parameter(W_q)
        self.register_buffer("K_const", crt_pe(seq_len, d_model))
        self.rng_state = s

    def forward(self, x):
        q = x @ self.W_q
        k = self.K_const
        scores = q @ k.T
        attn = F.softmax(scores, dim=-1)
        return attn @ x      # V = identity


class AttentionL6(nn.Module):
    """K = CRT-PE, Q = (x @ W_Q) + alpha * CRT_PE, V = identity.
    Q starts substrate-biased; alpha learns whether to lean on the
    substrate prior or the learned content path."""
    def __init__(self, d_model: int, seq_len: int, seed: int):
        super().__init__()
        W_q, s = make_matrix(d_model, d_model, 0.3, seed + 11)
        self.W_q = nn.Parameter(W_q)
        # alpha starts at 1.0 (pure substrate prior at init); learns to drift.
        self.alpha = nn.Parameter(torch.tensor(1.0))
        pe = crt_pe(seq_len, d_model)
        self.register_buffer("K_const", pe)
        self.register_buffer("Q_const", pe)
        self.rng_state = s

    def forward(self, x):
        q = x @ self.W_q + self.alpha * self.Q_const
        k = self.K_const
        scores = q @ k.T
        attn = F.softmax(scores, dim=-1)
        return attn @ x


class TransformerModelExt(TransformerModel):
    """Extends TransformerModel with L5 + L6 attention options."""
    def __init__(self, variant: str, vocab: int, d_model: int, ff_dim: int,
                 seq_len: int, seed: int):
        if variant in ("L5", "L6"):
            super().__init__("L3", vocab, d_model, ff_dim, seq_len, seed)
            attn_cls = {"L5": AttentionL5, "L6": AttentionL6}[variant]
            self.attn = attn_cls(d_model, seq_len, seed)
        else:
            super().__init__(variant, vocab, d_model, ff_dim, seq_len, seed)


def train_with_val(variant, train_ids, val_ids, vocab_size, seq_len, d_model,
                   ff_dim, lr, steps, seed, val_every=100, n_val_batches=20):
    torch.manual_seed(seed)
    random.seed(seed)
    model = TransformerModelExt(variant, vocab_size, d_model, ff_dim,
                                seq_len, seed)
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
    parser.add_argument("--seeds", type=str, default="42,7,123")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--ff-dim", type=int, default=64)
    parser.add_argument("--variants", type=str, default="L0,L1,L3,L5,L6")
    parser.add_argument("--out", type=str, default="results_torch_q_unfrozen.json")
    args = parser.parse_args()

    corpus_path = Path(__file__).parent.parent / "transformerless_lm" / "tinyshakespeare.txt"
    text = corpus_path.read_text()
    chars, lookup = build_vocab(text)
    vocab_size = len(chars)
    ids = [lookup[c] for c in text]

    # 90/10 split.
    split = int(len(ids) * 0.9)
    train_ids = ids[:split]
    val_ids = ids[split:]

    seeds = [int(s) for s in args.seeds.split(",")]
    variants = args.variants.split(",")

    print("=== TinyShakespeare with train/val split + unfrozen-Q variants ===")
    print(f"corpus: {len(text):,} chars; train {len(train_ids):,}; val {len(val_ids):,}")
    print(f"vocab={vocab_size} seq={args.seq_len} d={args.d_model} ff={args.ff_dim}")
    print(f"steps={args.steps} lr={args.lr} seeds={seeds} variants={variants}\n",
          flush=True)

    results = {}
    for v in variants:
        train_means, val_means = [], []
        n_params = 0
        for seed in seeds:
            tm, vm, n_params, _ = train_with_val(
                v, train_ids, val_ids, vocab_size, args.seq_len,
                args.d_model, args.ff_dim, args.lr, args.steps, seed,
            )
            train_means.append(tm)
            val_means.append(vm)
        results[v] = {
            "train": train_means, "val": val_means, "n_params": n_params,
            "train_mean": sum(train_means) / len(train_means),
            "val_mean": sum(val_means) / len(val_means),
        }
        print(f"[{v}] params={n_params:5d}  "
              f"train={results[v]['train_mean']:.3f}  "
              f"val={results[v]['val_mean']:.3f}  "
              f"per-seed val={[f'{x:.2f}' for x in val_means]}",
              flush=True)

    print("\n=== Train/Val comparison ===")
    print(f"{'variant':<8} {'params':>6} {'train':>8} {'val':>8} {'gap':>8}")
    for v in variants:
        r = results[v]
        gap = r["val_mean"] - r["train_mean"]
        print(f"{v:<8} {r['n_params']:>6} {r['train_mean']:>8.3f} "
              f"{r['val_mean']:>8.3f} {gap:>+8.3f}")

    print("\n=== Val-loss verdict ===")
    if "L0" in results:
        base_val = results["L0"]["val_mean"]
        for v in variants:
            if v == "L0":
                continue
            vmean = results[v]["val_mean"]
            rel = (vmean - base_val) / base_val * 100
            marker = "BETTER" if vmean < base_val else "worse "
            print(f"  {v}: val={vmean:.3f}  vs L0: {rel:+.1f}%   [{marker}]")

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump({"results": results, "config": vars(args)}, f,
                  indent=2, default=float)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

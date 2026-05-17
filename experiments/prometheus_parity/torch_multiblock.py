"""Multi-block transformer A/B: does L3-vs-L0 hold when stacking?

Single attention layer is the easiest test. Stacking exposes whether
the substrate-only attention COMPOSES across depth, or whether
deeper models reveal a need for learned attention that single-block
hid.

Architecture: stack `n_blocks` of (Attn + Residual + LN + FFN +
Residual + LN), same as the single-block model except repeated.

If L3 still beats L0 at depth=4, substrate attention isn't just
useful at the layer level — it's a structurally valid architectural
component.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_4way import (
    lcg, make_matrix, crt_pe,
    AttentionL0, AttentionL1, AttentionL2, AttentionL3,
    build_vocab,
)


class TransformerBlock(nn.Module):
    """One transformer block: Attn → +residual → LN → FFN → +residual → LN."""
    def __init__(self, variant: str, d_model: int, ff_dim: int,
                 seq_len: int, seed: int):
        super().__init__()
        attn_cls = {"L0": AttentionL0, "L1": AttentionL1,
                    "L2": AttentionL2, "L3": AttentionL3}[variant]
        self.attn = attn_cls(d_model, seq_len, seed)
        s = self.attn.rng_state
        self.ln1_g = nn.Parameter(torch.ones(d_model))
        self.ln1_b = nn.Parameter(torch.zeros(d_model))
        W_up, s = make_matrix(d_model, ff_dim, 0.3, s + 13)
        W_down, s = make_matrix(ff_dim, d_model, 0.3, s)
        self.ff_up = nn.Parameter(W_up)
        self.ff_up_b = nn.Parameter(torch.zeros(ff_dim))
        self.ff_down = nn.Parameter(W_down)
        self.ff_down_b = nn.Parameter(torch.zeros(d_model))
        self.ln2_g = nn.Parameter(torch.ones(d_model))
        self.ln2_b = nn.Parameter(torch.zeros(d_model))
        self.rng_state = s

    def forward(self, x):
        attn_out = self.attn(x)
        x_post_attn = x + attn_out
        normed1 = F.layer_norm(x_post_attn, (x.size(-1),),
                               weight=self.ln1_g, bias=self.ln1_b)
        up = normed1 @ self.ff_up + self.ff_up_b
        activated = F.relu(up)
        down = activated @ self.ff_down + self.ff_down_b
        x_post_ff = x_post_attn + down
        normed2 = F.layer_norm(x_post_ff, (x.size(-1),),
                               weight=self.ln2_g, bias=self.ln2_b)
        return normed2


class MultiBlockTransformer(nn.Module):
    def __init__(self, variant: str, vocab: int, d_model: int, ff_dim: int,
                 seq_len: int, n_blocks: int, seed: int):
        super().__init__()
        s = seed
        E, s = make_matrix(vocab, d_model, 0.3, s)
        self.embedding = nn.Parameter(E)
        self.register_buffer("pe_table", crt_pe(seq_len, d_model))
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            block = TransformerBlock(variant, d_model, ff_dim, seq_len, s + 100 * (i + 1))
            self.blocks.append(block)
            s = block.rng_state
        W_head, _ = make_matrix(d_model, vocab, 0.3, s + 17)
        self.head = nn.Parameter(W_head)
        self.head_b = nn.Parameter(torch.zeros(vocab))

    def forward(self, token_ids):
        x = self.embedding[token_ids] + self.pe_table[:token_ids.size(0)]
        for block in self.blocks:
            x = block(x)
        return x @ self.head + self.head_b


def train_arm(variant: str, ids: list, vocab_size: int, seq_len: int,
              d_model: int, ff_dim: int, n_blocks: int, lr: float,
              steps: int, seed: int):
    torch.manual_seed(seed)
    model = MultiBlockTransformer(variant, vocab_size, d_model, ff_dim,
                                   seq_len, n_blocks, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   betas=(0.9, 0.999), eps=1e-8)
    n_windows = len(ids) - seq_len - 1
    ids_tensor = torch.tensor(ids, dtype=torch.long)
    tail_losses = []
    for step in range(steps):
        start = step % n_windows
        window = ids_tensor[start:start + seq_len]
        targets = ids_tensor[start + 1:start + 1 + seq_len]
        logits = model(window)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step >= steps - 10:
            tail_losses.append(loss.item())
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(tail_losses) / len(tail_losses), n_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="42,7,123,2026,1")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--out", type=str, default="results_torch_multiblock.json")
    args = parser.parse_args()

    text = "the quick brown fox jumps over the lazy dog and the dog sleeps in the sun"
    chars, lookup = build_vocab(text)
    vocab_size = len(chars)
    ids = [lookup[c] for c in text]
    seq_len = 8
    d_model = 16
    ff_dim = 32
    seeds = [int(s) for s in args.seeds.split(",")]
    variants = ["L0", "L1", "L2", "L3"]

    print(f"=== Multi-block ({args.n_blocks} layers) attention A/B ===")
    print(f"setup: corpus={len(text)} vocab={vocab_size} seq={seq_len} "
          f"d={d_model} ff={ff_dim} n_blocks={args.n_blocks}")
    print(f"  steps={args.steps} lr={args.lr} seeds={seeds}\n", flush=True)

    results = {}
    for v in variants:
        losses = []
        n_params = 0
        for seed in seeds:
            loss, n_params = train_arm(v, ids, vocab_size, seq_len, d_model,
                                        ff_dim, args.n_blocks, args.lr,
                                        args.steps, seed)
            losses.append(loss)
        results[v] = {"losses": losses, "n_params": n_params,
                      "mean": sum(losses) / len(losses),
                      "std": statistics.stdev(losses) if len(losses) > 1 else 0.0}
        print(f"[{v}] params={n_params:5d}  mean={results[v]['mean']:.4f}  "
              f"std={results[v]['std']:.4f}", flush=True)

    print("\n=== Summary vs L0 ===")
    base_mean = results["L0"]["mean"]
    base_losses = results["L0"]["losses"]
    for v in variants:
        wins = sum(1 for x, b in zip(results[v]["losses"], base_losses) if x < b)
        rel = (results[v]["mean"] - base_mean) / base_mean * 100
        marker = "—" if v == "L0" else f"{rel:+.1f}%"
        print(f"  {v}: mean={results[v]['mean']:.4f}  vs L0: {marker:>8}  "
              f"wins={wins}/{len(base_losses)}")

    out = {"n_blocks": args.n_blocks, "seeds": seeds, "steps": args.steps,
           "results": results}
    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

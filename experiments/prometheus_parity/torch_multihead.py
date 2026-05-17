"""Multi-head L0 vs L1 at TinyShakespeare scale.

The production-shape validation. Yesterday: single-head L1 wins -8.0% val.
4-block-stacked single-head L1 wins -1.9% val.

This run: MULTI-HEAD (n_heads=4). Standard transformer pattern. If L1
still wins here, substrate-K is the production architecture
recommendation. If L0 catches up, multi-head's content-keying capacity
absorbed the substrate's advantage.

Setup:
  - TinyShakespeare 90/10 train/val
  - d_model=32, n_heads=4 (d_head=8), seq_len=32, ff=64
  - 1500 steps, AdamW lr=0.005
  - 3 seeds (matches yesterday's pattern)
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

from torch_4way import lcg, make_matrix, crt_pe, build_vocab


# ---- Multi-head attention variants ----


class AttentionL0_MH(nn.Module):
    """Standard multi-head: learned Q, K, V per head, then output projection."""
    def __init__(self, d_model: int, n_heads: int, seq_len: int, seed: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        s = seed + 11
        W_q, s = make_matrix(d_model, d_model, 0.3, s)
        W_k, s = make_matrix(d_model, d_model, 0.3, s)
        W_v, s = make_matrix(d_model, d_model, 0.3, s)
        W_o, s = make_matrix(d_model, d_model, 0.3, s)
        self.W_q = nn.Parameter(W_q)
        self.W_k = nn.Parameter(W_k)
        self.W_v = nn.Parameter(W_v)
        self.W_o = nn.Parameter(W_o)
        self.rng_state = s

    def forward(self, x):
        T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = (x @ self.W_q).view(T, H, dh).transpose(0, 1)  # [H, T, dh]
        k = (x @ self.W_k).view(T, H, dh).transpose(0, 1)
        v = (x @ self.W_v).view(T, H, dh).transpose(0, 1)
        scores = (q @ k.transpose(-2, -1)) / (dh ** 0.5)    # [H, T, T]
        attn = F.softmax(scores, dim=-1)
        out = attn @ v                                       # [H, T, dh]
        out = out.transpose(0, 1).contiguous().view(T, D)    # [T, D]
        return out @ self.W_o


class AttentionL1_MH(nn.Module):
    """Multi-head substrate-K: K replaced by CRT-PE (same per-head, shared
    across all heads) + learned Q, V, output projection. Each head still
    has its own Q + V — that's where content-keying happens. K is fixed
    structural prior.
    """
    def __init__(self, d_model: int, n_heads: int, seq_len: int, seed: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        s = seed + 11
        W_q, s = make_matrix(d_model, d_model, 0.3, s)
        W_v, s = make_matrix(d_model, d_model, 0.3, s)
        W_o, s = make_matrix(d_model, d_model, 0.3, s)
        self.W_q = nn.Parameter(W_q)
        self.W_v = nn.Parameter(W_v)
        self.W_o = nn.Parameter(W_o)
        # Substrate K: build a per-head [seq_len, d_head] CRT-PE table.
        # Same CRT-PE matrix, sliced by head.
        pe_full = crt_pe(seq_len, d_model)                   # [T, D]
        pe_per_head = pe_full.view(seq_len, n_heads,
                                    self.d_head).transpose(0, 1)  # [H, T, dh]
        self.register_buffer("K_const_mh", pe_per_head)
        self.rng_state = s

    def forward(self, x):
        T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = (x @ self.W_q).view(T, H, dh).transpose(0, 1)
        v = (x @ self.W_v).view(T, H, dh).transpose(0, 1)
        k = self.K_const_mh                                  # [H, T, dh]
        scores = (q @ k.transpose(-2, -1)) / (dh ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(0, 1).contiguous().view(T, D)
        return out @ self.W_o


# ---- Transformer block + model ----


class TransformerBlockMH(nn.Module):
    def __init__(self, variant: str, d_model: int, n_heads: int,
                 ff_dim: int, seq_len: int, seed: int):
        super().__init__()
        attn_cls = {"L0": AttentionL0_MH, "L1": AttentionL1_MH}[variant]
        self.attn = attn_cls(d_model, n_heads, seq_len, seed)
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


class MultiHeadModel(nn.Module):
    def __init__(self, variant: str, vocab: int, d_model: int,
                 n_heads: int, ff_dim: int, seq_len: int,
                 n_blocks: int, seed: int):
        super().__init__()
        s = seed
        E, s = make_matrix(vocab, d_model, 0.3, s)
        self.embedding = nn.Parameter(E)
        self.register_buffer("pe_table", crt_pe(seq_len, d_model))
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            block = TransformerBlockMH(variant, d_model, n_heads, ff_dim,
                                        seq_len, s + 100 * (i + 1))
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


# ---- Train with val split ----


def train_with_val(variant, train_ids, val_ids, vocab_size, seq_len,
                   d_model, n_heads, ff_dim, n_blocks, lr, steps, seed,
                   val_every=200, n_val_batches=30):
    torch.manual_seed(seed)
    random.seed(seed)
    model = MultiHeadModel(variant, vocab_size, d_model, n_heads, ff_dim,
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
    val_mean = val_history[-1][1]
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return train_mean, val_mean, n_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="42,7,123")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--out", type=str,
                        default="results_torch_multihead_tinyshakespeare.json")
    args = parser.parse_args()

    corpus_path = (Path(__file__).parent.parent
                   / "transformerless_lm" / "tinyshakespeare.txt")
    text = corpus_path.read_text()
    chars, lookup = build_vocab(text)
    vocab_size = len(chars)
    ids = [lookup[c] for c in text]
    split = int(len(ids) * 0.9)
    train_ids = ids[:split]
    val_ids = ids[split:]
    seeds = [int(s) for s in args.seeds.split(",")]
    variants = ["L0", "L1"]

    print(f"=== Multi-head ({args.n_heads}h × {args.n_blocks}b) + TinyShakespeare ===")
    print(f"corpus: {len(text):,} chars; train {len(train_ids):,}; val {len(val_ids):,}")
    print(f"vocab={vocab_size} seq={args.seq_len} d_model={args.d_model} "
          f"n_heads={args.n_heads} d_head={args.d_model // args.n_heads} ff={args.ff_dim}")
    print(f"steps={args.steps} lr={args.lr} seeds={seeds}\n", flush=True)

    results = {}
    for v in variants:
        train_means, val_means = [], []
        n_params = 0
        for seed in seeds:
            tm, vm, n_params = train_with_val(
                v, train_ids, val_ids, vocab_size, args.seq_len,
                args.d_model, args.n_heads, args.ff_dim, args.n_blocks,
                args.lr, args.steps, seed,
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

    print("=== Multi-head + TinyShakespeare verdict ===")
    l0 = results["L0"]
    l1 = results["L1"]
    delta_val = l1["val_mean"] - l0["val_mean"]
    rel_val = delta_val / l0["val_mean"] * 100
    wins = sum(1 for x, b in zip(l1["val"], l0["val"]) if x < b)
    print(f"L0 params={l0['n_params']}  train={l0['train_mean']:.4f}  val={l0['val_mean']:.4f}")
    print(f"L1 params={l1['n_params']}  train={l1['train_mean']:.4f}  val={l1['val_mean']:.4f}")
    print(f"L1 vs L0 (val): {rel_val:+.2f}%  wins={wins}/{len(l0['val'])}")
    print(f"Param savings: {(l0['n_params'] - l1['n_params']) / l0['n_params'] * 100:.1f}%")
    if l1["val_mean"] < l0["val_mean"]:
        print(f"\n[L1 WINS @ MULTI-HEAD] Substrate-K composes with multi-head at scale.")
        print(f"  → Production recommendation: L1 multi-head is the default attention block.")
    else:
        print(f"\n[L0 wins at multi-head scale] — multi-head's per-head content-keying")
        print("  may absorb the substrate's advantage. Worth investigating.")

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump({"results": results, "config": vars(args)}, f, indent=2, default=float)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

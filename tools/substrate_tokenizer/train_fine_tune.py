"""Stage 3 of the substrate-tokenizer pipeline — CPU sanity training.

Trains a tiny (~3M-param) char-level Transformer to learn one
substrate-specific behavior: when a `<omc:N>` token would correctly
reference content, emit it instead of the content itself.

This is NOT a useful model. It's the pipeline-end-to-end proof.
For a real fine-tune on a meaningful base model, see gpu_fine_tune.md.

What this demonstrates:
  - The vocab table from build_vocab.py can be loaded
  - Training loop runs end-to-end on CPU in <5 minutes
  - Loss decreases (model is learning to emit reference tokens)
  - The trained model emits reference tokens at correct positions
    on a synthetic test set

Usage:
    python3 train_fine_tune.py --table hash_token_table.json \\
        --steps 500 --out tiny_model.pt
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True, type=Path,
                        help="hash_token_table.json from build_vocab.py")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("tiny_model.pt"))
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        print("PyTorch not installed. `pip install torch` then re-run.",
              file=sys.stderr)
        sys.exit(2)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    table = json.loads(args.table.read_text())
    base_tok = table["base_token_id"]
    n_refs = len(table["tokens"])
    if n_refs == 0:
        print("vocab table empty — re-run build_vocab.py first", file=sys.stderr)
        sys.exit(1)

    # Build a tiny synthetic corpus.
    # Each example is a context followed by either a reference token
    # (when content matches the vocab) or the raw content (when it
    # doesn't). The model learns to pick.
    #
    # Vocab:
    #   0-127: ASCII chars
    #   128: PAD
    #   129: BOS
    #   130: EOS
    #   base_tok+i (for i in 0..n_refs): reference tokens
    PAD, BOS, EOS = 128, 129, 130
    # Renumber reference tokens densely so we don't need a 128K-vocab embedding.
    # Tokens 131..131+n_refs-1 are the reference token slots in this tiny model.
    REF_BASE = 131
    vocab_size = REF_BASE + n_refs
    print(f"sanity train: vocab_size={vocab_size}, n_ref_tokens={n_refs}, "
          f"steps={args.steps}, batch={args.batch_size}", file=sys.stderr)

    def encode(s: str) -> list[int]:
        return [ord(c) & 0x7F for c in s]

    def random_chars(n: int) -> list[int]:
        # Random printable ASCII so the "novel content" tokens look plausible.
        return [random.randint(32, 126) for _ in range(n)]

    def make_batch():
        """Each example: context-window then either ref_tok or raw chars.
        Half the time we emit a ref token (mapping to one of the vocab slots);
        the other half we emit raw chars (no reference applicable).
        The label sequence is the input shifted by 1 (standard LM training).
        """
        xs, ys = [], []
        for _ in range(args.batch_size):
            slot = random.randint(0, n_refs - 1)
            use_ref = random.random() < 0.5
            # Use the slot ID as a "context cue" so the model has SOMETHING
            # to learn correlation against.
            cue = [BOS, slot % 26 + ord('a')]  # 2-token cue
            if use_ref:
                target_tok = REF_BASE + slot
                seq = cue + [target_tok, EOS]
            else:
                body = random_chars(8)
                seq = cue + body + [EOS]
            # Pad to seq_len.
            seq = seq[: args.seq_len]
            seq = seq + [PAD] * (args.seq_len - len(seq))
            xs.append(seq[:-1])
            ys.append(seq[1:])
        x = torch.tensor(xs, dtype=torch.long)
        y = torch.tensor(ys, dtype=torch.long)
        return x, y

    # Tiny model.
    class TinyTransformer(nn.Module):
        def __init__(self, vocab, d_model, n_blocks, seq_len):
            super().__init__()
            self.embed = nn.Embedding(vocab, d_model)
            self.pe = nn.Parameter(torch.zeros(seq_len, d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=4, dim_feedforward=d_model * 4,
                batch_first=True, dropout=0.0,
            )
            self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
            self.head = nn.Linear(d_model, vocab)
            self.seq_len = seq_len

        def forward(self, x):
            T = x.size(1)
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
            h = self.embed(x) + self.pe[:T]
            h = self.blocks(h, mask=mask)
            return self.head(h)

    model = TinyTransformer(vocab_size, args.d_model, args.n_blocks, args.seq_len)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"sanity train: model params={n_params:,}", file=sys.stderr)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    t0 = time.time()
    losses = []
    for step in range(args.steps):
        x, y = make_batch()
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size), y.reshape(-1), ignore_index=PAD
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 50 == 0 or step == args.steps - 1:
            elapsed = time.time() - t0
            avg = sum(losses[-50:]) / max(len(losses[-50:]), 1)
            print(f"  step {step:4d}  loss={loss.item():.3f}  avg50={avg:.3f}  ({elapsed:.1f}s)",
                  flush=True)

    # Evaluate: feed a few cue contexts; check whether the model
    # predicts the correct reference token in the third position.
    model.eval()
    correct = 0
    total = 30
    with torch.no_grad():
        for _ in range(total):
            slot = random.randint(0, n_refs - 1)
            cue = torch.tensor([[BOS, slot % 26 + ord('a')]], dtype=torch.long)
            logits = model(cue)
            pred = int(logits[0, -1].argmax().item())
            target = REF_BASE + slot
            if pred == target:
                correct += 1
    print(f"\nsanity eval: {correct}/{total} correct reference-token predictions "
          f"({100 * correct / total:.0f}%)")
    if correct >= total * 0.8:
        print("  ✓ pipeline works: model learned cue → reference-token mapping")
    elif correct >= total * 0.3:
        print("  ~ partial learning. More steps or richer cues would push this up.")
    else:
        print("  ✗ no learning. Hyperparameters / data may need adjustment.")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "vocab_size": vocab_size,
                "d_model": args.d_model,
                "n_blocks": args.n_blocks,
                "seq_len": args.seq_len,
                "n_refs": n_refs,
                "ref_base": REF_BASE,
            },
            "vocab_table": table,
        },
        args.out,
    )
    print(f"sanity train: saved {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

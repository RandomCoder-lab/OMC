"""Curriculum fine-tuning: load existing LM checkpoint, train on function definitions.

The key difference from plain pre-training: we use omc_fndefs.txt (complete function
definitions, signature + body) instead of random sliding windows over the full corpus.

Effect: the LM's context window almost always starts near a `fn name(...)` signature,
so it learns the conditional distribution p(body_char | fn_name in context).
This is exactly how GPT-style models learn to synthesize code from function names.

No architecture changes — same FibRecLMNavigator, same d_model=64, same seq_len=89.
Just a different training data distribution: intent-bearing windows (fn name → body).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from train_substrate_attention import FibRecLMSubsim
from train_address_navigator import FibRecLMNavigator, FibonacciAdamW


def load_curriculum(corpus_path: Path, seq_len: int, device: str):
    """Load fndefs corpus and encode as char ids.

    Intentionally keeps function defs contiguous (separated by double newline)
    so that training windows often start near a `fn` signature.
    """
    text = corpus_path.read_text(errors='replace')
    # Use same 168-char vocab as the base model — don't extend it
    chars = sorted(set(text))
    stoi  = {c: i for i, c in enumerate(chars)}
    ids   = torch.tensor([stoi.get(c, 0) for c in text], dtype=torch.long)
    ids   = ids.to(device)
    # 90/10 split
    n     = len(ids)
    split = int(n * 0.9)
    return chars, stoi, ids[:split], ids[split:]


def build_fn_start_offsets(corpus: torch.Tensor, fn_tok: List[int]) -> List[int]:
    """Find all positions where a function definition begins (`fn ` token sequence).

    Training windows anchored to function starts teach the model to complete
    function bodies given a signature prefix — much more aligned with synthesis
    than random windows.
    """
    ids    = corpus.tolist()
    n      = len(ids)
    flen   = len(fn_tok)
    starts: List[int] = []
    for i in range(n - flen):
        if ids[i:i + flen] == fn_tok:
            starts.append(i)
    return starts


def sample_batch(corpus: torch.Tensor, seq_len: int, batch: int,
                 gen: torch.Generator,
                 fn_starts: Optional[List[int]] = None):
    """Sample training batch.  When fn_starts provided, all windows begin at
    function signatures — teaching signature→body completion.
    """
    if fn_starts and len(fn_starts) >= batch:
        # 80% signature-anchored, 20% random (to avoid over-fitting to starts)
        n_anchored = int(batch * 0.8)
        n_random   = batch - n_anchored
        aidx = torch.randint(0, len(fn_starts), (n_anchored,), generator=gen)
        anchor_pos = [fn_starts[i] for i in aidx.tolist()]
        ridx = torch.randint(0, len(corpus) - seq_len - 1, (n_random,), generator=gen).tolist()
        all_pos = anchor_pos + ridx
    else:
        n   = len(corpus) - seq_len - 1
        all_pos = torch.randint(0, n, (batch,), generator=gen).tolist()

    x = torch.stack([corpus[i: i + seq_len] for i in all_pos])
    y = torch.stack([corpus[i + 1: i + seq_len + 1] for i in all_pos])
    return x, y


@torch.no_grad()
def eval_loss(model, corpus: torch.Tensor, seq_len: int, n_batches: int,
              gen: torch.Generator, device: str) -> float:
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        x, y = sample_batch(corpus, seq_len, 32, gen)
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total += loss.item()
    model.train()
    return total / n_batches


def train(args: argparse.Namespace) -> None:
    device = args.device
    torch.manual_seed(42)
    gen      = torch.Generator(); gen.manual_seed(43)
    eval_gen = torch.Generator(); eval_gen.manual_seed(44)

    # ── Load base checkpoint ──────────────────────────────────────────────────
    ckpt_path = _HERE / args.checkpoint
    print(f"Loading checkpoint: {ckpt_path} ...", flush=True)
    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)

    orig_vocab = ckpt['vocab']   # list of chars
    d_model    = ckpt.get('d_model',  64)
    n_blocks   = ckpt.get('n_blocks', 4)
    seq_len    = ckpt.get('seq_len',  89)
    K_init     = ckpt.get('K_init',   89)
    orig_stoi  = {c: i for i, c in enumerate(orig_vocab)}
    vocab_size = len(orig_vocab)

    # ── Load curriculum corpus ────────────────────────────────────────────────
    corpus_path = _HERE / args.corpus
    print(f"Loading curriculum corpus: {corpus_path} ...", flush=True)
    cur_chars, cur_stoi, train_ids, val_ids = load_curriculum(corpus_path, seq_len, device)

    # Remap curriculum ids to base vocab (chars not in base vocab → 0)
    print(f"  curriculum vocab size: {len(cur_chars)}", flush=True)
    print(f"  base vocab size: {vocab_size}", flush=True)

    # Remap: for each id in train_ids, convert char → base vocab id
    cur_itos  = {i: c for c, i in cur_stoi.items()}
    remap     = torch.zeros(len(cur_chars), dtype=torch.long)
    for ci, c in cur_itos.items():
        remap[ci] = orig_stoi.get(c, 0)   # unknown chars → 0 (first char in base vocab)
    train_ids = remap[train_ids.cpu()].to(device)
    val_ids   = remap[val_ids.cpu()].to(device)
    print(f"  train tokens: {len(train_ids):,}  val: {len(val_ids):,}", flush=True)

    # Build function-start offsets so 80% of training windows begin at `fn `
    fn_tok = [orig_stoi.get(c, 0) for c in 'fn ']
    fn_starts = build_fn_start_offsets(train_ids, fn_tok)
    fn_starts = [s for s in fn_starts if s + seq_len < len(train_ids)]
    print(f"  fn_starts: {len(fn_starts):,} function-definition anchors", flush=True)

    # ── Build model and load weights ──────────────────────────────────────────
    base = FibRecLMSubsim(
        vocab_size=vocab_size, d_model=d_model, n_blocks=n_blocks,
        seq_len=seq_len, K=K_init, mode="cross", K_sig=32,
    ).to(device)
    model = FibRecLMNavigator(base, d_model).to(device)

    # Load only the base LM weights (skip intent_head if present)
    base_state = {k: v for k, v in ckpt['model_state'].items()
                  if not k.startswith('intent_head.')}
    model.load_state_dict(base_state, strict=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params={n_params:,}  loaded from {args.checkpoint}", flush=True)

    # ── Optimizer — lower LR for fine-tuning ─────────────────────────────────
    optimizer = FibonacciAdamW(model.parameters(), lr=args.lr)
    # Cosine decay over fine-tune steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.1
    )

    # ── Baseline eval ─────────────────────────────────────────────────────────
    baseline = eval_loss(model, val_ids, seq_len, 20, eval_gen, device)
    print(f"\nBaseline val loss on fndefs: {baseline:.4f}")
    print(f"(Base LM was trained on full corpus, expects {ckpt.get('final_char_loss', '?')} on that)")
    print(f"\nFine-tuning for {args.steps} steps  lr={args.lr}")
    print()

    best_val  = baseline
    log_every = max(1, args.steps // 20)
    t0        = time.time()

    for step in range(args.steps):
        x, y = sample_batch(train_ids, seq_len, args.batch, gen, fn_starts=fn_starts)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % log_every == 0:
            val = eval_loss(model, val_ids, seq_len, 10, eval_gen, device)
            elapsed = time.time() - t0
            print(f"  step {step+1:4d}/{args.steps}  train={loss.item():.4f}"
                  f"  val={val:.4f}  ({elapsed:.0f}s)", flush=True)
            if val < best_val:
                best_val = val
                # Save checkpoint
                out_path = _HERE / args.out
                torch.save({
                    **{k: v for k, v in ckpt.items() if k != 'model_state'},
                    'model_state': model.state_dict(),
                    'final_char_loss': val,
                    'curriculum_finetuned': True,
                }, str(out_path))
                print(f"    ← BEST  saved to {args.out}")

    print(f"\nDone.  Best val={best_val:.4f}  Output: {args.out}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Curriculum fine-tune FibRecLMNavigator on fn definitions")
    p.add_argument('--checkpoint', default='bloom_checkpoint_model.pt')
    p.add_argument('--corpus',     default='omc_fndefs.txt')
    p.add_argument('--out',        default='bloom_curriculum_model.pt')
    p.add_argument('--steps',      type=int,   default=2000)
    p.add_argument('--batch',      type=int,   default=32)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--device',     type=str,   default='cpu')
    args = p.parse_args()
    train(args)

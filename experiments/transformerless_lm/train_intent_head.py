"""Train an intent-conditioning head on the existing FibRecLMNavigator.

The intent head is a small Linear(d_model → vocab_size) that maps the φ-hash
of a function name to a logit bias applied at every generation step.

Training data is mined from the corpus itself: every `fn name(...)` definition
gives one (φ-hash(name), function_body) pair — no word lists, no external labels.
This stays fully universal: the same head design works on any new corpus because
φ-hash is corpus-agnostic.

After training, the checkpoint gains an `intent_head` key.  omc_assistant.py
already checks `hasattr(model, 'intent_head')` and applies the bias automatically.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from corpus_address_index import substrate_fingerprint
from train_substrate_attention import FibRecLMSubsim
from train_address_navigator import FibRecLMNavigator


# ─────────────────────────────────────────────────────────────────────────────
# Intent head — plugs onto FibRecLMNavigator, biases logits per query
# ─────────────────────────────────────────────────────────────────────────────

class IntentHead(nn.Module):
    """φ-hash(intent) → logit bias over vocabulary.

    Two-layer MLP with bottleneck keeps param count low (~8K for d=64, v=168).
    The bottleneck forces the head to learn compact intent representations rather
    than memorising individual fingerprints.
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model, bias=False)
        self.fc2 = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, intent_fp: torch.Tensor) -> torch.Tensor:
        """intent_fp: [..., d_model] → [..., vocab_size] logit bias."""
        return self.fc2(F.gelu(self.fc1(intent_fp)))


# ─────────────────────────────────────────────────────────────────────────────
# Corpus mining — (fn_name, fn_body) pairs
# ─────────────────────────────────────────────────────────────────────────────

def _extract_body(text: str, open_brace: int, max_len: int = 512) -> str:
    """Extract function body (including braces) starting at open_brace."""
    depth = 0
    i = open_brace
    while i < min(open_brace + max_len, len(text)):
        c = text[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return text[open_brace: i + 1]
        i += 1
    return ""


def mine_pairs(corpus_path: Path, max_pairs: int = 10_000) -> List[Tuple[str, str]]:
    """Extract (fn_name, fn_signature+body) pairs from corpus.

    Returns at most max_pairs pairs, preferring longer / more complete bodies.
    """
    text = corpus_path.read_text(errors='replace')
    # Match: fn name( ... ) { — capture name and locate opening brace
    pattern = re.compile(r'\bfn\s+([a-zA-Z_]\w*)\s*\([^)]*\)\s*\{')
    pairs: List[Tuple[str, str]] = []
    seen_names: set = set()

    for m in pattern.finditer(text):
        fn_name = m.group(1)
        open_pos = m.end() - 1  # position of '{'
        body = _extract_body(text, open_pos, max_len=512)
        if not body or len(body) < 10:
            continue
        full = text[m.start(): m.start() + len(m.group(0)) - 1 + len(body)]
        if len(full) > 50:
            pairs.append((fn_name, full))
        if len(pairs) >= max_pairs * 3:
            break

    # Deduplicate by name, keep longest body per name
    by_name: dict = {}
    for name, body in pairs:
        if name not in by_name or len(body) > len(by_name[name]):
            by_name[name] = body

    unique = list(by_name.items())
    print(f"  mined {len(unique)} unique (name, body) pairs from corpus")
    return unique[:max_pairs]


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    ckpt_path = _HERE / args.checkpoint
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    corpus_path = _HERE / args.corpus
    if not corpus_path.exists():
        print(f"Corpus not found: {corpus_path}")
        sys.exit(1)

    # ── Load base LM ──────────────────────────────────────────────────────────
    print("Loading LM checkpoint ...")
    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    vocab   = ckpt['vocab']
    d_model = ckpt.get('d_model', 64)
    n_blocks = ckpt.get('n_blocks', 4)
    seq_len = ckpt.get('seq_len', 89)
    K_init  = ckpt.get('K_init', 89)
    stoi = {c: i for i, c in enumerate(vocab)}

    base = FibRecLMSubsim(
        vocab_size=len(vocab), d_model=d_model, n_blocks=n_blocks,
        seq_len=seq_len, K=K_init, mode="cross", K_sig=32,
    )
    lm = FibRecLMNavigator(base, d_model)
    lm.load_state_dict(ckpt['model_state'])
    lm.eval()
    for p in lm.parameters():
        p.requires_grad_(False)   # freeze base LM

    # ── Build intent head ─────────────────────────────────────────────────────
    head = IntentHead(d_model, len(vocab))
    print(f"  IntentHead: {sum(p.numel() for p in head.parameters())} params")

    # ── Mine training pairs ───────────────────────────────────────────────────
    print("Mining (fn_name, fn_body) pairs from corpus ...")
    pairs = mine_pairs(corpus_path, max_pairs=args.max_pairs)
    print(f"  {len(pairs)} training pairs")

    # Pre-compute φ-hashes for all function names (intent fingerprints)
    print("Pre-computing intent fingerprints ...")
    intent_fps: List[torch.Tensor] = []
    bodies: List[str] = []
    for name, body in pairs:
        fp = substrate_fingerprint(name, d_model)
        intent_fps.append(fp)
        bodies.append(body)
    intent_fps_t = torch.stack(intent_fps)   # [N, d_model]

    # ── Training loop ─────────────────────────────────────────────────────────
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    N = len(bodies)
    print(f"\nTraining intent head for {args.steps} steps  lr={args.lr}  batch={args.batch}")
    print(f"  base LM frozen — only IntentHead ({sum(p.numel() for p in head.parameters())} params) trains")
    print()

    t0 = time.time()
    total_loss = 0.0
    log_every = max(1, args.steps // 20)

    for step in range(args.steps):
        # Sample a random batch of (intent_fp, body) pairs
        idx = torch.randint(0, N, (args.batch,))
        batch_fps = intent_fps_t[idx]   # [B, d_model]

        # For each pair, pick a random window in the body as the target
        batch_loss = torch.tensor(0.0)
        for bi in range(args.batch):
            body = bodies[idx[bi].item()]
            intent_fp = batch_fps[bi]   # [d_model]

            # Encode body as char ids (up to seq_len + 1 chars)
            n = min(len(body), seq_len + 1)
            if n < 2:
                continue
            ids = torch.tensor(
                [stoi.get(c, 0) for c in body[:n]], dtype=torch.long
            )
            inp = ids[:-1].unsqueeze(0)    # [1, T]
            tgt = ids[1:]                  # [T-1]

            # LM forward (frozen)
            with torch.no_grad():
                logits, _ = lm(inp)        # [1, T-1, vocab]

            # Intent bias from head
            bias = head(intent_fp)         # [vocab]
            logits_biased = logits[0] + bias.unsqueeze(0)  # [T-1, vocab]

            loss = F.cross_entropy(logits_biased, tgt)
            batch_loss = batch_loss + loss

        batch_loss = batch_loss / args.batch
        opt.zero_grad()
        batch_loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        scheduler.step()

        total_loss += batch_loss.item()
        if (step + 1) % log_every == 0:
            avg = total_loss / log_every
            elapsed = time.time() - t0
            print(f"  step {step+1:4d}/{args.steps}  loss={avg:.4f}  ({elapsed:.0f}s)")
            total_loss = 0.0

    # ── Save updated checkpoint ───────────────────────────────────────────────
    out_path = _HERE / args.out
    print(f"\nSaving to {out_path} ...")

    # Merge intent_head into the LM model state
    lm.intent_head = head
    updated_model_state = {**ckpt['model_state'], **{f'intent_head.{k}': v for k, v in head.state_dict().items()}}

    torch.save({
        **ckpt,
        'model_state': updated_model_state,
        'has_intent_head': True,
    }, str(out_path))

    print(f"Done.  IntentHead saved into {out_path}")
    print(f"  Run omc_assistant.py — it will pick up intent_head automatically")
    print(f"  (checks hasattr(model, 'intent_head') at generation time)")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Train intent-conditioning head on FibRecLMNavigator")
    p.add_argument('--checkpoint', default='bloom_checkpoint_model.pt')
    p.add_argument('--corpus',     default='omc_corpus.txt')
    p.add_argument('--out',        default='bloom_intent_model.pt')
    p.add_argument('--steps',      type=int, default=500)
    p.add_argument('--batch',      type=int, default=8)
    p.add_argument('--lr',         type=float, default=3e-4)
    p.add_argument('--max-pairs',  type=int, default=5000)
    args = p.parse_args()
    train(args)

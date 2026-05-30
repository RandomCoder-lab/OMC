"""Address Navigator — the Proto LM learns to predict WHERE next, not just WHAT next.

Two-head architecture:
  1. Character head  (existing): logits [B, T, vocab] — what's the next char?
  2. Address head    (new):      addr_emb [B, d_model] — what address comes next?

Training signal:
  loss = char_CE + lambda_addr * (1 - cosine_sim(addr_pred, addr_target))

  addr_target = substrate_fingerprint(next_window_in_corpus)
    — the φ-hash of the next seq_len chars after the current window.
    Computed on-the-fly, no pre-built index needed during training.

Generation:
  For each hop:
    1. Run model on current context (89 chars) → char logits + addr_pred
    2. Sample next char from char logits (or greedy)
    3. addr_pred → HAddr via assign_address
    4. Retrieve real corpus text at that HAddr from corpus_address_index
    5. Output: generated chars + retrieved Shakespeare
    6. Use retrieved text as next context → repeat

This breaks the seq_len barrier: the model generates confidently within
89 chars but NAVIGATES to real corpus windows via substrate addresses.
"""

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from corpus import make_dataset
from train_substrate_attention import FibRecLMSubsim
from optimizers_fib import FibonacciAdamW
from train_K_shrink import K_schedule_tier_walk, set_K_active_recursive
from corpus_address_index import substrate_fingerprint, retrieve, batch_substrate_fingerprint
from hierarchical_address import assign_address, addr_str


# ── Corpus loader (supports file paths or built-in names) ─────────────────────

def load_corpus(source: str, seq_len: int, device: str):
    """Load any text corpus from a file path or built-in name.

    Returns (chars, itos, train_tensor, val_tensor, tok_ord).
    """
    if source == "tinyshakespeare" or not Path(source).exists():
        chars, stoi, itos_list, encoded = make_dataset(
            seq_len=seq_len, source="tinyshakespeare"
        )
        itos = {i: c for i, c in enumerate(chars)}
    else:
        text = Path(source).read_text(errors='replace')
        chars = sorted(set(text))
        stoi  = {c: i for i, c in enumerate(chars)}
        itos  = {i: c for i, c in enumerate(chars)}
        encoded = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    vocab = len(chars)
    n     = len(encoded)
    val_n = max(int(0.1 * n), seq_len * 4)
    train_corpus = encoded[:n - val_n].to(device)
    val_corpus   = encoded[n - val_n:].to(device)

    tok_ord = torch.zeros(vocab, dtype=torch.int64, device=device)
    for t, c in itos.items():
        tok_ord[t] = ord(c)

    return chars, itos, train_corpus, val_corpus, tok_ord


# ── Address head wrapper ──────────────────────────────────────────────────────

class FibRecLMNavigator(nn.Module):
    """FibRecLMSubsim + address prediction head.

    The address head is one Linear(d_model, d_model) applied to the
    mean-pooled final hidden state. It learns to project the context
    representation toward the φ-hash fingerprint of the next corpus window.
    """

    def __init__(self, base_model: FibRecLMSubsim, d_model: int):
        super().__init__()
        self.base   = base_model
        self.addr_head = nn.Linear(d_model, d_model, bias=False)
        # Init addr_head as identity-ish — don't disturb char training early on
        nn.init.eye_(self.addr_head.weight) if d_model == d_model else None

    def forward(self, token_ids: torch.Tensor):
        """Return (logits [B,T,vocab], addr_emb [B,d_model])."""
        B, T = token_ids.shape
        h = self.base.embed(token_ids) + self.base.pe[:T]
        m = self.base.mask[:T, :T]
        seeds_per_layer, _ = self.base._all_seeds_lucas_homeo()
        for n, seeds_n in enumerate(seeds_per_layer):
            h = self.base._layer_forward(h, m, n, seeds_n)
        h_norm  = self.base.ln_f(h)                        # [B, T, d]
        # Self-Witness AFTER LayerNorm: a per-position scalar on h here scales the
        # logits (on-manifold confidence), which LN cannot cancel. Placing it before
        # ln_f is a no-op (LN is scale-invariant per position). No-op when flag off.
        h_norm  = self.base.apply_self_witness(h_norm)
        logits  = self.base.head(h_norm)                   # [B, T, vocab]
        # Address prediction: pool over T, project
        addr_emb = self.addr_head(h_norm.mean(dim=1))      # [B, d]
        return logits, addr_emb

    def char_only(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Fast path: just logits, no addr computation (for eval)."""
        return self.forward(token_ids)[0]


# ── Address target (on-the-fly, vectorised) ───────────────────────────────────

def addr_target_for_batch(
    corpus: torch.Tensor,
    positions: torch.Tensor,
    seq_len: int,
    tok_ord: torch.Tensor,
    d_model: int,
    device: str,
) -> torch.Tensor:
    """Vectorised φ-hash fingerprint of the next window for each batch item.

    Replaces a Python loop over batch × seq_len character ordinals with a
    single int64 dot-product + broadcast. ~100× faster than the scalar version.
    tok_ord: [vocab] int64 tensor — ord(char) for each token id.
    """
    # Gather next-window token ids: [B, seq_len]
    B = positions.shape[0]
    next_starts = (positions + seq_len).clamp(0, corpus.shape[0] - seq_len)
    j_idx  = torch.arange(seq_len, device=corpus.device)                  # [T]
    idxs   = next_starts.unsqueeze(1) + j_idx.unsqueeze(0)                # [B, T]
    ids_batch = corpus[idxs]                                               # [B, T]
    return batch_substrate_fingerprint(ids_batch, tok_ord, d_model).to(device)


# ── Batch sampler ─────────────────────────────────────────────────────────────

def sample_batch_with_positions(
    corpus: torch.Tensor,
    batch_size: int,
    seq_len: int,
    gen: torch.Generator,
):
    """Sample random windows. Return (x, y, positions)."""
    n   = corpus.shape[0] - seq_len - seq_len - 1   # leave room for next window
    idx = torch.randint(0, n, (batch_size,), generator=gen)
    x   = torch.stack([corpus[i:i+seq_len]   for i in idx])
    y   = torch.stack([corpus[i+1:i+seq_len+1] for i in idx])
    return x, y, idx


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: FibRecLMNavigator, val_corpus: torch.Tensor,
             tok_ord: torch.Tensor, batch_size: int, seq_len: int,
             d_model: int, lambda_addr: float,
             gen: torch.Generator, n_batches: int = 8) -> dict:
    model.eval()
    char_losses, addr_losses = [], []
    device = next(model.parameters()).device
    for _ in range(n_batches):
        x, y, pos = sample_batch_with_positions(val_corpus, batch_size, seq_len, gen)
        x, y = x.to(device), y.to(device)
        logits, addr_pred = model(x)
        addr_tgt = addr_target_for_batch(val_corpus, pos, seq_len, tok_ord, d_model, device)
        cl = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        al = 1.0 - F.cosine_similarity(addr_pred, addr_tgt, dim=-1).mean()
        char_losses.append(cl.item())
        addr_losses.append(al.item())
    model.train()
    return {
        'char': sum(char_losses) / len(char_losses),
        'addr': sum(addr_losses) / len(addr_losses),
        'total': sum(char_losses)/len(char_losses) + lambda_addr * sum(addr_losses)/len(addr_losses),
    }


# ── Address-navigated generation ──────────────────────────────────────────────

@torch.no_grad()
def generate_navigated(
    model: FibRecLMNavigator,
    index: dict,
    stoi: dict,
    itos: dict,
    seed_text: str,
    seq_len: int,
    n_hops: int = 10,
    n_gen_chars: int = 40,
    temperature: float = 0.85,
    device: str = 'cpu',
) -> str:
    """Generate text using address navigation.

    Each hop:
      1. Model generates n_gen_chars autoregressively
      2. Model's addr_head predicts next address
      3. Retrieve real corpus text at that address
      4. Output: generated + retrieved
      5. Use retrieved text as next context
    """
    model.eval()
    vocab = len(stoi)
    ctx   = [stoi.get(c, 0) for c in seed_text[-seq_len:]]
    if len(ctx) < seq_len:
        ctx = [0] * (seq_len - len(ctx)) + ctx

    output_parts = [seed_text]

    for hop in range(n_hops):
        # ── Step 1: generate n_gen_chars autoregressively ──────────────────
        gen_chars = []
        run_ctx   = list(ctx)
        for _ in range(n_gen_chars):
            ids     = torch.tensor(run_ctx[-seq_len:],
                                   dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = model(ids)
            probs   = F.softmax(logits[0, -1] / temperature, dim=-1)
            tok     = torch.multinomial(probs, 1).item()
            gen_chars.append(tok)
            run_ctx.append(tok)

        gen_text = ''.join(itos.get(i, '?') for i in gen_chars)

        # ── Step 2: addr_head predicts next address ────────────────────────
        ids      = torch.tensor(run_ctx[-seq_len:],
                                dtype=torch.long, device=device).unsqueeze(0)
        _, addr_pred = model(ids)
        addr     = assign_address(addr_pred[0].cpu())

        # ── Step 3: retrieve real corpus text at that address ──────────────
        retrieved = retrieve(addr_pred[0].cpu(), index, top_k=1)
        if retrieved:
            ret_text = retrieved[0]['text']
            ret_addr = addr_str(retrieved[0]['addr'])
            ret_sim  = retrieved[0]['sim']
        else:
            ret_text = gen_text   # fallback: use generated
            ret_addr = addr_str(addr)
            ret_sim  = 0.0

        # ── Step 4: update context with retrieved text ─────────────────────
        ctx = [stoi.get(c, 0) for c in ret_text[-seq_len:]]
        if len(ctx) < seq_len:
            ctx = [0] * (seq_len - len(ctx)) + ctx

        output_parts.append(gen_text)
        output_parts.append(ret_text)

        print(f"  hop {hop+1:2d}: addr={ret_addr}  sim={ret_sim:.3f}  "
              f"gen={repr(gen_text[:30])}  →  {repr(ret_text[:40])}")

    return ''.join(output_parts)


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    device = args.device
    torch.manual_seed(args.seed)
    gen = torch.Generator(); gen.manual_seed(args.seed + 1)
    eval_gen = torch.Generator(); eval_gen.manual_seed(args.seed + 2)

    # ── Corpus ────────────────────────────────────────────────────────────────
    print(f"Loading corpus: {args.corpus} ...", flush=True)
    chars, itos, train_corpus, val_corpus, tok_ord = load_corpus(
        args.corpus, args.seq_len, device
    )
    vocab = len(chars)
    stoi  = {c: i for i, c in itos.items()}
    print(f"  vocab={vocab}  train={len(train_corpus):,}  val={len(val_corpus):,}")

    # ── Corpus address index ──────────────────────────────────────────────────
    here       = Path(__file__).parent
    index_path = here / "corpus_address_index.pt"
    if index_path.exists():
        print(f"Loading address index: {index_path}", flush=True)
        index = torch.load(str(index_path), map_location='cpu', weights_only=False)
        print(f"  {index['meta']['n_windows']:,} windows")
    else:
        print("No corpus_address_index.pt found — run corpus_address_index.py first")
        index = None

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Building model ...", flush=True)
    base = FibRecLMSubsim(
        vocab_size=vocab,
        d_model=args.d_model,
        n_blocks=args.n_blocks,
        seq_len=args.seq_len,
        K=args.K_init,
        mode="cross",
        K_sig=32,
    ).to(device)
    model    = FibRecLMNavigator(base, args.d_model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params={n_params:,}  d_model={args.d_model}  "
          f"n_blocks={args.n_blocks}  seq_len={args.seq_len}")

    # ── Warm-start from existing checkpoint (optional) ─────────────────────────
    init_from = getattr(args, 'init_from', None)
    if init_from:
        init_path = here / init_from
        if init_path.exists():
            print(f"Warm-starting from {init_from} ...", flush=True)
            ckpt_warm = torch.load(str(init_path), map_location='cpu', weights_only=False)
            src = ckpt_warm['model_state']
            dst = model.state_dict()
            transferred = 0
            for k, v in src.items():
                if k in dst and dst[k].shape == v.shape:
                    dst[k] = v
                    transferred += 1
            model.load_state_dict(dst)
            print(f"  transferred {transferred}/{len(dst)} tensors "
                  f"(pe/mask skipped — new seq_len)")
        else:
            print(f"  [warn] init_from={init_from} not found, training from scratch")

    optimizer = FibonacciAdamW(model.parameters(), lr=args.lr)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining {args.steps} steps  lambda_addr={args.lambda_addr}\n",
          flush=True)
    t0         = time.time()
    best_total    = float('inf')
    best_char_val = float('inf')
    eval_every = max(args.steps // 20, 200)
    cur_K      = None

    for step in range(args.steps):
        # K schedule (skip when --no-k-shrink → static full K, the control arm)
        if getattr(args, 'no_k_shrink', False):
            new_K = args.K_init
        else:
            new_K = K_schedule_tier_walk(step, args.steps,
                                         K_init=args.K_init, K_min=args.K_min)
        if new_K != cur_K:
            set_K_active_recursive(model.base, new_K)
            cur_K = new_K

        # Batch
        x, y, pos = sample_batch_with_positions(
            train_corpus, args.batch_size, args.seq_len, gen
        )
        x, y = x.to(device), y.to(device)

        # Forward
        logits, addr_pred = model(x)

        # Char loss
        char_loss = F.cross_entropy(
            logits.reshape(-1, vocab), y.reshape(-1)
        )

        # Address loss — ramp lambda up over first 20% of training
        ramp      = min(1.0, step / max(args.steps * 0.2, 1))
        lam       = args.lambda_addr * ramp
        addr_tgt  = addr_target_for_batch(
            train_corpus, pos, args.seq_len, tok_ord, args.d_model, device
        )
        addr_loss = 1.0 - F.cosine_similarity(addr_pred, addr_tgt, dim=-1).mean()

        loss = char_loss + lam * addr_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_every == 0 or step == args.steps - 1:
            metrics = evaluate(model, val_corpus, tok_ord, args.batch_size,
                               args.seq_len, args.d_model, args.lambda_addr,
                               eval_gen)
            marker = ""
            if metrics['total'] < best_total:
                best_total = metrics['total']
                best_char_val = metrics['char']
                marker = " ← BEST"
            print(f"  step {step:5d}  char={metrics['char']:.4f}  "
                  f"addr={metrics['addr']:.4f}  K={cur_K}  "
                  f"({time.time()-t0:.0f}s){marker}", flush=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_name = getattr(args, 'out', 'bloom_checkpoint_model.pt')
    ckpt_path = here / out_name
    torch.save({
        'model_state':      model.state_dict(),
        'base_state':       model.base.state_dict(),
        'vocab':            chars,
        'seq_len':          args.seq_len,
        'd_model':          args.d_model,
        'n_blocks':         args.n_blocks,
        'K_init':           args.K_init,
        'navigator':        True,
        'final_char_loss':  best_char_val,
    }, str(ckpt_path))
    print(f"\nSaved → {ckpt_path}")

    # ── Generation demo ───────────────────────────────────────────────────────
    if index is not None:
        print("\n" + "=" * 70)
        print("ADDRESS-NAVIGATED GENERATION:")
        print("=" * 70)
        output = generate_navigated(
            model, index, stoi, itos,
            seed_text="Before we proceed any further, hear me speak.",
            seq_len=args.seq_len,
            n_hops=12,
            n_gen_chars=45,
            temperature=0.85,
            device=device,
        )
        print("\n── OUTPUT ──────────────────────────────────────────────────")
        print(output)

    return model


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps',       type=int,   default=15000)
    parser.add_argument('--batch-size',  type=int,   default=32)
    parser.add_argument('--seq-len',     type=int,   default=89)
    parser.add_argument('--d-model',     type=int,   default=64)
    parser.add_argument('--n-blocks',    type=int,   default=4)
    parser.add_argument('--lr',          type=float, default=3e-4)
    parser.add_argument('--K-init',      type=int,   default=89)
    parser.add_argument('--K-min',       type=int,   default=13)
    parser.add_argument('--no-k-shrink', action='store_true',
                        help='Disable K-shrink: train at static full K_init (control arm). '
                             'K-shrink is now functional (was a no-op pre-2026-05-29).')
    parser.add_argument('--lambda-addr', type=float, default=0.25,
                        help='Weight on address prediction loss (0=char only)')
    parser.add_argument('--corpus',      type=str,   default='tinyshakespeare',
                        help='Corpus source: "tinyshakespeare" or path to a .txt file')
    parser.add_argument('--out',         type=str,   default='bloom_checkpoint_model.pt',
                        help='Output checkpoint filename')
    parser.add_argument('--init-from',   type=str,   default=None,
                        help='Warm-start from existing checkpoint (matching tensors only)')
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--device',      type=str,   default='cpu')
    args = parser.parse_args()

    train(args)

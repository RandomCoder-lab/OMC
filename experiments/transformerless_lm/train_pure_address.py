"""Pure address navigator — one matrix, no characters, no vocab.

The model IS the substrate transition function:
    fingerprint(context) → fingerprint(what_comes_next)

Architecture:  AddressNavigator = nn.Linear(d_model, d_model, bias=False)
Parameters:    d_model² = 4,096  (for d_model=64)
Loss:          1 - cosine_sim(W @ fp_current, fp_next)
Generation:    hop → retrieve → hop → retrieve  (no character sampling)

Works on any corpus unchanged. Code, prose, Shakespeare — same math.
The code itself could be one address. The model navigates between them.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from corpus_address_index import substrate_fingerprint, retrieve, retrieve_dual, batch_substrate_fingerprint
from hierarchical_address import assign_address, addr_str


def load_corpus(source: str, seq_len: int, device: str):
    """Load any text corpus from a file path or built-in name.

    Returns (chars, itos, train_tensor, val_tensor, tok_ord).
    """
    if source == "tinyshakespeare" or not Path(source).exists():
        from corpus import make_dataset
        chars, stoi, itos_list, encoded = make_dataset(
            seq_len=seq_len, source="tinyshakespeare"
        )
        itos = {i: c for i, c in enumerate(chars)}
        text_chars = chars
    else:
        text = Path(source).read_text(errors='replace')
        text_chars = sorted(set(text))
        stoi   = {c: i for i, c in enumerate(text_chars)}
        itos   = {i: c for i, c in enumerate(text_chars)}
        encoded = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    vocab = len(text_chars)
    n     = len(encoded)
    val_n = max(int(0.1 * n), seq_len * 4)
    train_corpus = encoded[:n - val_n].to(device)
    val_corpus   = encoded[n - val_n:].to(device)

    tok_ord = torch.zeros(vocab, dtype=torch.int64, device=device)
    for t, c in itos.items():
        tok_ord[t] = ord(c)

    return text_chars, itos, train_corpus, val_corpus, tok_ord


# ── Model ─────────────────────────────────────────────────────────────────────

class AddressNavigator(nn.Module):
    """One matrix that walks address space.

    Learns: fingerprint(current_context) → fingerprint(next_context).
    No vocabulary. No characters. No seq_len limit.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.W = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.W.weight)

    def forward(self, fp: torch.Tensor) -> torch.Tensor:
        """fp: [B, d_model] fingerprint → [B, d_model] predicted next fingerprint."""
        return self.W(fp)


# ── Batch sampling ────────────────────────────────────────────────────────────

def sample_fp_batch(
    corpus: torch.Tensor,    # [N] token ids on device
    tok_ord: torch.Tensor,   # [vocab] int64 ord(char) on device
    batch_size: int,
    seq_len: int,
    d_model: int,
    stride: int,
    gen: torch.Generator,
) -> tuple:
    """Sample random window pairs at given stride → (cur_fp, nxt_fp) [B, d_model].

    stride=13  (index stride): consecutive windows share 76/89 chars —
               high fingerprint correlation, strong learnable signal.
    stride=89  (non-overlapping): no shared chars — near-zero correlation,
               hard for a single linear map to learn.
    Training uses stride=13; generation hops through real index windows.
    """
    N   = corpus.shape[0] - seq_len - stride - 1
    idx = torch.randint(0, N, (batch_size,), generator=gen)
    j   = torch.arange(seq_len, device=corpus.device)

    cur_ids = corpus[(idx.unsqueeze(1) + j.unsqueeze(0))]                      # [B, T]
    nxt_ids = corpus[((idx + stride).unsqueeze(1) + j.unsqueeze(0))]           # [B, T]

    return (
        batch_substrate_fingerprint(cur_ids, tok_ord, d_model),
        batch_substrate_fingerprint(nxt_ids, tok_ord, d_model),
    )


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_corpus, tok_ord, batch_size, seq_len, d_model, stride,
             gen, n_batches=16):
    model.eval()
    losses = []
    dev = next(model.parameters()).device
    for _ in range(n_batches):
        cur_fp, nxt_fp = sample_fp_batch(
            val_corpus, tok_ord, batch_size, seq_len, d_model, stride, gen
        )
        pred = model(cur_fp.to(dev))
        loss = 1.0 - F.cosine_similarity(pred, nxt_fp.to(dev), dim=-1).mean()
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ── Generation: pure address navigation ───────────────────────────────────────

@torch.no_grad()
def generate_navigated(
    model,
    index: dict,
    seed_text: str,
    d_model: int,
    n_hops: int = 20,
    top_k: int = 8,
    device: str = 'cpu',
) -> str:
    """Navigate the corpus entirely in address space — no character sampling.

    Each hop:
      1. fingerprint(current context) = current address vector
      2. model(current_addr) = predicted next address
      3. retrieve top_k real windows from single- or dual-scale index;
         pick the least-recently-visited
      4. that window becomes the new context

    Dual-scale index (meta.dual=True): model's predicted fp searches BOTH
    T=13 and T=89 sub-indices → all 12 dodecahedral faces reachable.
    Single-scale index: falls back to retrieve().

    top_k + visited-ring-buffer prevents 2-cycle collapse.
    """
    is_dual = index.get('meta', {}).get('dual', False)

    model.eval()
    parts   = [seed_text]
    visited = []          # ring buffer of recently used window indices
    ctx_fp  = substrate_fingerprint(seed_text, d_model).to(device)

    for hop in range(n_hops):
        pred_fp = model(ctx_fp.unsqueeze(0)).squeeze(0).cpu()

        if is_dual:
            # Use predicted fp for BOTH sub-indices: searches all 12 faces
            candidates = retrieve_dual(pred_fp, pred_fp, index, top_k=top_k,
                                       exclude_idxs=visited)
        else:
            candidates = retrieve(pred_fp, index, top_k=top_k)

        # Pick first candidate not in visited ring
        ret = None
        for c in candidates:
            win_idx = c.get('idx', -1)
            if win_idx not in visited:
                ret = c
                visited.append(win_idx)
                if len(visited) > top_k * 2:
                    visited.pop(0)
                break
        if ret is None and candidates:
            ret = candidates[0]   # fallback: allow repeat

        ret_text = ret['text']
        ret_sim  = ret['sim']
        ret_addr = addr_str(ret['addr'])
        scale_tag = f"  scale={ret.get('scale','?')}" if is_dual else ""

        parts.append(ret_text)
        ctx_fp = substrate_fingerprint(ret_text, d_model).to(device)

        print(f"  hop {hop+1:2d}: addr={ret_addr}  sim={ret_sim:.3f}{scale_tag}  "
              f"{repr(ret_text[:60])}")

    return '\n'.join(parts)


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    device = args.device
    torch.manual_seed(args.seed)
    gen      = torch.Generator(); gen.manual_seed(args.seed + 1)
    eval_gen = torch.Generator(); eval_gen.manual_seed(args.seed + 2)

    # Corpus
    print(f"Loading corpus: {args.corpus} ...", flush=True)
    chars, itos, train_corpus, val_corpus, tok_ord = load_corpus(
        args.corpus, args.seq_len, device
    )
    vocab = len(chars)
    print(f"  vocab={vocab}  train={len(train_corpus):,}  val={len(val_corpus):,}")

    # Corpus address index for generation
    here       = Path(__file__).parent
    index_path = here / args.index
    if index_path.exists():
        print("Loading address index ...", flush=True)
        index = torch.load(str(index_path), map_location='cpu', weights_only=False)
        print(f"  {index['meta']['n_windows']:,} windows")
    else:
        index = None
        print("No index — generation will be skipped")

    # Model: one matrix
    model     = AddressNavigator(args.d_model).to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"\nAddressNavigator: d_model={args.d_model}  params={n_params:,}")

    optimizer  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    eval_every = max(args.steps // 20, 100)

    print(f"Training {args.steps} steps  batch={args.batch_size} ...\n", flush=True)
    t0   = time.time()
    best = float('inf')

    for step in range(args.steps):
        cur_fp, nxt_fp = sample_fp_batch(
            train_corpus, tok_ord, args.batch_size, args.seq_len, args.d_model,
            args.stride, gen
        )
        cur_fp = cur_fp.to(device)
        nxt_fp = nxt_fp.to(device)

        pred = model(cur_fp)
        loss = 1.0 - F.cosine_similarity(pred, nxt_fp, dim=-1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_every == 0 or step == args.steps - 1:
            val_loss = evaluate(
                model, val_corpus, tok_ord,
                args.batch_size, args.seq_len, args.d_model, args.stride, eval_gen
            )
            marker = ""
            if val_loss < best:
                best = val_loss
                marker = " ← BEST"
            print(f"  step {step:5d}  val={val_loss:.4f}  "
                  f"({time.time()-t0:.1f}s){marker}", flush=True)

    # Save
    corpus_tag = Path(args.corpus).stem if Path(args.corpus).exists() else args.corpus
    ckpt = here / f"pure_address_navigator_{corpus_tag}.pt"
    torch.save({
        'model_state': model.state_dict(),
        'd_model':     args.d_model,
        'seq_len':     args.seq_len,
        'vocab':       chars,
    }, str(ckpt))
    print(f"\nSaved → {ckpt}")

    # Generation demo
    if index is not None:
        print("\n" + "=" * 70)
        print("PURE ADDRESS NAVIGATION (no characters, no vocab):")
        print("=" * 70)
        seed = (args.seed_text if args.seed_text
                else "Before we proceed any further, hear me speak.")
        out = generate_navigated(
            model, index,
            seed_text=seed,
            d_model=args.d_model,
            n_hops=15,
            device=device,
        )
        print("\n── OUTPUT ──────────────────────────────────────────────────")
        print(out)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps',      type=int,   default=5000)
    parser.add_argument('--batch-size', type=int,   default=256)
    parser.add_argument('--seq-len',    type=int,   default=89)
    parser.add_argument('--d-model',    type=int,   default=64)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--stride',     type=int,   default=13,
                        help='Corpus stride between window pairs (13=index stride, 89=non-overlapping)')
    parser.add_argument('--corpus',     type=str,   default='tinyshakespeare',
                        help='Corpus source: "tinyshakespeare" or path to a .txt file')
    parser.add_argument('--index',      type=str,   default='corpus_address_index.pt',
                        help='Address index .pt file to use for generation')
    parser.add_argument('--seed-text',  type=str,   default='',
                        help='Seed text for generation demo')
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--device',     type=str,   default='cpu')
    args = parser.parse_args()

    train(args)

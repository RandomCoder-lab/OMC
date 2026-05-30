"""Train a MultiskillNavigator on any corpus.

Same cosine loss as train_pure_address.py but with N skill matrices.
After training: face→skill heatmap shows geometric specialisation.
Demo: UniversalAddressStore populated with text + tools + weights, navigated
      to show type-diverse retrieval from a single seed.

Usage:
    python3 train_multiskill.py
    python3 train_multiskill.py --corpus omc_corpus.txt --n-skills 4 --steps 5000
"""

import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from train_pure_address import load_corpus, sample_fp_batch, evaluate
from corpus_address_index import substrate_fingerprint
from hierarchical_address import assign_address, addr_str
from multiskill_navigator import MultiskillNavigator
from universal_store import UniversalAddressStore


# ── Face→Skill heatmap ────────────────────────────────────────────────────────

@torch.no_grad()
def face_skill_heatmap(
    model:      MultiskillNavigator,
    val_corpus: torch.Tensor,
    tok_ord:    torch.Tensor,
    seq_len:    int,
    d_model:    int,
    n_samples:  int = 2000,
) -> torch.Tensor:
    """Sample val windows → assign to (face, dominant_skill). Returns [12, n_skills]."""
    model.eval()
    from train_pure_address import sample_fp_batch
    gen = torch.Generator(); gen.manual_seed(99)
    counts = torch.zeros(12, model.n_skills, dtype=torch.long)
    done = 0
    while done < n_samples:
        b = min(256, n_samples - done)
        fp, _ = sample_fp_batch(val_corpus, tok_ord, b, seq_len, d_model, seq_len, gen)
        for i in range(fp.shape[0]):
            face  = assign_address(fp[i]).face
            skill = model.dominant_skill(fp[i])
            counts[face, skill] += 1
        done += b
    model.train()
    return counts


def print_heatmap(counts: torch.Tensor, model: MultiskillNavigator) -> None:
    n = model.n_skills
    header = "face | " + " ".join(f"sk{i:1d}" for i in range(n)) + "  | dominant"
    print(header)
    print("-" * len(header))
    for f in range(12):
        row   = counts[f]
        total = row.sum().item()
        if total == 0:
            print(f"  {f:2d} | " + " ".join("   0" for _ in range(n)) + "  | -")
            continue
        dom   = row.argmax().item()
        bars  = " ".join(f"{v:4d}" for v in row.tolist())
        print(f"  {f:2d} | {bars}  | sk{dom}")


# ── Demo navigation ───────────────────────────────────────────────────────────

@torch.no_grad()
def demo_navigation(
    model:   MultiskillNavigator,
    store:   UniversalAddressStore,
    seed:    str,
    d_model: int,
    n_hops:  int = 15,
) -> None:
    model.eval()
    ctx_fp   = substrate_fingerprint(seed, d_model)
    visited  = set()

    print(f"\nSeed: {repr(seed)}")
    print("-" * 72)

    for hop in range(n_hops):
        pred_fp  = model(ctx_fp.unsqueeze(0)).squeeze(0)
        results  = store.retrieve(pred_fp, top_k=8, exclude_eids=visited)
        if not results:
            print(f"  hop {hop+1:2d}: store exhausted"); break

        sim, entry = results[0]
        visited.add(entry.eid)

        skill = model.dominant_skill(ctx_fp)
        label = f"hop {hop+1:2d}  sk{skill}  {entry.type:6s}  sim={sim:.3f}  addr={addr_str(entry.addr)}"

        if entry.type == 'text':
            print(f"  {label}  {repr(entry.content[:55])}")
            ctx_fp = substrate_fingerprint(entry.content, d_model)
        elif entry.type == 'tool':
            fn = entry.content
            print(f"  {label}  fn={fn.__name__}")
            try:
                result = fn(entry.content.__doc__ or seed)
                ctx_fp = substrate_fingerprint(str(result)[:89], d_model)
            except Exception:
                ctx_fp = pred_fp
        elif entry.type == 'weight':
            name = entry.meta.get('name', '?')
            print(f"  {label}  weight={name}  |W|={entry.content.norm():.3f}")
            ctx_fp = entry.fp
        elif entry.type == 'memory':
            print(f"  {label}  {repr(entry.content[:55])}")
            ctx_fp = substrate_fingerprint(entry.content, d_model)
        else:
            print(f"  {label}  {repr(str(entry.content)[:55])}")
            ctx_fp = pred_fp

    # Show explicit type-filtered retrieval from final fp
    print("\n── Nearest by type from final fp ───────────────────────────────────")
    for t in ('text', 'tool', 'weight', 'memory'):
        results = store.retrieve(ctx_fp, top_k=1, type_filter=t)
        if results:
            sim, e = results[0]
            if t == 'text':
                tag = repr(e.content[:50])
            elif t == 'tool':
                tag = e.content.__name__
            elif t == 'weight':
                tag = f"weight={e.meta.get('name','?')}"
            else:
                tag = repr(e.content[:50])
            print(f"  {t:8s}: sim={sim:.3f}  {tag}")


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    device = args.device
    torch.manual_seed(args.seed)
    gen      = torch.Generator(); gen.manual_seed(args.seed + 1)
    eval_gen = torch.Generator(); eval_gen.manual_seed(args.seed + 2)

    print(f"Loading corpus: {args.corpus} ...", flush=True)
    chars, itos, train_corpus, val_corpus, tok_ord = load_corpus(
        args.corpus, args.seq_len, device
    )
    print(f"  vocab={len(chars)}  train={len(train_corpus):,}  val={len(val_corpus):,}")

    model = MultiskillNavigator(args.d_model, args.n_skills).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{model.skill_summary()}")
    print(f"  total params: {n_params:,}")

    optimizer = torch.optim.AdamW([
        {'params': model.W_bank,             'lr': args.lr},
        {'params': model.router.parameters(), 'lr': args.lr * 3},
    ], weight_decay=0.01)

    eval_every = max(args.steps // 20, 100)
    lb = args.lb_coeff  # load-balance coefficient
    print(f"\nTraining {args.steps} steps  batch={args.batch_size}  "
          f"stride={args.stride}  lb={lb}\n", flush=True)

    t0   = time.time()
    best = float('inf')

    for step in range(args.steps):
        cur_fp, nxt_fp = sample_fp_batch(
            train_corpus, tok_ord, args.batch_size,
            args.seq_len, args.d_model, args.stride, gen
        )
        cur_fp = cur_fp.to(device)
        nxt_fp = nxt_fp.to(device)

        pred = model(cur_fp)
        nav_loss = 1.0 - F.cosine_similarity(pred, nxt_fp, dim=-1).mean()

        # Load-balance regulariser: maximise routing entropy → equal skill usage
        w    = F.softmax(model.router(cur_fp), dim=-1)        # [B, n]
        avg  = w.mean(dim=0)                                   # [n] average usage
        lb_loss = (avg * (avg + 1e-8).log()).sum()             # negative entropy
        loss = nav_loss + lb * lb_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_every == 0 or step == args.steps - 1:
            val_loss = evaluate(
                model, val_corpus, tok_ord,
                args.batch_size, args.seq_len, args.d_model, args.stride, eval_gen
            )
            # Report dominant skill distribution on a val batch
            with torch.no_grad():
                fp_v, _ = sample_fp_batch(val_corpus, tok_ord, 512,
                                          args.seq_len, args.d_model, args.stride, eval_gen)
                usage = F.softmax(model.router(fp_v.to(device)), dim=-1).mean(0)
                usage_str = " ".join(f"{v:.2f}" for v in usage.tolist())
            marker = ""
            if val_loss < best:
                best = val_loss
                marker = " ← BEST"
            print(f"  step {step:5d}  val={val_loss:.4f}  skills=[{usage_str}]  "
                  f"({time.time()-t0:.1f}s){marker}", flush=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    here       = Path(__file__).parent
    corpus_tag = Path(args.corpus).stem if Path(args.corpus).exists() else args.corpus
    ckpt       = here / f"multiskill_navigator_{corpus_tag}.pt"
    torch.save({
        'model_state': model.state_dict(),
        'd_model':     args.d_model,
        'n_skills':    args.n_skills,
        'seq_len':     args.seq_len,
    }, str(ckpt))
    print(f"\nSaved → {ckpt}")

    # ── Face→Skill heatmap ────────────────────────────────────────────────────
    print("\n── Face → Skill Heatmap ─────────────────────────────────────────────")
    heatmap = face_skill_heatmap(model, val_corpus, tok_ord, args.seq_len, args.d_model)
    print_heatmap(heatmap, model)

    # ── Demo: UniversalAddressStore ───────────────────────────────────────────
    print("\n── Building UniversalAddressStore ───────────────────────────────────")
    store = UniversalAddressStore(d_model=args.d_model)

    # Text: sparse corpus windows
    here        = Path(__file__).parent
    corpus_path = here / args.corpus if not Path(args.corpus).exists() else Path(args.corpus)
    if not corpus_path.exists():
        corpus_path = here / 'tinyshakespeare.txt'
    corpus_text = corpus_path.read_text(errors='replace') if corpus_path.exists() else None
    if corpus_text:
        n = store.store_corpus(corpus_text, seq_len=args.seq_len, stride=args.seq_len)
        print(f"  text: {n:,} windows")

    # Tools: simple callables with docstrings
    def tool_reverse(text):
        """Reverse the characters in a string."""
        return text[::-1]

    def tool_uppercase(text):
        """Convert text to uppercase letters."""
        return text.upper()

    def tool_wordcount(text):
        """Count the number of words in a text string."""
        return str(len(text.split()))

    def tool_first_sentence(text):
        """Extract the first sentence of a text passage."""
        return text.split('.')[0]

    def tool_ord_sum(text):
        """Sum the ASCII ordinal values of all characters in the text."""
        return str(sum(ord(c) for c in text))

    for fn in (tool_reverse, tool_uppercase, tool_wordcount,
               tool_first_sentence, tool_ord_sum):
        store.store(fn, type='tool')

    # Skill weight matrices as addressable entries
    for i in range(model.n_skills):
        W = model.W_bank[i].detach().cpu()
        store.store(W, type='weight', name=f'skill_{i}')

    # Memory: a few representative sentences
    for mem in [
        "The corpus is the memory; the model navigates it.",
        "Parameters are addresses. Tools are addresses. Everything is addresses.",
        "Navigation across types — text, tool, weight — is now uniform.",
    ]:
        store.store(mem, type='memory')

    print(store.stats())

    # Navigate
    print("\n── Navigation Demo ─────────────────────────────────────────────────")
    seed = args.seed_text or "To be, or not to be, that is the question"
    demo_navigation(model, store, seed=seed, d_model=args.d_model, n_hops=args.n_hops)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps',     type=int,   default=5000)
    parser.add_argument('--batch-size', type=int,  default=256)
    parser.add_argument('--seq-len',   type=int,   default=89)
    parser.add_argument('--d-model',   type=int,   default=64)
    parser.add_argument('--n-skills',  type=int,   default=4)
    parser.add_argument('--lr',        type=float, default=1e-3)
    parser.add_argument('--stride',    type=int,   default=13)
    parser.add_argument('--corpus',    type=str,   default='tinyshakespeare')
    parser.add_argument('--seed-text', type=str,   default='')
    parser.add_argument('--n-hops',    type=int,   default=15)
    parser.add_argument('--lb-coeff',  type=float, default=0.05,
                        help='Load-balance regulariser coefficient (higher → more uniform skill usage)')
    parser.add_argument('--seed',      type=int,   default=42)
    parser.add_argument('--device',    type=str,   default='cpu')
    args = parser.parse_args()
    train(args)

"""New Shakespeare via address-space interpolation.

Two Shakespeare passages are run through the model simultaneously.
At each generation step, their logit distributions are blended:

    logits_mix = (1 - alpha) * logits_A + alpha * logits_B

Sampling from logits_mix produces text that is statistically "between"
the two source passages — in Shakespeare's address space but not in the
corpus. This is new Shakespeare.

Alpha controls the blend:
    0.0 → pure passage A
    0.5 → equal blend (maximally novel)
    1.0 → pure passage B

The address angle: each generated window gets an HAddr. If that address
exists in the corpus, the generated text is near real Shakespeare. If it
falls on an empty face (2, 5, 9, 10), it is in unexplored territory —
substrate regions Shakespeare never wrote from.
"""

import math
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from corpus import make_dataset
from train_substrate_attention import FibRecLMSubsim
from hierarchical_address import assign_address, addr_str
from corpus_address_index import substrate_fingerprint


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str = 'cpu') -> tuple:
    ck   = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab   = ck['vocab']
    seq_len = ck['seq_len']
    d_model = ck['d_model']
    n_blocks = sum(1 for k in ck['model_state'] if k.startswith('ln1s.'))

    stoi = {c: i for i, c in enumerate(vocab)}
    itos = {i: c for i, c in enumerate(vocab)}

    # Use actual pe shape from state dict (overrides stale metadata)
    state = ck['model_state']
    if 'pe' in state:
        seq_len = state['pe'].shape[0]

    has_substrate_embed = 'embed.gamma' in ck['model_state']
    model = FibRecLMSubsim(
        vocab_size=len(vocab),
        d_model=d_model,
        n_blocks=n_blocks,
        seq_len=seq_len,
        K=89,
        mode='cross',
        K_sig=32,
        substrate_embed=has_substrate_embed,
    )
    missing, unexpected = model.load_state_dict(ck['model_state'], strict=False)
    if missing:
        print(f"[model] {len(missing)} missing keys: {missing[:5]}")
    model.to(device).eval()
    return model, vocab, stoi, itos, seq_len, d_model


# ── Encoding helpers ──────────────────────────────────────────────────────────

def encode(text: str, stoi: dict, seq_len: int) -> torch.Tensor:
    """Encode text → token ids, pad/truncate to seq_len."""
    ids = [stoi.get(c, 0) for c in text[-seq_len:]]
    if len(ids) < seq_len:
        ids = [0] * (seq_len - len(ids)) + ids
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)   # [1, seq_len]


def decode(ids: List[int], itos: dict) -> str:
    return ''.join(itos.get(i, '?') for i in ids)


# ── Logit interpolation generation ───────────────────────────────────────────

@torch.no_grad()
def generate_interpolated(
    model,
    stoi: dict,
    itos: dict,
    seq_len: int,
    seed_A: str,
    seed_B: str,
    n_chars: int = 300,
    alpha: float = 0.5,
    temperature: float = 0.85,
    device: str = 'cpu',
) -> str:
    """Generate new Shakespeare by blending logits from two source passages.

    At each step:
      1. Run model on context_A → logits_A
      2. Run model on context_B → logits_B
      3. logits_mix = (1-α)×logits_A + α×logits_B
      4. Sample next char from softmax(logits_mix / T)
      5. Append to both contexts
    """
    ctx_A = list(seed_A[-seq_len:].ljust(seq_len)[:seq_len])
    ctx_B = list(seed_B[-seq_len:].ljust(seq_len)[:seq_len])
    generated: List[int] = []

    for _ in range(n_chars):
        ids_A = encode(''.join(ctx_A), stoi, seq_len).to(device)
        ids_B = encode(''.join(ctx_B), stoi, seq_len).to(device)

        logits_A = model(ids_A)[0, -1, :]     # [vocab]
        logits_B = model(ids_B)[0, -1, :]     # [vocab]

        logits_mix = (1.0 - alpha) * logits_A + alpha * logits_B
        probs = F.softmax(logits_mix / temperature, dim=-1)
        tok   = torch.multinomial(probs, 1).item()

        generated.append(tok)
        c = itos.get(tok, '?')
        ctx_A = ctx_A[1:] + [c]
        ctx_B = ctx_B[1:] + [c]

    return decode(generated, itos)


# ── Address analysis of generated text ───────────────────────────────────────

def analyse_addresses(text: str, d_model: int = 64, window: int = 89) -> None:
    """Show which addresses the generated text falls into."""
    EMPTY_FACES = {2, 5, 9, 10}   # faces with no corpus windows
    print(f"\n── Address map of generated text ({'window'}={window} chars) ──")
    for i in range(0, len(text) - window + 1, window):
        chunk = text[i:i + window]
        vec   = substrate_fingerprint(chunk, d_model)
        addr  = assign_address(vec)
        flag  = " ← UNEXPLORED FACE" if addr.face in EMPTY_FACES else ""
        print(f"  chars {i:3d}-{i+window:3d}: addr={addr_str(addr)}{flag}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='bloom_checkpoint_model.pt')
    parser.add_argument('--alpha',       type=float, default=0.5,
                        help='Blend weight (0=pure A, 1=pure B, 0.5=equal mix)')
    parser.add_argument('--temperature', type=float, default=0.85)
    parser.add_argument('--n-chars',     type=int,   default=400)
    parser.add_argument('--seed-a',      type=str,
                        default='Before we proceed any further, hear me speak.')
    parser.add_argument('--seed-b',      type=str,
                        default='Two households, both alike in dignity,')
    parser.add_argument('--device',      type=str,   default='cpu')
    args = parser.parse_args()

    here = Path(__file__).parent

    print("Loading model ...", flush=True)
    model, vocab, stoi, itos, seq_len, d_model = load_model(
        str(here / args.checkpoint), args.device
    )
    print(f"  vocab={len(vocab)}  seq_len={seq_len}  d_model={d_model}")

    print(f"\nSeed A: {repr(args.seed_a)}")
    print(f"Seed B: {repr(args.seed_b)}")
    print(f"Alpha:  {args.alpha}  (0=A only, 0.5=equal, 1=B only)")
    print(f"Temp:   {args.temperature}")

    # ── Pure A (baseline) ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Pure A (alpha=0.0):")
    print("=" * 70)
    out_a = generate_interpolated(model, stoi, itos, seq_len,
                                   args.seed_a, args.seed_b,
                                   n_chars=args.n_chars, alpha=0.0,
                                   temperature=args.temperature,
                                   device=args.device)
    print(out_a)

    # ── Pure B (baseline) ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Pure B (alpha=1.0):")
    print("=" * 70)
    out_b = generate_interpolated(model, stoi, itos, seq_len,
                                   args.seed_a, args.seed_b,
                                   n_chars=args.n_chars, alpha=1.0,
                                   temperature=args.temperature,
                                   device=args.device)
    print(out_b)

    # ── Interpolated (novel) ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"INTERPOLATED alpha={args.alpha} — NEW SHAKESPEARE:")
    print("=" * 70)
    out_mix = generate_interpolated(model, stoi, itos, seq_len,
                                     args.seed_a, args.seed_b,
                                     n_chars=args.n_chars, alpha=args.alpha,
                                     temperature=args.temperature,
                                     device=args.device)
    print(out_mix)

    # ── Show address map for the novel output ─────────────────────────────────
    analyse_addresses(out_mix, d_model=d_model, window=89)

    # ── Sweep alpha ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Alpha sweep — watching the style shift from A to B:")
    print("=" * 70)
    for sweep_alpha in [0.2, 0.4, 0.6, 0.8]:
        out = generate_interpolated(model, stoi, itos, seq_len,
                                     args.seed_a, args.seed_b,
                                     n_chars=120, alpha=sweep_alpha,
                                     temperature=args.temperature,
                                     device=args.device)
        print(f"\n  α={sweep_alpha}:")
        print(f"  {repr(out)}")

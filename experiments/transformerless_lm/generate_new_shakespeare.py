"""New Shakespeare via trigram-substrate generation.

Build a trigram transition model from the full corpus — pure statistical
substrate, no word list — then generate text that sounds like Shakespeare
but has never existed.

Each generated window gets an HAddr. Windows that land at addresses with
no verbatim corpus passage are genuinely new: same substrate statistics,
novel address territory.

The generative model is entirely substrate-math:
  - Trigram transitions: P(c | prev2) learned from Shakespeare corpus
  - Temperature: controls how "orthodox" vs "creative" the generation is
  - Seed: taken from real Shakespeare windows as style anchors
"""

import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from hierarchical_address import assign_address, addr_str, addr_distance
from corpus_address_index import substrate_fingerprint


# ── Trigram model ─────────────────────────────────────────────────────────────

def build_trigram_model(windows: List[str]) -> Dict[str, Dict[str, int]]:
    """Build trigram transition counts: context(2 chars) → next char counts."""
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for w in windows:
        for i in range(len(w) - 2):
            ctx  = w[i:i+2]
            next_c = w[i+2]
            counts[ctx][next_c] += 1
    return counts


def trigram_probs(model: Dict, ctx: str, temperature: float = 1.0) -> Dict[str, float]:
    """Softmax-temperature distribution over next chars given 2-char context."""
    dist = model.get(ctx, {})
    if not dist:
        dist = model.get(ctx[1:] + ' ', {})  # fallback: unigram-ish
    if not dist:
        return {}
    chars  = list(dist.keys())
    counts = [float(dist[c]) for c in chars]
    # Log → scale by temperature → softmax
    log_c  = [math.log(c + 1e-9) / temperature for c in counts]
    max_lc = max(log_c)
    exps   = [math.exp(l - max_lc) for l in log_c]
    total  = sum(exps)
    return {c: e / total for c, e in zip(chars, exps)}


def generate_trigram(
    model: Dict,
    seed: str,
    n_chars: int = 89,
    temperature: float = 0.9,
    rng: random.Random = None,
) -> str:
    """Generate n_chars of Shakespeare-textured text from a seed."""
    if rng is None:
        rng = random.Random()
    buf  = list(seed[-2:]) if len(seed) >= 2 else list('  ')
    out  = list(seed)
    for _ in range(n_chars):
        ctx   = ''.join(buf[-2:])
        probs = trigram_probs(model, ctx, temperature)
        if not probs:
            buf.append(' ')
            out.append(' ')
            continue
        chars   = list(probs.keys())
        weights = list(probs.values())
        next_c  = rng.choices(chars, weights=weights, k=1)[0]
        buf.append(next_c)
        out.append(next_c)
    return ''.join(out)


# ── Address novelty ───────────────────────────────────────────────────────────

def build_corpus_addr_set(index: Dict) -> set:
    """All HAddr tuples that exist in the verbatim corpus."""
    known = set()
    for addr in index['addrs']:
        known.add((addr.face, addr.sub_face, frozenset(addr.zeck)))
    return known


# ── Main demo ─────────────────────────────────────────────────────────────────

def run(
    index: Dict,
    n_samples: int = 60,
    n_chars: int = 200,
    temperature: float = 0.92,
    seed_int: int = 1623,   # Shakespeare's First Folio year
    d_model: int = 64,
):
    rng = random.Random(seed_int)
    windows = index['windows']
    faces   = index['face_index']
    OCCUPIED = [f for f in range(12) if faces.get(f)]

    print(f"Building trigram model from {len(windows):,} Shakespeare windows...",
          flush=True)
    tgm = build_trigram_model(windows)
    print(f"  {len(tgm):,} unique 2-char contexts learned")

    known_addrs = build_corpus_addr_set(index)
    print(f"  {len(known_addrs):,} verbatim corpus addresses")

    # Generate samples from diverse seeds (one from each occupied face)
    seeds = []
    for f in OCCUPIED:
        wi = rng.choice(faces[f])
        seeds.append(windows[wi][:30])

    print(f"\nGenerating {n_samples} samples at T={temperature} ...\n")

    novel   = []   # new address territory
    known_g = []   # known address but generated (not verbatim)

    for i in range(n_samples):
        seed   = seeds[i % len(seeds)]
        text   = generate_trigram(tgm, seed, n_chars=n_chars,
                                  temperature=temperature, rng=rng)
        vec    = substrate_fingerprint(text[:89], d_model)
        addr   = assign_address(vec)
        key    = (addr.face, addr.sub_face, frozenset(addr.zeck))
        is_new = key not in known_addrs

        entry = {'text': text, 'addr': addr, 'addr_str': addr_str(addr),
                 'seed': seed, 'novel': is_new}
        (novel if is_new else known_g).append(entry)

    print(f"Results: {len(novel)} novel-address samples  |  "
          f"{len(known_g)} known-address samples")
    print(f"Novel rate: {100*len(novel)/n_samples:.1f}%")

    # ── Show best novel samples ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("NEW SHAKESPEARE — text at addresses never occupied by the corpus:")
    print("=" * 70)

    EMPTY_FACES = {2, 5, 9, 10}
    for j, entry in enumerate(novel[:5]):
        flag = "  ← EMPTY FACE TERRITORY" if entry['addr'].face in EMPTY_FACES else ""
        print(f"\n── Sample {j+1}  addr={entry['addr_str']}{flag}")
        print(f"   seed: {repr(entry['seed'])}")
        print()
        print(entry['text'])

    # ── Show face distribution of generated text ──────────────────────────────
    print("\n" + "=" * 70)
    print("Face distribution of generated text vs corpus:")
    print("=" * 70)
    gen_faces = defaultdict(int)
    for e in novel + known_g:
        gen_faces[e['addr'].face] += 1
    for f in range(12):
        corp = len(faces.get(f, []))
        gen  = gen_faces.get(f, 0)
        empty_flag = " ← EMPTY (corpus)" if corp == 0 else ""
        new_flag   = " ← GENERATED ONLY" if corp == 0 and gen > 0 else ""
        print(f"  face {f:2d}: corpus={corp:6,}  generated={gen:3d}{empty_flag}{new_flag}")

    return novel, known_g


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-samples',   type=int,   default=60)
    parser.add_argument('--n-chars',     type=int,   default=250)
    parser.add_argument('--temperature', type=float, default=0.92)
    parser.add_argument('--seed',        type=int,   default=1623)
    parser.add_argument('--index',       type=str,
                        default='corpus_address_index.pt')
    args = parser.parse_args()

    here  = Path(__file__).parent
    index = torch.load(str(here / args.index), map_location='cpu',
                       weights_only=False)
    print(f"Loaded index: {index['meta']['n_windows']:,} windows\n")

    run(index, n_samples=args.n_samples, n_chars=args.n_chars,
        temperature=args.temperature, seed_int=args.seed)

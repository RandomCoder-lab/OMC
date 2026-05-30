"""PHASE 4 — DEVELOP ADDRESSING: hierarchical multi-scale chunk index.

The user's chunk-grouping idea (char→word→sentence→paragraph→corpus) applied to the
PROVEN-REAL retrieval side. Builds a φ-address index at multiple Fibonacci window
scales and measures:
  (1) per-scale + combined dodecahedral face coverage (does adding scales tile all 12?)
  (2) retrieval LOCALIZATION per scale — does nearest-by-face land sequentially near the
      query (finer scale = tighter localization)? This is the hierarchical-addressing payoff.

Uses the proven substrate_fingerprint + assign_address (canonical normals, post-fix).
Universal: pure φ math, no word lists.
"""
import statistics, time
from pathlib import Path
import torch
from corpus_address_index import substrate_fingerprint
from hierarchical_address import assign_address

HERE = Path(__file__).parent
SCALES = [13, 34, 89, 233]     # ≈ subword / word / sentence / paragraph (Fibonacci)
N_WINDOWS = 1500               # windows per scale (sampled)
import torch.nn.functional as F

def build_scale(corpus_ids, tok_ord, scale, stride, dmodel=64):
    """Fingerprint windows at one scale; return (vecs[N,3unit], faces[N], positions[N])."""
    from corpus_address_index import batch_substrate_fingerprint, build_fingerprint_cache
    positions = list(range(0, len(corpus_ids)-scale, stride))[:N_WINDOWS]
    wins = torch.stack([corpus_ids[p:p+scale] for p in positions])
    vecs = batch_substrate_fingerprint(wins, tok_ord, dmodel)   # [N,64]
    faces = [assign_address(v).face for v in vecs]
    return vecs, faces, positions

if __name__ == '__main__':
    t0 = time.time()
    text = (HERE/'omc_corpus.txt').read_text(errors='replace')[:4_000_000]
    chars = sorted(set(text)); stoi={c:i for i,c in enumerate(chars)}
    corpus_ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    tok_ord = torch.tensor([ord(c) for c in chars], dtype=torch.int64)
    print(f"[hier] corpus slice={len(text):,} scales={SCALES}", flush=True)

    all_faces = set()
    scale_data = {}
    for sc in SCALES:
        vecs, faces, pos = build_scale(corpus_ids, tok_ord, sc, stride=sc)
        occ = sorted(set(faces))
        all_faces |= set(occ)
        scale_data[sc] = (vecs, faces, pos)
        print(f"[hier] scale={sc:3d}  faces_occupied={occ}  ({len(occ)}/12)", flush=True)
    print(f"[hier] COMBINED coverage: {sorted(all_faces)}  ({len(all_faces)}/12)", flush=True)

    # Retrieval localization per scale: for sampled queries, nearest-by-cosine WITHIN the
    # query's face; measure |Δposition| (sequential distance). Tighter = better localization.
    print("[hier] retrieval localization (mean |Δpos| of nearest-in-face; lower=tighter):", flush=True)
    for sc in SCALES:
        vecs, faces, pos = scale_data[sc]
        vn = F.normalize(vecs.float(), dim=1)
        faces_t = torch.tensor(faces); pos_t = torch.tensor(pos, dtype=torch.float)
        dists = []
        import random; random.seed(0)
        qidx = random.sample(range(len(pos)), min(120, len(pos)))
        for qi in qidx:
            same = (faces_t == faces_t[qi]).nonzero().squeeze(1)
            same = same[same != qi]
            if len(same) < 1: continue
            sims = vn[same] @ vn[qi]
            nn = same[int(sims.argmax())]
            dists.append(abs(pos_t[nn].item() - pos_t[qi].item()))
        md = statistics.median(dists) if dists else float('nan')
        print(f"[hier] scale={sc:3d}  median|Δpos|={md:,.0f}  (n_faces={len(set(faces))})", flush=True)
    print(f"[hier] HIER-INDEX DONE ({time.time()-t0:.0f}s)", flush=True)

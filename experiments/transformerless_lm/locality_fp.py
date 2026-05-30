"""ADDR-1 — Locality-bearing fingerprint + navigation/retrieval layer.

φ-hash fingerprints are equidistributed (great for uniform keys/buckets, χ²=17) but have
NO content-locality — overlapping windows are near-orthogonal, so φ-cosine retrieval and
coarse-to-fine navigation fail (proven: navigation 0.29). A locality fingerprint (char /
char-bigram histogram) makes similar content similar → navigation 0.88, retrieval works.

Design principle: TWO fingerprints, two jobs.
  φ-fp        → addressing / uniform buckets / exact keys  (corpus_address_index)
  locality-fp → similarity retrieval / coarse-to-fine navigation  (this module)

Universal: pure char-count statistics, no word lists.
"""
from __future__ import annotations
import torch, torch.nn.functional as F
from typing import List, Dict, Optional

def build_vocab(text: str):
    chars = sorted(set(text))
    return {c: i for i, c in enumerate(chars)}, len(chars)

def hist_fp(ids: torch.Tensor, start: int, L: int, V: int, bigram: bool = False) -> torch.Tensor:
    """Char-count (or char-bigram) histogram over ids[start:start+L]. Locality-bearing:
    overlapping windows share most counts → high cosine. Normalized."""
    seg = ids[start:start+L]
    if not bigram:
        h = torch.zeros(V)
        h.scatter_add_(0, seg, torch.ones(len(seg)))
    else:
        # bigram index = a*V + b, hashed to a fixed dim to bound size
        DIM = min(V * V, 4096)
        h = torch.zeros(DIM)
        if len(seg) > 1:
            bg = (seg[:-1] * V + seg[1:]) % DIM
            h.scatter_add_(0, bg, torch.ones(len(bg)))
    n = h.norm()
    return h / n if n > 0 else h

def fp_text(text: str, stoi: Dict[str, int], V: int, bigram: bool = False) -> torch.Tensor:
    ids = torch.tensor([stoi.get(c, 0) for c in text], dtype=torch.long)
    return hist_fp(ids, 0, len(ids), V, bigram=bigram)

# ── navigation over a nested para→sent→word tree (locality fp + beam) ──────────
def build_tree(ids, V, PARA=233, SENT=89, WORD=34, max_paras=400, bigram=False):
    tree = []
    for p in range(0, min(len(ids)-PARA, max_paras*PARA), PARA):
        node = {'pos': p, 'fp': hist_fp(ids, p, PARA, V, bigram), 'sents': []}
        for s in range(p, p+PARA-SENT, SENT):
            node['sents'].append({'pos': s, 'fp': hist_fp(ids, s, SENT, V, bigram),
                'words': [{'pos': w, 'fp': hist_fp(ids, w, WORD, V, bigram)}
                          for w in range(s, s+SENT-WORD, WORD)]})
        tree.append(node)
    return tree

def _topb(qfp, nodes, b):
    if not nodes: return []
    M = F.normalize(torch.stack([n['fp'] for n in nodes]).float(), dim=1)
    sims = (M @ F.normalize(qfp.float().unsqueeze(0), dim=1).T).squeeze(1)
    return [nodes[i] for i in sims.argsort(descending=True)[:b].tolist()]

def navigate(ids, V, qpos, tree, b=8, PARA=233, SENT=89, WORD=34, bigram=False):
    """Coarse→fine beam descent with scale-matched locality fps. Returns final word pos."""
    paras = _topb(hist_fp(ids, qpos, PARA, V, bigram), tree, b)
    spool = [s for p in paras for s in p['sents']]
    sents = _topb(hist_fp(ids, qpos, SENT, V, bigram), spool, b)
    wpool = [w for s in sents for w in s['words']]
    words = _topb(hist_fp(ids, qpos, WORD, V, bigram), wpool, 1)
    return words[0]['pos'] if words else -1

# ── flat similarity retrieval (for relevance comparison) ──────────────────────
def retrieve(qfp, windows_fp: torch.Tensor, top_k=5):
    """Return indices of top_k windows most similar to qfp (cosine)."""
    sims = (F.normalize(windows_fp.float(), dim=1) @ F.normalize(qfp.float().unsqueeze(0), dim=1).T).squeeze(1)
    return sims.argsort(descending=True)[:top_k].tolist()

if __name__ == '__main__':
    from pathlib import Path
    text = (Path(__file__).parent/'omc_corpus.txt').read_text(errors='replace')[:500_000]
    stoi, V = build_vocab(text)
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    tree = build_tree(ids, V, max_paras=200)
    import random; random.seed(0)
    hit = 0
    for _ in range(40):
        nd = random.choice(tree); s = random.choice(nd['sents']); wd = random.choice(s['words'])
        if navigate(ids, V, wd['pos'], tree, b=8) == wd['pos']: hit += 1
    print(f"[locality_fp] self-test: nav exact-recovery beam8 = {hit}/40 = {hit/40:.2f} (V={V})")

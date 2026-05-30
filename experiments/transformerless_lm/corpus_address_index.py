"""Corpus Address Index — substrate-native retrieval from any corpus.

Each seq_len-char window of the corpus gets a Fibonacci-harmonic fingerprint
vector → HAddr (dodecahedral face + sub_face + Zeckendorf magnitude).

Retrieval is O(1) by face, then cosine-ranked within the face.
The model's job shrinks to: current context → fingerprint → address → retrieve.
No generation needed. The corpus IS the memory.

Index format (corpus_address_index.pt):
    windows:    List[str]              — raw text windows
    addrs:      List[HAddr]            — HAddr per window
    vecs:       Tensor[N, d_model]     — fingerprint vecs (for cosine ranking)
    face_index: Dict[int, List[int]]   — face → window indices (O(1))
    seq_index:  List[int]              — window_idx → corpus position
    meta:       dict                   — seq_len, stride, n_windows, corpus_len

The fingerprint is purely substrate-math: Fibonacci-harmonic decomposition of
character ordinals. Universal — works on any corpus without any word list.
"""

import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from hierarchical_address import HAddr, assign_address, addr_str, addr_distance
from addressed_memory import _NEIGHBORS, _NORMALS_STACK
from corpus import make_dataset


# ── φ-hash fingerprint ────────────────────────────────────────────────────────
# floor(φ × 2^32) — golden-ratio multiplicative hash constant.
# φ-based LCG is provably equidistributed (low-discrepancy sequence),
# so vec[:3] will spread uniformly across all 12 dodecahedral faces.
_PHI_INT = 2654435769

def substrate_fingerprint(text: str, d_model: int = 64) -> torch.Tensor:
    """φ-hash fingerprint: golden-ratio LCG over character ordinals.

    Each component i uses seed (i+1)×φ_int before mixing ordinals, so
    different components are independent → vec[:3] covers all 12 faces.
    Universal: no model, no word list, identical math on any corpus.
    """
    if not text:
        return torch.zeros(d_model)
    ordinals = [ord(c) for c in text]
    vec = torch.zeros(d_model)
    for i in range(d_model // 2):
        h = (i + 1) * _PHI_INT & 0xFFFFFFFF
        for o in ordinals:
            h = (h * _PHI_INT + o) & 0xFFFFFFFF
        angle = 2.0 * math.pi * h / 0x100000000
        vec[2 * i]     = math.sin(angle)
        vec[2 * i + 1] = math.cos(angle)
    return vec


# ── Vectorized batch fingerprint ─────────────────────────────────────────────
# Linearised LCG:  h_T = PHI_INT^T * h_0  +  Σ_j PHI_INT^(T-1-j) * o_j  (mod 2^32)
# The ordinal sum is the same for ALL components i — compute once per batch
# item as a dot product, then broadcast the per-component offset.
_FP_CACHE: dict = {}   # (T, d_model) → (reverse_powers [T], offsets [d//2])

def build_fingerprint_cache(seq_len: int, d_model: int) -> tuple:
    """Precompute the constant tensors for batch_substrate_fingerprint."""
    MOD = 0x100000000
    T   = seq_len
    # reverse_powers[j] = PHI_INT^(T-1-j) mod MOD  (weight for ordinal at pos j)
    pows = [0] * T
    pows[T - 1] = 1
    for j in range(T - 2, -1, -1):
        pows[j] = (pows[j + 1] * _PHI_INT) % MOD
    reverse_powers = torch.tensor(pows, dtype=torch.int64)   # [T]
    # offset_i = PHI_INT^T * h_0_i mod MOD, where h_0_i = (i+1)*PHI_INT mod MOD
    phi_T = pow(_PHI_INT, T, MOD)
    offsets = [
        (phi_T * (((i + 1) * _PHI_INT) % MOD)) % MOD
        for i in range(d_model // 2)
    ]
    offsets_t = torch.tensor(offsets, dtype=torch.int64)     # [d//2]
    return reverse_powers, offsets_t


def batch_substrate_fingerprint(
    ids_batch: torch.Tensor,          # [B, T] int64 token ids
    tok_ord:   torch.Tensor,          # [vocab] int64  ord(char) per token
    d_model:   int = 64,
) -> torch.Tensor:
    """φ-hash fingerprint for a batch of token-id windows. [B, d_model] float32.

    Mathematically identical to calling substrate_fingerprint(text, d_model)
    for each item. ~100× faster via vectorised LCG linearisation.

    ids_batch : [B, T] — already-gathered token-id windows (not positions).
    tok_ord   : [vocab] — precomputed ord(char) for each token id.
    """
    MOD = 0x100000000
    B, T = ids_batch.shape
    key = (T, d_model)
    if key not in _FP_CACHE:
        _FP_CACHE[key] = build_fingerprint_cache(T, d_model)
    rev_pow, offsets = _FP_CACHE[key]
    dev = ids_batch.device
    rev_pow = rev_pow.to(dev)
    offsets = offsets.to(dev)

    # Ordinals [B, T] int64
    ords = tok_ord[ids_batch]                                       # [B, T]

    # Ordinal-sum per batch item: Σ_j PHI_INT^(T-1-j) * o_j  →  [B] int64
    # Products stay < 127 × (2^32-1) ≈ 5.5e11; sum of T terms < 5e13 < int64.
    ordinal_sum = (ords * rev_pow.unsqueeze(0)).sum(dim=1)          # [B]

    # Add per-component offset, take mod: [B, 1] + [1, d//2] → [B, d//2]
    h_T = (ordinal_sum.unsqueeze(1) + offsets.unsqueeze(0)) % MOD  # [B, d//2]

    # Angles → sin/cos interleaved: vec[b, 2i]=sin, vec[b, 2i+1]=cos
    angles = (2.0 * math.pi / MOD) * h_T.float()                   # [B, d//2]
    vec = torch.stack([angles.sin(), angles.cos()], dim=-1)         # [B, d//2, 2]
    return vec.reshape(B, d_model)                                  # [B, d]


# ── Index build ───────────────────────────────────────────────────────────────

def build_corpus_index(
    corpus_text: str,
    seq_len: int = 89,
    d_model: int = 64,
    stride: Optional[int] = None,
    batch_size: int = 4096,
) -> Dict:
    """Build the full address index for corpus_text.

    stride=None uses stride=13 (Fibonacci-aligned, ~77K windows for 1M corpus).
    Uses vectorised batch_substrate_fingerprint — fast on large corpora.
    """
    if stride is None:
        stride = 13

    total     = len(corpus_text)
    positions = list(range(0, total - seq_len + 1, stride))
    n_windows = len(positions)
    print(f"[CorpusIndex] corpus={total:,} chars  seq_len={seq_len}  "
          f"stride={stride}  windows={n_windows:,}", flush=True)

    # Build char vocab and tok_ord for the vectorised fingerprint
    chars   = sorted(set(corpus_text))
    stoi    = {c: i for i, c in enumerate(chars)}
    tok_ord = torch.tensor([ord(c) for c in chars], dtype=torch.int64)

    # Encode full corpus as token ids — use numpy for speed on large corpora
    import numpy as np
    print(f"  Encoding corpus ({total:,} chars) ...", flush=True)
    # Map via lookup table: build a 0x110000 array keyed by code point
    _max_cp = max(ord(c) for c in chars)
    _lut    = np.full(_max_cp + 1, 0, dtype=np.int32)
    for c, i in stoi.items():
        _lut[ord(c)] = i
    # Encode in chunks to avoid one giant allocation
    CHUNK = 4_000_000
    parts = []
    for off in range(0, total, CHUNK):
        chunk = corpus_text[off:off + CHUNK]
        arr   = np.frombuffer(chunk.encode('utf-32-le'), dtype=np.uint32)
        arr   = np.where(arr <= _max_cp, _lut[np.minimum(arr, _max_cp)], 0)
        parts.append(arr.astype(np.int64))
    corpus_ids = torch.from_numpy(np.concatenate(parts))
    print(f"  Corpus encoded: {len(corpus_ids):,} tokens", flush=True)

    windows:    List[str]          = []
    addrs:      List[HAddr]        = []
    seq_index_list: List[int]      = []
    face_index: Dict[int, List[int]] = defaultdict(list)
    vecs_chunks: List[torch.Tensor] = []  # accumulate per batch, then cat once

    j_idx = torch.arange(seq_len, dtype=torch.long)

    # Pre-stack neighbor normals for vectorised sub_face assignment
    _NORMALS = _NORMALS_STACK  # [12, 3]

    # Precompute neighbor index tensor for all 12 faces: [12, 3]
    _NBRS_T = torch.tensor(_NEIGHBORS, dtype=torch.long)  # [12, 3]

    def _batch_assign_addresses(vecs_batch: torch.Tensor):
        """Vectorised face + sub_face + zeckendorf assignment for a batch.

        Returns (face_ids [B], sub_face_ids [B], zeck_list [B]).
        """
        import torch.nn.functional as _F
        v3 = vecs_batch[:, :3].float()                        # [B, 3]
        v3 = _F.normalize(v3, dim=1)                          # [B, 3]
        sims = v3 @ _NORMALS.T                                 # [B, 12]
        face_ids_t = sims.argmax(dim=1)                        # [B]

        # sub_face: for each item pick its 3 neighbour faces, take argmax sim
        nbr_faces = _NBRS_T[face_ids_t]                       # [B, 3]
        Bsz = vecs_batch.shape[0]
        nbr_norms = _NORMALS[nbr_faces.reshape(-1)].reshape(
            Bsz, 3, 3)                                        # [B, 3, 3]
        nbr_sims  = (nbr_norms * v3.unsqueeze(1)).sum(-1)    # [B, 3]
        sub_ids_t = nbr_sims.argmax(dim=1)                    # [B]

        # Zeckendorf: quantize full-vector norm
        full_norms = vecs_batch.float().norm(dim=1)
        ns = (full_norms * _MAG_SCALE).round().clamp(min=0).int().tolist()
        zeck_list  = [zeckendorf(int(n)) for n in ns]

        return face_ids_t.tolist(), sub_ids_t.tolist(), zeck_list

    # Import needed names
    from hierarchical_address import _MAG_SCALE, zeckendorf

    for batch_start in range(0, n_windows, batch_size):
        batch_pos = positions[batch_start:batch_start + batch_size]
        pos_t     = torch.tensor(batch_pos, dtype=torch.long)
        ids_batch = corpus_ids[(pos_t.unsqueeze(1) + j_idx.unsqueeze(0))]  # [B, T]
        vecs      = batch_substrate_fingerprint(ids_batch, tok_ord, d_model)  # [B, d]

        # Vectorised address assignment
        face_ids, sub_ids, zeck_list = _batch_assign_addresses(vecs)

        # Accumulate vec chunk (don't split into N individual tensors)
        vecs_chunks.append(vecs)

        B = len(batch_pos)
        base_idx = batch_start
        for k in range(B):
            start  = batch_pos[k]
            window = corpus_text[start:start + seq_len]
            addr   = HAddr(face=face_ids[k], sub_face=sub_ids[k], zeck=zeck_list[k])
            idx    = base_idx + k
            windows.append(window)
            addrs.append(addr)
            seq_index_list.append(start)
            face_index[addr.face].append(idx)

        done = min(batch_start + batch_size, n_windows)
        if done % 200000 < batch_size or done == n_windows:
            print(f"  [{done:,}/{n_windows:,}]", flush=True)

    vecs_tensor = torch.cat(vecs_chunks, dim=0)   # [N, d_model]
    seq_index   = seq_index_list

    print(f"\n[CorpusIndex] Face distribution:")
    for face in range(12):
        count = len(face_index.get(face, []))
        bar   = '#' * min(count // max(1, n_windows // 60), 50)
        empty = " ← EMPTY" if count == 0 else ""
        print(f"  face {face:2d}: {count:6,} windows  {bar}{empty}")

    return {
        'windows':    windows,
        'addrs':      addrs,
        'vecs':       vecs_tensor,
        'face_index': dict(face_index),
        'seq_index':  seq_index,
        'meta': {
            'seq_len':    seq_len,
            'd_model':    d_model,
            'stride':     stride,
            'n_windows':  n_windows,
            'corpus_len': total,
        },
    }


# ── Dual-scale index (T=13 + T=89 → all 12 faces) ────────────────────────────

def build_dual_scale_index(
    corpus_text: str,
    seq_len: int = 89,
    d_model: int = 64,
    stride: int = 89,
    batch_size: int = 4096,
    scales: tuple = (13, 89),
) -> Dict:
    """Build an index where each window is fingerprinted at TWO window lengths.

    T=13 covers faces {1,2,3,4,5,6,9,10}.
    T=89 covers faces {0,1,3,4,6,7,8,11}.
    Together: all 12 faces — the full dodecahedron.

    Each 89-char window gets two entries in the index:
      - one under its T=scales[0] face (short fingerprint)
      - one under its T=scales[1] face (long fingerprint)

    Index format has a 'scales' dict: {T: {vecs, addrs, face_index}}.
    Retrieval uses retrieve_dual(fp_short, fp_long, index) to search both.
    """
    total     = len(corpus_text)
    positions = list(range(0, total - seq_len + 1, stride))
    n_windows = len(positions)
    s0, s1    = scales[0], scales[1]
    print(f"[DualIndex] corpus={total:,}  windows={n_windows:,}  "
          f"scales=({s0},{s1})  stride={stride}", flush=True)

    chars   = sorted(set(corpus_text))
    stoi    = {c: i for i, c in enumerate(chars)}
    tok_ord = torch.tensor([ord(c) for c in chars], dtype=torch.int64)
    corpus_ids = torch.tensor([stoi[c] for c in corpus_text], dtype=torch.long)

    windows: List[str] = []
    seq_index: List[int] = []

    sub: Dict[int, Dict] = {
        s0: {'vecs': [], 'addrs': [], 'face_index': defaultdict(list)},
        s1: {'vecs': [], 'addrs': [], 'face_index': defaultdict(list)},
    }

    from hierarchical_address import _MAG_SCALE, zeckendorf
    _NORMALS = _NORMALS_STACK.float()
    _NBRS_T  = torch.tensor(
        [[n for n in _NEIGHBORS[f]] for f in range(12)], dtype=torch.long
    )

    def _assign_batch(vecs_batch):
        v3 = F.normalize(vecs_batch[:, :3].float(), dim=1)
        face_ids = (_NORMALS @ v3.T).argmax(dim=0).tolist()
        sub_ids  = []
        for fi, v in zip(face_ids, v3):
            nbrs = _NEIGHBORS[fi]
            sims = [(_NORMALS_STACK[n].float() @ v).item() for n in nbrs]
            sub_ids.append(sims.index(max(sims)))
        norms = vecs_batch.float().norm(dim=1)
        zeck  = [zeckendorf(int((nv * _MAG_SCALE).round().clamp(min=0).item()))
                 for nv in norms]
        return face_ids, sub_ids, zeck

    j0 = torch.arange(s0, dtype=torch.long)
    j1 = torch.arange(s1, dtype=torch.long)

    for batch_start in range(0, n_windows, batch_size):
        batch_pos = positions[batch_start:batch_start + batch_size]
        pos_t     = torch.tensor(batch_pos, dtype=torch.long)
        B         = len(batch_pos)
        base      = batch_start

        # Short fingerprint (T=s0)
        ids0  = corpus_ids[(pos_t.unsqueeze(1) + j0.unsqueeze(0))]
        vecs0 = batch_substrate_fingerprint(ids0, tok_ord, d_model)
        fi0, si0, z0 = _assign_batch(vecs0)

        # Long fingerprint (T=s1)
        ids1  = corpus_ids[(pos_t.unsqueeze(1) + j1.unsqueeze(0))]
        vecs1 = batch_substrate_fingerprint(ids1, tok_ord, d_model)
        fi1, si1, z1 = _assign_batch(vecs1)

        for k, start in enumerate(batch_pos):
            window = corpus_text[start:start + seq_len]
            idx    = base + k
            windows.append(window)
            seq_index.append(start)

            a0 = HAddr(face=fi0[k], sub_face=si0[k], zeck=z0[k])
            sub[s0]['vecs'].append(vecs0[k])
            sub[s0]['addrs'].append(a0)
            sub[s0]['face_index'][a0.face].append(idx)

            a1 = HAddr(face=fi1[k], sub_face=si1[k], zeck=z1[k])
            sub[s1]['vecs'].append(vecs1[k])
            sub[s1]['addrs'].append(a1)
            sub[s1]['face_index'][a1.face].append(idx)

        done = min(batch_start + batch_size, n_windows)
        if done % 100000 < batch_size or done == n_windows:
            print(f"  [{done:,}/{n_windows:,}]", flush=True)

    # Materialise
    for sc in scales:
        sub[sc]['vecs']       = torch.stack(sub[sc]['vecs'])
        sub[sc]['face_index'] = dict(sub[sc]['face_index'])

    print(f"\n[DualIndex] Face coverage:")
    all_faces = set()
    for sc in scales:
        fi = sub[sc]['face_index']
        occupied = sorted(f for f in fi if fi[f])
        empty    = [f for f in range(12) if f not in fi or not fi[f]]
        all_faces |= set(occupied)
        print(f"  T={sc:3d}: occupied={occupied}  empty={empty}")
    print(f"  combined: {sorted(all_faces)}  "
          f"{'ALL 12' if len(all_faces)==12 else str(len(all_faces))+'/12'}")

    return {
        'windows':   windows,
        'seq_index': seq_index,
        'scales':    {sc: sub[sc] for sc in scales},
        'meta': {
            'seq_len':    seq_len,
            'd_model':    d_model,
            'stride':     stride,
            'n_windows':  n_windows,
            'corpus_len': total,
            'scales':     list(scales),
            'dual':       True,
        },
    }


def retrieve_dual(
    fp_short: torch.Tensor,
    fp_long: torch.Tensor,
    index: Dict,
    top_k: int = 4,
    exclude_idxs: Optional[List[int]] = None,
) -> List[Dict]:
    """Retrieve from a dual-scale index using both fingerprints.

    fp_short fingerprint searches the T=scales[0] sub-index.
    fp_long  fingerprint searches the T=scales[1] sub-index.
    Results from both sub-indices are merged, de-duplicated, and ranked.
    """
    exclude_idxs = set(exclude_idxs or [])
    scales = index['meta']['scales']
    fps    = {scales[0]: fp_short, scales[1]: fp_long}

    all_results = {}   # idx → best result dict

    for sc, fp in fps.items():
        sub   = index['scales'][sc]
        addr  = assign_address(fp)
        faces = [addr.face] + list(_NEIGHBORS[addr.face])
        for face in faces:
            cand_idxs = [i for i in sub['face_index'].get(face, [])
                         if i not in exclude_idxs]
            if not cand_idxs:
                continue
            cand_vecs = sub['vecs'][cand_idxs].float()
            q   = F.normalize(fp.unsqueeze(0).float(), dim=1)
            c   = F.normalize(cand_vecs, dim=1)
            sims = (c @ q.T).squeeze(1)
            for sim_val, ci in zip(sims.tolist(), cand_idxs):
                if ci not in all_results or sim_val > all_results[ci]['sim']:
                    all_results[ci] = {
                        'text':  index['windows'][ci],
                        'addr':  sub['addrs'][ci],
                        'sim':   sim_val,
                        'idx':   ci,
                        'pos':   index['seq_index'][ci],
                        'scale': sc,
                    }
            break   # found hits at this face — stop expanding

    ranked = sorted(all_results.values(), key=lambda r: r['sim'], reverse=True)
    return ranked[:top_k]


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_at_face(
    query_vec: torch.Tensor,
    face: int,
    index: Dict,
    top_k: int = 5,
    exclude_idx: Optional[int] = None,
) -> List[Dict]:
    """Retrieve top_k windows at face, ranked by cosine similarity to query_vec.

    Falls back to neighboring faces if face is empty.
    """
    candidate_indices = list(index['face_index'].get(face, []))
    if exclude_idx is not None and exclude_idx in candidate_indices:
        candidate_indices = [i for i in candidate_indices if i != exclude_idx]

    if not candidate_indices:
        return []

    cand_vecs = index['vecs'][candidate_indices].float()          # [M, d]
    q         = F.normalize(query_vec.unsqueeze(0).float(), dim=1)  # [1, d]
    c         = F.normalize(cand_vecs, dim=1)                     # [M, d]
    sims      = (c @ q.T).squeeze(1)                               # [M]

    k    = min(top_k, len(candidate_indices))
    top  = sims.topk(k)
    return [
        {
            'text':     index['windows'][candidate_indices[li]],
            'addr':     index['addrs'][candidate_indices[li]],
            'sim':      sv.item(),
            'idx':      candidate_indices[li],
            'pos':      index['seq_index'][candidate_indices[li]],
        }
        for sv, li in zip(top.values, top.indices)
    ]


def retrieve(
    query_vec: torch.Tensor,
    index: Dict,
    top_k: int = 3,
    exclude_idx: Optional[int] = None,
) -> List[Dict]:
    """Retrieve from the address at query_vec, falling back to neighbors."""
    addr = assign_address(query_vec)
    results = retrieve_at_face(query_vec, addr.face, index, top_k, exclude_idx)
    if not results:
        for nbr in _NEIGHBORS[addr.face]:
            results = retrieve_at_face(query_vec, nbr, index, top_k, exclude_idx)
            if results:
                break
    return results


# ── Address-driven generation ─────────────────────────────────────────────────

def generate_by_address(
    index: Dict,
    seed_text: str,
    n_hops: int = 20,
    d_model: int = 64,
    sequential_bias: float = 0.5,
    history_window: int = 12,
    verbose: bool = True,
) -> str:
    """Navigate the address space to produce long Shakespeare text.

    Each hop:
      1. Fingerprint current window → HAddr
      2. Retrieve closest corpus window at that face (excluding recent visits)
      3. With sequential_bias probability, prefer corpus-adjacent window
      4. Repeat

    history_window: exclude the last N visited window indices from retrieval,
    preventing the oscillation / loop problem.
    """
    seq_len = index['meta']['seq_len']
    segments: List[str] = []
    current   = seed_text[:seq_len].ljust(seq_len)[:seq_len]
    last_idx:  Optional[int] = None
    visited:   list = []          # ring buffer of recent indices

    for hop in range(n_hops):
        vec  = substrate_fingerprint(current, d_model)
        addr = assign_address(vec)

        # Retrieve top candidates, excluding recently visited windows
        candidate_indices = [
            i for i in index['face_index'].get(addr.face, [])
            if i not in visited
        ]
        if not candidate_indices:
            # Fall back to neighboring faces
            for nbr in _NEIGHBORS[addr.face]:
                candidate_indices = [
                    i for i in index['face_index'].get(nbr, [])
                    if i not in visited
                ]
                if candidate_indices:
                    break
        if not candidate_indices:
            # Global fallback: any unvisited window
            candidate_indices = [i for i in range(len(index['windows']))
                                  if i not in visited]

        cand_vecs = index['vecs'][candidate_indices].float()
        q         = F.normalize(vec.unsqueeze(0).float(), dim=1)
        c         = F.normalize(cand_vecs, dim=1)
        sims      = (c @ q.T).squeeze(1)
        top       = sims.topk(min(5, len(candidate_indices)))
        results   = [
            {
                'text': index['windows'][candidate_indices[li]],
                'addr': index['addrs'][candidate_indices[li]],
                'sim':  sv.item(),
                'idx':  candidate_indices[li],
                'pos':  index['seq_index'][candidate_indices[li]],
            }
            for sv, li in zip(top.values, top.indices)
        ]

        # Sequential bias: prefer corpus-adjacent window if available in top-5
        chosen = results[0]
        if sequential_bias > 0.0 and last_idx is not None:
            import random
            seq_match = next(
                (r for r in results if r['idx'] == last_idx + 1), None
            )
            if seq_match is not None and random.random() < sequential_bias:
                chosen = seq_match

        segments.append(chosen['text'])
        last_idx = chosen['idx']
        current  = chosen['text']

        # Update visited ring buffer
        visited.append(chosen['idx'])
        if len(visited) > history_window:
            visited.pop(0)

        if verbose:
            print(f"  hop {hop+1:2d}: face={addr.face}  "
                  f"pos={chosen['pos']:7,}  sim={chosen['sim']:.3f}  "
                  f"text={repr(current[:55])}", flush=True)

    return ''.join(segments)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Build corpus address index and optionally generate by address navigation"
    )
    parser.add_argument('--seq-len',     type=int,   default=89)
    parser.add_argument('--stride',      type=int,   default=None,
                        help='Window stride; default=13 (Fibonacci)')
    parser.add_argument('--d-model',     type=int,   default=64)
    parser.add_argument('--source',      type=str,   default='tinyshakespeare')
    parser.add_argument('--corpus-file', type=str,   default=None,
                        help='Path to a local UTF-8 text file to use as corpus '
                             '(overrides --source)')
    parser.add_argument('--out',         type=str,   default='corpus_address_index.pt')
    parser.add_argument('--output',      type=str,   default=None,
                        help='Alias for --out')
    parser.add_argument('--rebuild',     action='store_true',
                        help='Force rebuild even if index exists')
    parser.add_argument('--generate',  action='store_true',
                        help='Run address-navigation generation after building index')
    parser.add_argument('--n-hops',    type=int,   default=20,
                        help='Number of address hops for generation')
    parser.add_argument('--seed-text', type=str,
                        default='Before we proceed any further, hear me speak.')
    parser.add_argument('--seq-bias',  type=float, default=0.5,
                        help='Sequential-order bias [0=cosine only, 1=sequential only]')
    parser.add_argument('--dual',      action='store_true',
                        help='Build dual-scale index (uses --scales, saves extra vectors)')
    parser.add_argument('--scales',    type=str,   default=None,
                        help='Comma-separated short,long scale pair for --dual '
                             '(default: 13,<seq-len>)')
    args = parser.parse_args()

    here     = Path(__file__).parent
    # --output is an alias for --out
    out_name = args.output if args.output is not None else args.out
    out_path = Path(out_name) if Path(out_name).is_absolute() else here / out_name

    # ── Build or load ─────────────────────────────────────────────────────────
    if out_path.exists() and not args.rebuild:
        print(f"Loading existing index: {out_path}", flush=True)
        index = torch.load(str(out_path), map_location='cpu', weights_only=False)
        meta  = index['meta']
        print(f"  windows={meta['n_windows']:,}  seq_len={meta['seq_len']}  "
              f"stride={meta['stride']}  corpus={meta['corpus_len']:,} chars")
        # Print face distribution
        print("\n[Loaded] Face distribution:")
        for face in range(12):
            count = len(index['face_index'].get(face, []))
            bar   = '#' * min(count // max(meta['n_windows'] // 60, 1), 50)
            print(f"  face {face:2d}: {count:6,}  {bar}")
    else:
        print("Building corpus address index ...", flush=True)
        if args.corpus_file is not None:
            corpus_path = Path(args.corpus_file)
            if not corpus_path.is_absolute():
                corpus_path = here / corpus_path
            print(f"Reading corpus from file: {corpus_path}", flush=True)
            with open(corpus_path, 'r', encoding='utf-8') as _f:
                corpus_text = _f.read()
        else:
            chars, stoi, itos, encoded = make_dataset(
                seq_len=args.seq_len, source=args.source
            )
            corpus_text = ''.join(itos[tok.item()] for tok in encoded)
        if args.dual:
            if args.scales:
                s0, s1 = [int(x) for x in args.scales.split(',')]
            else:
                s0, s1 = 13, args.seq_len
            stride_val = args.stride if args.stride is not None else args.seq_len
            index = build_dual_scale_index(
                corpus_text,
                seq_len=args.seq_len,
                d_model=args.d_model,
                stride=stride_val,
                scales=(s0, s1),
            )
        else:
            index = build_corpus_index(
                corpus_text,
                seq_len=args.seq_len,
                d_model=args.d_model,
                stride=args.stride,
            )
        torch.save(index, str(out_path))
        print(f"\nSaved → {out_path}  "
              f"({index['meta']['n_windows']:,} windows, "
              f"{out_path.stat().st_size // 1024:,} KB)", flush=True)

    # ── Generate ──────────────────────────────────────────────────────────────
    if args.generate:
        print("\n" + "=" * 70)
        print("Address-Navigation Generation")
        print("=" * 70)
        print(f"Seed   : {repr(args.seed_text)}")
        print(f"Hops   : {args.n_hops}")
        print(f"SeqBias: {args.seq_bias}")
        print()

        output = generate_by_address(
            index,
            seed_text=args.seed_text,
            n_hops=args.n_hops,
            d_model=args.d_model,
            sequential_bias=args.seq_bias,
            verbose=True,
        )

        print()
        print("=" * 70)
        print(f"Output ({len(output):,} chars):")
        print("=" * 70)
        print(output)

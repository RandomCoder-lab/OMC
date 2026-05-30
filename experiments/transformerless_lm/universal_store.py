"""Universal Address Store — one geometric space for everything.

Text, tools, weights, memory — all fingerprinted the same way.
Navigation is uniform: query_fp → closest address → dispatch on type.

The model navigating this store can land on:
  'text'   → read it, becomes context
  'tool'   → call it, result becomes context
  'weight' → activate that parameter slice for next steps
  'memory' → inject past turn into context
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from hierarchical_address import HAddr, assign_address, addr_str
from addressed_memory import _NEIGHBORS, _NORMALS_STACK
from corpus_address_index import substrate_fingerprint, batch_substrate_fingerprint


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tensor_to_fp_str(W: torch.Tensor) -> str:
    """Flatten a tensor, quantize to printable ASCII, return string for fingerprinting."""
    flat = W.detach().float().flatten()[:512]          # cap at 512 values
    mn, mx = flat.min().item(), flat.max().item()
    rng = mx - mn + 1e-8
    return ''.join(chr(32 + int((v - mn) / rng * 94)) for v in flat.tolist())


def _tool_to_schema_str(fn: Callable) -> str:
    """Docstring + signature string — stable fingerprint for a callable."""
    doc = (fn.__doc__ or '').strip()
    try:
        sig = str(inspect.signature(fn))
    except (ValueError, TypeError):
        sig = '(...)'
    return f"{fn.__name__}{sig}: {doc}"


def fingerprint_any(obj: Any, d_model: int = 64) -> torch.Tensor:
    """Fingerprint anything into a d_model-dim substrate vector."""
    if isinstance(obj, str):
        text = obj
    elif isinstance(obj, bytes):
        text = obj.decode('utf-8', errors='replace')
    elif isinstance(obj, torch.Tensor):
        text = _tensor_to_fp_str(obj)
    elif callable(obj):
        text = _tool_to_schema_str(obj)
    elif isinstance(obj, dict):
        import json
        text = json.dumps(obj, sort_keys=True)
    else:
        text = str(obj)
    return substrate_fingerprint(text, d_model)


# ── Entry ─────────────────────────────────────────────────────────────────────

@dataclass(eq=False)
class Entry:
    eid:     str                    # unique id, e.g. "text_000042", "tool_0003"
    type:    str                    # 'text' | 'tool' | 'weight' | 'memory'
    content: Any                    # str, callable, Tensor, ...
    fp:      torch.Tensor           # [d_model] — substrate fingerprint
    addr:    HAddr                  # dodecahedral address
    meta:    dict = field(default_factory=dict)

    def __repr__(self) -> str:
        tag = f"{self.type}:{addr_str(self.addr)}"
        if self.type in ('text', 'memory') and isinstance(self.content, str):
            return f"<Entry {tag} {repr(self.content[:40])}>"
        if self.type == 'tool' and callable(self.content):
            return f"<Entry {tag} fn={self.content.__name__}>"
        if self.type == 'weight':
            return f"<Entry {tag} skill={self.meta.get('name', '?')}>"
        return f"<Entry {tag}>"


# ── UniversalAddressStore ─────────────────────────────────────────────────────

class UniversalAddressStore:
    """All things live here. Navigation is uniform across types.

    Usage:
        store = UniversalAddressStore(d_model=64)
        store.store("some text", type='text')
        store.store(my_fn, type='tool', name='search')
        store.store(W, type='weight', name='skill_0')
        store.store("I said this earlier", type='memory', turn=3)

        results = store.retrieve(query_fp, top_k=8)
        results = store.retrieve(query_fp, type_filter='tool')
    """

    def __init__(self, d_model: int = 64):
        self.d_model = d_model
        self._entries:    Dict[str, Entry]         = {}
        self._vecs:       Dict[str, torch.Tensor]  = {}   # eid → fp [d]
        self._face_index: Dict[int, List[str]]     = {}   # face → [eid]
        self._type_sets:  Dict[str, Set[str]]      = {}   # type → {eid}
        self._type_face:  Dict[str, Dict[int, List[str]]] = {}  # type → face → [eid]
        self._counters:   Dict[str, int]           = {}   # type → next id int

    # ── store ─────────────────────────────────────────────────────────────────

    def store(
        self,
        content: Any,
        type: str,
        fp: Optional[torch.Tensor] = None,
        **meta,
    ) -> Entry:
        """Fingerprint content and add to store. Returns the Entry."""
        if fp is None:
            fp = fingerprint_any(content, self.d_model)

        # Guard against degenerate fingerprints (zero-norm)
        if fp.norm() < 1e-8:
            fp = fp + 1e-6 * torch.ones_like(fp)

        addr = assign_address(fp)

        n = self._counters.get(type, 0)
        eid = f"{type}_{n:06d}"
        self._counters[type] = n + 1

        entry = Entry(eid=eid, type=type, content=content, fp=fp, addr=addr, meta=meta)
        self._entries[eid] = entry
        self._vecs[eid]    = fp

        self._face_index.setdefault(addr.face, []).append(eid)

        self._type_sets.setdefault(type, set()).add(eid)
        self._type_face.setdefault(type, {}).setdefault(addr.face, []).append(eid)

        return entry

    # ── bulk corpus loader ────────────────────────────────────────────────────

    def store_corpus(
        self,
        text: str,
        seq_len: int = 89,
        stride: int = 89,
        batch_size: int = 4096,
        verbose: bool = True,
    ) -> int:
        """Bulk-load seq_len-char windows into the store.  Returns count added."""
        import numpy as np

        total     = len(text)
        positions = list(range(0, total - seq_len + 1, stride))
        n_windows = len(positions)
        if verbose:
            print(f"[UniversalStore] corpus={total:,}  seq_len={seq_len}  "
                  f"stride={stride}  windows={n_windows:,}", flush=True)

        # Build char vocab + tok_ord
        chars   = sorted(set(text))
        stoi    = {c: i for i, c in enumerate(chars)}
        tok_ord = torch.tensor([ord(c) for c in chars], dtype=torch.int64)

        # Encode full corpus as token ids (numpy LUT approach for speed)
        _max_cp = max(ord(c) for c in chars)
        _lut    = np.full(_max_cp + 1, 0, dtype=np.int32)
        for c, i in stoi.items():
            _lut[ord(c)] = i
        CHUNK = 4_000_000
        parts = []
        for off in range(0, total, CHUNK):
            chunk = text[off:off + CHUNK]
            arr   = np.frombuffer(chunk.encode('utf-32-le'), dtype=np.uint32)
            arr   = np.where(arr <= _max_cp, _lut[np.minimum(arr, _max_cp)], 0)
            parts.append(arr.astype(np.int64))
        corpus_ids = torch.from_numpy(np.concatenate(parts))

        # Vectorised face assignment
        _NORMALS = _NORMALS_STACK.float()
        j_idx    = torch.arange(seq_len, dtype=torch.long)

        type_str = 'text'
        n_added  = 0

        for batch_start in range(0, n_windows, batch_size):
            batch_pos = positions[batch_start:batch_start + batch_size]
            pos_t     = torch.tensor(batch_pos, dtype=torch.long)
            ids_batch = corpus_ids[(pos_t.unsqueeze(1) + j_idx.unsqueeze(0))]   # [B, T]
            vecs      = batch_substrate_fingerprint(ids_batch, tok_ord, self.d_model)  # [B, d]

            # Vectorised face ids
            v3       = F.normalize(vecs[:, :3].float(), dim=1)
            face_ids = (_NORMALS @ v3.T).argmax(dim=0).tolist()

            for k, (start, face_id) in enumerate(zip(batch_pos, face_ids)):
                window = text[start:start + seq_len]
                n      = self._counters.get(type_str, 0)
                eid    = f"{type_str}_{n:06d}"
                self._counters[type_str] = n + 1

                fp   = vecs[k]
                addr = HAddr(face=face_id, sub_face=0, zeck=frozenset())

                entry = Entry(eid=eid, type=type_str, content=window,
                              fp=fp, addr=addr, meta={'pos': start, 'seq_len': seq_len})
                self._entries[eid] = entry
                self._vecs[eid]    = fp
                self._face_index.setdefault(face_id, []).append(eid)
                self._type_sets.setdefault(type_str, set()).add(eid)
                self._type_face.setdefault(type_str, {}).setdefault(face_id, []).append(eid)
                n_added += 1

            done = min(batch_start + batch_size, n_windows)
            if verbose and (done % 50_000 < batch_size or done == n_windows):
                print(f"  [{done:,}/{n_windows:,}]", flush=True)

        return n_added

    # ── retrieve ──────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_fp: torch.Tensor,
        top_k:       int             = 8,
        type_filter: Optional[str]   = None,
        exclude_eids: Optional[set]  = None,
    ) -> List[Tuple[float, Entry]]:
        """Return top_k entries closest to query_fp, ordered by cosine sim.

        type_filter=None searches all types.
        exclude_eids is a set of eid strings to skip.
        """
        exclude_eids = exclude_eids or set()
        addr         = assign_address(query_fp)

        face_idx = self._type_face.get(type_filter, self._face_index) if type_filter \
                   else self._face_index

        # Face-first search: this face, then neighbors
        candidates: List[str] = []
        for face in [addr.face] + list(_NEIGHBORS[addr.face]):
            eids = [e for e in face_idx.get(face, []) if e not in exclude_eids]
            if eids:
                candidates.extend(eids)
                break  # found at this face — stop expanding

        # Fallback: all entries of the requested type
        if not candidates:
            all_eids = list(self._type_sets.get(type_filter, set())) if type_filter \
                       else list(self._entries.keys())
            candidates = [e for e in all_eids if e not in exclude_eids]

        if not candidates:
            return []

        fps = torch.stack([self._vecs[e] for e in candidates]).float()   # [M, d]
        q   = F.normalize(query_fp.unsqueeze(0).float(), dim=1)          # [1, d]
        c   = F.normalize(fps, dim=1)                                     # [M, d]
        sims = (c @ q.T).squeeze(1).tolist()                             # [M]

        ranked = sorted(zip(sims, candidates), reverse=True)
        return [(sim, self._entries[eid]) for sim, eid in ranked[:top_k]]

    # ── utility ───────────────────────────────────────────────────────────────

    def stats(self) -> str:
        lines = [f"UniversalAddressStore: {len(self._entries):,} entries  d_model={self.d_model}"]
        for t in sorted(self._type_sets):
            eids  = self._type_sets[t]
            faces = sorted(set(self._entries[e].addr.face for e in eids))
            lines.append(f"  {t:8s}: {len(eids):6,} entries  faces={faces}")
        occupied = sorted(self._face_index)
        lines.append(f"  faces occupied: {occupied}  ({len(occupied)}/12)")
        return '\n'.join(lines)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"UniversalAddressStore(entries={len(self._entries)}, d_model={self.d_model})"

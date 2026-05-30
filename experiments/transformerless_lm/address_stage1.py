"""Stage 1: Bloom Direct Addressing.

Assigns each bloom vector a direct dode face address (0-11) so retrieval is
O(1) by face rather than O(n) by cosine search.

The face is determined by which dode face's member tokens have the highest
combined softmax probability when the bloom vector is projected through the
model's LM head (if available), or by geometric nearest-normal in 3-D (fallback).
"""

import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ── Dode face normals (fixed by substrate math) ───────────────────────────────

PHI = (1.0 + 5.0 ** 0.5) / 2.0

_RAW_NORMALS = [
    ( 1,  1,  1), (-1,  1,  1), ( 1, -1,  1), ( 1,  1, -1),
    (-1, -1,  1), (-1,  1, -1), ( 1, -1, -1), (-1, -1, -1),
    ( 0,  1/PHI,  PHI), ( 0, -1/PHI,  PHI),
    ( 0,  1/PHI, -PHI), ( 0, -1/PHI, -PHI),
]

DODE_NORMALS: List[torch.Tensor] = [
    (lambda t: t / t.norm())(torch.tensor(n, dtype=torch.float))
    for n in _RAW_NORMALS
]

DODE_NORMALS_STACK = torch.stack(DODE_NORMALS)   # [12, 3]

# ── Face assignment helpers ───────────────────────────────────────────────────

def assign_face_via_head(
    bv: torch.Tensor,
    head_weight: torch.Tensor,
    token_face_map: List[int],
) -> int:
    """Project bloom vec through LM head → vocab logits → highest-prob face.

    head_weight: [vocab_size, d_model]  (standard weight layout for F.linear)
    token_face_map: list[int] of length vocab_size, face index for each token
    """
    with torch.no_grad():
        logits = F.linear(bv.unsqueeze(0), head_weight).squeeze(0)  # [vocab_size]
        probs = torch.softmax(logits, dim=0)
    face_probs = torch.zeros(12)
    for tok_idx, face in enumerate(token_face_map):
        face_probs[face] += probs[tok_idx]
    return int(face_probs.argmax().item())


def assign_face_via_geometry(bv: torch.Tensor) -> int:
    """Assign bloom vec to nearest dode face by cosine similarity on first 3 dims."""
    bv3 = bv[:3].float()
    norm = bv3.norm()
    if norm > 1e-8:
        bv3 = bv3 / norm
    sims = DODE_NORMALS_STACK @ bv3   # [12]
    return int(sims.argmax().item())


# ── Token → face mapping (reuse Stage 3 substrate math) ──────────────────────

import math

_FIBS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]


def _collatz_depth(n: int, _max: int = 500) -> int:
    steps = 0
    while n != 1 and steps < _max:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps


def _zeckendorf_complexity(n: int) -> int:
    if n <= 0:
        return 1
    a, b, fibs = 1, 2, []
    while a <= n:
        fibs.append(a)
        a, b = b, a + b
    count = 0
    for f in reversed(fibs):
        if f <= n:
            n -= f
            count += 1
    return max(count, 1)


def _build_zeta_table(max_s: int, n_terms: int = 150, sigma: float = 0.75) -> list:
    ln_ns = [math.log(k) for k in range(1, n_terms + 1)]
    wt = [k ** (-sigma) for k in range(1, n_terms + 1)]
    table = []
    for s in range(max_s + 1):
        t = 14.0 + (50.0 - 14.0) * s / max(max_s, 1)
        val = sum(w * math.cos(t * lk) for w, lk in zip(wt, ln_ns))
        table.append(val)
    return table


def _char_resonance(ordinal: int, zeta_table: list, coll_max: int, max_ord: int) -> float:
    n = max(ordinal, 1)
    coll_norm = min(_collatz_depth(n) / max(coll_max, 1), 1.0)
    idx = min(n, max_ord)
    zeta_norm = min(abs(zeta_table[idx]) / 10.0, 1.0)
    nearest_fib = min(_FIBS, key=lambda f: abs(f - n))
    fib_prox = PHI ** (-(n - nearest_fib) ** 2 / PHI)
    zeck_score = 1.0 / _zeckendorf_complexity(n)
    return (PHI ** -1 * coll_norm +
            PHI ** -2 * zeta_norm +
            PHI ** -3 * fib_prox +
            PHI ** -4 * zeck_score)


def build_token_face_map(vocab: List[str]) -> List[int]:
    """Assign each vocab token to a dode face using substrate math."""
    ords = [ord(c) for c in vocab]
    max_ord = max(ords, default=130)
    coll_max = max(_collatz_depth(o) for o in ords) if ords else 200
    zeta_tab = _build_zeta_table(max_ord, n_terms=150, sigma=0.75)
    scored = sorted(
        ((c, _char_resonance(ord(c), zeta_tab, coll_max, max_ord)) for c in vocab),
        key=lambda x: x[1],
        reverse=True,
    )
    n_chars = len(scored)
    face_assign = {c: idx * 12 // n_chars for idx, (c, _) in enumerate(scored)}
    return [face_assign.get(c, 0) for c in vocab]


# ── BloomAddressTable ─────────────────────────────────────────────────────────

class BloomAddressTable:
    """Maps bloom vectors to dode face addresses. Stage 1: O(1) retrieval by face."""

    def __init__(
        self,
        checkpoint_path: str,
        model_path: Optional[str] = None,
    ) -> None:
        ckpt_path = Path(checkpoint_path)

        # ── Load bloom checkpoint ─────────────────────────────────────────
        print(f"[Stage1] Loading bloom checkpoint: {ckpt_path}", flush=True)
        ck = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)

        self.bloom_vecs: List[torch.Tensor] = ck.get('bloom_vecs', [])
        self.bloom_crystal_ages: List[int] = ck.get('bloom_crystal_ages', [])

        print(f"[Stage1] Bloom vecs: {len(self.bloom_vecs)}, d_model={self.bloom_vecs[0].shape[0] if self.bloom_vecs else 0}", flush=True)

        # ── Load model if available ───────────────────────────────────────
        self.head_weight: Optional[torch.Tensor] = None
        self.vocab: Optional[List[str]] = None
        self.token_face_map: Optional[List[int]] = None

        # Auto-detect model path if not specified
        if model_path is None:
            candidate = ckpt_path.parent / 'bloom_checkpoint_model.pt'
            if candidate.exists():
                model_path = str(candidate)

        if model_path is not None and Path(model_path).exists():
            print(f"[Stage1] Loading model: {model_path}", flush=True)
            mck = torch.load(model_path, map_location='cpu', weights_only=False)
            self.vocab = mck.get('vocab', None)
            state = mck.get('model_state', {})
            # head.weight is [vocab_size, d_model] — standard F.linear layout
            if 'head.weight' in state:
                self.head_weight = state['head.weight'].float()
                print(f"[Stage1] LM head loaded: {list(self.head_weight.shape)}", flush=True)
            else:
                print("[Stage1] No head.weight in model_state; will use geometry.", flush=True)
            if self.vocab is not None:
                self.token_face_map = build_token_face_map(self.vocab)
                print(f"[Stage1] Token face map built for {len(self.vocab)} tokens", flush=True)
        else:
            print("[Stage1] No model checkpoint found; head method unavailable.", flush=True)

        # ── Assignments (populated by assign_faces) ───────────────────────
        self.bloom_face_assignments: List[int] = []
        self.face_index: Dict[int, List[int]] = defaultdict(list)

    # ── Face assignment ───────────────────────────────────────────────────────

    def assign_faces(self, method: str = 'geometry') -> None:
        """Compute bloom_face_assignments and face_index.

        method='geometry': use bv[:3] vs DODE_NORMALS (always available)
        method='head':     use model head projection (requires model)
        """
        if method == 'head':
            if self.head_weight is None or self.token_face_map is None:
                print("[Stage1] head method requested but model not loaded; falling back to geometry.", flush=True)
                method = 'geometry'

        print(f"[Stage1] Assigning faces via '{method}' method...", flush=True)
        assignments: List[int] = []
        for bv in self.bloom_vecs:
            if method == 'head':
                face = assign_face_via_head(bv.float(), self.head_weight, self.token_face_map)
            else:
                face = assign_face_via_geometry(bv.float())
            assignments.append(face)

        self.bloom_face_assignments = assignments

        # Rebuild face_index
        self.face_index = defaultdict(list)
        for idx, face in enumerate(assignments):
            self.face_index[face].append(idx)

        print(f"[Stage1] Face assignment complete. Faces occupied: {sorted(self.face_index.keys())}", flush=True)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get_at_face(self, face: int) -> List[Tuple[torch.Tensor, int]]:
        """Return list of (bloom_vec, crystal_age) for all bloom vecs at this face. O(1)."""
        indices = self.face_index.get(face, [])
        return [(self.bloom_vecs[i], self.bloom_crystal_ages[i]) for i in indices]

    def get_face_for_vec(self, idx: int) -> int:
        """Return the face assignment for bloom vec at index idx."""
        if not self.bloom_face_assignments:
            raise RuntimeError("Call assign_faces() first.")
        return self.bloom_face_assignments[idx]

    # ── Benchmarking ─────────────────────────────────────────────────────────

    def retrieval_benchmark(self, n_queries: int = 1000) -> Dict:
        """Benchmark O(1) face lookup vs O(n) cosine search.

        Reports avg query time for each method and per-face distribution.
        """
        if not self.bloom_face_assignments:
            raise RuntimeError("Call assign_faces() first.")

        import random
        vecs_stack = torch.stack([bv.float() for bv in self.bloom_vecs])   # [N, 64]
        vecs_norm = F.normalize(vecs_stack, dim=1)                          # [N, 64]

        faces_to_query = [random.randint(0, 11) for _ in range(n_queries)]
        # Pick random query vecs matching those faces (some may be empty, that's OK)
        query_vecs = [vecs_stack[random.randint(0, len(self.bloom_vecs) - 1)]
                      for _ in range(n_queries)]

        # ── O(1) face lookup ──────────────────────────────────────────────
        t0 = time.perf_counter()
        for face in faces_to_query:
            _ = self.face_index.get(face, [])
        t_face = (time.perf_counter() - t0) / n_queries * 1e6   # microseconds

        # ── O(n) cosine search ────────────────────────────────────────────
        t0 = time.perf_counter()
        for qv in query_vecs:
            qv_norm = F.normalize(qv.unsqueeze(0), dim=1)           # [1, 64]
            sims = (vecs_norm @ qv_norm.T).squeeze(1)                # [N]
            _ = int(sims.argmax().item())
        t_cosine = (time.perf_counter() - t0) / n_queries * 1e6     # microseconds

        # ── Per-face distribution ─────────────────────────────────────────
        n_total = len(self.bloom_vecs)
        distribution = {
            f: len(self.face_index.get(f, [])) / max(n_total, 1)
            for f in range(12)
        }

        results = {
            'n_bloom_vecs':           n_total,
            'n_queries':              n_queries,
            'avg_face_lookup_us':     t_face,
            'avg_cosine_search_us':   t_cosine,
            'speedup_x':              t_cosine / max(t_face, 1e-12),
            'face_distribution':      distribution,
        }

        print("\n── Retrieval Benchmark ────────────────────────────────────────")
        print(f"  Bloom vecs          : {n_total}")
        print(f"  Queries             : {n_queries}")
        print(f"  Avg face lookup     : {t_face:.3f} µs")
        print(f"  Avg cosine search   : {t_cosine:.3f} µs")
        print(f"  Speedup             : {results['speedup_x']:.1f}x")
        print("\n  Per-face fraction:")
        for f in range(12):
            bar_len = int(distribution[f] * 40)
            bar = '#' * bar_len
            count = len(self.face_index.get(f, []))
            print(f"    face {f:2d}: {count:4d} vecs  ({distribution[f]*100:5.1f}%)  {bar}")

        return results

    # ── Coverage ──────────────────────────────────────────────────────────────

    def face_coverage(self) -> Dict:
        """Report how many faces have at least one bloom vec and which are empty."""
        if not self.bloom_face_assignments:
            raise RuntimeError("Call assign_faces() first.")

        populated = sorted(f for f in range(12) if self.face_index.get(f))
        empty = sorted(f for f in range(12) if not self.face_index.get(f))
        counts = {f: len(self.face_index.get(f, [])) for f in range(12)}

        print("\n── Face Coverage ──────────────────────────────────────────────")
        print(f"  Total bloom vecs    : {len(self.bloom_vecs)}")
        print(f"  Faces populated     : {len(populated)}/12  {populated}")
        print(f"  Faces EMPTY (gaps)  : {len(empty)}/12  {empty}")
        print("\n  Face → count:")
        for f in range(12):
            status = "POPULATED" if f in populated else "EMPTY (unexplored)"
            print(f"    face {f:2d}: {counts[f]:4d}  [{status}]")

        return {
            'populated_faces': populated,
            'empty_faces':     empty,
            'counts':          counts,
        }

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> None:
        """Print total bloom vecs, face assignments, and coverage stats."""
        n = len(self.bloom_vecs)
        n_assigned = len(self.bloom_face_assignments)
        n_faces = sum(1 for f in range(12) if self.face_index.get(f))
        print("\n── BloomAddressTable Summary ──────────────────────────────────")
        print(f"  Total bloom vecs    : {n}")
        print(f"  Face assignments    : {n_assigned}")
        print(f"  Faces populated     : {n_faces}/12")
        print(f"  Model head loaded   : {self.head_weight is not None}")
        print(f"  Vocab size          : {len(self.vocab) if self.vocab else 'N/A'}")
        if n_assigned:
            from collections import Counter
            dist = Counter(self.bloom_face_assignments)
            most_common = dist.most_common(3)
            print(f"  Top-3 faces         : {most_common}")


# ── Checkpoint persistence ────────────────────────────────────────────────────

def save_face_assignments(checkpoint_path: str, assignments: List[int]) -> None:
    """Add bloom_face_assignments to existing checkpoint without losing other data."""
    data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    data['bloom_face_assignments'] = assignments
    torch.save(data, checkpoint_path)
    print(f"Saved {len(assignments)} face assignments to {checkpoint_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    here = Path(__file__).parent
    ckpt = str(here / 'bloom_checkpoint.pt')
    model_ckpt = str(here / 'bloom_checkpoint_model.pt')

    do_save = '--save' in sys.argv

    print("=" * 62)
    print("Stage 1: Bloom Direct Addressing")
    print("=" * 62)

    # 1. Load
    bat = BloomAddressTable(ckpt, model_ckpt if Path(model_ckpt).exists() else None)

    # 2. Assign faces using geometry method
    bat.assign_faces(method='geometry')

    # 3. Face coverage — populated vs empty
    coverage = bat.face_coverage()

    # 4. First 5 assignments
    print("\n── First 5 bloom vec face assignments ─────────────────────────")
    for i in range(min(5, len(bat.bloom_vecs))):
        bv = bat.bloom_vecs[i]
        face = bat.bloom_face_assignments[i]
        age = bat.bloom_crystal_ages[i]
        print(f"  bloom[{i}]: face={face:2d}  age={age:4d}  bv[:3]={bv[:3].tolist()}")

    # 5. Retrieval benchmark
    bat.retrieval_benchmark(n_queries=1000)

    # 6. Summary
    bat.summary()

    # 7. Optionally save
    if do_save:
        save_face_assignments(ckpt, bat.bloom_face_assignments)
    else:
        print("\n[Stage1] Pass --save to write face assignments back to checkpoint.")

    # Report gaps clearly
    print("\n── Unexplored Territory (empty faces) ─────────────────────────")
    empty = coverage['empty_faces']
    if empty:
        print(f"  Empty faces: {empty}")
        print("  These face indices have NO bloom vectors assigned to them.")
        print("  They represent unexplored regions of the dode address space.")
    else:
        print("  All 12 faces are populated — full coverage achieved.")
    print("=" * 62)

"""Geometric episodic memory for a transformerless LM.

Assigns each bloom vector a dodecahedral face address (0-11) via φ-built
face normals, mirroring address_stage1.py and substrate_tokenizer.py.
"""

import math
from typing import ClassVar, Dict, List, Optional, Set

import torch
import torch.nn.functional as F

# ── Dodecahedral constants ────────────────────────────────────────────────────

PHI = (1.0 + math.sqrt(5.0)) / 2.0

# 12 face normals = the 12 vertices of a regular icosahedron = the face normals
# of its dual, the regular dodecahedron: (0,±1,±φ) and cyclic permutations.
# This is the CANONICAL set — by icosahedral symmetry every face subtends an
# equal solid angle, so a uniform fingerprint lands on each face with equal
# probability (verified: χ²≈17 on 20k uniform points, vs χ²≈1570 for the older
# mixed cube+golden-rectangle set which had grossly unequal cells). Equal-area
# faces = balanced address buckets = better retrieval discrimination.
_RAW_NORMALS = [
    [ 0,  1,  PHI], [ 0,  1, -PHI], [ 0, -1,  PHI], [ 0, -1, -PHI],
    [ 1,  PHI,  0], [ 1, -PHI,  0], [-1,  PHI,  0], [-1, -PHI,  0],
    [ PHI,  0,  1], [ PHI,  0, -1], [-PHI,  0,  1], [-PHI,  0, -1],
]

with torch.no_grad():
    _NORMALS_STACK: torch.Tensor = F.normalize(
        torch.tensor(_RAW_NORMALS, dtype=torch.float32), dim=1
    )  # [12, 3]

# Face adjacency, computed from the geometry: the 3 nearest other faces by normal
# dot product. (On the icosahedron each vertex has 5 equidistant neighbors; we keep
# the deterministic first 3 for the 3-way sub_face partition. Computed, not
# hand-transcribed, so it stays correct if the normal ordering ever changes.)
def _compute_neighbors(normals: torch.Tensor, k: int = 3) -> List[List[int]]:
    with torch.no_grad():
        sims = normals @ normals.T            # [12, 12]
        sims.fill_diagonal_(-2.0)             # exclude self
        nbrs = sims.argsort(dim=1, descending=True)[:, :k]
    return [row.tolist() for row in nbrs]

_NEIGHBORS: List[List[int]] = _compute_neighbors(_NORMALS_STACK, k=3)

_MAX_PER_FACE = 32


# ── Helper ────────────────────────────────────────────────────────────────────

def _assign_face(vec: torch.Tensor) -> int:
    """Return the dode face id (0-11) nearest to `vec` using first 3 dims."""
    with torch.no_grad():
        v3 = vec[:3].float()
        norm = v3.norm()
        if norm > 1e-8:
            v3 = v3 / norm
        sims = _NORMALS_STACK @ v3  # [12]
        return int(sims.argmax().item())


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    with torch.no_grad():
        # Align dimensions — truncate to shorter if shapes differ
        d = min(a.shape[0], b.shape[0])
        an = F.normalize(a[:d].float().unsqueeze(0), dim=1)
        bn = F.normalize(b[:d].float().unsqueeze(0), dim=1)
        return float((an @ bn.T).item())


# ── Entry dataclass (plain dict for JSON compatibility) ───────────────────────

class AddressedMemoryLayer:
    """Geometric episodic memory on a 12-face dodecahedral address space.

    Each face holds up to 32 entries; near-duplicate writes (cosine_sim > 0.95)
    increment the visit counter instead of creating duplicates.
    """

    VERSION: ClassVar[str] = "1.0"

    def __init__(self, d_model: int = 64, n_faces: int = 12) -> None:
        self.d_model = d_model
        self.n_faces = n_faces
        # faces[f] = list of dicts: {'vec': Tensor[d], 'age': int, 'visits': int}
        self.faces: Dict[int, List[dict]] = {f: [] for f in range(n_faces)}

    # ── Core write ────────────────────────────────────────────────────────────

    def write(self, bloom_vec: torch.Tensor, crystal_age: int = 0) -> int:
        """Store bloom_vec in its face bucket; return face id.

        Deduplicates at cosine_sim > 0.95.  Evicts lowest-visits (then age) when
        the bucket exceeds 32 entries.
        """
        face = _assign_face(bloom_vec)
        bucket = self.faces[face]

        with torch.no_grad():
            # Check for near-duplicate
            for entry in bucket:
                if _cosine_sim(bloom_vec, entry['vec']) > 0.95:
                    entry['visits'] += 1
                    entry['age'] = crystal_age
                    print(
                        f"[AML] write: face={face} dedup hit "
                        f"(visits={entry['visits']}, age={crystal_age})"
                    )
                    return face

            # New entry
            bucket.append({
                'vec': bloom_vec.detach().clone(),
                'age': crystal_age,
                'visits': 1,
            })

            # Evict if over capacity
            if len(bucket) > _MAX_PER_FACE:
                bucket.sort(key=lambda e: (e['visits'], e['age']))
                evicted = bucket.pop(0)
                print(
                    f"[AML] write: face={face} evict "
                    f"(visits={evicted['visits']}, age={evicted['age']})"
                )

        total = sum(len(b) for b in self.faces.values())
        print(
            f"[AML] write: face={face} size={len(bucket)} "
            f"age={crystal_age} total={total}"
        )
        return face

    # ── Core read ─────────────────────────────────────────────────────────────

    def read(self, query_vec: torch.Tensor, k: int = 3) -> List[torch.Tensor]:
        """Return top-k vectors by cosine similarity from query face + 3 neighbors."""
        face = _assign_face(query_vec)
        candidate_faces = [face] + _NEIGHBORS[face]

        candidates: List[dict] = []
        for f in candidate_faces:
            candidates.extend(self.faces[f])

        if not candidates:
            print(f"[AML] read: face={face} no candidates yet")
            return []

        with torch.no_grad():
            ranked = sorted(
                candidates,
                key=lambda e: _cosine_sim(query_vec, e['vec']),
                reverse=True,
            )

        result = [e['vec'] for e in ranked[:k]]
        print(
            f"[AML] read: face={face} candidates={len(candidates)} "
            f"returning={len(result)}"
        )
        return result

    # ── Gap map ───────────────────────────────────────────────────────────────

    def gap_map(self) -> Set[int]:
        """Return the set of face ids with zero stored memories (known unknowns)."""
        return {f for f in range(self.n_faces) if len(self.faces[f]) == 0}

    # ── Gap logit bias ────────────────────────────────────────────────────────

    def gap_logit_bias(
        self,
        query_vec: torch.Tensor,
        face_ids: torch.Tensor,
        strength: float = 0.3,
    ) -> torch.Tensor:
        """Return [vocab_size] logit additions guiding generation toward explored faces.

        Zeros if current face is explored.  If it's a gap: `strength` toward the
        nearest explored face; `strength*0.5` toward all other unexplored faces.
        """
        with torch.no_grad():
            vocab_size = face_ids.shape[0]
            bias = torch.zeros(vocab_size, dtype=torch.float32)

            current_face = _assign_face(query_vec)
            gaps = self.gap_map()

            if current_face not in gaps:
                # Already explored — no bias needed
                return bias

            # Find nearest explored face by normal-to-normal cosine similarity
            explored = [f for f in range(self.n_faces) if f not in gaps]
            if not explored:
                # Nothing explored at all — mild curiosity toward every face is redundant
                return bias

            cur_normal = _NORMALS_STACK[current_face]  # [3]
            best_face: Optional[int] = None
            best_sim = -2.0
            for f in explored:
                sim = float((_NORMALS_STACK[f] @ cur_normal).item())
                if sim > best_sim:
                    best_sim = sim
                    best_face = f

            # Bias toward tokens on the nearest explored face
            if best_face is not None:
                mask_near = (face_ids == best_face)
                bias[mask_near] += strength

            # Mild curiosity toward tokens on all unexplored faces
            for f in gaps:
                if f == current_face:
                    continue
                mask_gap = (face_ids == f)
                bias[mask_gap] += strength * 0.5

        return bias

    # ── Age / decay ───────────────────────────────────────────────────────────

    def age_all(self) -> None:
        """Increment age on all memories; remove unconsolidated (age>50, visits==1);
        halve visits on consolidated (age>50, visits>=3)."""
        for f in range(self.n_faces):
            survivors: List[dict] = []
            for entry in self.faces[f]:
                entry['age'] += 1
                if entry['age'] > 50:
                    if entry['visits'] == 1:
                        continue  # evict
                    elif entry['visits'] >= 3:
                        entry['visits'] = max(1, entry['visits'] // 2)
                survivors.append(entry)
            self.faces[f] = survivors

    # ── Neighbors ─────────────────────────────────────────────────────────────

    def neighbor_faces(self, face_id: int) -> List[int]:
        """Return the 3 adjacent face ids in the dodecahedron."""
        return list(_NEIGHBORS[face_id])

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize all memories to a JSON-compatible dict."""
        faces_serial: Dict[str, list] = {}
        for f in range(self.n_faces):
            faces_serial[str(f)] = [
                {
                    'vec': e['vec'].tolist(),
                    'age': e['age'],
                    'visits': e['visits'],
                }
                for e in self.faces[f]
            ]
        return {
            'version': self.VERSION,
            'd_model': self.d_model,
            'n_faces': self.n_faces,
            'faces': faces_serial,
        }

    @classmethod
    def from_dict(cls, d: dict, d_model: int = 64) -> 'AddressedMemoryLayer':
        """Reconstruct an AddressedMemoryLayer from a serialized dict."""
        n_faces = d.get('n_faces', 12)
        layer = cls(d_model=d.get('d_model', d_model), n_faces=n_faces)
        faces_serial = d.get('faces', {})
        with torch.no_grad():
            for f in range(n_faces):
                entries_raw = faces_serial.get(str(f), [])
                layer.faces[f] = [
                    {
                        'vec': torch.tensor(e['vec'], dtype=torch.float32),
                        'age': int(e['age']),
                        'visits': int(e['visits']),
                    }
                    for e in entries_raw
                ]
        return layer

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """One-line summary: memories per face, gaps, total entries."""
        counts = [len(self.faces[f]) for f in range(self.n_faces)]
        total = sum(counts)
        gaps = sorted(f for f in range(self.n_faces) if counts[f] == 0)
        per_face = ' '.join(f'{c}' for c in counts)
        line = (
            f"[AML] total={total} per_face=[{per_face}] "
            f"gaps={gaps if gaps else 'none'}"
        )
        print(line)
        return line

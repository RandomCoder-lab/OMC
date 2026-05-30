"""HierarchicalAddress — three-level dodecahedral address for any vector.

Level 1: face      (0-11)   argmax dot with 12 φ-built normals
Level 2: sub_face  (0-2)    which of the face's 3 neighbors the vector leans toward
Level 3: zeckendorf          Fibonacci decomposition of quantized vector magnitude

Full address: HAddr(face, sub_face, zeck_frozenset)
  e.g.  HAddr(face=6, sub_face=2, zeck=frozenset({3, 8, 21}))
        → human-readable: "(6,2,F3+F8+F21)"

Address distance:
  Same face, sub, zeck → 0.0
  Face mismatch        → 3.0
  Sub-face mismatch    → 1.0
  Zeck bit diff        → 0.5 per element in symmetric difference
"""

import math
from typing import FrozenSet, NamedTuple

import torch

from addressed_memory import _NORMALS_STACK, _NEIGHBORS

# ── Fibonacci numbers for Zeckendorf (covers integers 0 to 610) ──────────────
_FIBS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]

# Scale factor: vec.float().norm() * _MAG_SCALE → integer for Zeckendorf.
# Trained embeddings (d=64, LayerNorm-ish) have norms ≈ 1–8; scale=10 maps to 10-80.
_MAG_SCALE = 10.0


# ── Address type ──────────────────────────────────────────────────────────────

class HAddr(NamedTuple):
    face:     int
    sub_face: int
    zeck:     FrozenSet[int]


# ── Zeckendorf decomposition ───────────────────────────────────────────────────

def zeckendorf(n: int) -> FrozenSet[int]:
    """Greedy Zeckendorf: unique sum of non-consecutive Fibonacci numbers.

    Input clamped to [0, 610].  Returns frozenset of Fibonacci values (not indices).
    """
    if n <= 0:
        return frozenset()
    n = min(n, _FIBS[-1])
    result: set = set()
    for f in reversed(_FIBS):
        if f <= n:
            result.add(f)
            n -= f
    return frozenset(result)


# ── Core address assignment ────────────────────────────────────────────────────

def assign_address(vec: torch.Tensor, magnitude_scale: float = _MAG_SCALE) -> HAddr:
    """Compute full three-level hierarchical address for a d-dimensional vector.

    Level 1 uses only vec[:3] (the substrate's 3-space).
    Level 3 uses the full-vector norm (energy fingerprint).

    Args:
        vec:             Any float tensor of length ≥ 3.
        magnitude_scale: Scales vec.norm() to an integer for Zeckendorf.
    """
    with torch.no_grad():
        v3 = vec[:3].float()
        v3_unit = v3 / v3.norm().clamp(min=1e-8)

        # ── Level 1: face ─────────────────────────────────────────────────────
        face = int((_NORMALS_STACK @ v3_unit).argmax().item())

        # ── Level 2: sub_face — which of the 3 neighbor normals is nearest? ──
        neighbors = _NEIGHBORS[face]
        nbr_sims = [float((_NORMALS_STACK[nf] @ v3_unit).item()) for nf in neighbors]
        sub_face = int(max(range(3), key=lambda i: nbr_sims[i]))

        # ── Level 3: Zeckendorf of quantized magnitude ─────────────────────────
        full_norm = vec.float().norm().item()
        n = max(0, round(full_norm * magnitude_scale))
        zeck = zeckendorf(n)

    return HAddr(face=face, sub_face=sub_face, zeck=zeck)


# ── Utilities ─────────────────────────────────────────────────────────────────

def addr_str(addr: HAddr) -> str:
    """Human-readable: (face,sub,F3+F8+F21) or (face,sub,F0) when empty."""
    zeck_str = '+'.join(f'F{f}' for f in sorted(addr.zeck)) if addr.zeck else 'F0'
    return f"({addr.face},{addr.sub_face},{zeck_str})"


def addr_distance(a: HAddr, b: HAddr) -> float:
    """Coarse semantic distance between two hierarchical addresses.

    Identical → 0.0.
    Face mismatch = 3.0 (dominates).
    Sub-face mismatch = 1.0.
    Zeck symmetric difference = 0.5 per bit.
    """
    d = 0.0
    if a.face != b.face:
        d += 3.0
    if a.sub_face != b.sub_face:
        d += 1.0
    d += 0.5 * len(a.zeck.symmetric_difference(b.zeck))
    return d


def addr_coarse_match(a: HAddr, b: HAddr) -> bool:
    """True when face and sub_face match (Zeck may differ — same region, different energy)."""
    return a.face == b.face and a.sub_face == b.sub_face

"""Geometric prompting: bias generation toward a target address in semantic space."""

import re
from typing import List

import torch

from hierarchical_address import HAddr, assign_address, addr_str, addr_distance, zeckendorf
from addressed_memory import _NEIGHBORS


def parse_addr_str(s: str) -> HAddr:
    """Parse a human-readable address string into HAddr.

    Accepts:
      "(6,2,F3+F8+F21)"   — full hierarchical
      "(6,2)"             — face+sub, zeck=frozenset()
      "6"                 — face only, sub_face=0, zeck=frozenset()
    Raises ValueError on unrecognisable format.
    """
    s = s.strip()

    # bare integer: face only
    if re.fullmatch(r'\d+', s):
        face = int(s)
        if not (0 <= face <= 11):
            raise ValueError(f"face must be 0-11, got {face}")
        return HAddr(face=face, sub_face=0, zeck=frozenset())

    m2 = re.fullmatch(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', s)
    if m2:
        face, sub_face = int(m2.group(1)), int(m2.group(2))
        if not (0 <= face <= 11):
            raise ValueError(f"face must be 0-11, got {face}")
        if not (0 <= sub_face <= 2):
            raise ValueError(f"sub_face must be 0-2, got {sub_face}")
        return HAddr(face=face, sub_face=sub_face, zeck=frozenset())

    m3 = re.fullmatch(r'\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(F0|(?:F\d+(?:\+F\d+)*))\s*\)', s)
    if m3:
        face, sub_face = int(m3.group(1)), int(m3.group(2))
        zeck_str = m3.group(3)
        if not (0 <= face <= 11):
            raise ValueError(f"face must be 0-11, got {face}")
        if not (0 <= sub_face <= 2):
            raise ValueError(f"sub_face must be 0-2, got {sub_face}")
        if zeck_str == 'F0':
            zeck = frozenset()
        else:
            zeck = frozenset(int(part[1:]) for part in zeck_str.split('+'))
        return HAddr(face=face, sub_face=sub_face, zeck=zeck)

    raise ValueError(f"Unrecognisable address format: {s!r}")


class GeometricPrompter:
    def __init__(
        self,
        embed_weight: torch.Tensor,
        vocab: List[str],
    ):
        self._vocab = vocab
        self._token_addrs: List[HAddr] = []
        self._recompute(embed_weight)

    def _recompute(self, embed_weight: torch.Tensor) -> None:
        with torch.no_grad():
            self._token_addrs = [
                assign_address(embed_weight[i]) for i in range(embed_weight.shape[0])
            ]

    def parse_addr(self, s: str) -> HAddr:
        return parse_addr_str(s)

    def tokens_near(self, addr: HAddr, top_k: int = 5) -> List[str]:
        scored = sorted(
            range(len(self._token_addrs)),
            key=lambda i: addr_distance(self._token_addrs[i], addr),
        )
        return [self._vocab[i] for i in scored[:top_k]]

    def logit_bias(
        self,
        addr: HAddr,
        strength: float = 0.5,
    ) -> torch.Tensor:
        vocab_size = len(self._token_addrs)
        bias = torch.zeros(vocab_size, dtype=torch.float32)

        adj_faces = set(_NEIGHBORS[addr.face])

        for i, tok_addr in enumerate(self._token_addrs):
            if tok_addr == addr:
                bias[i] = strength
            elif tok_addr.face == addr.face and tok_addr.sub_face == addr.sub_face:
                bias[i] = strength * 0.6
            elif tok_addr.face == addr.face:
                bias[i] = strength * 0.3
            elif tok_addr.face in adj_faces:
                bias[i] = strength * 0.15

        return bias

    def describe(self, addr: HAddr) -> str:
        near = self.tokens_near(addr, top_k=5)
        zeck_str = '+'.join(f'F{f}' for f in sorted(addr.zeck)) if addr.zeck else 'F0'
        return (
            f"face={addr.face} sub={addr.sub_face} zeck={zeck_str}"
            f"  nearest tokens: {near}"
        )

    def refresh(self, embed_weight: torch.Tensor) -> None:
        self._recompute(embed_weight)

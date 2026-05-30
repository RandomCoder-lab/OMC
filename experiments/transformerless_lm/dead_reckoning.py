from collections import deque, Counter
from typing import Deque, List, Optional

import torch

from hierarchical_address import HAddr, assign_address
from addressed_memory import _NEIGHBORS


class AddressPredictor:
    def __init__(self, window: int = 5) -> None:
        self.window = window
        self._history: Deque[HAddr] = deque(maxlen=window)
        self._cached_embed_shape: Optional[torch.Size] = None
        self._cached_token_faces: List[int] = []

    def update(self, addr: HAddr) -> None:
        self._history.append(addr)

    def projected_face(self) -> Optional[int]:
        if len(self._history) < 2:
            return None

        faces = [a.face for a in self._history]

        # Momentum: last two face transitions in same direction → project continuation.
        if len(faces) >= 3:
            d1 = faces[-2] - faces[-3]
            d2 = faces[-1] - faces[-2]
            if d1 != 0 and d2 != 0 and (d1 > 0) == (d2 > 0):
                projected = faces[-1] + d2
                # Dodecahedron has 12 faces; clamp to valid range.
                if 0 <= projected <= 11:
                    return projected

        # No clear momentum — return most common face in window.
        return Counter(faces).most_common(1)[0][0]

    def logit_bias(
        self,
        embed_weight: torch.Tensor,
        strength: float = 0.4,
    ) -> torch.Tensor:
        target_face = self.projected_face()
        vocab = embed_weight.shape[0]
        bias = torch.zeros(vocab, dtype=torch.float32)

        if target_face is None:
            return bias

        if embed_weight.shape != self._cached_embed_shape:
            self._cached_embed_shape = embed_weight.shape
            with torch.no_grad():
                self._cached_token_faces = [
                    assign_address(embed_weight[i]).face for i in range(vocab)
                ]

        adjacent = set(_NEIGHBORS[target_face])

        for i, face in enumerate(self._cached_token_faces):
            if face == target_face:
                bias[i] = strength
            elif face in adjacent:
                bias[i] = strength * 0.5

        return bias

    def summary(self) -> str:
        pf = self.projected_face()
        faces = [a.face for a in self._history]
        if len(faces) >= 2:
            delta = faces[-1] - faces[-2]
            vel_str = f"face{'+' if delta >= 0 else ''}{delta}"
        else:
            vel_str = "face+0"
        line = (
            f"[DEAD-RECK] window={self.window} "
            f"projected_face={pf} velocity={vel_str}"
        )
        print(line)
        return line

from __future__ import annotations

from typing import List, Optional

import torch

from hierarchical_address import HAddr, assign_address, addr_distance


class SemanticTrajectory:
    def __init__(self):
        self._addrs: List[HAddr] = []
        self._tokens: List[str] = []

    def record(self, vec: torch.Tensor, token_str: str = '') -> HAddr:
        addr = assign_address(vec)
        self._addrs.append(addr)
        self._tokens.append(token_str)
        return addr

    def coherence_score(self) -> float:
        if len(self._addrs) < 2:
            return 1.0
        dists = [
            addr_distance(self._addrs[i], self._addrs[i + 1])
            for i in range(len(self._addrs) - 1)
        ]
        mean_dist = sum(dists) / len(dists)
        return max(0.0, min(1.0, 1.0 - mean_dist / 3.0))

    def phase_transitions(self) -> List[int]:
        return [
            i + 1
            for i in range(len(self._addrs) - 1)
            if self._addrs[i].face != self._addrs[i + 1].face
        ]

    def face_path(self) -> List[int]:
        return [a.face for a in self._addrs]

    def current_velocity(self) -> Optional[float]:
        if len(self._addrs) < 2:
            return None
        return addr_distance(self._addrs[-2], self._addrs[-1])

    def summary(self) -> str:
        n = len(self._addrs)
        faces = self.face_path()

        # Run-length compress face sequence
        if not faces:
            rle = ''
        else:
            runs: List[tuple] = []
            cur, count = faces[0], 1
            for f in faces[1:]:
                if f == cur:
                    count += 1
                else:
                    runs.append((cur, count))
                    cur, count = f, 1
            runs.append((cur, count))
            rle = '→'.join(
                str(face) if cnt == 1 else f'{face}×{cnt}'
                for face, cnt in runs
            )

        coherence = self.coherence_score()
        n_trans = len(self.phase_transitions())
        line = f'[TRAJ] len={n} faces={rle} coherence={coherence:.2f} transitions={n_trans}'
        print(line)
        return line

    def reset(self):
        self._addrs.clear()
        self._tokens.clear()


_active_trajectory: Optional[SemanticTrajectory] = None


def get_trajectory() -> SemanticTrajectory:
    global _active_trajectory
    if _active_trajectory is None:
        _active_trajectory = SemanticTrajectory()
    return _active_trajectory


def reset_trajectory():
    global _active_trajectory
    if _active_trajectory is not None:
        _active_trajectory.reset()
    else:
        _active_trajectory = SemanticTrajectory()

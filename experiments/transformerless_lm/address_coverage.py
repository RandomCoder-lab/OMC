"""AddressCoverage — tracks which HAddrs have been populated across training cycles.

Answers: "how much of semantic space has this model explored?"
"""

import re
from typing import Dict, List

from hierarchical_address import HAddr, addr_str

_VALID_TYPES = {'token', 'word', 'sentence', 'bloom'}
_ALL_FACES = list(range(12))


def _parse_addr_str(s: str) -> HAddr:
    """Inverse of addr_str. Parses '(face,sub,F3+F8+F21)' or '(face,sub,F0)'."""
    m = re.fullmatch(r'\((\d+),(\d+),(.*?)\)', s)
    if not m:
        raise ValueError(f"Cannot parse HAddr from: {s!r}")
    face = int(m.group(1))
    sub_face = int(m.group(2))
    zeck_raw = m.group(3)
    if zeck_raw == 'F0':
        zeck: frozenset = frozenset()
    else:
        zeck = frozenset(int(tok[1:]) for tok in zeck_raw.split('+'))
    return HAddr(face=face, sub_face=sub_face, zeck=zeck)


class AddressCoverage:
    def __init__(self) -> None:
        self._store: Dict[str, set] = {t: set() for t in _VALID_TYPES}

    def record(self, addr: HAddr, type_: str = 'bloom') -> None:
        """Mark an address as populated. type_ is one of: token, word, sentence, bloom."""
        if type_ not in _VALID_TYPES:
            raise ValueError(f"Unknown type {type_!r}; expected one of {_VALID_TYPES}")
        self._store[type_].add(addr)

    def _all_addrs(self) -> set:
        result: set = set()
        for s in self._store.values():
            result |= s
        return result

    def n_addresses(self) -> int:
        """Total distinct HAddrs recorded across all types."""
        return len(self._all_addrs())

    def n_by_type(self) -> Dict[str, int]:
        """Count distinct addresses per object type."""
        return {t: len(s) for t, s in self._store.items()}

    def face_coverage(self) -> Dict[int, int]:
        """face_id -> number of distinct HAddrs recorded on that face."""
        counts: Dict[int, int] = {f: 0 for f in _ALL_FACES}
        for addr in self._all_addrs():
            counts[addr.face] += 1
        return counts

    def empty_faces(self) -> List[int]:
        """Face ids (0-11) with zero recorded addresses (geometric blind spots)."""
        fc = self.face_coverage()
        return [f for f in _ALL_FACES if fc[f] == 0]

    def density_map(self) -> str:
        """12-char string, one char per face: '.' = empty, '1'-'9' = count (capped at 9),
        '*' = 10+."""
        fc = self.face_coverage()
        chars = []
        for f in _ALL_FACES:
            n = fc[f]
            if n == 0:
                chars.append('.')
            elif n >= 10:
                chars.append('*')
            else:
                chars.append(str(n))
        return ''.join(chars)

    def growth_delta(self, previous_n: int) -> int:
        """How many new addresses were added since previous_n total."""
        return self.n_addresses() - previous_n

    def report(self) -> str:
        """Multi-line coverage report. Prints and returns the string."""
        total = self.n_addresses()
        fc = self.face_coverage()
        active = sum(1 for c in fc.values() if c > 0)
        empty = self.empty_faces()
        dm = self.density_map()
        nbt = self.n_by_type()
        type_parts = '  '.join(f"{t}s={nbt[t]}" for t in _VALID_TYPES if nbt[t] > 0)
        lines = [
            f"[COVER] {total} addresses  faces_active={active}/12  empty_faces={empty}",
            f"density: {dm}  {type_parts}",
        ]
        text = '\n'.join(lines)
        print(text)
        return text

    def to_dict(self) -> dict:
        """Serialize for checkpoint saving (JSON-compatible)."""
        return {
            t: sorted(addr_str(a) for a in s)
            for t, s in self._store.items()
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'AddressCoverage':
        """Reconstruct from serialized dict."""
        obj = cls()
        for type_, str_list in d.items():
            for s in str_list:
                obj._store[type_].add(_parse_addr_str(s))
        return obj

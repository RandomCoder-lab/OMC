"""CrossTypeResonanceMap — tracks co-location of multiple object types at each address.

Object types tracked at each hierarchical address:
  'token'    — vocabulary character / subword (key = token string repr)
  'word'     — word beacon from corpus (key = word string)
  'sentence' — corpus sentence beacon (key = str(sentence_index))
  'bloom'    — model episodic bloom memory (key = 'cycleN_faceF')

The map answers:
  richest_addresses()    → which addresses have the most diverse object types
  find_deep_resonances() → bloom↔sentence and bloom↔token pairs above threshold
  convergence_report()   → grouped report of all deep resonances
  summary()              → concise report of the map state

Deep resonance (bloom ↔ sentence/token sim > 0.95) means the model has internally
reconstructed a corpus coordinate — not imitation, convergence.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import re

import torch
import torch.nn.functional as F

from hierarchical_address import HAddr, addr_str, assign_address

# Object types that can be registered
VALID_TYPES = frozenset(['token', 'word', 'sentence', 'bloom'])


# ── Address string parser ──────────────────────────────────────────────────────

def _parse_addr_str(s: str) -> HAddr:
    """Parse an addr_str like '(6,2,F3+F8+F21)' back to HAddr.

    Handles:
      '(6,2,F3+F8+F21)' → HAddr(face=6, sub_face=2, zeck=frozenset({3,8,21}))
      '(6,2,F0)'         → HAddr(face=6, sub_face=2, zeck=frozenset())
    """
    m = re.fullmatch(r'\((\d+),(\d+),(F[0-9+F]*)\)', s.strip())
    if m is None:
        raise ValueError(f"Cannot parse HAddr string: {s!r}")
    face     = int(m.group(1))
    sub_face = int(m.group(2))
    zeck_raw = m.group(3)  # e.g. 'F3+F8+F21' or 'F0'
    if zeck_raw == 'F0':
        zeck = frozenset()
    else:
        zeck = frozenset(int(part[1:]) for part in zeck_raw.split('+') if part)
    return HAddr(face=face, sub_face=sub_face, zeck=zeck)


# ── Entry ─────────────────────────────────────────────────────────────────────

class _Entry:
    __slots__ = ('type_', 'key', 'vec')

    def __init__(self, type_: str, key: str, vec: Optional[torch.Tensor] = None):
        self.type_ = type_
        self.key   = key
        self.vec   = vec.detach().cpu().float() if vec is not None else None


# ── Cosine similarity helper ───────────────────────────────────────────────────

def _cosim(a: torch.Tensor, b: torch.Tensor) -> float:
    with torch.no_grad():
        d  = min(a.shape[0], b.shape[0])
        an = F.normalize(a[:d].unsqueeze(0), dim=1)
        bn = F.normalize(b[:d].unsqueeze(0), dim=1)
        return float((an @ bn.T).item())


# ── CrossTypeResonanceMap ─────────────────────────────────────────────────────

class CrossTypeResonanceMap:
    """Tracks object co-location across a hierarchical dodecahedral address space.

    Usage::

        rmap = CrossTypeResonanceMap()
        rmap.register(addr, 'token', "'e'", embed_vec)
        rmap.register(addr, 'bloom', 'cycle12_face6', bloom_vec)
        rmap.find_deep_resonances()
        rmap.summary()
    """

    def __init__(self, deep_resonance_threshold: float = 0.95) -> None:
        self.threshold = deep_resonance_threshold
        self._store: Dict[HAddr, List[_Entry]] = defaultdict(list)
        self._deep_resonances: List[dict] = []
        # Track first-registration silence per beacon type
        self._sentence_beacon_logged: bool = False
        self._word_beacon_logged: bool = False

    # ── Registration ──────────────────────────────────────────────────────────

    def register(
        self,
        addr:  HAddr,
        type_: str,
        key:   str,
        vec:   Optional[torch.Tensor] = None,
    ) -> None:
        """Add an object at a hierarchical address."""
        if type_ not in VALID_TYPES:
            raise ValueError(f"Unknown type '{type_}'. Must be one of {sorted(VALID_TYPES)}")
        self._store[addr].append(_Entry(type_=type_, key=key, vec=vec))

    def register_sentence_beacon(self, sent_idx: int, vec: torch.Tensor) -> None:
        """Convenience wrapper: assign address from vec, register as 'sentence'.

        Prints a log line only on the first call, then stays silent.
        """
        addr = assign_address(vec)
        if not self._sentence_beacon_logged:
            print(f"[RMAP] sentence beacon {sent_idx} → {addr_str(addr)}", flush=True)
            self._sentence_beacon_logged = True
        self.register(addr, 'sentence', str(sent_idx), vec)

    def register_word_beacon(self, word: str, vec: torch.Tensor) -> None:
        """Convenience wrapper: assign address from vec, register as 'word'.

        Prints a log line only on the first call, then stays silent.
        """
        addr = assign_address(vec)
        if not self._word_beacon_logged:
            print(f"[RMAP] word beacon {word!r} → {addr_str(addr)}", flush=True)
            self._word_beacon_logged = True
        self.register(addr, 'word', word, vec)

    # ── Deep resonance ────────────────────────────────────────────────────────

    def find_deep_resonances(self) -> List[dict]:
        """Scan all addresses for bloom↔sentence and bloom↔token co-location above threshold.

        Returns list of event dicts.
        Prints any newly detected convergence events.
        """
        events: List[dict] = []

        # Build known-signature sets per resonance type to avoid duplicates
        known_bs = {
            (e['addr'], e['bloom'], e['sentence'])
            for e in self._deep_resonances
            if e.get('type') == 'bloom_sentence'
        }
        known_bt = {
            (e['addr'], e['bloom'], e['token'])
            for e in self._deep_resonances
            if e.get('type') == 'bloom_token'
        }

        for addr, entries in self._store.items():
            blooms    = [e for e in entries if e.type_ == 'bloom'    and e.vec is not None]
            sentences = [e for e in entries if e.type_ == 'sentence' and e.vec is not None]
            tokens    = [e for e in entries if e.type_ == 'token'    and e.vec is not None]

            # ── bloom ↔ sentence ─────────────────────────────────────────────
            if blooms and sentences:
                for b in blooms:
                    for s in sentences:
                        sim = _cosim(b.vec, s.vec)
                        if sim >= self.threshold:
                            event = {
                                'type': 'bloom_sentence',
                                'addr': addr,
                                'bloom': b.key,
                                'sentence': s.key,
                                'sim': sim,
                            }
                            events.append(event)
                            sig = (addr, b.key, s.key)
                            if sig not in known_bs:
                                self._deep_resonances.append(event)
                                known_bs.add(sig)
                                print(
                                    f"  [RESONANCE] DEEP CONVERGENCE  "
                                    f"bloom={b.key} ↔ sent={s.key}  "
                                    f"@ {addr_str(addr)}  sim={sim:.4f}",
                                    flush=True,
                                )

            # ── bloom ↔ token ─────────────────────────────────────────────────
            if blooms and tokens:
                for b in blooms:
                    for t in tokens:
                        sim = _cosim(b.vec, t.vec)
                        if sim >= self.threshold:
                            event = {
                                'type': 'bloom_token',
                                'addr': addr,
                                'bloom': b.key,
                                'token': t.key,
                                'sim': sim,
                            }
                            events.append(event)
                            sig = (addr, b.key, t.key)
                            if sig not in known_bt:
                                self._deep_resonances.append(event)
                                known_bt.add(sig)
                                print(
                                    f"  [RESONANCE] BLOOM→TOKEN "
                                    f"bloom={b.key} ↔ token={t.key}  "
                                    f"@ {addr_str(addr)}  sim={sim:.4f}",
                                    flush=True,
                                )

        return events

    # ── Convergence report ────────────────────────────────────────────────────

    def convergence_report(self) -> str:
        """Print and return all deep resonances grouped by type.

        Example output::

            [CONVERGENCE] 3 bloom↔sentence  1 bloom↔token
              bloom↔sentence: b3_f6 ↔ sent=4821 @ (6,2,F8+F21) sim=0.9712
              bloom↔token:    b7_f9 ↔ token='a' @ (9,1,F3)     sim=0.9631
        """
        bs_events = [e for e in self._deep_resonances if e.get('type') == 'bloom_sentence']
        bt_events = [e for e in self._deep_resonances if e.get('type') == 'bloom_token']

        header = f"[CONVERGENCE] {len(bs_events)} bloom↔sentence  {len(bt_events)} bloom↔token"
        lines  = [header]

        for e in bs_events:
            lines.append(
                f"  bloom↔sentence: {e['bloom']} ↔ sent={e['sentence']}"
                f" @ {addr_str(e['addr'])} sim={e['sim']:.4f}"
            )
        for e in bt_events:
            lines.append(
                f"  bloom↔token:    {e['bloom']} ↔ token={e['token']}"
                f" @ {addr_str(e['addr'])} sim={e['sim']:.4f}"
            )

        result = '\n'.join(lines)
        print(result, flush=True)
        return result

    # ── Richness queries ───────────────────────────────────────────────────────

    def richest_addresses(self, top_n: int = 5) -> List[Tuple[HAddr, int, set]]:
        """Return top_n addresses by number of distinct object types present.

        Returns list of (addr, n_distinct_types, type_set).
        """
        scored = [
            (addr, len({e.type_ for e in entries}), {e.type_ for e in entries})
            for addr, entries in self._store.items()
        ]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_n]

    def address_count(self) -> int:
        return len(self._store)

    def total_entries(self) -> int:
        return sum(len(v) for v in self._store.values())

    def entries_by_type(self) -> Dict[str, int]:
        counts: Dict[str, int] = {t: 0 for t in VALID_TYPES}
        for entries in self._store.values():
            for e in entries:
                counts[e.type_] += 1
        return counts

    # ── Summary ────────────────────────────────────────────────────────────────

    def summary(self, top_n: int = 5) -> str:
        """Print and return a concise map report."""
        by_type  = self.entries_by_type()
        n_addr   = self.address_count()
        n_total  = self.total_entries()
        n_res    = len(self._deep_resonances)

        type_line = '  '.join(f"{t}={by_type[t]}" for t in sorted(VALID_TYPES) if by_type[t] > 0)
        lines = [
            f"[RMAP] {n_total} objects @ {n_addr} addresses  "
            f"[{type_line}]  deep_resonances={n_res}"
        ]

        rich = self.richest_addresses(top_n)
        if rich:
            lines.append("  richest:")
            for addr, n_types, types_present in rich:
                es = self._store[addr]
                tc = {t: sum(1 for e in es if e.type_ == t) for t in types_present}
                tc_str = ' '.join(f"{t}×{tc[t]}" for t in sorted(tc))
                lines.append(f"    {addr_str(addr)}  [{tc_str}]  ({n_types} types)")

        if self._deep_resonances:
            lines.append("  deep resonances (latest 3):")
            for ev in self._deep_resonances[-3:]:
                ev_type = ev.get('type', 'bloom_sentence')
                if ev_type == 'bloom_token':
                    lines.append(
                        f"    {addr_str(ev['addr'])}  "
                        f"bloom={ev['bloom']} ↔ token={ev['token']}  "
                        f"sim={ev['sim']:.4f}"
                    )
                else:
                    lines.append(
                        f"    {addr_str(ev['addr'])}  "
                        f"bloom={ev['bloom']} ↔ sent={ev['sentence']}  "
                        f"sim={ev['sim']:.4f}"
                    )

        result = '\n'.join(lines)
        print(result, flush=True)
        return result

    # ── Serialization (compact — vectors are not saved) ───────────────────────

    def to_dict(self) -> dict:
        """Serialize address→type counts (no vectors) for logging."""
        out = {}
        for addr, entries in self._store.items():
            by_type = {}
            for e in entries:
                by_type[e.type_] = by_type.get(e.type_, 0) + 1
            out[addr_str(addr)] = by_type
        return {
            'counts_by_address': out,
            'n_deep_resonances': len(self._deep_resonances),
            'deep_resonances': [
                {
                    'type':     e.get('type', 'bloom_sentence'),
                    'addr':     addr_str(e['addr']),
                    'bloom':    e['bloom'],
                    'sentence': e.get('sentence'),
                    'token':    e.get('token'),
                    'sim':      round(e['sim'], 6),
                }
                for e in self._deep_resonances
            ],
        }

    @classmethod
    def from_dict(cls, d: dict, deep_resonance_threshold: float = 0.95) -> 'CrossTypeResonanceMap':
        """Reconstruct a CrossTypeResonanceMap from to_dict() output.

        Vectors are not stored in the serialized form, so all re-created entries
        have vec=None.  This is sufficient for coverage/reporting; deep resonance
        scanning only fires when vec is not None, so no spurious events are raised.

        Args:
            d:                        Dict produced by to_dict().
            deep_resonance_threshold: Passed through to the new instance.
        """
        rmap = cls(deep_resonance_threshold=deep_resonance_threshold)

        # ── Rebuild store from counts_by_address ──────────────────────────────
        for addr_s, type_counts in d.get('counts_by_address', {}).items():
            addr = _parse_addr_str(addr_s)
            for type_, count in type_counts.items():
                for _ in range(count):
                    # key unknown after serialization — use placeholder
                    rmap._store[addr].append(_Entry(type_=type_, key='<restored>', vec=None))

        # ── Rebuild deep_resonances list ──────────────────────────────────────
        for ev in d.get('deep_resonances', []):
            addr_s   = ev.get('addr', '')
            ev_type  = ev.get('type', 'bloom_sentence')
            restored = {
                'type':  ev_type,
                'addr':  _parse_addr_str(addr_s),
                'bloom': ev.get('bloom', ''),
                'sim':   ev.get('sim', 0.0),
            }
            if ev_type == 'bloom_token':
                restored['token']    = ev.get('token', '')
            else:
                restored['sentence'] = ev.get('sentence', '')
            rmap._deep_resonances.append(restored)

        return rmap

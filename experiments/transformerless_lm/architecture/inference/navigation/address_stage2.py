"""Stage 2: Unified Address Space.

Build a single lookup table that stores every addressable object in the
transformerless-LM system under a common address format:

    Address = (face: int, depth: float, velocity: tuple)

    face     - dodecahedral face 0-11  (primary bucket)
    depth    - Collatz depth normalised to [0,1]  (ordering within face)
    velocity - (vx, vy, vz) unit-vector in R^3  (exact location)

Object types stored:
    'bloom'    - bloom crystal vectors (bloom_vecs)
    'word'     - word_registry entries
    'sentence' - sentence beacons from SubstrateTokenizer
    'corpus'   - single corpus attractor vector
    'collatz'  - per-token Collatz depth scalars
    'fib'      - FibPosTable [10x8] counts (one entry per row)

Run as __main__ to produce the gap map and navigation query.
"""

import io
import contextlib
import math
import sys
from collections import Counter, defaultdict, namedtuple
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHI = (1.0 + 5.0 ** 0.5) / 2.0

# All known object types in the system
ALL_OBJECT_TYPES = {'bloom', 'word', 'sentence', 'corpus', 'collatz', 'fib'}


def _build_dode_normals() -> torch.Tensor:
    """12 face normals of a regular dodecahedron (icosahedron vertices)."""
    p = PHI
    m = (2.0 + p) ** 0.5
    raw = [
        (0,  1,  p), (0,  1, -p), (0, -1,  p), (0, -1, -p),
        (1,  p,  0), (1, -p,  0), (-1,  p,  0), (-1, -p,  0),
        (p,  0,  1), (p,  0, -1), (-p,  0,  1), (-p,  0, -1),
    ]
    return torch.tensor([[x/m, y/m, z/m] for x, y, z in raw], dtype=torch.float32)


DODE_NORMALS = _build_dode_normals()  # [12, 3]

# ---------------------------------------------------------------------------
# Entry type
# ---------------------------------------------------------------------------

Entry = namedtuple('Entry', ['object_type', 'object_data', 'address', 'age'])


# ---------------------------------------------------------------------------
# Mathematical helpers
# ---------------------------------------------------------------------------

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


def build_face_assignment(vocab: List[str]) -> List[int]:
    """Assign each character to a dode face using substrate math.

    Matches SubstrateTokenizer v4 exactly (rank-band -> face 0-11).
    """
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


def build_token_face_map(word_registry: dict, vocab: List[str]) -> List[int]:
    """Infer token->face from word_registry character data.

    Each word's address gives (char, face) pairs.  We tally per-char face
    counts, then assign each vocab token its most-common face.  Falls back
    to the substrate-math assignment for chars not seen in the registry.
    """
    char_face_counts: Dict[str, List[int]] = {}
    for word, data in word_registry.items():
        addr = data.get('address', ())
        for char, face in zip(word.lower(), addr):
            if char not in char_face_counts:
                char_face_counts[char] = [0] * 12
            if 0 <= face < 12:
                char_face_counts[char][face] += 1

    char_to_face: Dict[str, int] = {}
    for c, counts in char_face_counts.items():
        char_to_face[c] = max(range(12), key=lambda f: counts[f])

    # Fallback: use substrate-math assignment for any chars not in registry
    math_faces = build_face_assignment(vocab)

    result = []
    for i, c in enumerate(vocab):
        if c.lower() in char_to_face:
            result.append(char_to_face[c.lower()])
        elif c in char_to_face:
            result.append(char_to_face[c])
        else:
            result.append(math_faces[i])
    return result


def vec_to_face(vx: float, vy: float, vz: float) -> int:
    """Return the dode face index whose normal has highest cosine sim with (vx,vy,vz)."""
    v = torch.tensor([vx, vy, vz], dtype=torch.float32)
    norm = v.norm()
    if norm < 1e-8:
        return 0
    v = v / norm
    sims = DODE_NORMALS @ v   # [12]
    return int(sims.argmax().item())


# ---------------------------------------------------------------------------
# Unified Address Table
# ---------------------------------------------------------------------------

class UnifiedAddressTable:
    """Single lookup table for all objects in the system.

    Primary bucket: dode face (0-11).
    Secondary ordering: Collatz depth normalised to [0,1].
    Exact position: unit velocity in R^3.
    """

    def __init__(self, checkpoint_path: str) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        print(f"[Stage2] Loading bloom checkpoint: {self.checkpoint_path}", flush=True)
        ck = torch.load(str(self.checkpoint_path), map_location='cpu', weights_only=False)

        # Unpack checkpoint fields
        self._bloom_vecs: List[torch.Tensor] = ck.get('bloom_vecs', [])
        self._bloom_crystal_ages: List[int] = ck.get('bloom_crystal_ages', [])
        self._word_registry: dict = ck.get('word_registry', {})
        self._fib_table_counts: torch.Tensor = ck.get('fib_table_counts',
                                                        torch.zeros(10, 8))
        self._fib_table_token_counts: torch.Tensor = ck.get('fib_table_token_counts',
                                                              torch.zeros(65, 10))

        # face -> list[Entry]
        self.table: Dict[int, List[Entry]] = {f: [] for f in range(12)}

        # Vocab: try model checkpoint first, then derive from word_registry
        model_path = self.checkpoint_path.parent / 'bloom_checkpoint_model.pt'
        if model_path.exists():
            mck = torch.load(str(model_path), map_location='cpu', weights_only=False)
            self.vocab: List[str] = mck['vocab']
            print(f"[Stage2] Vocab from model checkpoint: {len(self.vocab)} chars",
                  flush=True)
        else:
            chars: set = set()
            for word in self._word_registry:
                chars.update(word)
            chars.update(' \n.,!?;:\'"()-abcdefghijklmnopqrstuvwxyz')
            self.vocab = sorted(chars)
            print(f"[Stage2] Vocab derived: {len(self.vocab)} chars", flush=True)

        # Per-token Collatz depths
        self._vocab_collatz_depths: List[int] = [
            _collatz_depth(ord(c)) for c in self.vocab]
        _cd_max = max(self._vocab_collatz_depths) if self._vocab_collatz_depths else 1

        # Normalised depths keyed by token index
        self._collatz_norm: List[float] = [
            d / max(_cd_max, 1) for d in self._vocab_collatz_depths]

        # Token -> face (inferred from word_registry, falls back to substrate math)
        self._token_to_face: List[int] = build_token_face_map(
            self._word_registry, self.vocab)

        print(f"[Stage2] Token->face map built for {len(self.vocab)} tokens", flush=True)

        # Sentence / corpus data (loaded on demand)
        self._sentence_velocities: Optional[Dict[str, dict]] = None
        self._corpus_vel: Optional[Tuple[float, float, float]] = None
        self._load_sentence_data()

    # ------------------------------------------------------------------
    # Internal: load sentence + corpus data via SubstrateTokenizer
    # ------------------------------------------------------------------

    def _load_sentence_data(self) -> None:
        corpus_path = self.checkpoint_path.parent / 'tinyshakespeare.txt'
        if not corpus_path.exists():
            print("[Stage2] No corpus file — sentence/corpus data unavailable",
                  flush=True)
            return
        try:
            sys.path.insert(0, str(self.checkpoint_path.parent))
            from substrate_tokenizer import SubstrateTokenizer
            with open(str(corpus_path), 'r') as fh:
                corpus_text = fh.read()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                tok = SubstrateTokenizer(corpus_text)
            self._sentence_velocities = tok.sentence_registry
            self._corpus_vel = tok.corpus_address
            print(f"[Stage2] Loaded {len(tok.sentence_registry)} sentence velocities; "
                  f"corpus_address={self._corpus_vel}", flush=True)
        except Exception as exc:
            print(f"[Stage2] Could not load SubstrateTokenizer: {exc}", flush=True)

    # ------------------------------------------------------------------
    # Population methods
    # ------------------------------------------------------------------

    def populate_bloom(
        self,
        bloom_vecs: List[torch.Tensor],
        bloom_crystal_ages: List[int],
        bloom_face_assignments: Optional[List[int]] = None,
    ) -> None:
        """Add all bloom crystal vectors to their face buckets."""
        if bloom_face_assignments is None:
            bloom_face_assignments = []
            for bv in bloom_vecs:
                bv3 = bv[:3].float()
                norm = bv3.norm()
                if norm > 1e-8:
                    bv3 = bv3 / norm
                sims = DODE_NORMALS @ bv3
                bloom_face_assignments.append(int(sims.argmax().item()))

        _cd_max = max(self._vocab_collatz_depths) if self._vocab_collatz_depths else 1

        for i, (bv, age, face) in enumerate(
                zip(bloom_vecs, bloom_crystal_ages, bloom_face_assignments)):
            # Depth: use crystal age as a proxy for Collatz resonance depth.
            # Normalise age to [0,1] against the observed max.
            depth = float(age) / max(max(bloom_crystal_ages), 1)

            # Velocity: first 3 dims of the bloom vector (already normalised)
            bv3 = bv[:3].float()
            norm = bv3.norm()
            if norm > 1e-8:
                bv3 = bv3 / norm
            vel = tuple(float(x) for x in bv3)

            addr = (face, depth, vel)
            entry = Entry(
                object_type='bloom',
                object_data={'vec': bv, 'age': age, 'index': i},
                address=addr,
                age=age,
            )
            self.table[face].append(entry)

    def populate_words(self, word_registry: dict) -> None:
        """Add each word to its face bucket.

        Face = face of the first character in the word's address tuple.
        Depth = average Collatz depth of character ordinals normalised to [0,1].
        Velocity = unit vector reached after reflecting through the word's face sequence.
        """
        # Pre-compute max Collatz depth across all vocab ordinals for normalisation
        all_ords = set()
        for word in word_registry:
            all_ords.update(ord(c) for c in word.lower())
        _max_cd = max((_collatz_depth(o) for o in all_ords), default=1)

        for word, data in word_registry.items():
            addr_seq = data.get('address', ())
            if not addr_seq:
                continue
            face = int(addr_seq[0])
            if not (0 <= face < 12):
                face = 0

            # Depth: normalised mean Collatz depth of the word's characters
            char_depths = [_collatz_depth(ord(c)) for c in word.lower() if c]
            depth = (sum(char_depths) / len(char_depths) / max(_max_cd, 1)
                     if char_depths else 0.0)

            # Velocity: simulate ball reflecting through each face in the address
            vx, vy, vz = DODE_NORMALS[0].tolist()
            for f in addr_seq:
                if not (0 <= f < 12):
                    continue
                nx, ny, nz = DODE_NORMALS[f].tolist()
                dot = vx * nx + vy * ny + vz * nz
                vx -= 2.0 * dot * nx
                vy -= 2.0 * dot * ny
                vz -= 2.0 * dot * nz
                mag = (vx**2 + vy**2 + vz**2) ** 0.5
                if mag > 1e-8:
                    vx /= mag
                    vy /= mag
                    vz /= mag
            vel = (vx, vy, vz)

            addr = (face, depth, vel)
            entry = Entry(
                object_type='word',
                object_data={'word': word, 'count': data.get('count', 0),
                             'cycle': data.get('cycle', -1),
                             'address_seq': addr_seq},
                address=addr,
                age=0,
            )
            self.table[face].append(entry)

    def populate_sentences(
        self,
        sentence_velocities: dict,
        corpus_vel: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """Add sentence beacons to the table.

        Face = argmax cosine(vel, dode_normals).
        Depth = 0.5 (sentences don't have a natural Collatz depth; use mid-range).
        Velocity = the sentence's 3D velocity.
        """
        for text, info in sentence_velocities.items():
            vx, vy, vz = info['velocity']
            face = vec_to_face(vx, vy, vz)
            vel = (vx, vy, vz)
            addr = (face, 0.5, vel)
            entry = Entry(
                object_type='sentence',
                object_data={'text': text, 'count': info.get('count', 0),
                             'velocity': vel},
                address=addr,
                age=0,
            )
            self.table[face].append(entry)

    def populate_corpus(
        self, corpus_vel: Tuple[float, float, float]
    ) -> None:
        """Add the single corpus attractor to the table."""
        vx, vy, vz = corpus_vel
        face = vec_to_face(vx, vy, vz)
        vel = (vx, vy, vz)
        addr = (face, 1.0, vel)   # depth=1.0: corpus is the deepest attractor
        entry = Entry(
            object_type='corpus',
            object_data={'velocity': vel},
            address=addr,
            age=0,
        )
        self.table[face].append(entry)

    def populate_collatz(
        self,
        vocab_collatz_depths: List[int],
        vocab: List[str],
    ) -> None:
        """Add per-token Collatz depths.

        Face = token's dode face (from _token_to_face).
        Depth = normalised Collatz depth.
        Velocity = DODE_NORMALS[face] (single-token has no trajectory beyond its face).
        """
        _max_cd = max(vocab_collatz_depths) if vocab_collatz_depths else 1
        for tok_id, (char, cd) in enumerate(zip(vocab, vocab_collatz_depths)):
            if tok_id >= len(self._token_to_face):
                continue
            face = self._token_to_face[tok_id]
            depth = cd / max(_max_cd, 1)
            vel = tuple(float(x) for x in DODE_NORMALS[face].tolist())
            addr = (face, depth, vel)
            entry = Entry(
                object_type='collatz',
                object_data={'char': char, 'token_id': tok_id,
                             'collatz_depth': cd, 'depth_norm': depth},
                address=addr,
                age=0,
            )
            self.table[face].append(entry)

    def populate_fib(
        self,
        fib_table_counts: torch.Tensor,
    ) -> None:
        """Add FibPosTable rows to the table.

        fib_table_counts: [10, 8] tensor (Fibonacci tiers x leaf buckets).
        Each of the 10 Fibonacci tiers is one entry.
        Face = tier * 12 // 10  (distributes across faces evenly).
        Depth = tier / 10.
        Velocity = DODE_NORMALS[face].
        """
        n_tiers = fib_table_counts.shape[0]
        for tier in range(n_tiers):
            face = (tier * 12) // n_tiers
            depth = tier / max(n_tiers - 1, 1)
            vel = tuple(float(x) for x in DODE_NORMALS[face].tolist())
            addr = (face, depth, vel)
            row_data = fib_table_counts[tier].tolist()
            entry = Entry(
                object_type='fib',
                object_data={'tier': tier, 'counts': row_data,
                             'total': float(fib_table_counts[tier].sum().item())},
                address=addr,
                age=0,
            )
            self.table[face].append(entry)

    # ------------------------------------------------------------------
    # Convenience: populate all object types at once
    # ------------------------------------------------------------------

    def populate_all(self) -> None:
        """Populate the table with all object types from the loaded checkpoint."""
        print("[Stage2] Populating bloom vectors ...", flush=True)
        self.populate_bloom(self._bloom_vecs, self._bloom_crystal_ages)
        print(f"[Stage2]   bloom: {len(self._bloom_vecs)} entries", flush=True)

        print("[Stage2] Populating words ...", flush=True)
        self.populate_words(self._word_registry)
        n_words = sum(1 for f in self.table.values()
                      for e in f if e.object_type == 'word')
        print(f"[Stage2]   words: {n_words} entries", flush=True)

        if self._sentence_velocities is not None:
            print("[Stage2] Populating sentences ...", flush=True)
            self.populate_sentences(self._sentence_velocities, self._corpus_vel)
            n_sents = sum(1 for f in self.table.values()
                          for e in f if e.object_type == 'sentence')
            print(f"[Stage2]   sentences: {n_sents} entries", flush=True)

        if self._corpus_vel is not None:
            print("[Stage2] Populating corpus attractor ...", flush=True)
            self.populate_corpus(self._corpus_vel)

        print("[Stage2] Populating Collatz depths ...", flush=True)
        self.populate_collatz(self._vocab_collatz_depths, self.vocab)
        print(f"[Stage2]   collatz: {len(self.vocab)} entries", flush=True)

        print("[Stage2] Populating Fibonacci position table ...", flush=True)
        self.populate_fib(self._fib_table_counts)

        total = sum(len(v) for v in self.table.values())
        print(f"[Stage2] Total entries in unified table: {total}", flush=True)

    # ------------------------------------------------------------------
    # Lookup methods
    # ------------------------------------------------------------------

    def lookup_face(self, face: int) -> List[Entry]:
        """Return ALL objects at this face."""
        return list(self.table.get(face, []))

    def lookup_face_typed(self, face: int, object_type: str) -> List[Entry]:
        """Return objects of a specific type at this face."""
        return [e for e in self.table.get(face, []) if e.object_type == object_type]

    def assign_address(
        self,
        object_type: str,
        object_data: dict,
        face: int,
        depth: float,
        velocity: tuple,
        age: int = 0,
    ) -> Entry:
        """Add any new object to the table.  Returns the created Entry."""
        addr = (face, depth, velocity)
        entry = Entry(object_type=object_type, object_data=object_data,
                      address=addr, age=age)
        self.table.setdefault(face, []).append(entry)
        return entry

    # ------------------------------------------------------------------
    # Analysis methods
    # ------------------------------------------------------------------

    def gap_map(self) -> Dict[int, Dict[str, set]]:
        """For each of 12 faces: what object types are present? What's missing?

        'missing' = object types that exist in the system but have 0 entries
                    at this face.

        Returns dict: face -> {'has': set[str], 'missing': set[str]}
        """
        result = {}
        for face in range(12):
            entries = self.table.get(face, [])
            has_types = {e.object_type for e in entries}
            # Determine which types are missing at this face
            # (only flag types that have at least one entry somewhere in the table)
            present_globally = {
                e.object_type
                for face_entries in self.table.values()
                for e in face_entries
            }
            missing = present_globally - has_types
            result[face] = {'has': has_types, 'missing': missing}
        return result

    def navigation_query(
        self, ball_vel: Tuple[float, float, float]
    ) -> Tuple[int, List[Entry], Optional[Entry]]:
        """Given a current ball velocity (R^3 unit vector):

        1. Find the nearest face (highest cosine similarity).
        2. Return all objects at that face.
        3. Return the nearest word entry in address space (highest cosine
           similarity between ball_vel and the word's velocity component).

        Returns: (nearest_face, objects_at_face, nearest_word_entry)
        """
        vx, vy, vz = ball_vel
        v = torch.tensor([vx, vy, vz], dtype=torch.float32)
        norm = v.norm()
        if norm > 1e-8:
            v = v / norm

        # 1. Nearest face
        sims = DODE_NORMALS @ v
        nearest_face = int(sims.argmax().item())

        # 2. All objects at nearest face
        objects_at_face = self.lookup_face(nearest_face)

        # 3. Nearest word: scan all word entries and pick max cosine sim
        best_sim = -2.0
        nearest_word: Optional[Entry] = None
        for face_entries in self.table.values():
            for entry in face_entries:
                if entry.object_type != 'word':
                    continue
                _, _, (wx, wy, wz) = entry.address
                wv = torch.tensor([wx, wy, wz], dtype=torch.float32)
                wn = wv.norm()
                if wn < 1e-8:
                    continue
                wv = wv / wn
                sim = float(v @ wv)
                if sim > best_sim:
                    best_sim = sim
                    nearest_word = entry

        return nearest_face, objects_at_face, nearest_word

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print total entries per face, types present, and gaps."""
        print()
        print("=" * 70)
        print("UNIFIED ADDRESS TABLE — SUMMARY")
        print("=" * 70)

        type_counts: Dict[str, int] = defaultdict(int)
        for face_entries in self.table.values():
            for e in face_entries:
                type_counts[e.object_type] += 1

        print(f"\nGlobal type counts:")
        for t in sorted(type_counts.keys()):
            print(f"  {t:<12s}: {type_counts[t]}")
        print(f"  {'TOTAL':<12s}: {sum(type_counts.values())}")

        print("\nPer-face breakdown:")
        header = f"  {'face':>4s}  {'total':>6s}  " + "  ".join(
            f"{t[:6]:>6s}" for t in sorted(ALL_OBJECT_TYPES))
        print(header)
        print("  " + "-" * (len(header) - 2))

        for face in range(12):
            entries = self.table.get(face, [])
            per_type = Counter(e.object_type for e in entries)
            cols = "  ".join(
                f"{per_type.get(t, 0):>6d}" for t in sorted(ALL_OBJECT_TYPES))
            print(f"  {face:>4d}  {len(entries):>6d}  {cols}")

        gap = self.gap_map()
        print("\nGap map (faces with missing object types):")
        any_gap = False
        for face in range(12):
            missing = gap[face]['missing']
            if missing:
                any_gap = True
                print(f"  face {face:2d}: MISSING {sorted(missing)}")
        if not any_gap:
            print("  (no gaps — all faces have all types)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    here = Path(__file__).parent
    ckpt = str(here / 'bloom_checkpoint.pt')

    print("=" * 70)
    print("Stage 2: Unified Address Space")
    print("=" * 70)

    # 1. Build table and populate all object types
    uat = UnifiedAddressTable(ckpt)
    uat.populate_all()

    # 2. Summary table
    uat.summary()

    # 3. Full gap map printout
    print("\n" + "=" * 70)
    print("GAP MAP (detailed, all 12 faces)")
    print("=" * 70)
    gap = uat.gap_map()
    for face in range(12):
        has = sorted(gap[face]['has'])
        missing = sorted(gap[face]['missing'])
        n_total = len(uat.lookup_face(face))
        print(f"\n  Face {face:2d}  ({n_total} entries)")
        print(f"    has    : {has}")
        print(f"    missing: {missing if missing else '(none)'}")

    # 4. Navigation query: what lives at the corpus's home face?
    print("\n" + "=" * 70)
    print("NAVIGATION QUERY: corpus velocity -> nearest face")
    print("=" * 70)
    if uat._corpus_vel is not None:
        corpus_vel = uat._corpus_vel
        print(f"\n  corpus_vel = {corpus_vel}")
        nearest_face, objs, nearest_word = uat.navigation_query(corpus_vel)
        print(f"  Nearest face: {nearest_face}")

        type_counts_at_face = Counter(e.object_type for e in objs)
        print(f"  Object types at face {nearest_face}:")
        for t, cnt in sorted(type_counts_at_face.items()):
            print(f"    {t:<12s}: {cnt}")

        if nearest_word is not None:
            wd = nearest_word.object_data
            print(f"\n  Nearest word in address space:")
            print(f"    word    : {wd.get('word', '?')!r}")
            print(f"    count   : {wd.get('count', '?')}")
            print(f"    addr    : {nearest_word.address[0]} "
                  f"(face), {nearest_word.address[1]:.4f} (depth)")
            print(f"    velocity: ({nearest_word.address[2][0]:.4f}, "
                  f"{nearest_word.address[2][1]:.4f}, "
                  f"{nearest_word.address[2][2]:.4f})")
        else:
            print("  (no word entries found)")
    else:
        print("  (no corpus velocity available)")

    # 5. Face with most bloom vecs AND most words — do they overlap?
    print("\n" + "=" * 70)
    print("FACE CONCENTRATION ANALYSIS")
    print("=" * 70)

    bloom_counts = {f: len(uat.lookup_face_typed(f, 'bloom')) for f in range(12)}
    word_counts = {f: len(uat.lookup_face_typed(f, 'word')) for f in range(12)}
    sent_counts = {f: len(uat.lookup_face_typed(f, 'sentence')) for f in range(12)}
    coll_counts = {f: len(uat.lookup_face_typed(f, 'collatz')) for f in range(12)}

    top_bloom_face = max(range(12), key=lambda f: bloom_counts[f])
    top_word_face = max(range(12), key=lambda f: word_counts[f])
    top_sent_face = max(range(12), key=lambda f: sent_counts[f])
    top_coll_face = max(range(12), key=lambda f: coll_counts[f])

    print(f"\n  Face with most bloom vecs : {top_bloom_face}  "
          f"({bloom_counts[top_bloom_face]} bloom entries)")
    print(f"  Face with most words      : {top_word_face}  "
          f"({word_counts[top_word_face]} word entries)")
    print(f"  Face with most sentences  : {top_sent_face}  "
          f"({sent_counts[top_sent_face]} sentence entries)")
    print(f"  Face with most collatz    : {top_coll_face}  "
          f"({coll_counts[top_coll_face]} collatz entries)")

    overlap_bloom_word = (top_bloom_face == top_word_face)
    overlap_bloom_sent = (top_bloom_face == top_sent_face)
    overlap_word_sent = (top_word_face == top_sent_face)

    print()
    print(f"  bloom peak == word peak   : {overlap_bloom_word}")
    print(f"  bloom peak == sent peak   : {overlap_bloom_sent}")
    print(f"  word  peak == sent peak   : {overlap_word_sent}")

    # Show bloom + word counts side by side for every face
    print("\n  All-face bloom vs word counts:")
    print(f"    {'face':>4s}  {'bloom':>6s}  {'words':>8s}  {'sents':>8s}  {'collatz':>7s}")
    for f in range(12):
        marker = " <-- max bloom" if f == top_bloom_face else ""
        marker2 = " <-- max word" if f == top_word_face else ""
        print(f"    {f:>4d}  {bloom_counts[f]:>6d}  "
              f"{word_counts[f]:>8d}  "
              f"{sent_counts[f]:>8d}  "
              f"{coll_counts[f]:>7d}"
              f"{marker}{marker2}")

    # Corpus face
    if uat._corpus_vel is not None:
        corpus_face = vec_to_face(*uat._corpus_vel)
        print(f"\n  Corpus attractor lives at face: {corpus_face}")
        print(f"    bloom entries there: {bloom_counts[corpus_face]}")
        print(f"    word  entries there: {word_counts[corpus_face]}")
        print(f"    sent  entries there: {sent_counts[corpus_face]}")


if __name__ == '__main__':
    main()

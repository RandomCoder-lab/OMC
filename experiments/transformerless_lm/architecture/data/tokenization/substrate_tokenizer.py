"""Substrate tokenizer v4 — dodecahedral character navigation.

The vocabulary is CHARACTERS ONLY. No words extracted from corpus.
The corpus teaches what paths THROUGH the dodecahedron are meaningful,
not which words exist.

The 12 faces of the dodecahedron are MATHEMATICAL ATTRACTORS defined
purely by the properties of each character's ordinal value:

    Collatz depth C(ord)  — wave-collapse depth of the character
    Zeta weight |ζ(σ+it)| — prime resonance at this ordinal position
    Fibonacci distance    — how close ord is to a Fibonacci number
    Zeckendorf complexity — how many Fibonacci terms sum to ord

These properties exist before any corpus. They are the same for
every language, every text, every corpus. The faces are universal.

When the model generates 'p','o','w','e','r', it is tracing a
specific path through these 12 mathematical attractors. The path
is the word. The corpus reinforces certain paths. Different corpora
reinforce different paths — but through the SAME dodecahedron.

Navigation is not recognition. The model doesn't look up 'power' —
it follows the mathematical trajectory until it reaches the face
that 'r' lives on. The path itself IS the meaning.

Words are emergent. Faces are eternal.
"""

import math
import re
from typing import List

PHI   = (1.0 + 5.0 ** 0.5) / 2.0
_FIBS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]


# ── Mathematical primitives ────────────────────────────────────────────────

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


def _build_zeta_table(max_s: int, n_terms: int = 150,
                      sigma: float = 0.75) -> list:
    """Re(ζ(σ+it)) for t linearly mapped to [14, 50] — Spell XIII's range."""
    ln_ns = [math.log(k) for k in range(1, n_terms + 1)]
    wt    = [k ** (-sigma) for k in range(1, n_terms + 1)]
    table = []
    for s in range(max_s + 1):
        t   = 14.0 + (50.0 - 14.0) * s / max(max_s, 1)
        val = sum(w * math.cos(t * lk) for w, lk in zip(wt, ln_ns))
        table.append(val)
    return table


def _char_resonance(ordinal: int, zeta_table: list,
                    coll_max: int, max_ord: int) -> float:
    """
    Corpus-independent mathematical resonance of a single character
    by its Unicode ordinal.  Used only for face assignment.
    """
    n = max(ordinal, 1)

    # 1. Collatz depth
    coll_norm = min(_collatz_depth(n) / max(coll_max, 1), 1.0)

    # 2. Zeta at this ordinal
    idx       = min(n, max_ord)
    zeta_norm = min(abs(zeta_table[idx]) / 10.0, 1.0)

    # 3. Fibonacci proximity: e^{-dist²/φ} where dist = min distance to Fib number
    nearest_fib = min(_FIBS, key=lambda f: abs(f - n))
    fib_prox    = PHI ** (-(n - nearest_fib) ** 2 / PHI)

    # 4. Zeckendorf complexity (fewer terms = more resonant)
    zeck_score  = 1.0 / _zeckendorf_complexity(n)

    return (PHI ** -1 * coll_norm  +
            PHI ** -2 * zeta_norm  +
            PHI ** -3 * fib_prox   +
            PHI ** -4 * zeck_score)


# ── Nested Platonic solid normals ─────────────────────────────────────────
# Five Platonic solids = five linguistic scales.  Each solid nests inside
# the next.  Navigation at the inner solid graduates to the outer solid at
# each structural boundary (char → word → sentence → paragraph → discourse).
#
# Tetrahedron (4 faces)  — character primitives        inner
# Cube        (6 faces)  — syllable / word              ↕
# Dodecahedron(12 faces) — clause / sentence            ↕  (current outer)
#
# Dual relationships already encoded:
#   tetra inscribes in cube  (4 of 8 cube vertices)
#   cube  inscribes in dode  (8 of 20 dode vertices)
#   dode  dual = icosahedron (for future discourse level)

def build_tetra_normals() -> List[List[float]]:
    """4 face normals of regular tetrahedron — alternate vertices of unit cube."""
    m = 3.0 ** 0.5
    return [[ 1/m,  1/m,  1/m],
            [ 1/m, -1/m, -1/m],
            [-1/m,  1/m, -1/m],
            [-1/m, -1/m,  1/m]]

def build_cube_normals() -> List[List[float]]:
    """6 face normals of cube — axis-aligned unit vectors."""
    return [[ 1., 0., 0.], [-1., 0., 0.],
            [ 0., 1., 0.], [ 0.,-1., 0.],
            [ 0., 0., 1.], [ 0., 0.,-1.]]

def build_dode_normals() -> List[List[float]]:
    """12 face normals of regular dodecahedron — icosahedron vertices."""
    p = PHI
    m = (2.0 + p) ** 0.5
    raw = [
        ( 0,  1,  p), ( 0,  1, -p), ( 0, -1,  p), ( 0, -1, -p),
        ( 1,  p,  0), ( 1, -p,  0), (-1,  p,  0), (-1, -p,  0),
        ( p,  0,  1), ( p,  0, -1), (-p,  0,  1), (-p,  0, -1),
    ]
    return [[x/m, y/m, z/m] for x, y, z in raw]


TETRA_NORMALS = build_tetra_normals()  #  4 × 3, character level
CUBE_NORMALS  = build_cube_normals()   #  6 × 3, word level
DODE_NORMALS  = build_dode_normals()   # 12 × 3, sentence level


# ── Tokenizer ──────────────────────────────────────────────────────────────

def _word_split(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+|\d+|[^\w\s]|\s", text)


class SubstrateTokenizer:
    """
    Character-level tokenizer with dodecahedral face navigation.

    Vocabulary = every character that appears in the corpus (universal atoms).
    No word extraction. No corpus-dependent vocabulary selection.

    Each character is assigned to one of 12 dodecahedral faces based on
    the mathematical resonance of its Unicode ordinal — a corpus-independent
    classification that is identical across any language or text.

    self.cube_faces[tok_id] ∈ {0, …, 11}   — face for each token
    self.cube_dim = 12                       — the dodecahedron
    self.cube_delta = 7                      — Hamiltonian rotation

    The model navigates the faces. Words are emergent trajectories.
    """

    CUBE_DIM   = 12
    CUBE_DELTA = 7    # prime ≈ D/2 → Hamiltonian cycle through all 12 faces

    def __init__(self, corpus_text: str,
                 face_budget: int = 1000,   # kept for API compatibility, unused
                 min_freq: int = 1):        # kept for API compatibility, unused

        # ── 1. Character vocabulary — the universal atoms ──────────────────
        chars      = sorted(set(corpus_text))
        self.vocab = chars

        # ── 2. Mathematical tables over ordinal range ──────────────────────
        ords      = [ord(c) for c in chars]
        max_ord   = max(ords, default=130)
        coll_max  = max(_collatz_depth(o) for o in ords) if ords else 200
        zeta_tab  = _build_zeta_table(max_ord, n_terms=150, sigma=0.75)

        # ── 3. Score every character by ordinal resonance ──────────────────
        scored = sorted(
            ((c, _char_resonance(ord(c), zeta_tab, coll_max, max_ord))
             for c in chars),
            key=lambda x: x[1],
            reverse=True)

        # ── 4. Assign each character to a face (rank-band → face 0–11) ─────
        n_chars     = len(scored)
        face_assign = {c: idx * self.CUBE_DIM // n_chars
                       for idx, (c, _) in enumerate(scored)}

        # ── 5. Assemble vocabulary ─────────────────────────────────────────
        self.token_to_id   = {c: i for i, c in enumerate(self.vocab)}
        self.vocab_size    = len(self.vocab)
        self.multichar_tokens  = set()     # no word tokens in v4
        self.lengths_desc      = []        # no multi-char tokens

        # ── 6. Face tensors for all three nested solids ───────────────────
        # Dodecahedron (12): primary face assignment from resonance ranking
        self.cube_faces: List[int] = [
            face_assign.get(c, 0) for c in self.vocab]
        self.cube_dim   = self.CUBE_DIM
        self.cube_delta = self.CUBE_DELTA
        # Cube (6): dode_face // 2  →  6 groups (pairs of adjacent dode faces)
        self.cube6_faces: List[int] = [f // 2 for f in self.cube_faces]
        # Tetrahedron (4): dode_face // 3  →  4 groups (trios of adjacent dode faces)
        self.tetra_faces: List[int] = [f // 3 for f in self.cube_faces]
        # Expose solid normals for use in generation
        self.tetra_normals = TETRA_NORMALS   #  4 × 3
        self.cube_normals  = CUBE_NORMALS    #  6 × 3
        self.dode_normals  = DODE_NORMALS    # 12 × 3
        # Boundary token ids (word / sentence) — set after vocab is built
        self.word_end_ids: set = set()
        self.sent_end_ids: set = set()
        for c in ' \t':
            if c in self.token_to_id: self.word_end_ids.add(self.token_to_id[c])
        for c in ".!?\n":
            if c in self.token_to_id:
                self.sent_end_ids.add(self.token_to_id[c])
                self.word_end_ids.add(self.token_to_id[c])
        for c in "',;:-":
            if c in self.token_to_id: self.word_end_ids.add(self.token_to_id[c])

        # ── 7. North Star: char with deepest Collatz depth of ordinal ──────
        # In char-level vocab, the North Star is the most Collatz-resonant char.
        collatz_by_tok = [_collatz_depth(ord(c)) for c in self.vocab]
        ns_idx   = max(range(len(collatz_by_tok)), key=lambda i: collatz_by_tok[i])
        self.north_star_token = ns_idx
        ns_char  = self.vocab[ns_idx]
        ns_face  = self.cube_faces[ns_idx]

        # Diagnostics
        print(f"  [SubstrateTok v4] vocab={self.vocab_size} chars"
              f"  (ZERO corpus word extraction)", flush=True)
        print(f"  [SubstrateTok v4] cube_dim={self.cube_dim}"
              f"  delta={self.cube_delta}"
              f"  rotation=[0,7,2,9,4,11,6,1,8,3,10,5]", flush=True)
        print(f"  [SubstrateTok v4] North Star = "
              f"tok{ns_idx}='{ns_char}' (ord={ord(ns_char)}"
              f"  C={collatz_by_tok[ns_idx]}  face={ns_face})", flush=True)

        # ── 8. Word address registry — pre-populated from corpus ──────────
        # Every word in the corpus already has a dodecahedral address:
        # address = tuple of face indices for each character.
        # The address is a pure mathematical fact (depends only on char
        # ordinals, not corpus statistics) — no need to wait for bloom.
        # We compute all addresses at init so the sphere knows every valid
        # path from cycle 1.
        self.word_registry: dict = {}
        self._addr_trie:    dict = {}
        _corpus_words = re.findall(r'[a-zA-Z]+', corpus_text)
        for _w in _corpus_words:
            _wl = _w.lower()
            if len(_wl) < 2:
                continue
            if _wl in self.word_registry:
                self.word_registry[_wl]['count'] += 1
            else:
                _addr = tuple(
                    self.cube_faces[self.token_to_id[c]]
                    for c in _wl if c in self.token_to_id)
                self.word_registry[_wl] = {
                    'address': _addr, 'count': 1, 'cycle': -1}
                _node = self._addr_trie
                for _f in _addr:
                    if _f not in _node:
                        _node[_f] = {}
                    _node = _node[_f]
                _node['_word'] = _wl
        print(f"  [SubstrateTok v4] word registry pre-populated:"
              f" {len(self.word_registry)} unique words → {len(_corpus_words)} tokens",
              flush=True)

        # ── 9. Hierarchical velocity registry ─────────────────────────────
        # Each block level gets a 3D velocity = sphere state after reflecting
        # through all its characters. Velocity IS the address — compact (3
        # floats), corpus-independent in structure, corpus-informed in content.
        #
        #   char      → single face  (already in cube_faces)
        #   word      → face tuple   (already in word_registry)
        #   sentence  → 3D velocity  (new — one per unique line)
        #   paragraph → 3D velocity  (new — one per speaker turn / stanza)
        #   corpus    → 3D velocity  (new — single attractor for whole text)
        #
        # Generation uses these as navigation beacons: if the ball is near
        # a registered sentence velocity, it steers toward that structure.

        # Sentence registry: each unique line of the corpus
        # Shakespeare is naturally line-structured; a line = a sentence unit.
        self.sentence_registry: dict = {}   # text → {'velocity':(x,y,z),'count':N}
        for _sline in re.split(r'\n', corpus_text):
            _sline = _sline.strip()
            if len(_sline) < 3:
                continue
            if _sline in self.sentence_registry:
                self.sentence_registry[_sline]['count'] += 1
            else:
                self.sentence_registry[_sline] = {
                    'velocity': self._velocity(_sline), 'count': 1}

        # Paragraph registry: consecutive non-blank lines as speaker turns
        self.paragraph_registry: dict = {}  # key → {'velocity':(x,y,z),'count':N,'text':str}
        _para_buf: List[str] = []
        for _pline in corpus_text.split('\n'):
            if _pline.strip():
                _para_buf.append(_pline)
            else:
                if len(_para_buf) >= 2:
                    _ptxt = '\n'.join(_para_buf)
                    _pkey = _ptxt[:80]
                    if _pkey in self.paragraph_registry:
                        self.paragraph_registry[_pkey]['count'] += 1
                    else:
                        self.paragraph_registry[_pkey] = {
                            'velocity': self._velocity(_ptxt),
                            'count': 1, 'text': _ptxt[:200]}
                _para_buf = []
        if len(_para_buf) >= 2:
            _ptxt = '\n'.join(_para_buf)
            _pkey = _ptxt[:80]
            if _pkey not in self.paragraph_registry:
                self.paragraph_registry[_pkey] = {
                    'velocity': self._velocity(_ptxt),
                    'count': 1, 'text': _ptxt[:200]}

        # Corpus address: sphere's resting point after the entire corpus.
        # Every generated sequence should tend toward this attractor.
        self.corpus_address: tuple = self._velocity(corpus_text)

        print(f"  [SubstrateTok v4] sentences={len(self.sentence_registry)}"
              f"  paragraphs={len(self.paragraph_registry)}"
              f"  corpus_address=({self.corpus_address[0]:.3f},"
              f"{self.corpus_address[1]:.3f},{self.corpus_address[2]:.3f})",
              flush=True)

        # Show face assignments
        from collections import defaultdict
        face_chars = defaultdict(list)
        for c, f in zip(self.vocab, self.cube_faces):
            face_chars[f].append(repr(c))
        for f in range(self.cube_dim):
            print(f"  [SubstrateTok v4] face {f:2d}: {face_chars[f]}", flush=True)

    # ── Velocity primitive ────────────────────────────────────────────────
    def _velocity(self, text: str) -> tuple:
        """Sphere velocity (3-tuple) after reflecting off every char's face."""
        vx, vy, vz = DODE_NORMALS[0]
        for c in text:
            if c not in self.token_to_id:
                continue
            nx, ny, nz = DODE_NORMALS[self.cube_faces[self.token_to_id[c]]]
            dot = vx*nx + vy*ny + vz*nz
            vx -= 2.0*dot*nx;  vy -= 2.0*dot*ny;  vz -= 2.0*dot*nz
            mag = (vx*vx + vy*vy + vz*vz) ** 0.5
            if mag > 1e-8:
                vx /= mag;  vy /= mag;  vz /= mag
        return (vx, vy, vz)

    def nearest_sentence(self, velocity: tuple, threshold: float = 0.92) -> str:
        """Return the sentence whose address is closest to velocity, or ''."""
        best_sim, best_text = -1.0, ''
        vx, vy, vz = velocity
        for text, entry in self.sentence_registry.items():
            sx, sy, sz = entry['velocity']
            sim = vx*sx + vy*sy + vz*sz
            if sim > best_sim:
                best_sim, best_text = sim, text
        return best_text if best_sim >= threshold else ''

    def nearest_paragraph(self, velocity: tuple, threshold: float = 0.92) -> str:
        """Return the paragraph text whose address is closest to velocity, or ''."""
        best_sim, best_text = -1.0, ''
        vx, vy, vz = velocity
        for key, entry in self.paragraph_registry.items():
            sx, sy, sz = entry['velocity']
            sim = vx*sx + vy*sy + vz*sz
            if sim > best_sim:
                best_sim, best_text = sim, entry.get('text', key)
        return best_text if best_sim >= threshold else ''

    # ── Word Address Registry ──────────────────────────────────────────────
    # All corpus words are pre-registered at init time (cycle=-1).
    # register_bloom_text() increments counts on confirmed bloom words
    # and adds any words not present in the original corpus (cross-corpus).
    # The address is corpus-independent: 'power' always traces the same
    # face sequence regardless of which corpus it came from.

    def word_address(self, word: str) -> tuple:
        """Compute the dodecahedral address (face sequence) for a word."""
        return tuple(
            self.cube_faces[self.token_to_id[c]]
            for c in word.lower()
            if c in self.token_to_id)

    def register_bloom_text(self, text: str, cycle: int = 0) -> List[str]:
        """
        Extract words from a confirmed bloom sequence.
        For each new word, compute its face address and log it.
        Returns list of newly discovered words.
        """
        new_words = []
        for word in re.findall(r'[a-zA-Z]+', text):
            w = word.lower()
            if len(w) < 2:
                continue
            if w not in self.word_registry:
                addr = self.word_address(w)
                self.word_registry[w] = {
                    'address': addr,
                    'count':   1,
                    'cycle':   cycle,
                }
                # Extend address trie for fast prefix lookup
                node = self._addr_trie
                for f in addr:
                    if f not in node:
                        node[f] = {}
                    node = node[f]
                node['_word'] = w
                new_words.append(w)
            else:
                self.word_registry[w]['count'] += 1
        return new_words

    def address_lookup(self, face_sequence: list) -> str:
        """
        Given a sequence of face indices already navigated, return the
        registered word if the full address matches, else '' (still navigating).
        """
        node = self._addr_trie
        for f in face_sequence:
            if f not in node:
                return ''
            node = node[f]
        return node.get('_word', '')

    def address_prefix_alive(self, face_sequence: list) -> bool:
        """True if the face sequence is a valid prefix of any registered address."""
        node = self._addr_trie
        for f in face_sequence:
            if f not in node:
                return False
            node = node[f]
        return True

    # ── Encode / Decode (char-level) ───────────────────────────────────────

    def encode(self, text: str) -> List[int]:
        return [self.token_to_id.get(c, self.token_to_id.get(' ', 0))
                for c in text]

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.vocab[int(i)] for i in ids)

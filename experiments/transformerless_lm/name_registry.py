"""Name→address registry for OMC function definitions.

Each function is stored at the address of its NAME fingerprint, not its body fingerprint.
This makes φ("fibonacci") navigate directly to fn fibonacci.

Address assignment — all three levels are pure φ(name):
  face, sub_face, zeck = assign_address(substrate_fingerprint(name))

No template, no word list, no hand-coded face→type table. The dodecahedral
address simply IS the golden-ratio fingerprint of the identity string, so the
registry is identical math on any corpus or language. Faces carry no imposed
"type" meaning — two names share a face iff their φ-hashes land in the same
dodecahedral region.
"""
from __future__ import annotations

import re
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_HERE = Path(__file__).parent


# ── Face assignment: pure φ(name), zero dictionary ────────────────────────────
# The face is wherever the golden-ratio fingerprint of the name lands on the
# dodecahedron — assign_address(φ(name)).face.  No template table, no anchor
# word list, no hand-coded category→face map.  "fibonacci" → face 10 is then a
# true statement about the substrate, not a label imposed from outside.
#
# Consequence (accepted): faces carry no human-readable "type" meaning; they are
# pure φ-addresses of the identity string.  This is maximally corpus-agnostic —
# identical math on any corpus, any language.  Two functions share a face iff
# their names' φ-hashes land in the same dodecahedral region, nothing more.


# ── Uniform φ-address (equal-area faces) ──────────────────────────────────────
# The shared substrate_fingerprint maps a string to vec[:3]=(sin θ0,cos θ0,sin θ1),
# which is NOT uniform on S² → skewed face occupancy (χ²≈216 on the registry).
# Here we map two fmix32-finalized golden-ratio hashes to a UNIFORM sphere point
# (inverse-CDF) and read the face off the canonical icosahedral normals. Measured
# χ²≈17 (p≈0.11) — statistically uniform. Registry-local: it does NOT touch the
# shared fingerprint, so index/navigator retrieval (which use sin/cos cosine
# vectors) are unaffected. Global migration of the shared fingerprint = step A.1c.
import math as _math

_PHI_INT  = 2654435769          # ⌊φ·2³²⌋
_FMIX_C1  = 0x85ebca6b
_FMIX_C2  = 0xc2b2ae35
_MOD32    = 0x100000000

def _lcg(text: str, seed_i: int) -> int:
    h = (seed_i + 1) * _PHI_INT & 0xFFFFFFFF
    for c in text:
        h = (h * _PHI_INT + ord(c)) & 0xFFFFFFFF
    return h

def _fmix32(h: int) -> int:
    """murmur3 avalanche finalizer — decorrelates the LCG output."""
    h &= 0xFFFFFFFF
    h ^= h >> 16; h = (h * _FMIX_C1) & 0xFFFFFFFF
    h ^= h >> 13; h = (h * _FMIX_C2) & 0xFFFFFFFF
    h ^= h >> 16
    return h

def uniform_haddr(text: str, d_model: int = 64, fp=None):
    """Equal-area dodecahedral address for `text` (face/sub_face/zeck)."""
    from addressed_memory import _NORMALS_STACK, _NEIGHBORS
    from hierarchical_address import zeckendorf, HAddr, _MAG_SCALE
    from corpus_address_index import substrate_fingerprint
    u0 = _fmix32(_lcg(text, 0)) / _MOD32
    u1 = _fmix32(_lcg(text, 1)) / _MOD32
    z  = 2.0 * u0 - 1.0
    th = 2.0 * _math.pi * u1
    r  = _math.sqrt(max(0.0, 1.0 - z * z))
    v3 = torch.tensor([r * _math.cos(th), r * _math.sin(th), z])
    face = int((_NORMALS_STACK @ v3).argmax())
    nbrs = _NEIGHBORS[face]
    sub  = int(max(range(len(nbrs)), key=lambda k: float(_NORMALS_STACK[nbrs[k]] @ v3)))
    if fp is None:
        fp = substrate_fingerprint(text, d_model)
    zeck = zeckendorf(round(fp.float().norm().item() * _MAG_SCALE))
    return HAddr(face=face, sub_face=sub, zeck=zeck)


# ── Parser ────────────────────────────────────────────────────────────────────

def parse_fndefs(text: str) -> List[Tuple[str, str]]:
    """Parse omc_fndefs.txt into (name, full_code) pairs.

    Handles both one-liner and multiline function definitions by
    tracking brace depth.
    """
    results: List[Tuple[str, str]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'^(fn\s+(\w+)\s*\()', line)
        if m:
            name = m.group(2)
            buf = [line]
            depth = line.count('{') - line.count('}')
            i += 1
            while i < len(lines) and depth > 0:
                buf.append(lines[i])
                depth += lines[i].count('{') - lines[i].count('}')
                i += 1
            results.append((name, '\n'.join(buf)))
        else:
            i += 1
    return results


# ── Registry builder ──────────────────────────────────────────────────────────

def build_name_registry(
    fndefs_path: Path,
    d_model: int = 64,
    verbose: bool = True,
) -> Dict[str, dict]:
    """Build name→{addr, code, fp} registry.

    For each fn name:
      1. fp   = substrate_fingerprint(name, d_model)   — golden-ratio φ-hash
      2. addr = assign_address(fp)                      — face/sub_face/zeck, all φ
      3. Store: registry[name] = {addr, code, fp, name}

    Pure substrate: no template classification, no word list, no face table.
    Returns registry dict keyed by lowercase function name.
    """
    from corpus_address_index import substrate_fingerprint

    text = fndefs_path.read_text(errors='replace')
    fns = parse_fndefs(text)
    if verbose:
        print(f"[NameRegistry] parsed {len(fns)} functions from {fndefs_path.name}")

    registry: Dict[str, dict] = {}
    face_counts: Dict[int, int] = {}

    for name, code in fns:
        name_lc = name.lower()

        # Address is φ(name), all three levels — face, sub_face, zeck.
        # No template, no word list, no hand-coded face. The dodecahedral
        # address IS the golden-ratio fingerprint of the identity string.
        fp = substrate_fingerprint(name_lc, d_model)
        addr = uniform_haddr(name_lc, d_model, fp=fp)

        registry[name_lc] = {
            'addr': addr,
            'code': code,
            'fp':   fp,
            'name': name,
        }
        face_counts[addr.face] = face_counts.get(addr.face, 0) + 1

    if verbose:
        print(f"[NameRegistry] {len(registry)} entries across faces:")
        for face in sorted(face_counts):
            bar = '#' * min(face_counts[face] // 2, 40)
            print(f"  face {face:2d}: {face_counts[face]:4d}  {bar}")

    return registry


def save_registry(registry: Dict[str, dict], path: Path) -> None:
    torch.save(registry, str(path))
    print(f"[NameRegistry] saved → {path}  ({len(registry)} entries)")


def load_registry(path: Path) -> Dict[str, dict]:
    reg = torch.load(str(path), map_location='cpu', weights_only=False)
    print(f"[NameRegistry] loaded {len(reg)} entries from {path.name}")
    return reg


def load_or_build_registry(
    cache_path: Path,
    fndefs_path: Optional[Path] = None,
    d_model: int = 64,
    rebuild: bool = False,
) -> Dict[str, dict]:
    """Load cached registry or build from fndefs and save."""
    if cache_path.exists() and not rebuild:
        return load_registry(cache_path)
    fndefs_path = fndefs_path or (_HERE / 'omc_fndefs.txt')
    reg = build_name_registry(fndefs_path, d_model=d_model)
    save_registry(reg, cache_path)
    return reg


# ── Lookup ────────────────────────────────────────────────────────────────────

def lookup(registry: Dict[str, dict], query: str) -> Optional[dict]:
    """O(1) exact-name lookup. Returns registry entry or None."""
    return registry.get(query.lower().strip())


def lookup_terms(registry: Dict[str, dict], query: str) -> Optional[dict]:
    """Try each meaningful term in query as a name. Returns first hit.

    Underscore-insensitive: "quicksort" matches "quick_sort" and vice-versa.
    This is a universal string normalization (collapse separators), not a
    word list — it transfers to any corpus.
    """
    def _norm(s: str) -> str:
        return s.replace('_', '')

    # Build an underscore-collapsed view of the registry once per call.
    # registry keys are stable; the collapsed map lets "quicksort" find
    # "quick_sort" without any per-name special-casing.
    collapsed = {_norm(k): v for k, v in registry.items()}

    def _get(key: str) -> Optional[dict]:
        return registry.get(key) or collapsed.get(_norm(key))

    tokens = re.split(r'\W+', query.lower())
    tokens = [t for t in tokens if len(t) > 2]
    # Try compound names first (merge_sort before merge or sort)
    for i in range(len(tokens) - 1):
        for sep in ('_', ''):
            hit = _get(tokens[i] + sep + tokens[i + 1])
            if hit:
                return hit
            hit = _get(tokens[i + 1] + sep + tokens[i])
            if hit:
                return hit
    # Then single terms
    for t in tokens:
        hit = _get(t)
        if hit:
            return hit
    return None


def retrieve_by_address(
    registry: Dict[str, dict],
    face: int,
    sub_face: Optional[int] = None,
    top_k: int = 5,
) -> List[dict]:
    """Retrieve all functions at a given face (and optionally sub_face)."""
    results = []
    for entry in registry.values():
        if entry['addr'].face == face:
            if sub_face is None or entry['addr'].sub_face == sub_face:
                results.append(entry)
    return results[:top_k]


def face_summary(registry: Dict[str, dict]) -> str:
    """φ-face → functions mapping. Faces are pure φ(name) addresses; the
    grouping is whatever the golden-ratio hash produces, with no imposed
    type labels."""
    from collections import defaultdict
    by_face: Dict[int, List[str]] = defaultdict(list)
    for entry in registry.values():
        by_face[entry['addr'].face].append(entry['name'])
    lines = []
    for face in sorted(by_face):
        names = ', '.join(sorted(by_face[face])[:8])
        suffix = '...' if len(by_face[face]) > 8 else ''
        lines.append(f"  face {face:2d}  ({len(by_face[face]):4d} fns): {names}{suffix}")
    return '\n'.join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse, sys
    parser = argparse.ArgumentParser(description='Build/query name→address registry')
    parser.add_argument('--build',   action='store_true', help='Force rebuild')
    parser.add_argument('--query',   type=str, default=None, help='Lookup a name')
    parser.add_argument('--face',    type=int, default=None, help='List functions on face')
    parser.add_argument('--summary', action='store_true', help='Print face summary')
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--out',     type=str, default='omc_name_registry.pt')
    args = parser.parse_args()

    cache = _HERE / args.out
    fndefs = _HERE / 'omc_fndefs.txt'

    reg = load_or_build_registry(cache, fndefs, d_model=args.d_model, rebuild=args.build)

    if args.summary:
        print('\nFace → Function mapping:')
        print(face_summary(reg))

    if args.query:
        hit = lookup_terms(reg, args.query)
        if hit:
            print(f'\nQuery: {args.query!r}')
            print(f'  name:     {hit["name"]}')
            print(f'  addr:     {hit["addr"]}')
            print(f'  code:\n{hit["code"]}')
        else:
            print(f'Not found: {args.query!r}')
            sys.exit(1)

    if args.face is not None:
        hits = retrieve_by_address(reg, args.face)
        print(f'\nFunctions at face {args.face}:')
        for h in hits:
            print(f'  {h["name"]:30s}  addr={h["addr"]}')

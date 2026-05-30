"""address_transfer.py — cross-run knowledge transfer for the transformerless LM.

Merges bloom memories and resonance maps from different training runs,
compares address coverage, and serializes/deserializes both map types.
"""

import json


def merge_resonance_dicts(d1: dict, d2: dict) -> dict:
    """Merge two to_dict() outputs. Combine counts_by_address additively.
    Concatenate deep_resonances (deduplicating by addr+bloom+sentence).
    Returns a merged dict in the same format."""
    merged_counts: dict = {}
    for addr, type_counts in d1.get('counts_by_address', {}).items():
        merged_counts[addr] = dict(type_counts)
    for addr, type_counts in d2.get('counts_by_address', {}).items():
        if addr not in merged_counts:
            merged_counts[addr] = dict(type_counts)
        else:
            for type_, count in type_counts.items():
                merged_counts[addr][type_] = merged_counts[addr].get(type_, 0) + count

    seen: set = set()
    merged_resonances: list = []
    for ev in list(d1.get('deep_resonances', [])) + list(d2.get('deep_resonances', [])):
        # Use type-aware third element so bloom_token events don't KeyError on 'sentence'
        ev_type = ev.get('type', 'bloom_sentence')
        third = ev.get('token') if ev_type == 'bloom_token' else ev.get('sentence')
        sig = (ev['addr'], ev['bloom'], third)
        if sig not in seen:
            seen.add(sig)
            merged_resonances.append(ev)

    return {
        'counts_by_address': merged_counts,
        'n_deep_resonances': len(merged_resonances),
        'deep_resonances': merged_resonances,
    }


def coverage_diff(d1: dict, d2: dict) -> dict:
    """Compare two to_dict() outputs.
    Returns {
      'only_in_d1': [addr_str, ...],
      'only_in_d2': [addr_str, ...],
      'shared':     [addr_str, ...],
      'n_d1': int, 'n_d2': int, 'n_shared': int
    }
    """
    addrs1 = set(d1.get('counts_by_address', {}).keys())
    addrs2 = set(d2.get('counts_by_address', {}).keys())
    only1 = sorted(addrs1 - addrs2)
    only2 = sorted(addrs2 - addrs1)
    shared = sorted(addrs1 & addrs2)
    return {
        'only_in_d1': only1,
        'only_in_d2': only2,
        'shared':     shared,
        'n_d1':       len(addrs1),
        'n_d2':       len(addrs2),
        'n_shared':   len(shared),
    }


def save_resonance_map(rmap, path: str) -> None:
    """Save CrossTypeResonanceMap to path as JSON.
    Uses rmap.to_dict() — vectors not saved (expected)."""
    with open(path, 'w') as f:
        json.dump(rmap.to_dict(), f)


def load_resonance_map(path: str) -> 'CrossTypeResonanceMap':
    """Load from JSON. Returns a CrossTypeResonanceMap via from_dict().
    Requires resonance_map.CrossTypeResonanceMap.from_dict() to exist."""
    from resonance_map import CrossTypeResonanceMap
    with open(path, 'r') as f:
        d = json.load(f)
    return CrossTypeResonanceMap.from_dict(d)


def save_coverage(coverage, path: str) -> None:
    """Save AddressCoverage to path as JSON."""
    with open(path, 'w') as f:
        json.dump(coverage.to_dict(), f)


def load_coverage(path: str) -> 'AddressCoverage':
    """Load AddressCoverage from JSON. Requires AddressCoverage.from_dict()."""
    from address_coverage import AddressCoverage
    with open(path, 'r') as f:
        d = json.load(f)
    return AddressCoverage.from_dict(d)


def transfer_report(d1: dict, d2: dict, name1: str = 'run_A', name2: str = 'run_B') -> str:
    """Print and return a concise cross-run comparison report."""
    diff = coverage_diff(d1, d2)
    n1 = diff['n_d1']
    n2 = diff['n_d2']
    n_shared = diff['n_shared']
    n_only1 = len(diff['only_in_d1'])
    n_only2 = len(diff['only_in_d2'])

    res1 = d1.get('n_deep_resonances', 0)
    res2 = d2.get('n_deep_resonances', 0)
    sigs1 = {(e['addr'], e['bloom'], e['sentence']) for e in d1.get('deep_resonances', [])}
    n_new_res = sum(
        1 for e in d2.get('deep_resonances', [])
        if (e['addr'], e['bloom'], e['sentence']) not in sigs1
    )

    lines = [
        f"[TRANSFER] {name1}: {n1} addresses  {name2}: {n2} addresses",
        f"  shared: {n_shared}  only_{name1}: {n_only1}  only_{name2}: {n_only2}",
        f"  new knowledge in {name2}: {n_only2} addresses not in {name1}",
        f"  resonances: {name1}={res1}  {name2}={res2}  ({name2} found {n_new_res} new)",
    ]
    result = '\n'.join(lines)
    print(result, flush=True)
    return result

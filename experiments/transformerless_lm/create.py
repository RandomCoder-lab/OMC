#!/usr/bin/env python3
"""create.py — THE CREATION ENGINE: recombine DISTANT addresses into a thought no single source states.

Recall (engine.py) retrieves what the corpus already says — and verbatim is a CORRECT answer there.
CREATION is the frontier: a connection between two distant concepts that appears in NO single passage,
assembled by bridging them across DIFFERENT sources. Mechanism (all proven parts):
  1. BRIDGE  — nav the web from A to B with the DIRECT edge forbidden (force a real multi-hop path);
               each hop is grounded in its own source passage (often a different field).
  2. GATE 1 (coherence) — does the bridged concept-set RESOLVE? (the WHAT oracle verifies the novel
               combination is coherent, not word-salad).
  3. GRADE   — support = the WEAKEST hop's PMI (a chain is only as strong as its weakest link);
               novelty = no direct edge existed (never stated directly) × #distinct sources crossed.
  4. SPEAK   — realize the bridge into a fluent statement.
Honest: a bridge that RESOLVES is COHERENT-new, not yet TRUE-new. Support/strength is the second gate;
a weak-link bridge is a plausible stretch, not a warranted claim. We report both, and never hide a weak link.

  python create.py A B          # bridge two concepts
  python create.py --demo
"""
import sys, math, heapq
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from langexec import open_web, edge_w
from fluency import get_fluency_model, WORD
from thinkloop import resolve
from realize import hybrid_realize, heal_surface

CREATE_TRAIN = 80000


def connecting_span(k, pid, a, b, margin=12):
    """The grounded text window where a and b actually co-occur (for the recall-the-relation case)."""
    text, dom = k._passage(pid)
    toks = WORD.findall(text.lower())
    ia = [i for i, t in enumerate(toks) if t == a]
    ib = [i for i, t in enumerate(toks) if t == b]
    if ia and ib:
        lo, hi = max(0, min(ia[0], ib[0]) - margin), min(len(toks), max(ia[0], ib[0]) + margin + 1)
        return " ".join(toks[lo:hi]), dom
    return " ".join(toks[:40]), dom


def pmi_neighbors(k, a, fanout=25, scan=150):
    """Hub-DAMPED neighbors: scan the top-weight neighbors, keep only ABOVE-CHANCE (PMI>0) non-stopword
    ones, ranked by PMI. The bridge must hop through MEANINGFUL concepts, not co-occurrence hubs
    (their/men/women) — the same hub-damping that fixes every other layer."""
    out = []
    for b, w in k.db.execute("SELECT b,w FROM edges WHERE a=? ORDER BY w DESC LIMIT ?", (a, scan)):
        if b in k.stop:
            continue
        p = k.assoc(a, b, w)
        if p > 0:
            out.append((p, b))
    out.sort(reverse=True)
    return out[:fanout]


def hop_info(k, x, y):
    r = k.db.execute("SELECT src,pid,w FROM edges WHERE a=? AND b=?", (x, y)).fetchone()
    return r if r else ("?", -1, 1)


def bounded_bridge(k, a, b, max_expand=500, fanout=25):
    """Best-first nav A→B with the DIRECT edge forbidden, hopping only through high-PMI (meaningful)
    neighbors. Cost rewards strong (high-PMI) hops; relate() heuristic steers toward B."""
    pq, seen, n = [], {a: 0.0}, 0
    for p, nb in pmi_neighbors(k, a, fanout):
        if nb == b:
            continue                                  # forbid the direct edge → force a real bridge
        step = 1.0 / (1.0 + p)                         # a strong (high-PMI) hop costs less
        heapq.heappush(pq, (step - 2.0 * k.relate(nb, b), step, nb, [a, nb]))
    while pq and n < max_expand:
        _, g, node, path = heapq.heappop(pq); n += 1
        if node == b:
            return path
        if len(path) > 6:
            continue                                  # keep bridges short (a long chain is a weak claim)
        for p, nb in pmi_neighbors(k, node, fanout):
            ng = g + 1.0 / (1.0 + p)
            if nb not in seen or ng < seen[nb]:
                seen[nb] = ng
                heapq.heappush(pq, (ng - 2.0 * k.relate(nb, b), ng, nb, path + [nb]))
    return None


def create(k, model, a, b):
    a, b = a.lower(), b.lower()
    for w in (a, b):
        if w not in k.stoi:
            return f"  '{w}' is not addressable (not a concept node)."
    direct = edge_w(k, a, b)
    dpmi = k.assoc(a, b, direct) if direct is not None else -99.0
    if dpmi > 1.0:
        # strongly, directly connected → RECALL the grounded relation (don't force a bridge)
        src, pid, _w = hop_info(k, a, b)
        span, dom = connecting_span(k, pid, a, b)
        return (f"  {a} ⇄ {b}: directly connected (PMI {dpmi:+.2f}, src {dom}) → the grounded relation:\n"
                f"   « …{span}… »")
    path = bounded_bridge(k, a, b)                      # weak/no direct edge → bridge across sources
    if not path or len(path) < 3:
        return f"  {a} ⇄ {b}: no grounded bridge — too distant to connect."
    hops = []
    for x, y in zip(path, path[1:]):
        src, pid, w = hop_info(k, x, y)
        hops.append((x, y, src, pid, k.assoc(x, y, w)))
    coherence = resolve(k, [c for c in path if c in k.nodeset])["pos_pmi"]
    support = min(h[4] for h in hops)                  # weakest link = the claim's strength
    fields = sorted(set(h[2] for h in hops))
    said = heal_surface(hybrid_realize(k, model, path, k.stop))

    grade = ("WARRANTED" if support > 1.5 else
             "SUGGESTIVE" if support > 0.7 else "WEAK STRETCH")
    novelty = "NEW (no direct edge)" if direct is None else "weak direct edge exists"
    out = [f"  {a} ⇄ {b}   [{grade}; {novelty}; crosses {len(fields)} field(s): {', '.join(fields)}]",
           f"   bridge : {' → '.join(path)}",
           f"   gate1 coherence (resolve): {coherence:.2f}    gate2 support (weakest hop PMI): {support:+.2f}",
           f"   spoken : {said}"]
    weak = min(hops, key=lambda h: h[4])
    out.append(f"   weakest link: {weak[0]}–{weak[1]} (PMI {weak[4]:+.2f}, src {weak[2]})  ← the claim is only as strong as this")
    return "\n".join(out)


DEMO = [("gravity", "music"), ("sleep", "memory"), ("volcano", "climate"), ("light", "emotion")]


def main():
    k = open_web()
    max_pid = k.db.execute("SELECT MAX(pid) FROM passages").fetchone()[0]
    print("[create] loading fluency model (cached) ...", flush=True)
    model = get_fluency_model(k, CREATE_TRAIN, max_pid)
    print(f"[create] ready.\n", flush=True)
    if "--demo" in sys.argv or len(sys.argv) < 3:
        for a, b in DEMO:
            print(create(k, model, a, b)); print()
    else:
        print(create(k, model, sys.argv[1], sys.argv[2]))


if __name__ == "__main__":
    main()

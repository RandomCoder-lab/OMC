#!/usr/bin/env python3
"""navigator.py — the INTERNAL NAVIGATOR: an autonomous loop that walks the web and MAKES MEMORIES. It picks
seed concepts, explores (runner) and bridges/infers (wavefront) between them, and every grounded discovery
auto-remembers as a 'derived' memory (dedup→reinforce). This is the system experiencing its own geoform —
reason → remember → recall → reason better — write-don't-train, fully grounded, nothing invented.

Honest scope: it discovers GROUNDED connections (real edges/paths) and records them; it is not conscious and
not generating claims — it's an addressed walk that deposits what it finds, with provenance. Memories are
tagged 'derived' and never count as corpus evidence.

  python navigator.py --rounds 30           # 30 discovery rounds
  python navigator.py --rounds 50 --seeds gravity,light,number,mind,energy,cell
"""
import sys, math, time, random, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from kdb import KnowledgeDB, load_embedding
from memory import Memory


def open_web():
    s, E, n, e = load_embedding()
    return KnowledgeDB(str(Path(__file__).parent / "knowledge.db"), s, E, n, e)


HUB_DEG = 250_000   # demote promiscuous hubs (boilerplate/metadata)


def top_neighbors(k, x, n=6):
    rows = k.db.execute(
        "SELECT b,w FROM edges WHERE a=? AND src NOT LIKE 'align-%' AND src NOT LIKE 'parallel-%' AND src!='memory' "
        "ORDER BY w DESC LIMIT 200", (x,)).fetchall()
    rows = [(b, w) for b, w in rows if b not in k.stop and b != x and k.deg.get(b, 0) < HUB_DEG]
    return [b for sc, w, b in sorted(((k.assoc(x, b, w) * math.log1p(w), w, b) for b, w in rows), reverse=True)[:n]]


CONCEPT_MIN = 0.15   # intermediate meaning-coherence above this = a conceptual bridge


def derive_forms(k, path, axis, fields):
    """Derive MULTIPLE form-tags from DATA SIGNALS ONLY — agnostic, no hand-coded ontology:
      • the relation-axis (conceptual/institutional) — MEASURED from meaning-coherence
      • the raw source fields the path crosses — the DATA'S OWN provenance labels (genetics, arxiv-cs, …),
        recorded verbatim, not bucketed into human categories
      • scope (cross-domain/within-domain) — a COUNT of distinct fields
    Grouping fields into 'scientific'/'humanities' etc. would be a hand-ontology → forbidden. If you want
    such a lens, filter these raw field-tags at QUERY time (your choice, not baked in)."""
    forms = {axis, "cross-domain" if len(set(fields)) >= 2 else "within-domain"}
    forms |= {f"src:{f}" for f in fields}     # the data's own field labels, verbatim
    return forms


def classify_path(k, path):
    """AXIS-TAGGING (better than keep/reject): label a grounded path by what KIND of knowledge it is.
      • 'conceptual'    — intermediates are meaning-coherent ideas (refraction↔light, atom↔acid): the web's
                          conceptual structure. Requires endpoints to relate + intermediates coherent.
      • 'institutional' — grounded but low concept-coherence: named entities / places / affiliations
                          (cell↔lanzhou↔center = 'cells are studied at Lanzhou research centers'). A TRUE,
                          different axis of knowledge (bibliometric/geographic), kept and labeled, not noise.
      • None            — reject: a promiscuous hub intermediate (function-word/boilerplate), or a broken path.
    All signals derived (meaning cosine + degree), agnostic. The two axes are both real; we tag, not discard."""
    if len(path) < 3:
        return None
    inter = path[1:-1]
    if any(k.deg.get(m, 0) >= HUB_DEG for m in inter):    # promiscuous hub → true noise
        return None
    # coherence of each intermediate with its neighbors
    cohs = [max(k.relate(m, path[i - 1]), k.relate(m, path[i + 1])) for i, m in enumerate(path[1:-1], 1)]
    endpoints = k.relate(path[0], path[-1])
    if endpoints >= 0.12 and all(c >= CONCEPT_MIN for c in cohs):
        return "conceptual"
    if all(c > 0 for c in cohs):                          # grounded co-occurrence, low concept-coherence
        return "institutional"
    return None                                            # no real link at all


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--seeds", default="")
    ap.add_argument("--wander", action="store_true", help="roam: sample random concepts across the whole web")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    k = open_web()
    mem = Memory(k)
    if not k.deg:
        print("[nav] node_deg missing — refusing to make memories on ungrounded data."); return

    # seeds: user list, else wander (random concepts across the whole web), else domain anchors
    if a.seeds:
        pool = [s.strip() for s in a.seeds.split(",") if s.strip() and s.strip() in k.stoi]
    elif a.wander:
        # roam: sample well-grounded, non-stopword, non-hub concepts from anywhere in the geoform
        cands = [n for n in k.nodes if n not in k.stop and 50 < k.deg.get(n, 0) < 200_000]
        rnd = random.Random(int(time.time()))
        pool = rnd.sample(cands, min(a.rounds, len(cands)))
    else:
        pool = [s for s in ["gravity", "light", "number", "mind", "energy", "cell", "force", "time", "atom",
                "life", "water", "sound", "memory", "logic", "wave", "heat", "motion", "color", "language",
                "space"] if s in k.stoi]
    made = reinforced = rejected = 0
    t0 = time.time()
    for r in range(a.rounds):
        seed = pool[r % len(pool)]
        nbrs = top_neighbors(k, seed, 6)
        if len(nbrs) < 2:
            continue
        # explore: bridge the seed to a NON-adjacent strong concept (a discovery, not a restated edge)
        far = top_neighbors(k, random.Random(r).choice(nbrs), 6)
        targets = [t for t in far if t != seed and t not in nbrs][:2] or nbrs[1:2]
        for tgt in targets:
            # is there a grounded multi-hop path? (deep_connect forbids the direct edge)
            chain = k.deep_connect(seed, tgt)
            head = chain.split("\n")[0]
            if " → " in head and "only direct" not in head:
                path = [p.strip() for p in head.split("(")[0].split("→")]
                axis = classify_path(k, path)         # AXIS-TAG: conceptual | institutional | None(hub/noise)
                if axis is None:
                    rejected += 1
                    if not a.quiet:
                        print(f"[nav r{r}] {seed}⇝{tgt}: ✗ rejected (hub/no-link) {('→'.join(path))[:70]}", flush=True)
                    continue
                cm = re.search(r'crosses: ([^\]]+)\]', head)
                fields = cm.group(1).split('+') if cm else []
                forms = derive_forms(k, path, axis, fields)
                mid, act = mem.remember_discovery(path, f"navigated [{axis}]: {head}", axis, forms)
                made += (act == "new"); reinforced += (act == "reinforced")
                if not a.quiet:
                    tag = "◆" if axis == "conceptual" else "▣"
                    print(f"[nav r{r}] {seed}⇝{tgt}: {tag}{axis} #{mid} ({act}) {{{','.join(sorted(forms))[:50]}}}", flush=True)
    print(f"\n[nav] {a.rounds} rounds in {time.time()-t0:.0f}s → {made} new, {reinforced} reinforced, "
          f"{rejected} rejected by compass (hub/incoherent).", flush=True)
    tot = k.db.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
    print(f"[nav] total memories now: {tot}", flush=True)


if __name__ == "__main__":
    main()

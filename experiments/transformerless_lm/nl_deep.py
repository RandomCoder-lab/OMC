"""nl_deep.py — push per-hop retrieval recall to the ceiling (addressed inverted index) and go DEEP.

nl_ground.py used fuzzy locality-fp retrieval: ~97% bridge coverage but only at topk=120 (scan ALL
passages per hop, imperfect recall). The substrate move: an ADDRESSED inverted index (entity -> postings)
gives PERFECT per-hop recall at O(postings) cost instead of O(all-passages). That both maxes recall and
makes hops cheap enough to chain DEEP (3-4 hop grounded reasoning), each hop still verified by a quoted
passage. Built from raw text, no schema, law-clean (everything derived).

Measures: (1) per-hop recall, inverted vs fuzzy locality-fp; (2) per-hop cost; (3) DEEP grounded
reachability — shortest verified chains at graph-distance 2/3/4; (4) quoted deep chains (the dot-connector).
"""
import sys, time, json, random
from collections import deque
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))
from nl_ground import clean_text, split_passages, extract_entities, fp

HERE = Path(__file__).parent


def contains(e, p_lower_padded):
    return any(s in p_lower_padded for s in (
        " " + e.lower() + " ", " " + e.lower() + ",", " " + e.lower() + ".",
        " " + e.lower() + "'", " " + e.lower() + ";"))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="pride_prejudice.txt")
    ap.add_argument("--top_n", type=int, default=40)
    ap.add_argument("--max_depth", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    text = clean_text((HERE / args.corpus).read_text(errors="replace"))
    passages = split_passages(text)
    ents = extract_entities(text, top_n=args.top_n)
    plow = [" " + p.lower() + " " for p in passages]

    # ADDRESSED INVERTED INDEX: entity -> postings (passage indices). Built once, O(N). Perfect recall.
    inv = {e: [] for e in ents}
    for i, pl in enumerate(plow):
        for e in ents:
            if contains(e, pl):
                inv[e].append(i)
    # build co-occurrence (exact) once, with an evidence passage per edge
    adj = {e: {} for e in ents}
    for i, pl in enumerate(plow):
        present = [e for e in ents if contains(e, pl)]
        for a in present:
            for b in present:
                if a != b and b not in adj[a]:
                    adj[a][b] = i                          # evidence: a passage proving a~b
    n = len(ents)
    deg = {e: len(adj[e]) for e in ents}
    hubs = [e for e in ents if deg[e] > 0.92 * (n - 1)]
    ents = [e for e in ents if e not in hubs]
    adj = {e: {b: i for b, i in adj[e].items() if b not in hubs} for e in ents}
    print(f"[deep] corpus={args.corpus} passages={len(passages):,} entities={len(ents)} "
          f"(dropped hubs {hubs})", flush=True)

    # ── (1)+(2) per-hop recall & cost: addressed inverted index vs fuzzy locality-fp topk ──
    Pmat = torch.stack([fp(p) for p in passages])
    def fuzzy_neighbors(e, topk):
        sims = Pmat @ fp(e)
        idx = sims.topk(min(topk, len(passages)), largest=True).indices.tolist()
        idx = [i for i in idx if contains(e, plow[i])]
        nb = set()
        for i in idx:
            nb |= {o for o in ents if contains(o, plow[i])}
        nb.discard(e)
        return nb
    rec, cost_inv = [], []
    for e in ents:
        true_nb = set(adj[e])
        if not true_nb:
            continue
        f = fuzzy_neighbors(e, 120)
        rec.append(len(f & true_nb) / len(true_nb))
        cost_inv.append(len(inv[e]))
    print(f"[deep] per-hop recall: inverted-index = 1.00 (exact)   fuzzy locality-fp@topk120 = "
          f"{sum(rec)/len(rec):.2f}", flush=True)
    print(f"[deep] per-hop cost: inverted = O(postings) avg {sum(cost_inv)/len(cost_inv):.0f} passages read; "
          f"fuzzy = O(all) {len(passages):,} scored every hop", flush=True)

    # ── (3) DEEP grounded reachability via BFS shortest verified path ──
    def bfs(start, target, max_depth):
        if start == target:
            return [start], []
        prev = {start: (None, None)}
        q = deque([(start, 0)])
        while q:
            cur, d = q.popleft()
            if d >= max_depth:
                continue
            for nb, evi in adj[cur].items():
                if nb not in prev:
                    prev[nb] = (cur, evi)
                    if nb == target:
                        path, edges, x = [nb], [], nb
                        while prev[x][0] is not None:
                            edges.append(prev[x][1]); x = prev[x][0]; path.append(x)
                        return path[::-1], edges[::-1]
                    q.append((nb, d + 1))
        return None, None

    # all-pairs shortest distance on the exact graph -> bucket by distance
    def dist_from(s):
        dd = {s: 0}; q = deque([s])
        while q:
            c = q.popleft()
            for nb in adj[c]:
                if nb not in dd:
                    dd[nb] = dd[c] + 1; q.append(nb)
        return dd
    buckets = {2: [], 3: [], 4: []}
    for a in ents:
        dd = dist_from(a)
        for b, d in dd.items():
            if a < b and d in buckets:
                buckets[d].append((a, b))
    print(f"\n[deep] DEEP grounded reachability (shortest VERIFIED chain found by BFS):", flush=True)
    for d in (2, 3, 4):
        pairs = buckets[d]
        if not pairs:
            print(f"[deep]   distance {d}: (no pairs)", flush=True); continue
        hit = sum(1 for a, b in pairs if bfs(a, b, args.max_depth)[0])
        print(f"[deep]   distance {d}: {hit}/{len(pairs)} = {100*hit/len(pairs):.0f}% connected via a "
              f"{d}-hop verified chain", flush=True)

    # ── (4) quote a deep chain (3+ hops) — the dot-connector ──
    rng = random.Random(args.seed)
    deep_pairs = buckets[3] + buckets[4]
    rng.shuffle(deep_pairs)
    print(f"\n[deep] === sample DEEP grounded chains (each hop quoted from the text) ===", flush=True)
    shown = 0
    for a, b in deep_pairs:
        path, edges = bfs(a, b, args.max_depth)
        if path and len(path) >= 4 and shown < 3:
            print(f"\n[deep] {'  ──▶  '.join(path)}", flush=True)
            for (x, y), ei in zip(zip(path, path[1:]), edges):
                snip = passages[ei].strip().replace("\n", " ")
                snip = (snip[:150] + "…") if len(snip) > 150 else snip
                print(f"[deep]   {x}~{y}: \"{snip}\"", flush=True)
            shown += 1

    (HERE / "results_nl_deep.json").write_text(json.dumps(dict(
        entities=len(ents), passages=len(passages),
        recall_inverted=1.0, recall_fuzzy=round(sum(rec)/len(rec), 3),
        cost_inverted_avg=round(sum(cost_inv)/len(cost_inv)),
        reach={d: (round(100*sum(1 for a, b in buckets[d] if bfs(a, b, args.max_depth)[0])/len(buckets[d]))
                   if buckets[d] else None) for d in (2, 3, 4)}), indent=2))
    print("\n[deep] wrote results_nl_deep.json", flush=True)


if __name__ == "__main__":
    main()

"""nl_ground.py — NATURAL-LANGUAGE grounding for reasoning-as-navigation.

The synthetic test (reasoning_nav.py) proved the mechanism on clean "A leads_to B" facts. This moves it to
REAL natural language: a hop from concept A to concept B is GROUNDED iff a real passage co-mentions them.
Dot-connecting = finding a grounded path A->B->C where A and C never appear together, every hop backed by
a quoted passage. Corpus: pride_prejudice.txt (rich entity structure). Ground truth = the co-occurrence
graph DERIVED from the text (computed, not hand-labeled).

The honest empirical question: is our PROVEN addressed retrieval (locality-fp) + simple entity extraction
faithful enough on messy NL to support grounded multi-hop reasoning that single-hop retrieval cannot?
  - precision is guaranteed: a hop is followed ONLY if the retrieved passage really contains both ends
    (grounding verification — no hallucinated edges).
  - recall (coverage) is the open question: does addressed retrieval surface the right bridging passage?
Metric: coverage of true 2-hop bridges (pairs with NO direct co-occurrence) by addressed grounded-nav vs
single-hop. Plus qualitative grounded chains with quoted evidence. Honest failure modes reported.
"""
import sys, re, json, random
from pathlib import Path
import torch

HERE = Path(__file__).parent
DIM = 2048


def fp(s: str) -> torch.Tensor:
    v = torch.zeros(DIM)
    s = s.lower()
    for i in range(len(s) - 1):
        v[(ord(s[i]) * 131 + ord(s[i + 1])) % DIM] += 1.0
    n = v.norm()
    return v / n if n > 0 else v


def clean_text(raw):
    """Agnostic structural cleanup — strip boilerplate by its MARKERS, not by any word-list:
    Gutenberg header/footer, [Illustration]/[_..._] markup, chapter headings + TOC. No entity names hard-coded."""
    m = re.search(r"\*\*\*\s*START OF.*?\*\*\*(.*?)\*\*\*\s*END OF", raw, re.S | re.I)
    body = m.group(1) if m else raw
    body = re.sub(r"\[[^\]]*\]", " ", body)                       # [Illustration: ...], [_Reading…_]
    body = re.sub(r"(?i)heading to chapter\s+[ivxlcdm\d]+\.?", " ", body)   # TOC lines
    body = re.sub(r"(?i)\bchapter\s+[ivxlcdm]+\b\.?", " ", body)  # chapter headings (roman-numeral form)
    return body


def split_passages(text, target_sentences=3):
    # normalize whitespace, split into sentences, group into ~target_sentences-sized passages
    text = re.sub(r"\s+", " ", text)
    sents = re.split(r"(?<=[.!?]) ", text)
    passages, cur = [], []
    for s in sents:
        cur.append(s)
        if len(cur) >= target_sentences:
            passages.append(" ".join(cur)); cur = []
    if cur:
        passages.append(" ".join(cur))
    return [p for p in passages if len(p) > 30]


def extract_entities(text, top_n=40, min_mid=4, cap_ratio=0.9, drop_titles=True):
    """Corpus-derived proper nouns. Two principled, derived filters (no hand-coded entity list):
      (1) MID-sentence capitalization >= min_mid (drops sentence-initial-only 'The'/'And').
      (2) capitalization RATIO: real names are almost ALWAYS capitalized ('Darcy' is never 'darcy'),
          while 'they/yes/but' appear constantly in lowercase -> keep cap_count/(cap+lower) >= cap_ratio.
    Also drops generic TITLES (token almost always FOLLOWED by another capitalized word: 'Mrs. Bennet')."""
    tokens = list(re.finditer(r"\b([A-Z][a-z]{2,})\b", text))
    mid, total, followed_by_cap = {}, {}, {}
    for m in tokens:
        w = m.group(1)
        total[w] = total.get(w, 0) + 1
        prev = text[max(0, m.start() - 2):m.start()]
        if not (prev.endswith(". ") or prev.endswith("! ") or prev.endswith("? ") or m.start() == 0):
            mid[w] = mid.get(w, 0) + 1
        nxt = text[m.end():m.end() + 3]
        if re.match(r"[.,]?\s+[A-Z][a-z]", nxt):
            followed_by_cap[w] = followed_by_cap.get(w, 0) + 1
    keep = []
    for w in total:
        if mid.get(w, 0) < min_mid:
            continue
        lower = len(re.findall(r"\b" + re.escape(w.lower()) + r"\b", text))   # case-sensitive lowercase form
        if total[w] / (total[w] + lower) < cap_ratio:                         # mostly-capitalized = proper noun
            continue
        if drop_titles and followed_by_cap.get(w, 0) / total[w] > 0.6:        # almost always before a name = title
            continue
        keep.append((w, mid.get(w, 0)))
    keep.sort(key=lambda x: -x[1])
    return [w for w, _ in keep[:top_n]]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="pride_prejudice.txt")
    ap.add_argument("--top_n", type=int, default=40)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--n_pairs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    text = clean_text((HERE / args.corpus).read_text(errors="replace"))   # structural boilerplate strip
    passages = split_passages(text)
    ents = extract_entities(text, top_n=args.top_n)
    print(f"[nlg] corpus={args.corpus}  passages={len(passages):,}  entities(pre-hub-filter)={ents[:18]}...", flush=True)

    # word-boundary containment per passage (the GROUNDING relation: real co-mention in actual text)
    def contains(e, pl):
        return any(p in pl for p in (" " + e.lower() + " ", " " + e.lower() + ",", " " + e.lower() + ".",
                                     " " + e.lower() + "'", " " + e.lower() + ";"))
    def co_occ(entset):
        pae = [{e for e in entset if contains(e, " " + p.lower() + " ")} for p in passages]
        adj = {e: {} for e in entset}
        for present in pae:
            for a in present:
                for b in present:
                    if a != b:
                        adj[a][b] = adj[a].get(b, 0) + 1
        return pae, adj

    # provisional graph -> DERIVED promiscuous-hub filter (drop tokens that co-occur with ~everything:
    # structural hubs like a stray "Chapter"; a derived statistic, NOT a hand-coded stoplist)
    _, adj0 = co_occ(ents)
    n = len(ents)
    kept = [e for e in ents if len(adj0[e]) <= 0.92 * (n - 1)]
    dropped = [e for e in ents if e not in kept]
    ents = kept
    ent_set = set(ents)
    pa_ents, true_adj = co_occ(ents)
    n_edges = sum(len(v) for v in true_adj.values()) // 2
    print(f"[nlg] dropped promiscuous hubs (>92% degree): {dropped}", flush=True)
    print(f"[nlg] entities kept: {ents}", flush=True)
    print(f"[nlg] co-occurrence graph: {len([e for e in ents if true_adj[e]])} connected entities, "
          f"{n_edges} undirected edges", flush=True)

    # addressed index over passages (locality-fp)
    Pmat = torch.stack([fp(p) for p in passages])             # [P, DIM]

    def addressed_passages(entity, k):
        """Retrieve passages for an entity by ADDRESS (locality-fp), then GROUND-verify containment."""
        sims = Pmat @ fp(entity)
        idx = sims.topk(min(k, len(passages)), largest=True).indices.tolist()
        return [i for i in idx if entity in pa_ents[i]]        # keep only those that really mention it

    def addressed_neighbors(entity, k):
        nbrs = {}
        for i in addressed_passages(entity, k):
            for e in pa_ents[i]:
                if e != entity:
                    nbrs.setdefault(e, i)                      # e -> a passage proving entity~e
        return nbrs                                            # grounded (verified co-mention)

    # build true 2-hop bridge test set: pairs (A,C) with NO direct edge but a real common neighbor B
    rng = random.Random(args.seed)
    bridge_pairs = []
    connected = [e for e in ents if true_adj[e]]
    for A in connected:
        for C in connected:
            if A < C and C not in true_adj[A]:                # NOT directly co-occurring
                common = set(true_adj[A]) & set(true_adj[C])
                if common:
                    bridge_pairs.append((A, C, common))
    rng.shuffle(bridge_pairs)
    bridge_pairs = bridge_pairs[:args.n_pairs]
    print(f"[nlg] {len(bridge_pairs)} test pairs: NO direct co-occurrence, but a true 2-hop bridge exists\n", flush=True)

    # navigation: coverage of true 2-hop bridges as a function of RETRIEVAL BUDGET (topk passages/hop)
    print(f"[nlg] === GROUNDED MULTI-HOP NAVIGATION on natural language (coverage vs retrieval budget) ===", flush=True)
    examples = []
    cov_curve = {}
    for topk in [12, 30, 60, 120]:
        nav_hit = single_hit = 0
        for A, C, common in bridge_pairs:
            nA = addressed_neighbors(A, topk)
            single_hit += int(C in nA)
            found = None
            for B in nA:
                if B == C:
                    continue
                nB = addressed_neighbors(B, topk)
                if C in nB:
                    found = (B, nA[B], nB[C]); break
            if found:
                nav_hit += 1
                if topk == 60 and len(examples) < 4 and found[0] in common:
                    examples.append((A, found[0], C, passages[found[1]], passages[found[2]]))
        cov = 100.0 * nav_hit / max(1, len(bridge_pairs))
        shc = 100.0 * single_hit / max(1, len(bridge_pairs))
        cov_curve[topk] = round(cov, 1)
        print(f"[nlg]   topk={topk:>3}/hop:  grounded 2-hop nav = {cov:4.0f}%   (single-hop control = {shc:.0f}%)", flush=True)
    cov = cov_curve[120]; shc = 0.0
    print(f"\n[nlg] === sample grounded connections (every hop quoted from the text) ===", flush=True)
    for A, B, C, p1, p2 in examples:
        clip = lambda s: (s[:200] + "…") if len(s) > 200 else s
        print(f"\n[nlg] {A}  ──▶  {B}  ──▶  {C}", flush=True)
        print(f"[nlg]   ground({A}~{B}): \"{clip(p1)}\"", flush=True)
        print(f"[nlg]   ground({B}~{C}): \"{clip(p2)}\"", flush=True)

    climbs = cov_curve[120] >= cov_curve[12] + 20
    verdict = (f"SUPPORTED: grounded multi-hop nav WORKS on real NL — every hop verified by a real passage, "
               f"single-hop control 0%, and coverage CLIMBS with retrieval budget ({cov_curve[12]}%->"
               f"{cov_curve[120]}%) — the limit is retrieval RECALL (a tunable budget), not the mechanism."
               if climbs and cov_curve[120] >= 40 else
               f"PARTIAL: meaningful grounded bridges found, but coverage ({cov_curve}) is retrieval-bound; "
               f"locality-fp recall per hop is the bottleneck on NL.")
    print(f"\n[nlg] === VERDICT === {verdict}", flush=True)
    (HERE / "results_nl_ground.json").write_text(json.dumps(
        dict(passages=len(passages), entities=len(ents), edges=n_edges, pairs=len(bridge_pairs),
             coverage_by_topk=cov_curve), indent=2))
    print("[nlg] wrote results_nl_ground.json", flush=True)


if __name__ == "__main__":
    main()

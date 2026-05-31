"""nl_xdomain.py — CROSS-DOMAIN grounded navigation: connect a concept in one corpus to a concept in a
DIFFERENT corpus, through a shared bridge term, with a real quote from EACH source.

A single dense corpus is small-world (protagonists bridge everything in <=2 hops — nl_deep.py). The
non-obvious dots live ACROSS domains. This puts two corpora in ONE addressed index and finds grounded
chains  A_entity --(passage in corpus A)--> bridge_term --(passage in corpus B)--> B_entity, where the
bridge is a contentful term appearing in BOTH (derived by frequency band — no hand-coded list). Every hop
is verified by a real passage; the chain literally crosses domains at the bridge. Law-clean (all derived).

This is the dot-connector the user envisioned, on real text: "what no one else can" = an agnostic,
grounded, cross-domain associative walk built from raw corpora with no schema.
"""
import sys, re, json, random, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nl_ground import clean_text, split_passages, extract_entities, fp

HERE = Path(__file__).parent


def contains(term, p_lower_padded):
    return (" " + term.lower() + " ") in p_lower_padded or (" " + term.lower() + ",") in p_lower_padded \
        or (" " + term.lower() + ".") in p_lower_padded or (" " + term.lower() + ";") in p_lower_padded


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="pride_prejudice.txt")
    ap.add_argument("--b", default="tinyshakespeare.txt")
    ap.add_argument("--n_show", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    A = clean_text((HERE / args.a).read_text(errors="replace"))
    B = clean_text((HERE / args.b).read_text(errors="replace"))
    pa = split_passages(A); pb = split_passages(B)
    pal = [" " + p.lower() + " " for p in pa]; pbl = [" " + p.lower() + " " for p in pb]
    entA = extract_entities(A, top_n=50); entB = extract_entities(B, top_n=50)
    a_only = [e for e in entA if e not in set(entB)]
    b_only = [e for e in entB if e not in set(entA)]
    print(f"[xdom] A={args.a} ({len(pa)} passages, {len(entA)} ents)  B={args.b} ({len(pb)} passages, "
          f"{len(entB)} ents)", flush=True)

    # bridge candidates: contentful words appearing in BOTH corpora, in a derived mid-frequency band
    # (drop ultra-rare and ultra-generic by passage-frequency — a statistic, NOT a hand-coded stoplist).
    def passfreq(words_in, plist):
        df = {}
        for pl in plist:
            for w in set(re.findall(r"[a-z]{4,}", pl)):
                df[w] = df.get(w, 0) + 1
        return df
    dfa, dfb = passfreq(None, pal), passfreq(None, pbl)
    na, nb = len(pa), len(pb)
    bridges = []
    for w in set(dfa) & set(dfb):
        fa, fb = dfa[w] / na, dfb[w] / nb
        if 0.004 < fa < 0.15 and 0.004 < fb < 0.15:        # in both, contentful (not rare, not generic)
            bridges.append((w, fa + fb))
    bridges = [w for w, _ in sorted(bridges, key=lambda x: -x[1])]
    print(f"[xdom] {len(bridges)} cross-corpus bridge terms (derived, mid-frequency in both). "
          f"e.g. {bridges[:20]}", flush=True)
    bset = set(bridges)

    # postings for A-entities (in A), B-entities (in B), and bridges (in both)
    def post(term, plist_lower):
        return [i for i, pl in enumerate(plist_lower) if contains(term, pl)]
    invA = {e: post(e, pal) for e in a_only}
    invB = {e: post(e, pbl) for e in b_only}
    brA = {w: post(w, pal) for w in bridges}
    brB = {w: post(w, pbl) for w in bridges}
    # which bridges each entity touches (in its own corpus)
    a_bridges = {e: {w for w in bridges if any(contains(w, pal[i]) for i in invA[e])} for e in a_only}
    b_bridges = {e: {w for w in bridges if any(contains(w, pbl[i]) for i in invB[e])} for e in b_only}

    # cross-domain coverage + examples: A_ent --bridge--> B_ent, quoted from each corpus
    rng = random.Random(args.seed)
    a_samp = [e for e in a_only if a_bridges[e]]
    b_samp = [e for e in b_only if b_bridges[e]]
    connectable = 0; total = 0; examples = []
    for a in a_samp:
        for c in b_samp:
            total += 1
            shared = a_bridges[a] & b_bridges[c]
            if shared:
                connectable += 1
    cov = 100.0 * connectable / max(1, total)
    print(f"\n[xdom] cross-domain connectivity: {connectable}/{total} = {cov:.0f}% of "
          f"(A-entity, B-entity) pairs share a grounded bridge term", flush=True)

    # AGNOSTIC thread-strength ranking (no learned encoder, no hand-list):
    #   strength = idf(bridge)  ×  passage_overlap_beyond_the_bridge
    #   - idf: a rare/specific shared term ('elopement') beats a generic one ('wish').
    #   - overlap: do the two passages share MORE than the single bridge word? (locality-fp cosine) ->
    #     a real shared THEME, not a lone coincident token.
    N = na + nb
    idf = {w: math.log(N / (dfa.get(w, 0) + dfb.get(w, 0) + 1)) for w in bridges}
    fpa_cache, fpb_cache = {}, {}
    def fpa(i):
        if i not in fpa_cache: fpa_cache[i] = fp(pa[i])
        return fpa_cache[i]
    def fpb(i):
        if i not in fpb_cache: fpb_cache[i] = fp(pb[i])
        return fpb_cache[i]
    threads = []
    pairs = [(a, c) for a in a_samp for c in b_samp if a_bridges[a] & b_bridges[c]]
    for a, c in pairs:
        best = None
        for t in (a_bridges[a] & b_bridges[c]):
            ia = next(i for i in invA[a] if contains(t, pal[i]))
            ib = next(i for i in invB[c] if contains(t, pbl[i]))
            sim = float((fpa(ia) * fpb(ib)).sum())          # locality-fp cosine (passages normalized)
            s = idf[t] * sim
            if best is None or s > best[0]:
                best = (s, t, ia, ib)
        threads.append((best[0], a, best[1], c, best[2], best[3]))
    threads.sort(key=lambda x: -x[0])
    strs = [t[0] for t in threads]
    med = strs[len(strs) // 2] if strs else 0
    print(f"\n[xdom] thread-strength spread (idf×passage-overlap): max={strs[0]:.3f} med={med:.3f} "
          f"min={strs[-1]:.3f}  -> {'DISCRIMINATIVE (strong vs weak separable)' if strs[0] > 3*max(med,1e-6) else 'flat'}", flush=True)
    clip = lambda s: (s.strip()[:140] + "…") if len(s.strip()) > 140 else s.strip()
    def show_thread(tag, th):
        s, a, t, c, ia, ib = th
        print(f"\n[xdom] [{tag} strength={s:.3f}] {a} ⟨{args.a.split('.')[0]}⟩ ──[{t}]──▶ {c} ⟨{args.b.split('.')[0]}⟩", flush=True)
        print(f"[xdom]   A: \"{clip(pa[ia])}\"", flush=True)
        print(f"[xdom]   B: \"{clip(pb[ib])}\"", flush=True)
    print(f"\n[xdom] === STRONGEST grounded cross-domain threads (ranked, agnostic) ===", flush=True)
    for th in threads[:args.n_show]:
        show_thread("STRONG", th)
    print(f"\n[xdom] === WEAKEST (what the ranking correctly demotes) ===", flush=True)
    for th in threads[-2:]:
        show_thread("weak", th)

    (HERE / "results_nl_xdomain.json").write_text(json.dumps(dict(
        a=args.a, b=args.b, passages_a=len(pa), passages_b=len(pb),
        bridges=len(bridges), cross_pairs=total, connectable=connectable, connectivity_pct=round(cov, 1),
        strength_max=round(strs[0], 3), strength_med=round(med, 3), strength_min=round(strs[-1], 3)), indent=2))
    print("\n[xdom] wrote results_nl_xdomain.json", flush=True)


if __name__ == "__main__":
    main()

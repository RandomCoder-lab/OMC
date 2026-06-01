#!/usr/bin/env python3
"""thinkloop.py — GENERATION AS HEALING-TO-RESOLUTION  (the thought-engine's first cycle)

langexec.py proved the EXECUTION ORACLE is real: a sentence's concepts, traversed over the weighted
web, RESOLVE (hub-damped PMI connectivity) iff coherent — AUC 0.91–0.98 vs salad. This asks the next
question: can we GENERATE by CLIMBING that oracle? i.e. take a FAULTING draft and have HEAL snap/route
its concepts until resolve rises into real-sentence territory.

Same honest method as heal_as_generator.rs + langexec.py: corrupt a coherent thought (keep half its
concepts, inject random real concepts → a faulting draft, resolve≈0), then heal-to-fixpoint and measure
resolve RECOVERY (corrupted → healed vs the original coherent level).

HEAL operators (all = addressing, nothing invented):
  fault  = a concept with NO above-chance PMI edge to the rest of the set (it doesn't belong)
  repair = replace the worst-faulting concept with the best COHERENT candidate — the highest-summed-PMI
           neighbor of the connected core (spreading activation: retrieval-by-address, the proven strength)
  fixpoint = the set stops changing / resolve stops climbing = a settled, coherent thought

  python thinkloop.py [N]      # N seed thoughts (default 40)
"""
import sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from langexec import open_web, edge_w, TOK
from kdb import KnowledgeDB  # noqa  (open_web returns one)

random.seed(2718)
MAX_CONCEPTS = 12


def resolve(k, concepts):
    """Concept-level execution score. coverage=1 (all are addresses) so resolve = pos_pmi:
    fraction of concept-PAIRS co-occurring ABOVE chance (the steelman-surviving, hub-damped signal)."""
    A = list(dict.fromkeys(concepts))[:MAX_CONCEPTS]
    if len(A) < 2:
        return dict(pos_pmi=0.0, mean_pmi=0.0, n=len(A))
    pmis, total = [], 0
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            total += 1
            w = edge_w(k, A[i], A[j])
            if w is not None:
                pmis.append(k.assoc(A[i], A[j], w))
    pos = sum(1 for p in pmis if p > 0) / total if total else 0.0
    mean = sum(pmis) / len(pmis) if pmis else 0.0
    return dict(pos_pmi=pos, mean_pmi=mean, n=len(A))


def connectivity(k, c, others):
    """Above-chance PMI strength of concept c to the rest (0 if it connects to none = a FAULT)."""
    pos = []
    for o in others:
        if o == c:
            continue
        w = edge_w(k, c, o)
        if w is not None:
            a = k.assoc(c, o, w)
            if a > 0:
                pos.append(a)
    return sum(pos) / len(pos) if pos else 0.0


def top_neighbors(k, a, lim=30):
    """Strongest direct neighbors of a (bounded; ranked by raw co-occurrence weight)."""
    return [(b, w) for b, w in k.db.execute(
        "SELECT b,w FROM edges WHERE a=? ORDER BY w DESC LIMIT ?", (a, lim))]


def candidates(k, core):
    """Spreading activation: pool the strong neighbors of the coherent core, rank by SUMMED above-chance
    PMI back to the core. This is generation-by-addressing — pick the concept most coherent with the core."""
    pool = set()
    for c in core:
        for b, _w in top_neighbors(k, c, 30):
            if b not in k.stop and b not in core:
                pool.add(b)
    ranked = []
    for b in pool:
        tot = 0.0
        for c in core:
            w = edge_w(k, b, c)
            if w is not None:
                a = k.assoc(b, c, w)
                if a > 0:
                    tot += a
        if tot > 0:
            ranked.append((tot, b))
    ranked.sort(reverse=True)
    return [b for _t, b in ranked]


def think_heal(k, draft, target, max_iter=14):
    """Heal a faulting draft to a resolved fixpoint. Returns (final_concepts, resolve_trajectory)."""
    S = list(dict.fromkeys(draft))[:target]
    traj = [resolve(k, S)["pos_pmi"]]
    for _ in range(max_iter):
        conns = sorted((connectivity(k, c, S), c) for c in S)
        worst_strength, worst = conns[0]
        core = [c for c in S if connectivity(k, c, S) > 0] or [S[0]]
        cands = [c for c in candidates(k, core) if c not in S]
        if not cands:
            break
        if worst_strength <= 0:
            # FAULT: snap the non-belonging concept to the best coherent address
            S[S.index(worst)] = cands[0]
        else:
            # all connected — try to raise coherence; accept only if resolve improves (else fixpoint)
            trial = S[:]
            trial[trial.index(worst)] = cands[0]
            if resolve(k, trial)["pos_pmi"] > traj[-1] + 1e-9:
                S = trial
            else:
                break
        traj.append(resolve(k, S)["pos_pmi"])
    return S, traj


def candidates_anchored(k, core, anchor):
    """Like candidates(), but rank by summed above-chance PMI to the ANCHOR (the intent), not to the
    whole current set. Junk concepts can't nucleate their own cluster — every candidate must cohere
    with the seed, so the thought resolves TOWARD intent instead of drifting to the tightest cluster."""
    pool = set()
    for c in core:
        for b, _w in top_neighbors(k, c, 30):
            if b not in k.stop and b not in core:
                pool.add(b)
    ranked = []
    for b in pool:
        tot = 0.0
        for a in anchor:
            w = edge_w(k, b, a)
            if w is not None:
                x = k.assoc(b, a, w)
                if x > 0:
                    tot += x
        if tot > 0:
            ranked.append((tot, b))
    ranked.sort(reverse=True)
    return [b for _t, b in ranked]


def think_heal_anchored(k, draft, anchor, target, max_iter=14):
    """Heal to resolution while PINNED to the anchor: the anchor concepts are protected (never replaced),
    the core is what connects to the anchor, and candidates must cohere with the anchor."""
    anchor = list(dict.fromkeys(anchor))
    S = list(dict.fromkeys(list(anchor) + [c for c in draft if c not in anchor]))[:max(target, len(anchor))]

    def anchor_conn(c):
        pos = [k.assoc(c, a, w) for a in anchor if a != c
               and (w := edge_w(k, c, a)) is not None and k.assoc(c, a, w) > 0]
        return sum(pos) / len(pos) if pos else 0.0

    traj = [resolve(k, S)["pos_pmi"]]
    for _ in range(max_iter):
        core = anchor + [c for c in S if c not in anchor and anchor_conn(c) > 0]
        cands = [c for c in candidates_anchored(k, core, anchor) if c not in S]
        repl = [(anchor_conn(c), c) for c in S if c not in anchor]
        if not cands or not repl:
            break
        worst_strength, worst = min(repl)
        if worst_strength <= 0:
            S[S.index(worst)] = cands[0]
        else:
            trial = S[:]
            trial[trial.index(worst)] = cands[0]
            if resolve(k, trial)["pos_pmi"] > traj[-1] + 1e-9:
                S = trial
            else:
                break
        traj.append(resolve(k, S)["pos_pmi"])
    return S, traj


def fidelity(k, concepts, ref):
    """Topic-fidelity: mean learned-meaning similarity (embedding) of each concept to the reference
    thought. High = stayed on the intended topic; low = drifted."""
    vals = []
    for c in concepts:
        rs = [r for r in (k.relate(c, x) for x in ref if x != c) if r > -1]
        if rs:
            vals.append(sum(rs) / len(rs))
    return sum(vals) / len(vals) if vals else 0.0


def concepts_of(k, text):
    return list(dict.fromkeys(
        t for t in TOK.findall(text.lower()) if t not in k.stop and t in k.nodeset))[:MAX_CONCEPTS]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    k = open_web()
    from langexec import sample_real
    max_pid = k.db.execute("SELECT MAX(pid) FROM passages").fetchone()[0]
    print(f"[thinkloop] web nodes={len(k.nodes):,}  N={n} seed thoughts", flush=True)

    pool = [w for w in k.nodes if w not in k.stop]
    sents = sample_real(k, n, max_pid)
    # head-to-head on IDENTICAL drafts: does anchoring keep resolve high AND stop the drift?
    res_u, res_a, fid_u, fid_a = [], [], [], []
    fid_core, fid_rand = [], []
    examples = []
    for s in sents:
        core = concepts_of(k, s)
        if len(core) < 6:
            continue
        target = len(core)
        anchor = core[:2]                                            # the SEED / intent (small)
        inject = random.sample(pool, target - len(anchor))           # the rest is random junk (faults)
        draft = anchor + inject
        random.shuffle(draft)

        healed_u, _ = think_heal(k, draft, target)                   # unanchored: pure resolve-max
        healed_a, _ = think_heal_anchored(k, draft, anchor, target)  # anchored: pinned to the seed

        res_u.append(resolve(k, healed_u)["pos_pmi"])
        res_a.append(resolve(k, healed_a)["pos_pmi"])
        fid_u.append(fidelity(k, healed_u, core))                    # did it stay on the original topic?
        fid_a.append(fidelity(k, healed_a, core))
        fid_core.append(fidelity(k, core, core))                     # upper bound (the real thought)
        fid_rand.append(fidelity(k, inject, core))                   # lower bound (the junk)
        if len(examples) < 3:
            examples.append((core, anchor, healed_u, healed_a))

    m = len(res_u)
    avg = lambda v: sum(v) / len(v) if v else 0.0
    print(f"\n  thoughts measured: {m}   (anchor = 2 seed concepts; rest of draft = random junk)")
    print(f"  {'':<26}{'UNANCHORED':>12}{'ANCHORED':>12}")
    print(f"  {'resolve (pos_pmi)':<26}{avg(res_u):>12.3f}{avg(res_a):>12.3f}   ← both should climb")
    print(f"  {'topic-fidelity':<26}{avg(fid_u):>12.3f}{avg(fid_a):>12.3f}   ← anchored should WIN")
    print(f"\n  fidelity calibration:  original-thought {avg(fid_core):.3f} (max)   |   junk {avg(fid_rand):.3f} (min)")
    won = sum(1 for a, u in zip(fid_a, fid_u) if a > u)
    print(f"  anchored kept more on-topic than unanchored: {won}/{m} ({100*won/max(1,m):.0f}%)")

    for core, anchor, hu, ha in examples:
        print(f"\n  ── example ── seed/anchor: {', '.join(anchor)}")
        print(f"   original topic   : {', '.join(core[:8])}")
        print(f"   UNANCHORED healed: {', '.join(hu[:8])}")
        print(f"   ANCHORED   healed: {', '.join(ha[:8])}")


if __name__ == "__main__":
    main()

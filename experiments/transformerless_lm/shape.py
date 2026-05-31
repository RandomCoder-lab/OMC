"""shape.py — relation-SHAPE resonance: the seed of analogy and humor.

The user's insight (2026-05-30): "it may find a relation SHAPE in addressing with other off-topic things —
kind of like how jokes are formed in people." A relationship has a SHAPE in the learned address space:
the vector r(A->B) = vec(B) - vec(A). Two relationships RHYME when their shape vectors are parallel
(cosine high) — that's analogy ("A is to B as X is to Y"), and the structural pivot behind a joke (two
frames sharing a hidden shape). word2vec's king-man+woman=queen is exactly this arithmetic.

As the system learns more addresses, more shapes become available to rhyme. This finds resonant relation
pairs — including DISTANT/off-topic ones (the surprising rhymes = the funny/insightful ones). Agnostic:
shapes come from the learned embedding (corpus-derived); resonance is geometry.

Honest: this surfaces STRUCTURAL rhymes (parallel relation vectors). Whether a rhyme is profound, apt, or
merely funny is for a mind to judge — the system proposes the resonance + its strength; we don't oversell.
"""
import sys, itertools
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))
from connect import ConceptSpace

HERE = Path(__file__).parent


def relation_vec(cs, a, b):
    va, vb = cs.vec(a), cs.vec(b)
    if va is None or vb is None:
        return None
    r = vb - va
    n = r.norm()
    return r / n if n > 1e-6 else None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="pride_prejudice.txt")
    ap.add_argument("--min_cos", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cs = ConceptSpace((HERE / args.corpus).read_text(errors="replace"), seed=args.seed)
    ents = [e for e in cs.ents if cs.vec(e) is not None]

    # All GROUNDED relations (real co-occurring pairs) -> their shape vectors.
    rels = []
    for a in ents:
        for b in cs.adj[a]:
            if a < b:
                rv = relation_vec(cs, a, b)
                if rv is not None:
                    rels.append((a, b, rv))
    print(f"[shape] {len(rels)} grounded relations -> shape vectors in learned address space", flush=True)

    # Resonance = parallel shape vectors between relations that DON'T share an endpoint (distinct pairs).
    # The most resonant-yet-disjoint rhymes are the analogies / joke-seeds.
    M = torch.stack([r[2] for r in rels])                 # [R, d]
    S = M @ M.t()                                          # cosine (unit vectors)
    rhymes = []
    for i in range(len(rels)):
        for j in range(i + 1, len(rels)):
            a1, b1, _ = rels[i]; a2, b2, _ = rels[j]
            if {a1, b1} & {a2, b2}:                        # disjoint endpoints -> a real cross-pair rhyme
                continue
            c = float(S[i, j])
            if c >= args.min_cos:
                # "distance" of the two relations in concept space (off-topic-ness): low overlap of company
                rhymes.append((c, (a1, b1), (a2, b2)))
    rhymes.sort(reverse=True)
    print(f"[shape] {len(rhymes)} resonant relation-pairs (parallel shapes, disjoint endpoints, "
          f"cos>={args.min_cos})\n", flush=True)

    print(f"[shape] === STRONGEST shape rhymes: 'A is to B as X is to Y' (analogy / the joke-pivot) ===", flush=True)
    for c, (a1, b1), (a2, b2) in rhymes[:10]:
        print(f"[shape]   [{c:.2f}]  {a1.title()}→{b1.title()}   rhymes with   {a2.title()}→{b2.title()}", flush=True)

    # analogy completion as a capability: given A->B and X, find Y s.t. r(X->Y) ~ r(A->B)
    def analogy(a, b, x, topn=3):
        rv = relation_vec(cs, a, b)
        vx = cs.vec(x)
        if rv is None or vx is None:
            return []
        target = vx + rv                                   # X + (B-A): "complete the shape"
        target = target / (target.norm() + 1e-9)
        scored = [(float(cs.vec(y) @ target), y) for y in ents if y not in (a, b, x)]
        return sorted(scored, reverse=True)[:topn]

    print(f"\n[shape] === analogy completion (shape arithmetic: X + (B−A) ≈ ?) ===", flush=True)
    if len(rels) >= 2:
        # show a few using the strongest grounded relations as the template
        seeds = rels[:4]
        others = [e for e in ents]
        for a, b, _ in seeds:
            x = next((e for e in others if e not in (a, b)), None)
            if x is None:
                continue
            comp = analogy(a, b, x)
            if comp:
                ans = ", ".join(f"{y.title()}({s:.2f})" for s, y in comp)
                print(f"[shape]   {a.title()} is to {b.title()} as {x.title()} is to → {ans}", flush=True)

    import json
    (HERE / "results_shape.json").write_text(json.dumps(dict(
        corpus=args.corpus, relations=len(rels), rhymes=len(rhymes),
        top_rhyme=(round(rhymes[0][0], 3) if rhymes else None)), indent=2))
    print(f"\n[shape] wrote results_shape.json", flush=True)
    print(f"[shape] NOTE: the system PROPOSES structural rhymes + strengths; aptness/humor is a mind's "
          f"call. This is the analogy/joke MECHANISM (parallel shapes), not a claim each rhyme is profound.", flush=True)


if __name__ == "__main__":
    main()

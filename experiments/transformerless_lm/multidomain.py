"""multidomain.py — point the full stack at GENUINELY distant domains; hunt cross-domain analogies.

Five domains (Gutenberg, distinct fields): science(Darwin) · detective(Holmes) · history(Gibbon) ·
philosophy(Nietzsche) · romance(Austen). ALL trained into ONE shared meaning-space (so vectors are
comparable across domains), but GROUNDING stays within-domain (no passage spans two books). So a
cross-domain connection CANNOT come from a shared passage — it must come from:
  MEANING   — two concepts play structurally similar roles (learned vector similarity across domains).
  SHAPE     — a relationship in one domain RHYMES with one in another (parallel shape vectors) = the
              cross-domain ANALOGY / the joke-pivot the user pointed at. Each side quoted from its book.

Honest design: cross-domain rhymes are PROPOSED with strength; aptness is a mind's call. We measure
whether real cross-domain shape-cosines exceed a shuffled-vector baseline (is the structure real or noise?)
and show the strongest, each grounded by a quote from each domain + narrated with calibrated confidence.
"""
import sys, re, time, json, random
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))
from nl_ground import clean_text, split_passages, extract_entities
from connect import train_embedding

HERE = Path(__file__).parent
CORP = HERE / "corpora"


def contains(e, pl):
    return any(s in pl for s in (" " + e + " ", " " + e + ",", " " + e + ".", " " + e + "'", " " + e + ";"))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_domain_ents", type=int, default=30)
    ap.add_argument("--embed_steps", type=int, default=7000)
    ap.add_argument("--min_cos", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    files = sorted(CORP.glob("*.txt"))
    domains, passages, dom_of_pass, big = [], [], [], []
    for f in files:
        dom = f.stem.split("_")[0]
        domains.append(dom)
        txt = clean_text(f.read_text(errors="replace"))
        ps = split_passages(txt)
        for p in ps:
            passages.append(p); dom_of_pass.append(dom)
        big.append(txt)
        print(f"[md] {dom:12} {len(ps):5} passages", flush=True)
    plow = [" " + p.lower() + " " for p in passages]
    full = "\n".join(big)

    # ONE shared meaning space across all domains
    t0 = time.time()
    vec, stoi = train_embedding(full, dim=64, steps=args.embed_steps, max_vocab=12000, seed=args.seed, strip_top=1)
    print(f"[md] shared meaning-space trained over {len(domains)} domains ({time.time()-t0:.0f}s)", flush=True)

    # per-domain entities, tagged with home domain (skip cross-domain-common tokens = weak signal)
    ent_dom = {}
    for f, dom in zip(files, domains):
        for e in extract_entities(clean_text(f.read_text(errors="replace")), top_n=args.per_domain_ents):
            el = e.lower()
            if vec(el) is None:
                continue
            ent_dom.setdefault(el, set()).add(dom)
    # keep entities that are characteristic of ONE domain (drop pan-domain commons like 'god','england')
    ents_by_dom = {d: [] for d in domains}
    for e, ds in ent_dom.items():
        if len(ds) == 1:
            ents_by_dom[next(iter(ds))].append(e)
    for d in domains:
        print(f"[md] {d:12} concepts: {ents_by_dom[d][:10]}", flush=True)

    # within-domain grounded relations -> shape vectors (each tagged with domain + evidence passage)
    rels = []   # (dom, a, b, shape_vec, evidence_passage_idx)
    for di, dom in enumerate(domains):
        E = ents_by_dom[dom]
        idxs = [i for i in range(len(passages)) if dom_of_pass[i] == dom]
        seen = set()
        for i in idxs:
            present = [e for e in E if contains(e, plow[i])]
            for a in present:
                for b in present:
                    if a < b and (a, b) not in seen:
                        va, vb = vec(a), vec(b)
                        r = vb - va; n = r.norm()
                        if n > 1e-6:
                            rels.append((dom, a, b, r / n, i)); seen.add((a, b))
    print(f"\n[md] {len(rels)} within-domain grounded relations across {len(domains)} domains", flush=True)

    # ── CROSS-DOMAIN SHAPE RHYMES: parallel relation-shapes from DIFFERENT domains ──
    M = torch.stack([r[3] for r in rels])
    S = M @ M.t()
    cross = []
    for i in range(len(rels)):
        for j in range(i + 1, len(rels)):
            if rels[i][0] == rels[j][0]:
                continue                                   # different domains only
            if {rels[i][1], rels[i][2]} & {rels[j][1], rels[j][2]}:
                continue
            c = float(S[i, j])
            if c >= args.min_cos:
                cross.append((c, i, j))
    cross.sort(reverse=True)

    # baseline: is cross-domain structure real? compare to shuffled (random) shape vectors
    g = torch.Generator().manual_seed(args.seed)
    Mr = torch.randn(M.shape, generator=g); Mr = Mr / Mr.norm(dim=1, keepdim=True)
    Sr = (Mr @ Mr.t())
    real_top = sorted((float(S[i, j]) for i in range(len(rels)) for j in range(i + 1, len(rels))
                       if rels[i][0] != rels[j][0]), reverse=True)[:200]
    rand_top = sorted((float(Sr[i, j]) for i in range(min(len(rels), 600))
                       for j in range(i + 1, min(len(rels), 600))), reverse=True)[:200]
    print(f"[md] cross-domain shape-cosine: real top-200 mean={sum(real_top)/len(real_top):.3f}  "
          f"vs shuffled baseline={sum(rand_top)/len(rand_top):.3f}  "
          f"-> {'STRUCTURE IS REAL' if sum(real_top)/len(real_top) > sum(rand_top)/len(rand_top)+0.1 else 'near noise (honest)'}", flush=True)

    def q(i, w=120):
        s = passages[i].strip().replace("\n", " ")
        return (s[:w] + "…") if len(s) > w else s

    print(f"\n[md] === CROSS-DOMAIN ANALOGIES (a relationship in one field rhyming with another) ===", flush=True)
    shown, used = 0, set()
    for c, i, j in cross:
        d1, a1, b1, _, e1 = rels[i]; d2, a2, b2, _, e2 = rels[j]
        key = (d1, d2, a1, a2)
        if (d1, d2) in used or (d2, d1) in used:           # diversify across domain-pairs
            continue
        used.add((d1, d2))
        print(f"\n[md] [{c:.2f}]  {a1.title()}→{b1.title()} ⟨{d1}⟩   rhymes with   {a2.title()}→{b2.title()} ⟨{d2}⟩", flush=True)
        print(f"[md]   ⟨{d1}⟩ \"{q(e1)}\"", flush=True)
        print(f"[md]   ⟨{d2}⟩ \"{q(e2)}\"", flush=True)
        shown += 1
        if shown >= 8:
            break

    # cross-domain MEANING neighbors: which concept in another domain plays a similar role?
    print(f"\n[md] === cross-domain MEANING neighbors (roles that rhyme across fields) ===", flush=True)
    for d in domains:
        for a in ents_by_dom[d][:2]:
            best = []
            for d2 in domains:
                if d2 == d:
                    continue
                for b in ents_by_dom[d2]:
                    best.append((float(vec(a) @ vec(b)), b, d2))
            best.sort(reverse=True)
            tops = ", ".join(f"{b.title()}⟨{d2}⟩{s:.2f}" for s, b, d2 in best[:3])
            print(f"[md]   {a.title():>12} ⟨{d}⟩  ~  {tops}", flush=True)

    (HERE / "results_multidomain.json").write_text(json.dumps(dict(
        domains=domains, passages=len(passages), relations=len(rels),
        cross_rhymes=len(cross), real_top_mean=round(sum(real_top)/len(real_top), 3),
        rand_top_mean=round(sum(rand_top)/len(rand_top), 3)), indent=2))
    print(f"\n[md] wrote results_multidomain.json", flush=True)


if __name__ == "__main__":
    main()

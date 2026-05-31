"""process.py — abstract RELATIONS, not proper nouns: rhyme PROCESSES (verbs) across domains.

BUILD-20 rhymed entities/places ("Rome is like Pemberley"). The deep analogies live in VERBS — the
actions/processes a field is built on: science SELECTS/VARIES/INHERITS, detection OBSERVES/DEDUCES,
history CONQUERS/REIGNS, romance MARRIES/ESTEEMS. This:
  1. extracts verbs AGNOSTICALLY (grammatical signal: words that follow pronouns/modals + verb morphology
     — closed-class function words only, NO domain dictionary).
  2. names each domain's SIGNATURE processes (domain-TF-IDF: verbs disproportionately this field's).
  3. rhymes processes ACROSS domains via the shared meaning-space (nearest signature-verb pairs) — the
     step toward "natural selection is like detective deduction".
Each cross-domain process-rhyme is grounded by an example sentence from each field; a mind judges aptness.
Honest baseline: are real cross-domain verb-similarities above a shuffled baseline (real structure or noise)?
"""
import sys, re, time, json, math
from collections import Counter, defaultdict
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))
from nl_ground import clean_text, split_passages
from connect import train_embedding

HERE = Path(__file__).parent
CORP = HERE / "corpora"
WORD = re.compile(r"[a-z]+")

# NO hand-coded word lists (law: no dictionaries in the authoritative path). Verb-ness is DERIVED from the
# corpus by the INFLECTION PARADIGM: a word is a verb iff its gerund (-ing) AND past (-ed) forms both occur
# in the text. Uses only general suffix OPERATIONS (e-drop, consonant-doubling, y->i) + corpus statistics —
# language-general in principle, zero hand-listed vocabulary. Function words don't inflect, so they're
# excluded automatically; 'red'/'something' fail the paradigm and drop out for free.


def _ing_stems(w):
    """plausible base forms for a gerund 'w' (…ing), via general spelling rules."""
    b = w[:-3]
    out = {b, b + "e"}                       # play->playing ; produc(e)->producing
    if len(b) >= 2 and b[-1] == b[-2]:
        out.add(b[:-1])                      # run->running (undouble)
    return out


def _ed_stems(w):
    """plausible base forms for a past tense 'w' (…ed)."""
    b = w[:-2]
    out = {b, b + "e"}                        # play->played ; produc(e)->produced
    if len(b) >= 2 and b[-1] == b[-2]:
        out.add(b[:-1])                       # stop->stopped (undouble)
    if b.endswith("i"):
        out.add(b[:-1] + "y")                 # marri->marry (i->y)
    return out


def derive_verb_stems(token_counts, min_count=2):
    """A stem is VERBAL iff BOTH a gerund and a past form reduce to it (the verb inflection paradigm)."""
    ing, ed = set(), set()
    for w, c in token_counts.items():
        if c < min_count:
            continue
        if w.endswith("ing") and len(w) >= 5:
            ing |= _ing_stems(w)
        if w.endswith("ed") and len(w) >= 4:
            ed |= _ed_stems(w)
    return ing & ed                           # has both -> participates in the verbal paradigm


def to_stem(w, verbal):
    """Reduce a surface token to its verbal stem if it belongs to the derived verbal set, else None."""
    if w in verbal:
        return w
    if w.endswith("ing") and len(w) >= 5:
        for s in _ing_stems(w):
            if s in verbal:
                return s
    if w.endswith("ed") and len(w) >= 4:
        for s in _ed_stems(w):
            if s in verbal:
                return s
    if w.endswith("es") and w[:-2] in verbal:
        return w[:-2]
    if w.endswith("s") and len(w) >= 4 and w[:-1] in verbal:
        return w[:-1]
    return None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_verbs", type=int, default=12)
    ap.add_argument("--cand", type=int, default=120)
    ap.add_argument("--embed_steps", type=int, default=7000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = sorted(CORP.glob("*.txt"))
    dom_tokens, dom_pass, big = {}, {}, []
    for f in files:
        dom = f.stem.split("_")[0]
        txt = clean_text(f.read_text(errors="replace"))
        dom_pass[dom] = split_passages(txt)
        dom_tokens[dom] = WORD.findall(txt.lower())
        big.append(txt)
    domains = list(dom_tokens)
    full = "\n".join(big)

    # ── (1) agnostic verb extraction by the DERIVED inflection paradigm (no word lists) ──
    full_counts = Counter(WORD.findall(full.lower()))
    verbal = derive_verb_stems(full_counts)                 # corpus-derived verb stems
    print(f"[proc] derived {len(verbal)} verb stems from the inflection paradigm (no hand-coded lists)", flush=True)
    dom_verbstem = {}
    for dom, toks in dom_tokens.items():
        cnt = Counter(); surface = defaultdict(Counter)
        for w in toks:
            s = to_stem(w, verbal)
            if s is not None:
                cnt[s] += 1; surface[s][w] += 1
        cand = [s for s, _ in cnt.most_common(args.cand)]
        dom_verbstem[dom] = {s: (cnt[s], surface[s].most_common(1)[0][0]) for s in cand}

    # ── (2) domain SIGNATURE verbs: TF-IDF over domains (disproportionately this field's process) ──
    df = Counter()
    for dom in domains:
        for s in dom_verbstem[dom]:
            df[s] += 1
    sig = {}
    for dom in domains:
        tot = sum(c for c, _ in dom_verbstem[dom].values()) or 1
        scored = []
        for s, (c, surf) in dom_verbstem[dom].items():
            tf = c / tot
            idf = math.log(len(domains) / df[s])
            scored.append((tf * (idf + 0.01), surf, s))
        scored.sort(reverse=True)
        sig[dom] = scored[:args.top_verbs]
    print("[proc] === each domain's SIGNATURE processes (verbs it does disproportionately) ===", flush=True)
    for dom in domains:
        print(f"[proc]   {dom:11}: {', '.join(surf for _, surf, _ in sig[dom])}", flush=True)

    # ── (3) shared meaning-space; rhyme signature-verbs ACROSS domains by nearest meaning ──
    t0 = time.time()
    vec, stoi = train_embedding(full, dim=64, steps=args.embed_steps, max_vocab=14000, seed=args.seed, strip_top=1)
    print(f"\n[proc] shared meaning-space trained ({time.time()-t0:.0f}s)", flush=True)

    # collect embeddable signature verbs per domain (use the surface form present in the embedding)
    sigv = {}
    for dom in domains:
        vs = []
        for _, surf, st in sig[dom]:
            v = vec(surf) if vec(surf) is not None else vec(st)
            if v is not None:
                vs.append((surf, st, v))
        sigv[dom] = vs

    # cross-domain process rhymes: nearest signature-verb pairs from different domains
    rh = []
    for i, d1 in enumerate(domains):
        for d2 in domains[i + 1:]:
            for s1, st1, v1 in sigv[d1]:
                for s2, st2, v2 in sigv[d2]:
                    if st1 == st2:
                        continue
                    rh.append((float(v1 @ v2), d1, s1, d2, s2))
    rh.sort(reverse=True)

    # honest baseline: real cross-domain verb sims vs shuffled random vectors of same count
    g = torch.Generator().manual_seed(args.seed)
    allv = [v for dom in domains for _, _, v in sigv[dom]]
    R = torch.randn(len(allv), 64, generator=g); R = R / R.norm(dim=1, keepdim=True)
    rand_sims = sorted((float(R[i] @ R[j]) for i in range(len(allv)) for j in range(i + 1, len(allv))), reverse=True)
    real_top = [c for c, *_ in rh[:40]]
    rand_topn = rand_sims[:40]
    print(f"[proc] cross-domain verb-sim: real top-40 mean={sum(real_top)/len(real_top):.3f} vs shuffled "
          f"{sum(rand_topn)/len(rand_topn):.3f} -> {'REAL STRUCTURE' if sum(real_top)/40 > sum(rand_topn)/40+0.1 else 'near noise'}", flush=True)

    def example(dom, surf):
        for p in dom_pass[dom]:
            if re.search(r"\b" + re.escape(surf) + r"\b", p.lower()):
                s = p.strip().replace("\n", " ")
                return (s[:130] + "…") if len(s) > 130 else s
        return ""

    print(f"\n[proc] === CROSS-DOMAIN PROCESS RHYMES ('field A's <verb> is like field B's <verb>') ===", flush=True)
    shown, used = 0, set()
    for c, d1, s1, d2, s2 in rh:
        if (d1, d2) in used:
            continue
        used.add((d1, d2))
        print(f"\n[proc] [{c:.2f}]  {d1}:{s1.upper()}  ≈  {d2}:{s2.upper()}", flush=True)
        print(f"[proc]   ⟨{d1}⟩ \"{example(d1, s1)}\"", flush=True)
        print(f"[proc]   ⟨{d2}⟩ \"{example(d2, s2)}\"", flush=True)
        shown += 1
        if shown >= 8:
            break

    (HERE / "results_process.json").write_text(json.dumps(dict(
        domains=domains, signature={d: [s for _, s, _ in sig[d]] for d in domains},
        cross_rhymes=len(rh), real_top_mean=round(sum(real_top)/len(real_top), 3),
        rand_top_mean=round(sum(rand_topn)/len(rand_topn), 3)), indent=2))
    print(f"\n[proc] wrote results_process.json", flush=True)


if __name__ == "__main__":
    main()

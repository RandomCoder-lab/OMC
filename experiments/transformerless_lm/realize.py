#!/usr/bin/env python3
"""realize.py — THE REALIZER: resolved concepts → fluent, grounded sentence  (WHAT meets HOW)

Three strategies, head-to-head, all over the web-learned (trigram) fluency model:
  COMPOSE  — concept-constrained beam over the fluency model; glue = the web's derived stopwords (agnostic).
             Perfect coverage, weak fluency (a model can only order; it can't borrow real structure).
  TEMPLATE — "the web says it this way": retrieve the grounded passage covering the most concepts, trim to
             span. Real-sentence fluency BY CONSTRUCTION; partial coverage (it lacks some target concepts).
  HYBRID   — template STRUCTURE healed to the exact concept set: graft each missing concept into the most-
             fluent filler slot (fluency-gated swap). Template fluency × compose coverage.

Round-trip: re-execute the output — a good utterance still RESOLVES (means it) AND reads fluently.

  python realize.py [N]
"""
import sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from langexec import open_web, sample_real
from fluency import get_fluency_model, fluency as seq_fluency, logp, WORD
from thinkloop import resolve, concepts_of

random.seed(2718)
REALIZE_TRAIN = 80000          # fluency model size (cached after first run)


# ── COMPOSE ──────────────────────────────────────────────────────────────────
def top_glue(model, last, stop, n=6):
    if last is None or last not in model["bi"]:
        return [w for w, _ in model["uni"].most_common(60) if w in stop][:n]
    ranked = sorted(model["bi"][last].items(), key=lambda x: -x[1])
    return [w for w, _ in ranked if w in stop][:n]


def compose_realize(model, concepts, stop, beam=14, cov_bonus=9.0, max_glue=8):
    concepts = list(dict.fromkeys(concepts))
    if not concepts:
        return []
    full = set(concepts)
    cap = len(concepts) + max_glue
    beams = sorted([(cov_bonus, [c], full - {c}) for c in concepts], reverse=True)[:beam]
    terminals = []
    for _ in range(cap):
        nxt = []
        for score, seq, remaining in beams:
            if not remaining:
                terminals.append((score, seq)); continue
            a = seq[-2] if len(seq) >= 2 else None
            b = seq[-1]
            cands = list(remaining) + [g for g in top_glue(model, b, stop, 6) if g not in seq[-2:]]
            for c in cands:
                s2 = score + logp(model, a, b, c) + (cov_bonus if c in remaining else 0.0)
                nxt.append((s2, seq + [c], remaining - {c}))
        if not nxt:
            break
        beams = sorted(nxt, reverse=True)[:beam]
    terminals += [(s, seq) for s, seq, rem in beams if not rem]
    if not terminals:
        terminals = [(s, seq) for s, seq, _ in beams]
    terminals.sort(key=lambda x: seq_fluency(model, x[1]), reverse=True)
    return terminals[0][1]


# ── TEMPLATE ─────────────────────────────────────────────────────────────────
def template_realize(k, concepts, margin=3, max_cand=16):
    cs = list(dict.fromkeys(concepts))
    cset = set(cs)
    cand = []
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            r = k.db.execute("SELECT pid,w FROM edges WHERE a=? AND b=?", (cs[i], cs[j])).fetchone()
            if r:
                cand.append((r[1] or 1, r[0]))
    cand.sort(reverse=True)
    best, best_cov = None, -1
    for _w, pid in cand[:max_cand]:
        row = k.db.execute("SELECT text FROM passages WHERE pid=?", (pid,)).fetchone()
        if not row:
            continue
        toks = WORD.findall(row[0].lower())
        idx = [i for i, t in enumerate(toks) if t in cset]
        if not idx:
            continue
        cov = len(set(toks[i] for i in idx) & cset)
        if cov > best_cov:
            lo, hi = max(0, idx[0] - margin), min(len(toks), idx[-1] + 1 + margin)
            best, best_cov = toks[lo:hi], cov
    return best or cs


# ── HYBRID: template structure, healed to the exact concepts ──────────────────
def hybrid_realize(k, model, concepts, stop):
    """Take the template's fluent STRUCTURE, then HEAL it toward the exact concept set: graft each missing
    concept into the filler slot (a content word that isn't a target) whose replacement keeps fluency
    highest. Glue/structure words stay — so fluency≈template while coverage→1."""
    span = list(template_realize(k, concepts))
    target = set(concepts)
    for c in concepts:
        if c in span:
            continue
        fillers = [i for i, w in enumerate(span) if w not in stop and w not in target]
        if not fillers:
            span = span + [c]
            continue
        best = None
        for i in fillers:
            trial = span[:]; trial[i] = c
            f = seq_fluency(model, trial)
            if best is None or f > best[0]:
                best = (f, i)
        span[best[1]] = c
    return span


def heal_surface(toks):
    if not toks:
        return ""
    return (toks[0].capitalize() + " " + " ".join(toks[1:])).strip() + "."


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    k = open_web()
    max_pid = k.db.execute("SELECT MAX(pid) FROM passages").fetchone()[0]
    print(f"[realize] loading fluency model (~{REALIZE_TRAIN:,} passages, cached) ...", flush=True)
    model = get_fluency_model(k, REALIZE_TRAIN, max_pid)
    print(f"[realize] vocab={model['V']:,}  trigrams={len(model['tri']):,}", flush=True)

    sents = sample_real(k, n, max_pid)
    fl_real, fl_bag = [], []
    F = {"compose": [], "template": [], "hybrid": []}
    C = {"compose": [], "template": [], "hybrid": []}
    R = {"compose": [], "template": [], "hybrid": []}
    examples = []
    for s in sents:
        core = concepts_of(k, s)[:7]
        if len(core) < 4:
            continue
        outs = {
            "compose": compose_realize(model, core, k.stop),
            "template": template_realize(k, core),
            "hybrid": hybrid_realize(k, model, core, k.stop),
        }
        fl_real.append(seq_fluency(model, WORD.findall(s.lower())[:24]))
        fl_bag.append(seq_fluency(model, core))
        for name, o in outs.items():
            F[name].append(seq_fluency(model, o))
            C[name].append(sum(1 for c in core if c in o) / len(core))
            R[name].append(resolve(k, [w for w in o if w in k.nodeset])["pos_pmi"])
        if len(examples) < 5:
            examples.append((core, {n2: heal_surface(o) for n2, o in outs.items()}))

    m = len(fl_real)
    avg = lambda v: sum(v) / len(v) if v else 0.0
    print(f"\n  realized {m} thoughts    (bag {avg(fl_bag):.2f} → real-web {avg(fl_real):.2f})")
    print(f"  {'':<20}{'COMPOSE':>10}{'TEMPLATE':>10}{'HYBRID':>10}")
    print(f"  {'fluency':<20}{avg(F['compose']):>10.2f}{avg(F['template']):>10.2f}{avg(F['hybrid']):>10.2f}")
    print(f"  {'coverage':<20}{avg(C['compose']):>10.2f}{avg(C['template']):>10.2f}{avg(C['hybrid']):>10.2f}")
    print(f"  {'round-trip resolve':<20}{avg(R['compose']):>10.3f}{avg(R['template']):>10.3f}{avg(R['hybrid']):>10.3f}")
    print(f"\n  → HYBRID aims for TEMPLATE's fluency AND COMPOSE's coverage.")

    print("\n  ── concepts → sentence ──")
    for core, outs in examples:
        print(f"   concepts: {', '.join(core)}")
        print(f"   hybrid  : {outs['hybrid']}")
        print(f"   template: {outs['template']}\n")


if __name__ == "__main__":
    main()

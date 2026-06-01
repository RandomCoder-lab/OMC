#!/usr/bin/env python3
"""fluency.py — "TRAIN IT ON THE WEB ON HOW TO SPEAK"  (the HOW oracle)

The web's own passages TEACH fluency: derive word-transition statistics from them (counted, agnostic — no
hand-coded grammar) and a fluent sentence scores HIGH while the SAME words SCRAMBLED score LOW. If the
web-learned model separates fluent word-ORDER from scrambled, "how to speak" is learnable from the web.

v2: TRIGRAM with stupid-backoff (Brants et al.) — P(w|a,b) → 0.4·P(w|b) → 0.16·P(w). Trigrams see two
words of context (local grammar/agreement) so they separate better than bigrams. Singleton trigrams are
pruned (they don't generalize and they cost memory; backoff covers them). Held-out: learn on TRAIN pids,
score on DISJOINT TEST pids.

  python fluency.py [N_test]
"""
import sys, re, math, random, pickle
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from kdb import KnowledgeDB, load_embedding
from langexec import auc

HERE = Path(__file__).parent
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
random.seed(31415)
TRAIN_PASSAGES = 40000
ALPHA = 0.4          # stupid-backoff weight per dropped order


def get_fluency_model(k, n_train, max_pid, cache=True):
    """Train (or load cached) the web fluency model. Cached to disk so realize/engine reuse the GOOD
    (large-n) model without retraining each run."""
    path = HERE / f".fluencycache_{n_train}"
    if cache and path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    pids = [random.randint(0, max_pid) for _ in range(n_train)]
    m = learn_transitions(k, pids)
    if cache:
        with open(path, "wb") as f:
            pickle.dump(m, f)
    return m


def open_web():
    s, E, n, e = load_embedding()
    k = KnowledgeDB(str(Path(__file__).parent / "knowledge.db"), s, E, n, e)
    k.db.execute("PRAGMA busy_timeout=8000")
    return k


def learn_transitions(k, pids):
    """COUNT uni/bi/tri from the web's passages = learn 'how to speak'. Pure counting, agnostic. Singleton
    trigrams pruned (backoff covers them) to bound memory."""
    uni = Counter()
    bi = defaultdict(Counter)
    tri = defaultdict(int)
    bictx = defaultdict(int)
    for pid in pids:
        r = k.db.execute("SELECT text FROM passages WHERE pid=?", (pid,)).fetchone()
        if not r:
            continue
        t = WORD.findall(r[0].lower())
        uni.update(t)
        for a, b in zip(t, t[1:]):
            bi[a][b] += 1
        for a, b, c in zip(t, t[1:], t[2:]):
            tri[(a, b, c)] += 1
            bictx[(a, b)] += 1
    ctx = {a: sum(c.values()) for a, c in bi.items()}
    tri = {key: v for key, v in tri.items() if v >= 2}      # prune singletons (memory; they don't generalize)
    return dict(uni=uni, bi=bi, ctx=ctx, tri=tri, bictx=dict(bictx),
                V=len(uni), N=sum(uni.values()))


def logp(model, a, b, w):
    """log P(w | a, b) with stupid-backoff. a may be None (sentence start / <2 history)."""
    tri, bictx, bi, ctx, uni = model["tri"], model["bictx"], model["bi"], model["ctx"], model["uni"]
    V, N = model["V"], model["N"]
    if a is not None:
        bc = bictx.get((a, b), 0)
        if bc and (a, b, w) in tri:
            return math.log(tri[(a, b, w)] / bc)
    if b in ctx and bi[b].get(w, 0) > 0:
        return math.log(ALPHA * bi[b][w] / ctx[b])
    return math.log(ALPHA * ALPHA * (uni.get(w, 0) + 0.5) / (N + 0.5 * V))


def fluency(model, toks):
    """Mean log P per word under the web-learned trigram model. Higher = the web would more likely say it."""
    if len(toks) < 2:
        return -99.0
    lp = 0.0
    for i in range(1, len(toks)):
        a = toks[i - 2] if i >= 2 else None
        lp += logp(model, a, toks[i - 1], toks[i])
    return lp / (len(toks) - 1)


def main():
    n_test = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else TRAIN_PASSAGES
    k = open_web()
    max_pid = k.db.execute("SELECT MAX(pid) FROM passages").fetchone()[0]
    print(f"[fluency] trigram; training ~{n_train:,} passages, testing {n_test}", flush=True)
    test_pids = set(random.sample(range(max_pid + 1), n_test * 3))
    train_pids = [p for p in (random.randint(0, max_pid) for _ in range(n_train)) if p not in test_pids]
    model = learn_transitions(k, train_pids)
    print(f"[fluency] vocab={model['V']:,}  bigram-ctx={len(model['ctx']):,}  trigrams(kept)={len(model['tri']):,}", flush=True)

    real, shuf, rand = [], [], []
    vocab = [w for w, _ in model["uni"].most_common(50000)]
    ex = []
    for pid in test_pids:
        if len(real) >= n_test:
            break
        r = k.db.execute("SELECT text FROM passages WHERE pid=?", (pid,)).fetchone()
        if not r:
            continue
        toks = WORD.findall(r[0].lower())[:20]
        if len(toks) < 6:
            continue
        sh = toks[:]
        random.shuffle(sh)
        rd = random.sample(vocab, len(toks)) if len(vocab) >= len(toks) else toks
        real.append(fluency(model, toks)); shuf.append(fluency(model, sh)); rand.append(fluency(model, rd))
        if len(ex) < 2:
            ex.append((toks, sh))

    avg = lambda v: sum(v) / len(v) if v else 0.0
    print(f"\n  test sentences: {len(real)}    (mean log-prob/word; higher=more fluent)")
    print(f"    REAL (fluent web order) {avg(real):>8.2f}")
    print(f"    SHUFFLED (same words)   {avg(shuf):>8.2f}")
    print(f"    RANDOM words            {avg(rand):>8.2f}")
    print(f"\n  SEPARATION (AUC: P[REAL higher]; 0.5=nothing learned, →1.0='how to speak' learned):")
    print(f"    REAL vs SHUFFLED  AUC = {auc(real, shuf):.3f}   (pure word-ORDER / grammar)   [bigram was 0.827]")
    print(f"    REAL vs RANDOM    AUC = {auc(real, rand):.3f}")
    print(f"\n  example REAL: {' '.join(ex[0][0])}")
    print(f"  example SHUF: {' '.join(ex[0][1])}")


if __name__ == "__main__":
    main()

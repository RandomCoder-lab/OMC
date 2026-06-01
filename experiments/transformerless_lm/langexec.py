#!/usr/bin/env python3
"""langexec.py — IS LANGUAGE EXECUTABLE OVER THE WEB?  (the load-bearing experiment)

Principle under test (omc_addressing_is_execution): addressing = executability. To "execute" a
sentence is to traverse its concept-addresses along the web's WEIGHTED edges and see whether it
RESOLVES (the concepts actually connect, above chance, not just through hubs) or FAULTS (no path).

Falsifiable prediction: a real, coherent sentence resolves to a tight, hub-damped, well-connected
subgraph; a word-salad of RANDOM REAL words (the HARD negative — every word is a valid address, so
lexical coverage can't be the giveaway) FAULTS — its concepts don't co-occur, so edges are absent
and PMI collapses. If the executed-score SEPARATES the two, language-execution is real. If a real
sentence and a salad score the same, the principle is wrong — and we learn that cheaply.

Execution score per sentence (all from the existing KnowledgeDB primitives, nothing new invented):
  coverage     fraction of content tokens that are real concept-addresses
  edge_density fraction of concept PAIRS joined by a direct weighted edge
  mean_pmi     mean assoc() PMI over connected pairs (hub-damped: log(w·N/(deg_a·deg_b)))
  pos_pmi      fraction of connected pairs co-occurring ABOVE chance (PMI>0)
  resolve      coverage · edge_density  (the composite "did it execute")

  python langexec.py [N]     # N sentences per condition (default 60)
"""
import sys, re, math, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from kdb import KnowledgeDB, load_embedding

TOK = re.compile(r"[a-z][a-z']+")
random.seed(1729)   # deterministic (no Math.random equivalent abuse — fixed seed)


def open_web():
    s, E, n, e = load_embedding()
    k = KnowledgeDB(str(Path(__file__).parent / "knowledge.db"), s, E, n, e)
    k.db.execute("PRAGMA busy_timeout=8000")   # tolerate the live growth-loop writer (WAL)
    return k


def edge_w(k, a, b):
    """Direct weighted edge a–b (indexed point query). Fold inserts both directions; check both."""
    r = k.db.execute("SELECT w FROM edges WHERE a=? AND b=?", (a, b)).fetchone()
    if r:
        return r[0] or 1
    r = k.db.execute("SELECT w FROM edges WHERE a=? AND b=?", (b, a)).fetchone()
    return (r[0] or 1) if r else None


def execute(k, text, max_concepts=14):
    """Traverse the sentence's concept-addresses over the weighted web. Returns the execution metrics."""
    content = [t for t in TOK.findall(text.lower()) if t not in k.stop]
    if not content:
        return None
    addressed = [t for t in content if t in k.nodeset]
    coverage = len(addressed) / len(content)
    # unique concepts, capped (bounds the O(n^2) pair scan; comparable across conditions)
    A = list(dict.fromkeys(addressed))[:max_concepts]
    if len(A) < 2:
        return dict(coverage=coverage, edge_density=0.0, mean_pmi=0.0, pos_pmi=0.0,
                    resolve=0.0, n_concepts=len(A))
    pmis = []
    total = 0
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            total += 1
            w = edge_w(k, A[i], A[j])
            if w is not None:
                pmis.append(k.assoc(A[i], A[j], w))
    edge_density = len(pmis) / total if total else 0.0
    mean_pmi = sum(pmis) / len(pmis) if pmis else 0.0
    pos_pmi = sum(1 for p in pmis if p > 0) / len(pmis) if pmis else 0.0
    # RESOLVE (corrected): coverage × ABOVE-CHANCE connectivity. The steelman proved raw edge_density is
    # hub-fooled (common words always have edges); pos_pmi only counts pairs co-occurring MORE than their
    # promiscuity predicts — the hub-damped signal. This is "did it execute meaningfully", not "are there edges".
    resolve = coverage * pos_pmi
    return dict(coverage=coverage, edge_density=edge_density, mean_pmi=mean_pmi,
                pos_pmi=pos_pmi, resolve=resolve, n_concepts=len(A))


def sample_real(k, n, max_pid):
    """Sample real passages that have >=5 addressed content concepts (so execution is meaningful)."""
    out = []
    tries = 0
    while len(out) < n and tries < n * 60:
        tries += 1
        pid = random.randint(0, max_pid)
        r = k.db.execute("SELECT text FROM passages WHERE pid=?", (pid,)).fetchone()
        if not r:
            continue
        text = r[0]
        addressed = [t for t in TOK.findall(text.lower()) if t not in k.stop and t in k.nodeset]
        if len(set(addressed)) >= 5:
            # truncate to first ~14 content tokens so REAL and RAND have comparable concept counts
            toks = TOK.findall(text)
            out.append(" ".join(toks[:24]))
    return out


def make_salad(k, real_sentences, pool):
    """Replace each real sentence with the SAME number of random concept-addresses drawn from `pool`.
    Coverage stays ~1.0 (every word is a valid address) so ONLY connectivity can separate them."""
    salads = []
    for s in real_sentences:
        addressed = [t for t in TOK.findall(s.lower()) if t not in k.stop and t in k.nodeset]
        kk = max(5, len(set(addressed)))
        salads.append(" ".join(random.sample(pool, min(kk, len(pool)))))
    return salads


def auc(real_scores, rand_scores):
    """Mann–Whitney AUC: P(a random REAL outranks a random RAND). 0.5=no separation, 1.0=perfect."""
    alls = sorted([(s, 0) for s in real_scores] + [(s, 1) for s in rand_scores])
    ranks = {}
    i = 0
    while i < len(alls):
        j = i
        while j < len(alls) and alls[j][0] == alls[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for t in range(i, j):
            ranks[t] = avg
        i = j
    rsum = sum(ranks[idx] for idx, (_, lbl) in enumerate(alls) if lbl == 0)
    nr, nd = len(real_scores), len(rand_scores)
    return (rsum - nr * (nr + 1) / 2.0) / (nr * nd) if nr and nd else 0.5


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    k = open_web()
    max_pid = k.db.execute("SELECT MAX(pid) FROM passages").fetchone()[0]
    print(f"[langexec] web max_pid={max_pid:,}  nodes={len(k.nodes):,}  stop={len(k.stop)}  N={n}/condition", flush=True)

    real_txt = sample_real(k, n, max_pid)
    pool_rand = [w for w in k.nodes if w not in k.stop]
    # STEELMAN negative: the most-connected COMMON content words (highest degree, non-stop). They have
    # the most edges, so spurious connectivity is most likely here — the hardest salad to tell apart.
    pool_freq = [w for w, _ in sorted(((w, k.deg.get(w, 0)) for w in pool_rand),
                                       key=lambda x: -x[1])[:3000]]
    rand_txt = make_salad(k, real_txt, pool_rand)
    freq_txt = make_salad(k, real_txt, pool_freq)

    real = [m for s in real_txt if (m := execute(k, s))]
    rand = [m for s in rand_txt if (m := execute(k, s))]
    freq = [m for s in freq_txt if (m := execute(k, s))]

    def col(rows, key):
        vals = [r[key] for r in rows]
        return sum(vals) / len(vals) if vals else 0.0

    print(f"\n  {'metric':<13}{'REAL':>10}{'RAND-salad':>13}{'FREQ-salad':>13}  (FREQ = common-word steelman)")
    for key in ("coverage", "edge_density", "mean_pmi", "pos_pmi", "resolve"):
        print(f"  {key:<13}{col(real, key):>10.3f}{col(rand, key):>13.3f}{col(freq, key):>13.3f}")

    print("\n  SEPARATION (AUC: P[REAL outranks salad]; 0.5=principle FALSE, →1.0=execution real):")
    print(f"    {'metric':<13}{'vs RAND':>10}{'vs FREQ':>10}")
    for key in ("resolve", "edge_density", "mean_pmi", "pos_pmi"):
        a1 = auc([r[key] for r in real], [r[key] for r in rand])
        a2 = auc([r[key] for r in real], [r[key] for r in freq])
        print(f"    {key:<13}{a1:>10.3f}{a2:>10.3f}")

    print("\n  example REAL: ", real_txt[0][:80])
    print("    →", {kk: round(real[0][kk], 3) for kk in ('coverage', 'edge_density', 'mean_pmi', 'resolve')})
    print("  example FREQ: ", freq_txt[0][:80])
    print("    →", {kk: round(freq[0][kk], 3) for kk in ('coverage', 'edge_density', 'mean_pmi', 'resolve')})


if __name__ == "__main__":
    main()

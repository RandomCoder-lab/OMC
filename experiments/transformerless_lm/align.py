#!/usr/bin/env python3
"""align.py — IBM Model 1 unsupervised word aligner: turn the noisy parallel-verse co-occurrence into CLEAN
translation probabilities, so the geoform translates reliably (not just bridges).

Why: verse-level bag-of-words co-occurrence links a word to EVERY word in its verse, so it can't separate the
translation (rei↔king) from the grammar (rei↔the/to). IBM Model 1 fixes this with COMPETITION: via EM it
learns t(e|f) = P(English word e | foreign word f), where each English word's alignment mass is split among
the foreign words of its verse. A foreign function word that co-occurs with everything gets a flat,
high-entropy t(·|f) (mass spread thin); a real pair like rei→king concentrates. Convex objective → a few EM
iterations converge to the global optimum, regardless of init.

Reads the parallel pairs straight from knowledge.db passages (no re-fetch): bridge stored them as
"[BK c:v] <TAG>: <foreign> || ENG: <english>".  Adds a NULL foreign token (absorbs English words with no
foreign counterpart — standard, sharpens the rest).

  python align.py --lang portuguese                  # rei→? , top translations
  python align.py --lang latin --test rex,lux,deus,aqua,bellum
  python align.py --lang portuguese --write           # bake clean t(e|f) back as 'align-portuguese' edges
"""
import sys, math, argparse, sqlite3
from collections import defaultdict
from pathlib import Path
import langtok

HERE = Path(__file__).parent
DB = HERE / "knowledge.db"
NULL = "∅"
# language -> (passage dom, in-text tag). Latin used a custom tag; others follow add_language's "<NAME>:".
SPECIAL = {"latin": ("bible-parallel", "LAT:")}


def lang_meta(lang):
    if lang in SPECIAL:
        dom, tag = SPECIAL[lang]
    else:
        dom, tag = f"parallel-{lang}", f"{lang.upper()}:"
    return dom, tag, langtok.mode_for(lang)


def load_pairs(dom, tag, mode):
    db = sqlite3.connect(str(DB)); db.execute("PRAGMA query_only=1")
    pairs = []
    for (text,) in db.execute("SELECT text FROM passages WHERE dom=?", (dom,)):
        left, sep, eng = text.partition("|| ENG:")
        if not sep:
            continue
        foreign = left.partition(tag)[2]
        F = list(dict.fromkeys(langtok.tokenize(foreign, mode)))
        Eng = list(dict.fromkeys(langtok.tokenize_en(eng)))
        if F and Eng:
            pairs.append((F, Eng))
    db.close()
    return pairs


def ibm1(pairs, n_iter=8):
    # t[f][e] = P(e|f). sparse over co-occurring pairs; add NULL to every foreign side.
    cooc = defaultdict(set)
    for F, Eng in pairs:
        for f in F + [NULL]:
            cooc[f].update(Eng)
    t = {f: {e: 1.0 / len(es) for e in es} for f, es in cooc.items()}
    for it in range(n_iter):
        count = defaultdict(lambda: defaultdict(float))
        total = defaultdict(float)
        loglik = 0.0
        for F, Eng in pairs:
            Fn = F + [NULL]
            for e in Eng:
                Z = sum(t[f][e] for f in Fn if e in t[f])
                if Z <= 0:
                    continue
                loglik += math.log(Z)
                for f in Fn:
                    if e in t[f]:
                        c = t[f][e] / Z
                        count[f][e] += c
                        total[f] += c
        for f in count:
            tf = total[f]
            if tf > 0:
                t[f] = {e: c / tf for e, c in count[f].items()}
        print(f"  EM iter {it+1}/{n_iter}  loglik={loglik:,.0f}", flush=True)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--test", default="")
    ap.add_argument("--write", action="store_true", help="write clean alignments back as 'align-<lang>' edges")
    ap.add_argument("--min-t", type=float, default=0.15, help="min t(e|f) to write an edge")
    a = ap.parse_args()
    dom, tag, mode = lang_meta(a.lang)
    pairs = load_pairs(dom, tag, mode)
    if not pairs:
        sys.exit(f"[align:{a.lang}] no parallel passages (dom={dom}) — run add_language/bible_bridge first")
    print(f"[align:{a.lang}] {len(pairs):,} verse pairs (mode={mode}); running IBM Model 1 ({a.iters} iters)...", flush=True)
    t = ibm1(pairs, a.iters)

    defaults = {"latin": ["rex", "lux", "deus", "aqua", "bellum", "terra", "vita", "amor", "pater", "nox"],
                "portuguese": ["rei", "rainha", "luz", "deus", "agua", "guerra", "vida", "amor", "filho", "irmao"]}
    probes = [w.strip() for w in a.test.split(",") if w.strip()] or defaults.get(a.lang, [])
    print(f"\n[align:{a.lang}] top English translations  t(e|f):")
    for f in probes:
        if f not in t:
            print(f"  {f:10}: (not in parallel corpus)"); continue
        top = sorted(t[f].items(), key=lambda kv: -kv[1])[:5]
        print(f"  {f:10}-> " + "  ".join(f"{e}·{p:.2f}" for e, p in top))

    if a.write:
        con = sqlite3.connect(str(DB))
        cur = con.cursor()
        n = 0
        for f, te in t.items():
            if f == NULL:
                continue
            for e, p in te.items():
                if p >= a.min_t and e != f:
                    w = int(round(p * 100))
                    cur.executemany(
                        "INSERT INTO edges(a,b,src,pid,w) VALUES(?,?,?,?,?) "
                        "ON CONFLICT(a,b) DO UPDATE SET w=MAX(w,excluded.w), src=excluded.src",
                        [(f, e, f"align-{a.lang}", -1, w), (e, f, f"align-{a.lang}", -1, w)])
                    n += 2
        con.commit()
        con.execute("DROP TABLE IF EXISTS node_deg")
        con.execute("CREATE TABLE node_deg AS SELECT a AS node, SUM(w) AS deg FROM edges GROUP BY a")
        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_node_deg ON node_deg(node)")
        con.commit(); con.close()
        print(f"\n[align:{a.lang}] wrote {n:,} clean align edges (t>={a.min_t}); node_deg refreshed.", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""kask.py — SEE THE WEB through addressing. A CLI to query the live knowledge.db (RAM-flat, reads OK
even while it's growing — SQLite WAL). This is the tool that lets me (the assistant) look into the
addressed knowledge web during a conversation.

  python kask.py connect  A B     grounded multi-hop chain A→…→B (each hop quoted from its source, field-tagged)
  python kask.py path     A B     shortest grounded path (may be direct)
  python kask.py like     X       nearest concepts by learned meaning (the address neighborhood)
  python kask.py near     X       direct addressed neighbors (who X co-occurs with, + source)
  python kask.py discover X       meaningful but NON-obvious links (high meaning, no direct edge), grounded
  python kask.py search   TERM    concepts whose name contains TERM
  python kask.py stat             web size (passages/edges/fields)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from kdb import KnowledgeDB, load_embedding

DB = Path(__file__).parent / "knowledge.db"


def open_web():
    s, E, n, e = load_embedding()
    return KnowledgeDB(str(DB), s, E, n, e)


def like(k, x, topk=10):
    x = x.lower()
    if x not in k.stoi:
        return f"'{x}' not addressed (not a concept node)"
    scored = sorted(((k.relate(x, c), c) for c in k.nodes if c != x), reverse=True)[:topk]
    return f"{x} ~ " + "  ".join(f"{c}·{s:.2f}" for s, c in scored if s > 0.15)


def near(k, x, topk=14):
    x = x.lower()
    nb = k.neighbors(x)
    nb = {b: v for b, v in nb.items() if b not in k.stop}   # drop derived function-word neighbors
    if not nb:
        return f"'{x}' has no direct addressed neighbors"
    # rank by ASSOCIATION (w normalized by node degrees) — "apply their weight at scale" without letting
    # function-word hubs dominate. Show raw w too.
    ranked = sorted(((k.assoc(x, b, w), w, b, src) for b, (src, pid, w) in nb.items()), reverse=True)[:topk]
    return f"{x} relates to: " + "  ".join(f"{b}·a{a:.3f}·w{w}⟨{src}⟩" for a, w, b, src in ranked)


def discover(k, x, topn=6):
    x = x.lower()
    if x not in k.stoi:
        return f"'{x}' not addressed"
    direct = set(k.neighbors(x))
    cands = sorted(((k.relate(x, c), c) for c in k.nodes if c != x and c not in direct), reverse=True)
    out = [f"{x} — meaningful but not directly stated:"]
    found = 0
    for r, c in cands:
        if found >= topn:
            break
        # is there a grounded path?
        line = k.deep_connect(x, c)
        if "→" in line.split(chr(10))[0] and " → " in line.split(chr(10))[0]:
            out.append(f"  ⤳{r:.2f}  " + line.split(chr(10))[0]); found += 1
    return chr(10).join(out) if found else f"{x}: no non-obvious grounded links surfaced"


def translate(k, x, topn=6):
    # translation-by-addressing: query ONLY the IBM-Model-1 alignment edges (src='align-*'), which carry
    # clean t(e|f) translation probability as weight — not noisy raw co-occurrence.
    x = x.lower()
    rows = k.db.execute("SELECT b,src,w FROM edges WHERE a=? AND src LIKE 'align-%' ORDER BY w DESC", (x,)).fetchall()
    if not rows:
        return f"'{x}' has no alignment (not in a bridged parallel corpus)"
    langs = sorted({src.split("-", 1)[1] for _, src, _ in rows})
    best = [(b, w) for b, src, w in rows][:topn]
    return f"{x} ⇒ " + "  ".join(f"{b}·{w/100:.2f}" for b, w in best) + f"   ⟨from {'+'.join(langs)}↔english⟩"


def search(k, term, topn=30):
    term = term.lower()
    hits = [n for n in k.nodes if term in n][:topn]
    return f"concepts matching '{term}': " + (", ".join(hits) if hits else "(none)")


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    cmd = sys.argv[1]; a = sys.argv[2:]
    k = open_web()
    if cmd == "stat":
        p, e, f = k.stats(); print(f"web: {p:,} passages, {e:,} edges, {f} fields")
    elif cmd == "connect":
        print(k.deep_connect(a[0], a[1]))
    elif cmd == "path":
        pth = k.nav(a[0].lower(), a[1].lower())
        print(" → ".join(pth) if pth else f"no grounded path {a[0]}~{a[1]} (meaning {k.relate(a[0].lower(),a[1].lower()):+.2f})")
    elif cmd == "like":
        print(like(k, a[0]))
    elif cmd == "near":
        print(near(k, a[0]))
    elif cmd == "discover":
        print(discover(k, a[0]))
    elif cmd == "search":
        print(search(k, a[0]))
    elif cmd == "translate" or cmd == "tr":
        print(translate(k, a[0]))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()

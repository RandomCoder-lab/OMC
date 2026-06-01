#!/usr/bin/env python3
"""stitch.py — RETRIEVE→STITCH over the address-geoform. Tests the thesis (omc_addressed_cognition_thesis):
the LM does not GENERATE an answer token-by-token; it ADDRESSES the query's concepts, RETRIEVES the grounded
pieces attached to the strongest relations between them, and STITCHES those pieces into a response. Every
content word comes from a real source passage (quoted, field-tagged, pid-traceable) — the only non-quoted
text is structural labels (the relation, its PMI/weight), which are data, not authored prose. So this cannot
hallucinate a fact the corpus doesn't contain: an unsupported claim has no address to retrieve from.

Three query shapes (dispatched on how many query words are addressable concept nodes):
  • 1 concept  → ABOUT x : the strongest PMI relations of x, each with its grounded example passage = facets.
  • 2 concepts → RELATE a b : the grounded connection — direct edge (quoted example) or a multi-hop path
                  (deep_connect), each hop quoted from its source.
  • N concepts → assemble pairwise relations among them, strongest first.

The compressed weighted-edge store (kdb v2) keeps ONE example pid per distinct (a,b) pair + weight w; that
example IS the retrievable grounded piece. PMI (kdb.assoc) ranks relations so function-word hubs don't
dominate the stitch.

  python stitch.py "how does gravity relate to light"
  python stitch.py about quantum
  python stitch.py relate gene disease
"""
import sys, re, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from kdb import KnowledgeDB, load_embedding

DB = Path(__file__).parent / "knowledge.db"
TOK = re.compile(r"[a-z][a-z']+")


def open_web():
    s, E, n, e = load_embedding()
    return KnowledgeDB(str(DB), s, E, n, e)


class Stitcher:
    def __init__(self, k):
        self.k = k

    # ── address: which query words are concept nodes (singular + naive de-plural) ──
    def concepts(self, query):
        out, seen = [], set()
        for w in TOK.findall(query.lower()):
            for cand in (w, w[:-1] if w.endswith("s") else None, w[:-2] if w.endswith("es") else None):
                if cand and cand in self.k.stoi and cand not in seen:
                    out.append(cand); seen.add(cand); break
        return out

    # ── retrieve one grounded piece: the example passage for an (a,b) relation, span around b ──
    def _piece(self, a, b, src, pid, w, radius=120):
        text, dom = self.k._passage(pid)
        if not text:
            return None
        m = re.search(r"\b" + re.escape(b) + r"\b", text.lower()) or re.search(r"\b" + re.escape(a) + r"\b", text.lower())
        if m:
            lo = max(0, m.start() - radius); hi = min(len(text), m.end() + radius)
            span = re.sub(r"\s+", " ", text[lo:hi]).strip()
        else:
            span = re.sub(r"\s+", " ", text[:2 * radius]).strip()
        return {"a": a, "b": b, "span": span, "dom": dom or src, "pid": pid, "w": w,
                "pmi": self.k.assoc(a, b, w)}

    # relation score = PMI * log1p(w): rewards a relation that is both SURPRISING (high PMI, genuinely
    # associated not just a hub) and WELL-ATTESTED (seen many times). Pure PMI over-ranks rare cross-language
    # co-occurrences (quantum~tantum from Latin classics); weighting by evidence damps them. Fully derived.
    def score(self, a, b, w):
        return self.k.assoc(a, b, w) * math.log1p(w)

    # ── ABOUT one concept: top relations by weighted-PMI, each a grounded facet ──
    def about(self, x, topn=8):
        nb = self.k.neighbors(x)
        ranked = sorted(((self.score(x, b, w), b, src, pid, w)
                         for b, (src, pid, w) in nb.items()), reverse=True)
        pieces, used = [], set()
        for pmi, b, src, pid, w in ranked:
            if pid in used:
                continue
            p = self._piece(x, b, src, pid, w)
            if p and len(p["span"]) > 30:
                pieces.append(p); used.add(pid)
            if len(pieces) >= topn:
                break
        return pieces

    # ── RELATE two concepts: direct grounded edge, else multi-hop grounded path ──
    def relate(self, a, b):
        nb = self.k.neighbors(a)
        if b in nb:                                   # direct relation — one grounded piece
            src, pid, w = nb[b]
            p = self._piece(a, b, src, pid, w)
            return {"kind": "direct", "pmi": self.k.assoc(a, b, w), "pieces": [p] if p else []}
        path = self.k.nav(a, b)                       # addressed multi-hop path
        if not path or len(path) < 2:
            return {"kind": "none", "meaning": self.k.relate(a, b), "pieces": []}
        pieces = []
        for x, y in zip(path, path[1:]):
            nx = self.k.neighbors(x); src, pid, w = nx.get(y, ("?", -1, 1))
            pc = self._piece(x, y, src, pid, w)
            if pc:
                pieces.append(pc)
        return {"kind": "path", "path": path, "pieces": pieces}

    # ── assemble an answer from addressed pieces (the STITCH) ──
    def answer(self, query):
        cs = self.concepts(query)
        out = [f"⌖ addressed concepts: {', '.join(cs) if cs else '(none — query has no concept nodes)'}"]
        if not cs:
            out.append("  cannot retrieve: nothing in the query is an address in the web.")
            return "\n".join(out)
        if len(cs) == 1:
            out.append(f"\n■ {cs[0]} — assembled from its strongest grounded relations:")
            for p in self.about(cs[0]):
                out.append(f"  ·{p['b']}· (pmi {p['pmi']:.1f}, ×{p['w']}) ⟨{p['dom']}⟩")
                out.append(f"      “…{p['span']}…”")
        else:
            # pairwise relations among all query concepts, strongest first
            rels = []
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    rels.append((cs[i], cs[j], self.relate(cs[i], cs[j])))
            for a, b, r in rels:
                if r["kind"] == "direct":
                    out.append(f"\n■ {a} ←→ {b}  (direct, pmi {r['pmi']:.1f})")
                elif r["kind"] == "path":
                    out.append(f"\n■ {a} → {b}  via grounded path: {' → '.join(r['path'])}")
                else:
                    out.append(f"\n■ {a} ⋯ {b}  no grounded path (meaning {r.get('meaning',0):+.2f})")
                for p in r["pieces"]:
                    out.append(f"      {p['a']}–{p['b']} ⟨{p['dom']}⟩: “…{p['span']}…”")
        out.append(f"\n— stitched from {self._count(query)} grounded pieces; every span is quoted + pid-traceable, none generated.")
        return "\n".join(out)

    def _count(self, query):
        cs = self.concepts(query)
        if len(cs) == 1:
            return len(self.about(cs[0]))
        n = 0
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                n += len(self.relate(cs[i], cs[j]).get("pieces", []))
        return n


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    k = open_web()
    st = Stitcher(k)
    a = sys.argv[1:]
    if a[0] == "about" and len(a) >= 2:
        print(st.answer(a[1]))
    elif a[0] == "relate" and len(a) >= 3:
        print(st.answer(f"{a[1]} {a[2]}"))
    else:
        print(st.answer(" ".join(a)))


if __name__ == "__main__":
    main()

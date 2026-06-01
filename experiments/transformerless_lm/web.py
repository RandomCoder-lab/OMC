#!/usr/bin/env python3
"""web.py — the cross-field knowledge web: connect CONCEPTS across domains, cross-verified by addressing.

The vision (user, 2026-05-30): every field is a corpus; one SHARED addressed space cross-verifies a
concept by all its uses across fields (physics/biology/history/philosophy/astronomy/language/…); the
dictionary is the vocabulary backbone. Connections become grounded paths through shared explanatory text.

Agnostic throughout: knowledge enters as DATA (corpus files), never hardcoded. Concepts = content-word
vocabulary DERIVED by frequency band (no list). Meaning = learned embedding (cross-verification). Grounding
= real co-occurrence, evidence = the passage, tagged with its field. Voice = ZERO authored sentences:
concept words (corpus) + the connective words BETWEEN them (corpus) + the score (derived) + symbols only.
"""
import sys, re, time, json, heapq
from collections import deque, Counter
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))
from nl_ground import clean_text, split_passages
from connect import train_embedding

HERE = Path(__file__).parent
CORP = HERE / "corpora"
TOK = re.compile(r"[a-z][a-z']+")


def derive_nodes(passages, max_nodes=320, max_df_frac=0.30, min_count=4):
    """Content-word concepts, DERIVED: frequent (>=min_count) but not promiscuous (<max_df_frac of
    passages — drops 'the/of/and/is'). No external list; the corpus is the dictionary of its own concepts."""
    df, tf = Counter(), Counter()
    for p in passages:
        seen = set()
        for w in TOK.findall(p.lower()):
            if len(w) >= 4:
                tf[w] += 1
                if w not in seen:
                    df[w] += 1; seen.add(w)
    n = len(passages)
    cand = [(w, tf[w]) for w in df if df[w] >= min_count and df[w] < max_df_frac * n]
    cand.sort(key=lambda x: -x[1])
    return [w for w, _ in cand[:max_nodes]]


class KnowledgeWeb:
    def __init__(self, labeled_texts, max_nodes=320, embed_steps=6000, seed=0, log=print):
        passages, dom, parts = [], [], []
        for label, raw in labeled_texts:
            t = clean_text(raw); ps = split_passages(t)
            passages += ps; dom += [label] * len(ps); parts.append(t)
            log(f"[web] {label:11} {len(ps):5} passages")
        self.passages, self.dom = passages, dom
        full = "\n".join(parts)
        nodes = derive_nodes(passages, max_nodes=max_nodes)
        nodeset = set(nodes)
        # cross-field grounded graph: richest span (concepts closest) per edge, evidence tagged by field
        self.adj = {e: {} for e in nodes}
        best = {}
        for i, p in enumerate(passages):
            plo = p.lower()
            pos = {}
            for m in TOK.finditer(plo):
                w = m.group()
                if w in nodeset:
                    pos.setdefault(w, []).append(m.start())
            present = list(pos)
            for a in present:
                for b in present:
                    if a == b:
                        continue
                    d = min(abs(x - y) for x in pos[a] for y in pos[b])
                    if b not in self.adj[a] or d < best.get((a, b), 1 << 30):
                        self.adj[a][b] = i; best[(a, b)] = d
        self.nodes = nodes
        log(f"[web] {len(passages)} passages, {len(nodes)} concepts; training shared meaning-space...")
        t0 = time.time()
        _, self.stoi, self.E = train_embedding(full, steps=embed_steps, max_vocab=20000, seed=seed,
                                               return_matrix=True)
        self.nodes = [e for e in nodes if e in self.stoi]
        log(f"[web] cross-verified meaning learned ({time.time()-t0:.0f}s). {len(self.nodes)} concepts live.")

    def vec(self, w):
        i = self.stoi.get(w); return self.E[i] if i is not None else None

    def relate(self, a, b):
        va, vb = self.vec(a), self.vec(b)
        return float(va @ vb) if va is not None and vb is not None else -1.0

    def resolve(self, w):
        w = w.lower().strip(".,!?;:'\"")
        if w in self.adj:
            return w, 1.0
        v = self.vec(w)
        if v is None:
            return None, 0.0
        best = max(((float(v @ self.vec(e)), e) for e in self.nodes), default=(0, None))
        return (best[1], best[0]) if best[0] >= 0.45 else (None, best[0])

    def nav(self, a, b, max_expand=6000):
        pq = [(-self.relate(a, b), 0, a, [a], [])]; seen_n = {a: 0}; n = 0
        while pq and n < max_expand:
            _, g, node, path, edges = heapq.heappop(pq); n += 1
            if node == b:
                return path, edges
            for nb, evi in self.adj.get(node, {}).items():
                ng = g + 1
                if nb not in seen_n or ng < seen_n[nb]:
                    seen_n[nb] = ng
                    heapq.heappush(pq, (ng - 2.0 * self.relate(nb, b), ng, nb, path + [nb], edges + [evi]))
        return None, None

    def discover(self, a, k=4):
        direct = set(self.adj.get(a, {}))
        cands = sorted(((self.relate(a, c), c) for c in self.nodes if c != a and c not in direct), reverse=True)
        out = []
        for r, c in cands:
            path, edges = self.nav(a, c)
            if path and len(path) >= 3:
                out.append((r, c, path, edges))
            if len(out) >= k:
                break
        return out

    # ── zero-authored-English voice: concept (corpus) + between-words (corpus) + score + field tag ──
    def _between(self, a, b, i, maxlen=90):
        p = self.passages[i]; plo = p.lower()
        ai = [m.start() for m in re.finditer(r"\b" + re.escape(a) + r"\b", plo)]
        bi = [m.start() for m in re.finditer(r"\b" + re.escape(b) + r"\b", plo)]
        if not ai or not bi:
            return "…"
        _, x, y = min((abs(x - y), x, y) for x in ai for y in bi)
        if x <= y:
            mid = p[x + len(a):y]
        else:
            mid = p[y + len(b):x]
        mid = re.sub(r"\s+", " ", mid).strip()
        return (mid[:maxlen] + "…") if len(mid) > maxlen else (mid or "·")

    def render_path(self, a, b, path, edges):
        lines = [f"{' → '.join(path)}   ({self.relate(a, b):+.2f})"]
        for (x, y), ei in zip(zip(path, path[1:]), edges):
            lines.append(f"   {x} ⟪ {self._between(x, y, ei)} ⟫ {y}   ⟨{self.dom[ei]}⟩")
        return "\n".join(lines)

    def connect(self, aw, bw):
        a, _ = self.resolve(aw); b, _ = self.resolve(bw)
        if a is None or b is None:
            return f"{aw if a is None else bw} ∉ web"
        r = self.relate(a, b)
        path, edges = self.nav(a, b)
        if path and r >= 0.15:
            cross = len({self.dom[e] for e in edges})
            tag = f"  [crosses {cross} fields]" if cross > 1 else ""
            return self.render_path(a, b, path, edges) + tag
        if path:
            return f"{a} → … → {b}   ({r:+.2f})   [path via generic terms; meaning ~0]"
        return f"{a} ∥ {b}   (no grounded path; meaning {r:+.2f})"

    def explore(self, xw, k=4):
        x, _ = self.resolve(xw)
        if x is None:
            return f"{xw} ∉ web"
        found = self.discover(x, k=k)
        if not found:
            return f"{x}: (no indirect links)"
        return "\n\n".join(self.render_path(x, c, path, edges) for r, c, path, edges in found)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--max-nodes", type=int, default=320, dest="max_nodes")
    ap.add_argument("--exclude", default="dict_webster", help="comma substrings to skip (dict too big for v1)")
    args = ap.parse_args()
    skip = set(args.exclude.split(","))
    labeled = [(f.stem.split("_")[0], f.read_text(errors="replace"))
               for f in sorted(CORP.glob("*.txt")) if not any(s in f.stem for s in skip)]
    web = KnowledgeWeb(labeled, max_nodes=args.max_nodes, embed_steps=args.steps)
    print(f"\n[web] === cross-field concept connections (⟨field⟩ per hop; the corpus's own words between) ===")
    for q in [("force", "life"), ("star", "time"), ("light", "mind"),
              ("war", "nature"), ("number", "world"), ("motion", "matter")]:
        print(f"\n? {q[0]} ~ {q[1]}\n{web.connect(*q)}")
    json.dump({"fields": sorted(set(web.dom)), "passages": len(web.passages), "concepts": len(web.nodes)},
              open(HERE / "results_web.json", "w"), indent=2)


if __name__ == "__main__":
    main()

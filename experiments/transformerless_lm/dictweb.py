#!/usr/bin/env python3
"""dictweb.py — the DICTIONARY as a knowledge web: every word addressed, cross-linked by definition.

The backbone of the cross-field vision (user, 2026-05-30). A dictionary is the agnostic, corpus-as-DATA
way to address the whole language WITH meaning: each word's DEFINITION is its context (so even rare words
get a real address — meaning IS use), and definitions cross-reference each other, forming a concept graph.
Grounded edge A→B = "B appears in A's definition" (evidence = the definition itself). Meaning = embedding
learned over all definitions (cross-verification across the entire vocabulary). The code stays agnostic:
it parses the dictionary's STRUCTURE (no hardcoded words) and derives everything.

Connect any two concepts through definitional chains — love↔pride, force↔life — grounded in how the
language itself defines them. Voice = concept words (dict) + the dict's own words between them + score.
"""
import sys, re, time, json, heapq
from collections import Counter
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))
from connect import train_embedding

HERE = Path(__file__).parent
HEAD = re.compile(r"^[A-Z][A-Z'\- ]+$")
WORD = re.compile(r"[a-z][a-z]+")


def parse_webster(text):
    """Structural parse (no hardcoded vocabulary): caps line = headword; following text = its entry.
    Keep single common-word headwords (alpha, len>=3). Returns {headword_lower: definition_text}."""
    body = text
    m = re.search(r"\*\*\*\s*START OF.*?\*\*\*(.*)", text, re.S)
    if m:
        body = m.group(1)
    entries = {}
    cur, buf = None, []
    for ln in body.split("\n"):
        s = ln.strip()
        if s and HEAD.fullmatch(s):
            if cur and buf:
                hw = cur.lower()
                if re.fullmatch(r"[a-z]{3,}", hw) and hw not in entries:
                    entries[hw] = " ".join(buf)
            cur, buf = s, []
        elif s:
            s = re.sub(r"^Defn:\s*", "", s)
            if not s.startswith("Etym:"):
                buf.append(s)
    return entries


class DictWeb:
    def __init__(self, dict_text, max_nodes=6000, embed_steps=8000, seed=0, log=print):
        t0 = time.time()
        entries = parse_webster(dict_text)
        log(f"[dict] parsed {len(entries):,} single-word headwords ({time.time()-t0:.0f}s)")
        # definition word-sets (only words that are themselves headwords = the concept vocabulary)
        head = set(entries)
        defwords = {h: (set(WORD.findall(d.lower())) & head) - {h} for h, d in entries.items()}
        # nodes = most-REFERENCED headwords (core vocabulary: appears in many definitions) — derived
        ref = Counter()
        for h, ws in defwords.items():
            for w in ws:
                ref[w] += 1
        nodes = [w for w, _ in ref.most_common(max_nodes)]
        nodeset = set(nodes)
        # grounded definitional graph: A→B iff B (a node) is in A's definition (evidence = A's own entry)
        self.adj = {a: [b for b in defwords.get(a, ()) if b in nodeset] for a in nodes}
        self.entries = entries
        self.nodes = nodes
        deg = sum(len(v) for v in self.adj.values())
        log(f"[dict] concept graph: {len(nodes):,} nodes, {deg:,} definitional edges")
        log(f"[dict] training meaning over all definitions...")
        t0 = time.time()
        alldefs = " ".join(entries.values())
        _, self.stoi, self.E = train_embedding(alldefs, steps=embed_steps, max_vocab=30000, seed=seed,
                                               return_matrix=True)
        self.nodes = [w for w in nodes if w in self.stoi]
        log(f"[dict] meaning learned ({time.time()-t0:.0f}s). {len(self.nodes):,} concepts live, agnostic.")

    def save(self, path):
        p = Path(path); p.mkdir(parents=True, exist_ok=True)
        torch.save(self.E, p / "E.pt")
        node_entries = {a: self.entries[a] for a in self.nodes if a in self.entries}
        (p / "web.json").write_text(json.dumps(dict(stoi=self.stoi, nodes=self.nodes,
                                                     adj=self.adj, entries=node_entries)))

    @classmethod
    def load(cls, path):
        p = Path(path); self = cls.__new__(cls)
        self.E = torch.load(p / "E.pt")
        d = json.loads((p / "web.json").read_text())
        self.stoi = d["stoi"]; self.nodes = d["nodes"]; self.adj = d["adj"]; self.entries = d["entries"]
        return self

    def vec(self, w):
        i = self.stoi.get(w); return self.E[i] if i is not None else None

    def relate(self, a, b):
        va, vb = self.vec(a), self.vec(b)
        return float(va @ vb) if va is not None and vb is not None else -1.0

    def has(self, w):
        return w.lower() in self.adj

    def nav(self, a, b, max_expand=20000):
        pq = [(-self.relate(a, b), 0, a, [a])]; seen = {a: 0}; n = 0
        while pq and n < max_expand:
            _, g, node, path = heapq.heappop(pq); n += 1
            if node == b:
                return path
            for nb in self.adj.get(node, []):
                ng = g + 1
                if nb not in seen or ng < seen[nb]:
                    seen[nb] = ng
                    heapq.heappush(pq, (ng - 2.0 * self.relate(nb, b), ng, nb, path + [nb]))
        return None

    def _between(self, a, b, maxlen=80):
        """The dictionary's OWN words: the span of A's definition around the reference to B."""
        d = self.entries.get(a, "")
        m = re.search(r"\b" + re.escape(b) + r"\b", d.lower())
        if not m:
            return "·"
        lo = max(0, m.start() - maxlen // 2); hi = min(len(d), m.end() + maxlen // 2)
        return re.sub(r"\s+", " ", d[lo:hi]).strip()

    def connect(self, aw, bw):
        a, b = aw.lower(), bw.lower()
        if a not in self.adj:
            return f"{aw} ∉ dictionary-web"
        if b not in self.adj:
            return f"{bw} ∉ dictionary-web"
        path = self.nav(a, b)
        if not path:
            return f"{a} ∥ {b}   (no definitional path; meaning {self.relate(a,b):+.2f})"
        lines = [f"{' → '.join(path)}   ({self.relate(a, b):+.2f})"]
        for x, y in zip(path, path[1:]):
            lines.append(f"   {x} ⟪ …{self._between(x, y)}… ⟫ {y}")
        return "\n".join(lines)

    def like(self, xw, k=8):
        x = xw.lower()
        if x not in self.stoi:
            return f"{xw} ∉ dictionary-web"
        nb = sorted(((self.relate(x, e), e) for e in self.nodes if e != x), reverse=True)[:k]
        return f"{x} ~ " + "  ".join(f"{e}·{s:.2f}" for s, e in nb if s > 0.2)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-nodes", type=int, default=6000, dest="max_nodes")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    cache = HERE / ".dictcache"
    if cache.exists() and not args.rebuild:
        t0 = time.time(); w = DictWeb.load(cache)
        print(f"[dict] loaded cached web ({len(w.nodes):,} concepts, {time.time()-t0:.1f}s) — no rebuild")
    else:
        text = (HERE / "corpora" / "dict_webster.txt").read_text(errors="replace")
        w = DictWeb(text, max_nodes=args.max_nodes, embed_steps=args.steps)
        w.save(cache)
        print(f"[dict] built + SAVED to {cache} — instant load next time")
    print(f"\n[dict] === concept connections through the language's own definitions ===")
    for a, b in [("force", "life"), ("light", "mind"), ("love", "pride"),
                 ("time", "motion"), ("number", "music"), ("fear", "courage"), ("water", "fire")]:
        print(f"\n? {a} ~ {b}\n{w.connect(a, b)}")
    print(f"\n[dict] === meaning-neighbors (learned from definitions) ===")
    for x in ["force", "mind", "justice", "wave"]:
        print(f"  {w.like(x)}")
    json.dump({"nodes": len(w.nodes), "headwords": len(w.entries)}, open(HERE / "results_dictweb.json", "w"))


if __name__ == "__main__":
    main()

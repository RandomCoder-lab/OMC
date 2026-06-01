#!/usr/bin/env python3
"""kweb.py — the unified KNOWLEDGE WEB: dictionary backbone + every field, one cross-verified space, saved.

The full vision (user, 2026-05-30). The dictionary (dictweb) gave the connective backbone — concepts
defined against each other. web.py (fields only) was weak — narrow corpora don't define/cross-link
concepts. The fix: BOTH in ONE addressed space —
  * NODES + definitional edges + broad meaning  ← the dictionary (agnostic data, structural parse).
  * domain co-occurrence edges + domain meaning  ← each field corpus (physics/history/biology/…).
A concept ("force") is now DEFINED (dict) AND USED (physics): its address is triangulated by every field
that touches it (cross-verified addressing), and connections run through definitions OR domain text, each
hop tagged with its source. Voice = concept words + the source's own words between them + score. Agnostic
throughout (knowledge = data; nothing hardcoded). Persisted: build once, reload instantly.
"""
import sys, re, time, json, heapq
from collections import Counter
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))
from connect import train_embedding
from dictweb import parse_webster, WORD

HERE = Path(__file__).parent
CORP = HERE / "corpora"
from nl_ground import clean_text, split_passages
TOK = re.compile(r"[a-z][a-z']+")


class KnowledgeWeb:
    def __init__(self, dict_text=None, field_texts=None, max_nodes=6000, embed_steps=9000,
                 edge_window=14, seed=0, log=print):
        if dict_text is None:
            return                                              # shell for load()
        t0 = time.time()
        entries = parse_webster(dict_text)
        head = set(entries)
        defwords = {h: (set(WORD.findall(d.lower())) & head) - {h} for h, d in entries.items()}
        ref = Counter()
        for ws in defwords.values():
            for w in ws:
                ref[w] += 1
        nodes = [w for w, _ in ref.most_common(max_nodes)]
        nodeset = set(nodes)
        log(f"[kweb] dictionary: {len(entries):,} headwords -> {len(nodes):,} concept nodes ({time.time()-t0:.0f}s)")
        # edges: src='dict' (evidence=a's entry) or src=field (evidence=passage idx). Prefer a field edge
        # when present (domain grounding is more specific), else the definitional edge.
        self.adj = {a: {} for a in nodes}
        for a in nodes:                                         # definitional backbone
            for b in defwords.get(a, ()):
                if b in nodeset:
                    self.adj[a][b] = ("dict", -1)
        # field layer
        self.passages, self.dom = [], []
        embed_parts = [" ".join(entries.values())]
        for label, raw in (field_texts or []):
            t = clean_text(raw); ps = split_passages(t); embed_parts.append(t)
            base = len(self.passages)
            self.passages += ps; self.dom += [label] * len(ps)
            for j, p in enumerate(ps):
                toks = [(m.group(), m.start()) for m in TOK.finditer(p.lower())]
                present = [(w, s) for w, s in toks if w in nodeset]
                for ai in range(len(present)):
                    wa, sa = present[ai]
                    for bi in range(ai + 1, len(present)):
                        wb, sb = present[bi]
                        if wa == wb or abs(sa - sb) > edge_window * 6:   # ~within edge_window tokens (chars proxy)
                            continue
                        # field edge (overwrites dict edge → shows domain grounding)
                        self.adj[wa][wb] = (label, base + j)
                        self.adj[wb][wa] = (label, base + j)
            log(f"[kweb] +field {label:11} {len(ps):5} passages")
        self.entries = {a: entries[a] for a in nodes if a in entries}
        self.nodes = nodes
        deg = sum(len(v) for v in self.adj.values())
        log(f"[kweb] unified graph: {len(nodes):,} nodes, {deg:,} edges (dict + fields). training meaning...")
        t0 = time.time()
        _, self.stoi, self.E = train_embedding("\n".join(embed_parts), steps=embed_steps,
                                               max_vocab=30000, seed=seed, return_matrix=True)
        self.nodes = [w for w in nodes if w in self.stoi]
        log(f"[kweb] cross-verified meaning learned ({time.time()-t0:.0f}s). {len(self.nodes):,} concepts live.")

    # ── INCREMENTAL "stack then integrate": add a field WITHOUT a full rebuild ──
    def add_field(self, label, raw, edge_window=14):
        """STACK a new field: append its passages + co-occurrence edges (between existing concept nodes)
        to the live web. O(new text) — no full retrain. The 6,000 dict nodes are fixed, so their learned
        vectors stay valid; this only adds new grounded EDGES/passages. (Full retrain later = INTEGRATE,
        refreshing cross-verification.) Returns #edges added."""
        from nl_ground import clean_text, split_passages
        t = clean_text(raw); ps = split_passages(t)
        nodeset = set(self.nodes)
        base = len(self.passages)
        self.passages += ps; self.dom += [label] * len(ps)
        added = 0
        for j, p in enumerate(ps):
            toks = [(m.group(), m.start()) for m in TOK.finditer(p.lower())]
            present = [(w, s) for w, s in toks if w in nodeset]
            for ai in range(len(present)):
                wa, sa = present[ai]
                for bi in range(ai + 1, len(present)):
                    wb, sb = present[bi]
                    if wa == wb or abs(sa - sb) > edge_window * 6:
                        continue
                    if wb not in self.adj[wa]:
                        self.adj[wa][wb] = (label, base + j); added += 1
                    if wa not in self.adj[wb]:
                        self.adj[wb][wa] = (label, base + j); added += 1
        return added

    # ── persistence ──
    def save(self, path):
        p = Path(path); p.mkdir(parents=True, exist_ok=True)
        torch.save(self.E, p / "E.pt")
        (p / "web.json").write_text(json.dumps(dict(
            stoi=self.stoi, nodes=self.nodes, entries=self.entries, dom=self.dom,
            passages=self.passages, adj={a: {b: list(v) for b, v in d.items()} for a, d in self.adj.items()})))

    @classmethod
    def load(cls, path):
        p = Path(path); self = cls.__new__(cls)
        self.E = torch.load(p / "E.pt")
        d = json.loads((p / "web.json").read_text())
        self.stoi = d["stoi"]; self.nodes = d["nodes"]; self.entries = d["entries"]
        self.dom = d["dom"]; self.passages = d["passages"]
        self.adj = {a: {b: tuple(v) for b, v in dd.items()} for a, dd in d["adj"].items()}
        return self

    def vec(self, w):
        i = self.stoi.get(w); return self.E[i] if i is not None else None

    def relate(self, a, b):
        va, vb = self.vec(a), self.vec(b)
        return float(va @ vb) if va is not None and vb is not None else -1.0

    def nav(self, a, b, max_expand=30000):
        pq = [(-self.relate(a, b), 0, a, [a])]; seen = {a: 0}; n = 0
        while pq and n < max_expand:
            _, g, node, path = heapq.heappop(pq); n += 1
            if node == b:
                return path
            for nb in self.adj.get(node, {}):
                ng = g + 1
                if nb not in seen or ng < seen[nb]:
                    seen[nb] = ng
                    heapq.heappush(pq, (ng - 2.0 * self.relate(nb, b), ng, nb, path + [nb]))
        return None

    def _between(self, a, b, maxlen=80):
        src, ref = self.adj[a].get(b, ("dict", -1))
        if src == "dict":
            text = self.entries.get(a, ""); tag = "def"
        else:
            text = self.passages[ref]; tag = src
        m = re.search(r"\b" + re.escape(b) + r"\b", text.lower())
        if not m:
            return "·", tag
        lo = max(0, m.start() - maxlen // 2); hi = min(len(text), m.end() + maxlen // 2)
        return re.sub(r"\s+", " ", text[lo:hi]).strip(), tag

    def connect(self, aw, bw):
        a, b = aw.lower(), bw.lower()
        if a not in self.adj:
            return f"{aw} ∉ web"
        if b not in self.adj:
            return f"{bw} ∉ web"
        path = self.nav(a, b)
        if not path:
            return f"{a} ∥ {b}   (no path; meaning {self.relate(a,b):+.2f})"
        srcs = []
        lines = [f"{' → '.join(path)}   ({self.relate(a, b):+.2f})"]
        for x, y in zip(path, path[1:]):
            span, tag = self._between(x, y); srcs.append(tag)
            lines.append(f"   {x} ⟪ …{span}… ⟫ {y}   ⟨{tag}⟩")
        fields = {s for s in srcs if s != "def"}
        if fields:
            lines[0] += f"  [crosses: {'+'.join(sorted(fields))} + def]"
        return "\n".join(lines)

    def deep_connect(self, aw, bw):
        """MULTI-HOP cross-field chain: forbid the direct edge so the path must route through
        intermediate concepts (often across several fields) — the interesting connections, not 1-hop."""
        a, b = aw.lower(), bw.lower()
        if a not in self.adj or b not in self.adj:
            return f"{aw if a not in self.adj else bw} ∉ web"
        da = self.adj[a].pop(b, None); db = self.adj[b].pop(a, None)   # temporarily forbid direct edge
        path = self.nav(a, b)
        if da is not None:
            self.adj[a][b] = da
        if db is not None:
            self.adj[b][a] = db
        if not path or len(path) < 3:
            return f"{a} → {b}: only a direct/!no indirect link (meaning {self.relate(a,b):+.2f})"
        srcs, lines = [], [f"{' → '.join(path)}   ({self.relate(a, b):+.2f})"]
        for x, y in zip(path, path[1:]):
            span, tag = self._between(x, y); srcs.append(tag)
            lines.append(f"   {x} ⟪ …{span}… ⟫ {y}   ⟨{tag}⟩")
        fields = sorted({s for s in srcs if s != "def"})
        lines[0] += f"  [{len(path)-1} hops; crosses {len(set(srcs))} sources: {'+'.join(sorted(set(srcs)))}]"
        return "\n".join(lines)

    def like(self, xw, k=8):
        x = xw.lower()
        if x not in self.stoi:
            return f"{xw} ∉ web"
        nb = sorted(((self.relate(x, e), e) for e in self.nodes if e != x), reverse=True)[:k]
        return f"{x} ~ " + "  ".join(f"{e}·{s:.2f}" for s, e in nb if s > 0.2)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-nodes", type=int, default=6000, dest="max_nodes")
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    import shutil
    if args.rebuild and shutil.disk_usage(HERE).free < 1_500_000_000:
        print(f"[kweb] DISK GUARD: {shutil.disk_usage(HERE).free/1e9:.1f}GB free < 1.5GB — skipping rebuild "
              f"(keeping last .kwebcache checkpoint). Free space or remove old *.pt to continue.", flush=True)
        return
    cache = HERE / ".kwebcache"
    if cache.exists() and not args.rebuild:
        t0 = time.time(); w = KnowledgeWeb.load(cache)
        print(f"[kweb] loaded cached web ({len(w.nodes):,} concepts, {len(w.passages):,} field passages, {time.time()-t0:.1f}s)")
    else:
        dtext = (CORP / "dict_webster.txt").read_text(errors="replace")
        # fields = the original corpora/ books + the growing gitignored library/ (label = filename prefix)
        srcs = [f for f in sorted(CORP.glob("*.txt")) if "dict" not in f.stem]
        srcs += sorted((HERE / "library").glob("*.txt"))
        fields = [(f.stem.split("__")[0].split("_")[0], f.read_text(errors="replace")) for f in srcs]
        w = KnowledgeWeb(dtext, fields, max_nodes=args.max_nodes, embed_steps=args.steps)
        w.save(cache)
        # baseline for incremental stack.py: everything in this full build is already incorporated
        (cache / "stacked.json").write_text(json.dumps(sorted(f.name for f in (HERE / "library").glob("*.txt"))))
        print(f"[kweb] built + SAVED to {cache}")
    print(f"\n[kweb] === MULTI-HOP cross-field chains (direct edge forbidden → routes through fields) ===")
    for a, b in [("force", "life"), ("light", "mind"), ("star", "soul"), ("war", "justice"),
                 ("number", "god"), ("fear", "love"), ("water", "fire"), ("money", "virtue"),
                 ("light", "truth"), ("matter", "spirit")]:
        print(f"\n? {a} ~ {b}\n{w.deep_connect(a, b)}")


if __name__ == "__main__":
    main()

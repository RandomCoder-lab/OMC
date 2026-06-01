#!/usr/bin/env python3
"""kdb.py — disk-backed (SQLite) knowledge store: removes the RAM ceiling so the web can scale to ALL of it.

The master key for "all of human knowledge" (user, 2026-05-30). kweb held the whole web (passages + 12M-edge
graph) in RAM each cycle → capped ~1,500 texts on 30G. KnowledgeDB stores passages + edges ON DISK (SQLite),
keeps only the small node embedding (6k×64, ~7MB) in RAM, and answers nav/connect by querying the DB on
demand. Growth = streaming INSERTs (append-only, RAM-flat). Ceiling becomes DISK, not RAM — and scales with
storage. Agnostic: any source (Gutenberg, arXiv, PubMed, Wikipedia, code) feeds in as text.

Schema:  passages(pid, text, dom)   edges(a, b, src, pid)  [index on a]   meta(key, val)
Embedding (stoi + E) loaded from .kwebcache/E.pt (+ web.json's stoi/nodes/entries) — reused, not retrained.
"""
import sys, re, json, time, sqlite3, heapq, math
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))
from nl_ground import clean_text, split_passages

HERE = Path(__file__).parent
TOK = re.compile(r"[a-z][a-z']+")


class KnowledgeDB:
    def __init__(self, db_path, stoi, E, nodes, entries):
        self.db = sqlite3.connect(str(db_path))
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        # COMPRESSED (interned) schema detection: knowledge_compressed.db stores edges as int ids +
        # zlib'd passages, and presents the legacy TEXT schema (edges/passages/node_deg) as VIEWS. We
        # register the `unzip` fn the passages view needs, and SKIP the legacy table/index DDL (creating
        # an index on a view errors). Reads then work unchanged through the views. See compress_web.py /
        # finalize_compressed.py.
        import zlib as _zlib
        self.compressed = self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='nodes'").fetchone() is not None
        if self.compressed:
            self.db.create_function(
                "unzip", 1, lambda b: _zlib.decompress(b).decode("utf-8", "replace") if b else "")
        else:
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS passages(pid INTEGER PRIMARY KEY, text TEXT, dom TEXT);
                CREATE TABLE IF NOT EXISTS edges(a TEXT, b TEXT, src TEXT, pid INTEGER, w INTEGER DEFAULT 1);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_edges_ab ON edges(a,b);
                CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, val TEXT);
            """)
        self.stoi = stoi; self.E = E; self.nodes = nodes; self.nodeset = set(nodes)
        self.entries = entries
        self._npass = self.db.execute("SELECT COALESCE(MAX(pid),-1)+1 FROM passages").fetchone()[0]
        # node degree (sum of edge weights) for association/PMI ranking — demotes ubiquitous function-word
        # hubs (huge degree) so addressing surfaces informative neighbors, not 'the'/'and'. Lazy; rebuilt
        # by index(). Empty dict => assoc() falls back to raw weight.
        self.deg = {}
        self._degtot = 1.0
        try:
            self.deg = {r[0]: r[1] for r in self.db.execute("SELECT node,deg FROM node_deg")}
            self._degtot = float(sum(self.deg.values())) or 1.0
        except sqlite3.OperationalError:
            pass
        # derived stopwords (closed-class function words, df-ratio flagged by node_expand) — demoted globally
        self.stop = set()
        try:
            self.stop = {r[0] for r in self.db.execute("SELECT node FROM stop")}
        except sqlite3.OperationalError:
            pass

    def assoc(self, a, b, w):
        """Pointwise mutual information: log( w * total / (deg_a * deg_b) ). Measures how much more a and b
        co-occur than chance given how promiscuous each is. Function words co-occur with everything at ~chance
        -> PMI near 0 or negative (demoted even against very common content words like 'light'); specific
        pairs -> high PMI. Falls back to raw w when degrees aren't available."""
        da = self.deg.get(a); db_ = self.deg.get(b)
        if not da or not db_:
            return float(w)
        return math.log(w * self._degtot / (da * db_) + 1e-12)

    # ── meaning (in-RAM, small) ──
    def vec(self, w):
        i = self.stoi.get(w); return self.E[i] if i is not None else None

    def relate(self, a, b):
        va, vb = self.vec(a), self.vec(b)
        return float(va @ vb) if va is not None and vb is not None else -1.0

    # ── streaming add (RAM-flat): passages + edges INSERTed to disk, no whole-web load ──
    def add_field(self, label, raw, edge_window=14, commit=True):
        t = clean_text(raw); ps = split_passages(t)
        cur = self.db.cursor()
        added = 0
        for p in ps:
            pid = self._npass; self._npass += 1
            cur.execute("INSERT INTO passages VALUES(?,?,?)", (pid, p, label))
            toks = [(m.group(), m.start()) for m in TOK.finditer(p.lower())]
            present = [(w, s) for w, s in toks if w in self.nodeset]
            seen = set()
            for ai in range(len(present)):
                wa, sa = present[ai]
                for bi in range(ai + 1, len(present)):
                    wb, sb = present[bi]
                    if wa == wb or abs(sa - sb) > edge_window * 6 or (wa, wb) in seen:
                        continue
                    seen.add((wa, wb))
                    # COMPRESSED weighted edges: increment co-occurrence weight if the pair exists, else
                    # insert w=1. Keeps the edge table at distinct-pairs size as growth continues.
                    cur.executemany(
                        "INSERT INTO edges(a,b,src,pid,w) VALUES(?,?,?,?,1) "
                        "ON CONFLICT(a,b) DO UPDATE SET w=w+1",
                        [(wa, wb, label, pid), (wb, wa, label, pid)])
                    added += 2
        if commit:
            self.db.commit()
        return added, len(ps)

    def index(self):
        self.db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_edges_ab ON edges(a,b)")
        # refresh node degrees so association ranking tracks growth
        self.db.execute("DROP TABLE IF EXISTS node_deg")
        self.db.execute("CREATE TABLE node_deg AS SELECT a AS node, SUM(w) AS deg FROM edges GROUP BY a")
        self.db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_node_deg ON node_deg(node)")
        self.db.commit()
        self.deg = {r[0]: r[1] for r in self.db.execute("SELECT node,deg FROM node_deg")}

    # ── queries (read DB on demand, RAM-flat) ──
    def neighbors(self, a):
        # edges are now COMPRESSED to distinct pairs with weight w (co-occurrence count). Return w so
        # callers can rank/traverse by edge strength ("apply their weight at scale"). Back-compatible with
        # the old un-weighted schema (w defaults to 1).
        out = {}
        try:
            rows = self.db.execute("SELECT b,src,pid,w FROM edges WHERE a=?", (a,))
            for b, src, pid, w in rows:
                if b not in out or (w or 1) > out[b][2]:
                    out[b] = (src, pid, w or 1)
        except sqlite3.OperationalError:
            for b, src, pid in self.db.execute("SELECT b,src,pid FROM edges WHERE a=?", (a,)):
                if b not in out:
                    out[b] = (src, pid, 1)
        return out

    def nav(self, a, b, max_expand=20000):
        if a not in self.stoi or b not in self.stoi:
            return None
        pq = [(-self.relate(a, b), 0.0, a, [a])]; seen = {a: 0.0}; n = 0
        while pq and n < max_expand:
            _, g, node, path = heapq.heappop(pq); n += 1
            if node == b:
                return path
            # traverse weighted edges: a strongly-attested co-occurrence (high w) costs less than a hop,
            # so navigation prefers well-grounded connections at scale.
            for nb, (src, pid, w) in self.neighbors(node).items():
                # hop cost penalizes promiscuous hub nodes (high degree = function words), so paths route
                # through specific, informative intermediates rather than 'the'/'and'.
                ng = g + 1.0 + 0.06 * math.log1p(self.deg.get(nb, 1))
                if nb not in seen or ng < seen[nb]:
                    seen[nb] = ng
                    heapq.heappush(pq, (ng - 2.0 * self.relate(nb, b), ng, nb, path + [nb]))
        return None

    def _passage(self, pid):
        r = self.db.execute("SELECT text,dom FROM passages WHERE pid=?", (pid,)).fetchone()
        return r if r else ("", "?")

    def deep_connect(self, aw, bw):
        a, b = aw.lower(), bw.lower()
        nb = self.neighbors(a)
        saved = nb.pop(b, None)                     # forbid direct edge -> force multi-hop
        # temp: nav with a's direct-b edge removed (re-add after via a fresh neighbors call is fine;
        # we only suppress at the first expansion by checking)
        path = self._nav_nodirect(a, b)
        if not path or len(path) < 3:
            return f"{a} → {b}: only direct/no indirect link (meaning {self.relate(a,b):+.2f})"
        lines = [f"{' → '.join(path)}   ({self.relate(a,b):+.2f})"]
        srcs = []
        for x, y in zip(path, path[1:]):
            nx = self.neighbors(x); src, pid, w = nx.get(y, ("?", -1, 1)); srcs.append(src)
            text, dom = self._passage(pid)
            m = re.search(r"\b" + re.escape(y) + r"\b", text.lower())
            span = "·"
            if m:
                lo = max(0, m.start() - 40); span = re.sub(r"\s+", " ", text[lo:m.end() + 40]).strip()
            lines.append(f"   {x} ⟪ …{span}… ⟫ {y}   ⟨{src}×{w}⟩")
        lines[0] += f"  [crosses: {'+'.join(sorted(set(srcs)))}]"
        return "\n".join(lines)

    def _nav_nodirect(self, a, b, max_expand=20000):
        pq = []; seen = {a: 0.0}; n = 0
        for nb, (src, pid, w) in self.neighbors(a).items():
            if nb != b:
                step = 1.0 + 0.06 * math.log1p(self.deg.get(nb, 1))
                heapq.heappush(pq, (step - 2.0 * self.relate(nb, b), step, nb, [a, nb]))
        while pq and n < max_expand:
            _, g, node, path = heapq.heappop(pq); n += 1
            if node == b:
                return path
            for nb, (src, pid, w) in self.neighbors(node).items():
                ng = g + 1.0 + 0.06 * math.log1p(self.deg.get(nb, 1))
                if nb not in seen or ng < seen[nb]:
                    seen[nb] = ng
                    heapq.heappush(pq, (ng - 2.0 * self.relate(nb, b), ng, nb, path + [nb]))
        return None

    def stats(self):
        np_ = self.db.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
        ne = self.db.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        nd = self.db.execute("SELECT COUNT(DISTINCT dom) FROM passages").fetchone()[0]
        return np_, ne, nd


def load_embedding(cache=HERE / ".kwebcache"):
    d = json.loads((cache / "web.json").read_text())
    E = torch.load(cache / "E.pt")
    return d["stoi"], E, d["nodes"], d.get("entries", {})


def migrate_from_kweb(cache=HERE / ".kwebcache", db_path=HERE / "knowledge.db"):
    """One-time: pour the existing in-RAM .kwebcache web into the disk-backed SQLite store."""
    print("[kdb] loading current web for migration...", flush=True)
    from kweb import KnowledgeWeb
    w = KnowledgeWeb.load(cache)
    stoi = {k: int(v) for k, v in w.stoi.items()} if isinstance(next(iter(w.stoi.values())), str) else w.stoi
    kdb = KnowledgeDB(db_path, w.stoi, w.E, w.nodes, w.entries)
    cur = kdb.db.cursor()
    print(f"[kdb] migrating {len(w.passages):,} passages...", flush=True)
    cur.executemany("INSERT OR REPLACE INTO passages VALUES(?,?,?)",
                    ((i, w.passages[i], w.dom[i] if i < len(w.dom) else "?") for i in range(len(w.passages))))
    print(f"[kdb] migrating edges...", flush=True)
    rows = ((a, b, v[0], v[1]) for a, d in w.adj.items() for b, v in d.items())
    cur.executemany("INSERT INTO edges VALUES(?,?,?,?)", rows)
    kdb._npass = len(w.passages)
    kdb.db.commit(); kdb.index()
    np_, ne, nd = kdb.stats()
    print(f"[kdb] MIGRATED -> {db_path}: {np_:,} passages, {ne:,} edges, {nd} fields. RAM-flat from here.", flush=True)
    return kdb


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--migrate", action="store_true")
    args = ap.parse_args()
    if args.migrate:
        kdb = migrate_from_kweb()
        print("\n[kdb] test queries (disk-backed):")
        for a, b in [("force", "life"), ("war", "justice"), ("light", "mind")]:
            print(f"  {a}~{b}: " + kdb.deep_connect(a, b).split(chr(10))[0])

#!/usr/bin/env python3
"""refold.py — the FAST rebuild (replaces the I/O-thrashing UPSERT fold). Root cause of the slowness: every
new edge did ON CONFLICT(a,b) DO UPDATE against a 110M-row indexed table = a random disk seek per edge,
decelerating as it grows (process sat in D-state, ~92% I/O-wait). Fix = the load-then-aggregate pattern:
DROP the unique index → plain INSERT appends (sequential, no seek, fast) → ONE GROUP BY at the end collapses
duplicates (sequential scan + disk sort). Preserves ALL existing edges (align/parallel/code/memory just ride
through the aggregate). Skips Pass 1 — the 30k new nodes already persisted to the embedding.

  python refold.py
"""
import sys, re, json, time, shutil
from pathlib import Path
HERE = Path(__file__).parent
LIB = HERE / "library"
DB = HERE / "knowledge.db"
sys.path.insert(0, str(HERE))
from kdb import KnowledgeDB, load_embedding
from nl_ground import clean_text, split_passages

TOK = re.compile(r"[a-z][a-z']+")
WIN = 14 * 6
PAIR_CAP = 2_000_000
PASS_BUF = 8000
DF_RATIO_STOP = 0.05


def mem_gb():
    try:
        for l in open("/proc/meminfo"):
            if l.startswith("MemAvailable:"):
                return int(l.split()[1]) / 1_048_576
    except Exception:
        return 99.0
    return 99.0


def main():
    stoi, E, nodes, entries = load_embedding()      # already has the 30k new nodes
    nodeset = set(nodes)
    k = KnowledgeDB(str(DB), stoi, E, nodes, entries)
    k.db.execute("PRAGMA busy_timeout=600000")
    k.db.execute("PRAGMA synchronous=NORMAL")
    row = k.db.execute("SELECT val FROM meta WHERE key='stacked'").fetchone()
    stacked = set(json.loads(row[0])) if row else set()
    now = time.time()
    new = [f for f in sorted(LIB.glob("*.txt")) if f.name not in stacked and now - f.stat().st_mtime > 5]
    print(f"[refold] {len(new):,} files; {len(nodes):,} nodes (Pass 1 skipped). dropping unique index ...", flush=True)
    k.db.execute("DROP INDEX IF EXISTS idx_edges_ab")        # → appends become fast (no per-edge seek)
    k.db.commit()
    t0 = time.time()

    cur = k.db.cursor(); pairs = {}; pbuf = []; df = {}; npass = 0; tot = 0; flushes = 0

    def flush():
        nonlocal pairs, pbuf, tot, flushes
        if pbuf:
            cur.executemany("INSERT INTO passages VALUES(?,?,?)", pbuf); pbuf = []
        if pairs:
            cur.executemany("INSERT INTO edges(a,b,src,pid,w) VALUES(?,?,?,?,?)",
                            [(a, b, e[2], e[1], e[0]) for (a, b), e in pairs.items()])
            tot += len(pairs); pairs = {}
        k.db.commit(); flushes += 1

    for fi, f in enumerate(new):
        label = f.stem.split("__")[0]
        try:
            ps = split_passages(clean_text(f.read_text(errors="replace")))
        except Exception:
            ps = []
        for p in ps:
            pid = k._npass; k._npass += 1
            pbuf.append((pid, p, label)); npass += 1
            present = [(m.group(), m.start()) for m in TOK.finditer(p.lower()) if m.group() in nodeset]
            seen = set()
            for ai in range(len(present)):
                wa, sa = present[ai]
                if wa not in seen:
                    df[wa] = df.get(wa, 0) + 1
                for bi in range(ai + 1, len(present)):
                    wb, sb = present[bi]
                    if wa == wb or abs(sa - sb) > WIN:
                        continue
                    for key in ((wa, wb), (wb, wa)):
                        if key in seen:
                            continue
                        seen.add(key)
                        e = pairs.get(key)
                        if e: e[0] += 1
                        else: pairs[key] = [1, pid, label]
            if len(pairs) >= PAIR_CAP or len(pbuf) >= PASS_BUF:
                flush()
        if fi % 3000 == 0 and fi:
            print(f"[refold] {fi}/{len(new)} files, {npass:,} passages, {tot:,} appended, {flushes}f, {mem_gb():.1f}GB", flush=True)
    flush()
    stacked.update(f.name for f in new)
    k.db.execute("INSERT OR REPLACE INTO meta VALUES('stacked',?)", (json.dumps(sorted(stacked)),))
    k.db.commit()
    print(f"[refold] appended {tot:,} edge-rows from {npass:,} passages in {time.time()-t0:.0f}s. "
          f"aggregating (collapse dupes, sequential) ...", flush=True)

    # ONE aggregate: collapse duplicate (a,b) rows; prefer a knowledge src over align/parallel for the label.
    k.db.execute("DROP TABLE IF EXISTS edges2")
    k.db.execute("""CREATE TABLE edges2 AS
        SELECT a, b,
               MIN(CASE WHEN src LIKE 'align-%' OR src LIKE 'parallel-%' THEN 'zzz'||src ELSE src END) AS src,
               MIN(pid) AS pid, SUM(w) AS w
        FROM edges GROUP BY a, b""")
    k.db.execute("UPDATE edges2 SET src=substr(src,4) WHERE src LIKE 'zzz%'")
    k.db.execute("DROP TABLE edges")
    k.db.execute("ALTER TABLE edges2 RENAME TO edges")
    k.db.execute("CREATE UNIQUE INDEX idx_edges_ab ON edges(a,b)")
    k.db.commit()
    print(f"[refold] edges aggregated + indexed. rebuilding node_deg + stop ...", flush=True)
    k.db.execute("DROP TABLE IF EXISTS node_deg")
    k.db.execute("CREATE TABLE node_deg AS SELECT a AS node, SUM(w) AS deg FROM edges GROUP BY a")
    k.db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_node_deg ON node_deg(node)")
    stops = [w for w in df if w in nodeset and df[w] / max(npass, 1) > DF_RATIO_STOP]
    k.db.execute("CREATE TABLE IF NOT EXISTS stop(node TEXT PRIMARY KEY)")
    k.db.execute("DELETE FROM stop")
    k.db.executemany("INSERT OR IGNORE INTO stop VALUES(?)", ((w,) for w in stops))
    k.db.commit()
    np_, ne, nd = k.stats()
    print(f"[refold] DONE in {time.time()-t0:.0f}s. {np_:,} passages, {ne:,} edges, {nd} fields, "
          f"{len(nodes):,} nodes, {len(stops):,} stopwords.", flush=True)


if __name__ == "__main__":
    main()

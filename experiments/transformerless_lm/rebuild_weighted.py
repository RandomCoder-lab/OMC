#!/usr/bin/env python3
"""rebuild_weighted.py — rebuild knowledge.db from library/ with COMPRESSED, WEIGHTED edges.

Two things at once (user, 2026-05-31):
  • "why don't we just compress non-unique words?" — instead of storing one edge ROW per co-occurrence
    (the 700M-row explosion that filled the disk), store each DISTINCT directed pair (a,b) ONCE with a
    weight w = how many passages it co-occurred in. ~8x fewer rows, and w is a real signal.
  • "apply their weight at scale" — w becomes the edge strength used by addressing (neighbors/nav/connect
    rank by it). See kdb.py changes.

Also serves as crash RECOVERY: the live knowledge.db was corrupted (a killed re-index left an inconsistent
WAL). The corpus is reconstructable from library/*.txt (the source of truth); the 27,999-node embedding
(.kwebcache, from re-index Stage A+B) is intact and reused.

Memory-SAFE: passages stream to disk (SQLite); the only large RAM structure is the distinct-pair table
(dict packed-key -> index + parallel typed arrays), ~9GB for ~90M pairs, behind a hard RAM guard that
aborts well before the box thrashes. NEVER loads the whole graph into RAM.
"""
import sys, os, re, json, time, sqlite3
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nl_ground import clean_text, split_passages

HERE = Path(__file__).parent
LIB = HERE / "library"
CACHE = HERE / ".kwebcache"
NEWDB = HERE / "knowledge.db.new"
TOK = re.compile(r"[a-z][a-z']+")
WIN_CHARS = 14 * 6
RAM_FLOOR_GB = 3.0


def mem_avail_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1_048_576
    except Exception:
        return 99.0
    return 99.0


def guard(where):
    g = mem_avail_gb()
    if g < RAM_FLOOR_GB:
        sys.exit(f"[rebuild] RAM GUARD at {where}: {g:.1f}GB free — aborting cleanly (no crash, no corruption).")


def main():
    if NEWDB.exists():
        NEWDB.unlink()
    d = json.loads((CACHE / "web.json").read_text())
    stoi = d["stoi"]; nodes = d["nodes"]; entries = d.get("entries", {})
    V = len(nodes)
    print(f"[rebuild] {V:,} nodes (incl. modern terms). reading library/ ...", flush=True)

    db = sqlite3.connect(str(NEWDB))
    db.execute("PRAGMA journal_mode=WAL"); db.execute("PRAGMA synchronous=NORMAL")
    db.executescript("""
        CREATE TABLE passages(pid INTEGER PRIMARY KEY, text TEXT, dom TEXT);
        CREATE TABLE edges(a TEXT, b TEXT, src TEXT, pid INTEGER, w INTEGER);
        CREATE TABLE meta(key TEXT PRIMARY KEY, val TEXT);
    """)
    cur = db.cursor()

    # distinct directed pair store: key2idx[packed] -> row index; parallel arrays count/pid/srccode
    key2idx = {}
    cnt = array('i'); rpid = array('i'); rsrc = array('H')
    domcodes = {}; domlist = []

    def domcode(label):
        c = domcodes.get(label)
        if c is None:
            c = len(domlist); domcodes[label] = c; domlist.append(label)
        return c

    files = sorted(LIB.glob("*.txt"))
    pid = 0; npairs_touched = 0; t0 = time.time()
    pbuf = []
    for fi, f in enumerate(files):
        label = f.stem.split("__")[0]
        dc = domcode(label)
        try:
            text = clean_text(f.read_text(errors="replace"))
        except Exception as e:
            print(f"[rebuild] skip {f.name}: {e}"); continue
        for p in split_passages(text):
            pbuf.append((pid, p, label))
            present = [(stoi[m.group()], m.start()) for m in TOK.finditer(p.lower()) if m.group() in stoi]
            seen = set()
            for ai in range(len(present)):
                ia, sa = present[ai]
                for bi in range(ai + 1, len(present)):
                    ib, sb = present[bi]
                    if ia == ib or abs(sa - sb) > WIN_CHARS:
                        continue
                    for k in (ia * V + ib, ib * V + ia):
                        if k in seen:
                            continue
                        seen.add(k)
                        j = key2idx.get(k)
                        if j is None:
                            key2idx[k] = len(cnt)
                            cnt.append(1); rpid.append(pid); rsrc.append(dc)
                        else:
                            cnt[j] += 1
                        npairs_touched += 1
            pid += 1
            if len(pbuf) >= 5000:
                cur.executemany("INSERT INTO passages VALUES(?,?,?)", pbuf); pbuf = []; db.commit()
        if fi % 100 == 0:
            guard("fold")
            print(f"[rebuild]   file {fi}/{len(files)}  passages={pid:,}  distinct_pairs={len(cnt):,}  "
                  f"{mem_avail_gb():.1f}GB free", flush=True)
    if pbuf:
        cur.executemany("INSERT INTO passages VALUES(?,?,?)", pbuf); db.commit()
    print(f"[rebuild] folded {pid:,} passages; {len(cnt):,} distinct edges "
          f"(compressed from {npairs_touched:,} co-occurrences) in {time.time()-t0:.0f}s", flush=True)
    guard("write-edges")

    # write the compressed weighted edges
    print("[rebuild] writing weighted edges ...", flush=True)
    ebuf = []; written = 0
    for k, j in key2idx.items():
        a = k // V; b = k % V
        ebuf.append((nodes[a], nodes[b], domlist[rsrc[j]], rpid[j], cnt[j]))
        if len(ebuf) >= 100000:
            cur.executemany("INSERT INTO edges VALUES(?,?,?,?,?)", ebuf); ebuf = []; db.commit()
            written += 100000
            if written % 2_000_000 == 0:
                print(f"[rebuild]   wrote {written:,} edges, {mem_avail_gb():.1f}GB free", flush=True)
    if ebuf:
        cur.executemany("INSERT INTO edges VALUES(?,?,?,?,?)", ebuf); db.commit()
    db.execute("CREATE UNIQUE INDEX idx_edges_ab ON edges(a,b)")  # dedup + neighbor lookup + UPSERT growth
    cur.execute("INSERT OR REPLACE INTO meta VALUES('stacked',?)",
                (json.dumps(sorted(f.name for f in files)),))
    db.commit()
    ne = db.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    np_ = db.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
    wmax = db.execute("SELECT MAX(w) FROM edges").fetchone()[0]
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.close()
    print(f"[rebuild] DONE: {np_:,} passages, {ne:,} weighted edges (max w={wmax}). -> {NEWDB.name}", flush=True)


if __name__ == "__main__":
    main()

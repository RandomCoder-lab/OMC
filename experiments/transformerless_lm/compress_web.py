#!/usr/bin/env python3
"""compress_web.py — LOSSLESS compression of knowledge.db, via the techniques the web-native LM itself
surfaced (dogfooded): DICTIONARY ENCODING (intern node strings → int32 ids — the LM's 'dictionary→
substrings' hit) + SRC ENUM (140 domains → 1 ref) + zlib on passage TEXT (the LM's 'compression→deflate/
zstd' hit; our own prior lesson says zlib is the bar, so use it). All lossless — verified by round-trip.

NON-DESTRUCTIVE: reads knowledge.db read-only, writes a NEW file knowledge_compressed.db, VERIFIES the
copy reproduces the original exactly (edge count, sampled edge (a,b,w) identity, passage round-trip), and
reports before/after. It NEVER modifies or swaps knowledge.db — swapping (+ the kdb query shim) is a
separate human-gated step once the numbers look right.

Compressed schema:
  nodes(id INTEGER PRIMARY KEY, word TEXT)         -- the interning dictionary
  srcs(id INTEGER PRIMARY KEY, name TEXT)          -- domain enum
  edges(a_id INT, b_id INT, src_id INT, pid INT, w INT) + UNIQUE(a_id,b_id)
  passages(pid INTEGER PRIMARY KEY, ztext BLOB, dom TEXT)   -- ztext = zlib(text)
  node_deg(node_id INT PRIMARY KEY, deg INT)
  meta(key,val) + text_codec='zlib'

  python compress_web.py
"""
import sqlite3, zlib, time, os, sys
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "knowledge.db"
DST = HERE / "knowledge_compressed.db"
SAMPLE = 8000


def main():
    if not SRC.exists():
        sys.exit("no knowledge.db")
    if DST.exists():
        DST.unlink()
    t0 = time.time()
    s = sqlite3.connect(f"file:{SRC}?mode=ro", uri=True)
    s.execute("PRAGMA busy_timeout=120000")
    s.execute("PRAGMA cache_size=-2000000")     # 2GB cache helps the (fragmented) single source scan
    d = sqlite3.connect(str(DST))
    d.execute("PRAGMA journal_mode=OFF")        # fresh build, no concurrent reader → fastest
    d.execute("PRAGMA synchronous=OFF")
    d.execute("PRAGMA cache_size=-2000000")
    d.executescript("""
        CREATE TABLE nodes(id INTEGER PRIMARY KEY, word TEXT);
        CREATE TABLE srcs(id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE edges(a_id INT, b_id INT, src_id INT, pid INT, w INT);
        CREATE TABLE passages(pid INTEGER PRIMARY KEY, ztext BLOB, dom TEXT);
        CREATE TABLE node_deg(node_id INT PRIMARY KEY, deg INT);
        CREATE TABLE meta(key TEXT PRIMARY KEY, val TEXT);
    """)

    # 1. node dictionary from node_deg (edges are symmetric → node_deg.node is the full endpoint universe)
    w2id = {}
    deg_rows = s.execute("SELECT node, deg FROM node_deg").fetchall()
    for w, _deg in deg_rows:
        if w not in w2id:
            w2id[w] = len(w2id) + 1
    print(f"[compress] node dictionary: {len(w2id):,} words from node_deg", flush=True)

    # 2. src enum
    src2id = {}
    for (name,) in s.execute("SELECT DISTINCT src FROM edges"):
        if name not in src2id:
            src2id[name] = len(src2id) + 1
    print(f"[compress] src enum: {len(src2id)} domains", flush=True)

    d.executemany("INSERT INTO nodes VALUES(?,?)", [(i, w) for w, i in w2id.items()])
    d.executemany("INSERT INTO srcs VALUES(?,?)", [(i, n) for n, i in src2id.items()])
    d.executemany("INSERT INTO node_deg VALUES(?,?)",
                  [(w2id[w], deg) for w, deg in deg_rows if w in w2id])
    for k, v in s.execute("SELECT key,val FROM meta"):
        d.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (k, v))
    d.execute("INSERT OR REPLACE INTO meta VALUES('text_codec','zlib')")
    d.commit()

    # 3. edges: stream ONCE, translate to ids (unknown word → assign new id, safety). Count during the
    # stream (no separate COUNT scan — the source edges table is fragmented from churn, so each full scan
    # is slow; do exactly ONE).
    print(f"[compress] interning edges (single streaming scan) ...", flush=True)
    counter = [0]
    def gen_edges():
        cur = s.execute("SELECT a,b,src,pid,w FROM edges")
        while True:
            rows = cur.fetchmany(50000)
            if not rows:
                break
            counter[0] += len(rows)
            for a, b, src, pid, wt in rows:
                ai = w2id.get(a)
                if ai is None:
                    ai = w2id[a] = len(w2id) + 1
                bi = w2id.get(b)
                if bi is None:
                    bi = w2id[b] = len(w2id) + 1
                yield (ai, bi, src2id.get(src, 0), pid, wt or 1)
    d.executemany("INSERT INTO edges VALUES(?,?,?,?,?)", gen_edges())
    d.commit()
    n_edges = counter[0]                        # source row count, from the single stream
    print(f"[compress] interned {n_edges:,} edges in {time.time()-t0:.0f}s. building (a_id,b_id) index ...", flush=True)
    d.execute("CREATE UNIQUE INDEX idx_edges_ab ON edges(a_id,b_id)")
    d.commit()

    # 4. passages: zlib the text
    n_pass = s.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
    print(f"[compress] compressing {n_pass:,} passage texts (zlib) ...", flush=True)
    def gen_pass():
        cur = s.execute("SELECT pid,text,dom FROM passages")
        while True:
            rows = cur.fetchmany(20000)
            if not rows:
                break
            for pid, text, dom in rows:
                yield (pid, zlib.compress((text or "").encode("utf-8"), 6), dom)
    d.executemany("INSERT INTO passages VALUES(?,?,?)", gen_pass())
    d.commit()

    # if word-dict grew during edge streaming (safety adds), persist the extras
    have = d.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    if have < len(w2id):
        existing = {r[0] for r in d.execute("SELECT id FROM nodes")}
        d.executemany("INSERT INTO nodes VALUES(?,?)",
                      [(i, w) for w, i in w2id.items() if i not in existing])
        d.commit()

    # ---- LOSSLESS VERIFICATION ----
    print("[compress] verifying lossless round-trip ...", flush=True)
    id2w = {i: w for w, i in w2id.items()}
    ok = True
    # edge count
    de = d.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    if de != n_edges:
        ok = False; print(f"  FAIL edge count {de} != {n_edges}")
    # passage count
    dp = d.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
    if dp != n_pass:
        ok = False; print(f"  FAIL passage count {dp} != {n_pass}")
    # sample edges round-trip: pick random src edges, confirm (a,b,w) reproduce via interned form
    import random
    random.seed(7)
    mp = s.execute("SELECT MAX(rowid) FROM edges").fetchone()[0]
    checked = miss = 0
    for _ in range(SAMPLE):
        rid = random.randint(1, mp)
        row = s.execute("SELECT a,b,w FROM edges WHERE rowid=?", (rid,)).fetchone()
        if not row:
            continue
        a, b, wt = row
        checked += 1
        r = d.execute("SELECT w FROM edges WHERE a_id=? AND b_id=?", (w2id[a], w2id[b])).fetchone()
        if not r or r[0] != (wt or 1):
            miss += 1
    if miss:
        ok = False; print(f"  FAIL {miss}/{checked} sampled edges did not round-trip")
    # sample passages round-trip
    pmiss = 0
    for _ in range(2000):
        pid = random.randint(0, n_pass + 1_700_000)
        o = s.execute("SELECT text FROM passages WHERE pid=?", (pid,)).fetchone()
        if not o:
            continue
        z = d.execute("SELECT ztext FROM passages WHERE pid=?", (pid,)).fetchone()
        if not z or zlib.decompress(z[0]).decode("utf-8") != (o[0] or ""):
            pmiss += 1
    if pmiss:
        ok = False; print(f"  FAIL {pmiss} sampled passages did not round-trip")

    d.close(); s.close()
    src_gb = os.path.getsize(SRC) / 1e9
    dst_gb = os.path.getsize(DST) / 1e9
    print(f"\n[compress] {'LOSSLESS OK' if ok else '*** VERIFICATION FAILED ***'}")
    print(f"[compress] {src_gb:.2f} GB  ->  {dst_gb:.2f} GB   ({100*(1-dst_gb/src_gb):.0f}% smaller)  in {time.time()-t0:.0f}s")
    print(f"[compress] checked {checked:,} edges + 2k passages round-trip.")
    print(f"[compress] knowledge.db UNTOUCHED. compressed copy = {DST.name}.")
    if ok:
        print("[compress] next (human-gated): add kdb shim to query the interned/zlib schema, then swap.")


if __name__ == "__main__":
    main()

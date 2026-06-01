#!/usr/bin/env python3
"""ghost_fold.py — connect GHOST nodes (in vocab, zero edges) by folding their co-occurrence edges from
the passages they already appear in. STAGING design (avoids the UPSERT I/O-thrash on the 9GB edges table):

  1. SCAN passages once, APPEND ghost-involving (a,b) pairs to a fresh unindexed stage table — fast
     sequential writes, no random-access index lookups.
  2. MERGE once: aggregate the stage by (a,b) and INSERT into edges with ON CONFLICT (one sorted pass).
  Ghosts currently have zero edges, so their new edges essentially can't conflict (the ON CONFLICT just
  safely folds in the small partial set from an earlier run). Additive; node_deg rebuilt at the end.

  python ghost_fold.py
"""
import sys, re, time, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from langexec import open_web

TOK = re.compile(r"[a-z][a-z']+")
WIN = 14 * 6
MAX_NODES = 60
BATCH = 4000


def mem_free_gb():
    try:
        for l in open("/proc/meminfo"):
            if l.startswith("MemAvailable:"):
                return int(l.split()[1]) / 1_048_576
    except Exception:
        return 99.0
    return 99.0


def main():
    if shutil.disk_usage(Path(__file__).parent).free < 5_000_000_000:
        sys.exit("[ghost_fold] disk < 5GB — abort")
    k = open_web()
    k.db.execute("PRAGMA busy_timeout=120000")
    k.db.execute("PRAGMA synchronous=NORMAL")
    ghosts = set(n for n in k.nodes if k.deg.get(n, 0) == 0)
    nodeset = k.nodeset
    maxpid = k.db.execute("SELECT MAX(pid) FROM passages").fetchone()[0]
    print(f"[ghost_fold] {len(ghosts):,} ghost words, {maxpid+1:,} passages. staging ghost-involving edges.", flush=True)

    k.db.execute("DROP TABLE IF EXISTS ghost_stage")
    k.db.execute("CREATE TABLE ghost_stage(a TEXT, b TEXT, dom TEXT, pid INTEGER)")  # unindexed = fast append
    k.db.commit()

    read = k.db.cursor()
    wr = k.db.cursor()
    t0 = time.time()
    npass = touched = staged = 0
    buf = []
    for lo in range(0, maxpid + 1, BATCH):
        rows = read.execute("SELECT pid,text,dom FROM passages WHERE pid>=? AND pid<?", (lo, lo + BATCH)).fetchall()
        for pid, text, dom in rows:
            npass += 1
            present = [(m.group(), m.start()) for m in TOK.finditer(text.lower()) if m.group() in nodeset][:MAX_NODES]
            if not any(w in ghosts for w, _ in present):
                continue
            touched += 1
            seen = set()
            for ai in range(len(present)):
                wa, sa = present[ai]
                for bi in range(ai + 1, len(present)):
                    wb, sb = present[bi]
                    if wa == wb or abs(sa - sb) > WIN:
                        continue
                    if wa not in ghosts and wb not in ghosts:
                        continue
                    if (wa, wb) in seen:
                        continue
                    seen.add((wa, wb))
                    buf.append((wa, wb, dom, pid)); buf.append((wb, wa, dom, pid)); staged += 2
        if len(buf) >= 50000:
            wr.executemany("INSERT INTO ghost_stage VALUES(?,?,?,?)", buf); buf.clear()
        if lo % (BATCH * 25) == 0:
            k.db.commit()
            print(f"[ghost_fold] pid<{lo+BATCH:,} | {touched:,} ghost-passages | {staged:,} staged | "
                  f"{mem_free_gb():.1f}GB free | {time.time()-t0:.0f}s", flush=True)
    if buf:
        wr.executemany("INSERT INTO ghost_stage VALUES(?,?,?,?)", buf); buf.clear()
    k.db.commit()
    print(f"[ghost_fold] staged {staged:,} edge-rows over {touched:,} ghost-passages in {time.time()-t0:.0f}s. merging ...", flush=True)

    t1 = time.time()
    k.db.execute("""
        INSERT INTO edges(a,b,src,pid,w)
        SELECT a, b, MIN(dom), MIN(pid), COUNT(*) FROM ghost_stage GROUP BY a, b ORDER BY a, b
        ON CONFLICT(a,b) DO UPDATE SET w = w + excluded.w
    """)
    k.db.commit()
    print(f"[ghost_fold] merged in {time.time()-t1:.0f}s. rebuilding node_deg ...", flush=True)
    k.db.execute("DROP TABLE IF EXISTS ghost_stage")
    k.db.execute("DROP TABLE IF EXISTS node_deg")
    k.db.execute("CREATE TABLE node_deg AS SELECT a AS node, SUM(w) AS deg FROM edges GROUP BY a")
    k.db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_node_deg ON node_deg(node)")
    k.db.commit()
    for w in ["antibiotics", "antibiotic", "photosynthesis", "entanglement"]:
        d = k.db.execute("SELECT deg FROM node_deg WHERE node=?", (w,)).fetchone()
        print(f"[ghost_fold] '{w}' deg = {d[0] if d else 0}", flush=True)
    rem = sum(1 for n in k.nodes if not k.db.execute("SELECT 1 FROM node_deg WHERE node=?", (n,)).fetchone())
    print(f"[ghost_fold] DONE in {time.time()-t0:.0f}s. remaining ghost vocab: {rem:,} (was 48,628)", flush=True)


if __name__ == "__main__":
    main()

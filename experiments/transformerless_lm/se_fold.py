#!/usr/bin/env python3
"""se_fold.py — fold the FOCUSED SCIENCE subset of the StackExchange backlog (physics/electronics/
computing/signals/engineering/robotics/design/music/scientific_computing — everything EXCEPT the giant
14.9k math dump) into knowledge.db. This is the thin-science knowledge the corpus lacks (~1% science).

These files crashed every previous fold (I/O-thrash / OOM / disk-explosion: ~5.4M passages would stage
~3B raw co-occurrence rows = 122GB > disk). THE FIX, three guards combined:
  1. PER-CHUNK IN-RAM AGGREGATION: dedupe each chunk's co-occurrences in a Counter BEFORE touching disk,
     so recurring word-pairs (resistor–voltage ×1000s) collapse to ONE weighted edge. Raw 100M/chunk →
     deduped ~15-25M. Bounds disk (no 3B-row staging) AND merge cost (deduped flush, not raw).
  2. PER-PASSAGE NODE CAP (50): bounds O(n^2) pairs on dense passages.
  3. RAM + DISK guards between chunks; resumable via the DB meta('stacked') set.

  python se_fold.py [files_per_chunk]      # default 750
"""
import sys, re, json, time, shutil
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from kdb import KnowledgeDB, load_embedding
from nl_ground import clean_text, split_passages

HERE = Path(__file__).parent
LIB = HERE / "library"
TOK = re.compile(r"[a-z][a-z']+")
WIN = 14 * 6                 # co-occurrence char window (matches add_field edge_window=14)
MAX_NODES = 50               # per-passage node cap
DISK_FLOOR = 12_000_000_000  # abort a chunk if < 12GB free
RAM_FLOOR_GB = 5.0           # flush early if available RAM drops below this
# focused science subset = SE technical domains EXCEPT the 14.9k 'math' dump
SUBSET = {"physics", "electronics", "computing", "signals", "engineering",
          "robotics", "design", "music", "scientific_computing"}


def mem_avail_gb():
    try:
        for l in open("/proc/meminfo"):
            if l.startswith("MemAvailable:"):
                return int(l.split()[1]) / 1_048_576
    except Exception:
        return 99.0
    return 99.0


def main():
    chunk = int(sys.argv[1]) if len(sys.argv) > 1 else 750
    k = KnowledgeDB(str(HERE / "knowledge.db"), *load_embedding())
    k.db.execute("PRAGMA busy_timeout=120000")
    k.db.execute("PRAGMA synchronous=NORMAL")
    # 4GB page cache holds the hot edges(a,b) index in RAM (default 2MB → every probe hits disk = the
    # 11GB-read thrash). NO mmap_size: combined with the cache it caused page-reclaim stalls
    # (folio_wait). Cache alone is the right lever, leaving RAM headroom for the OS + the chunk Counter.
    k.db.execute("PRAGMA cache_size=-4000000")   # ~4GB
    nodeset = k.nodeset
    row = k.db.execute("SELECT val FROM meta WHERE key='stacked'").fetchone()
    stacked = set(json.loads(row[0])) if row else set()

    todo = [f for f in sorted(LIB.glob("*.txt"))
            if f.name not in stacked and f.name.split("__")[0] in SUBSET]
    print(f"[se_fold] {len(todo):,} focused-science files to fold (subset={sorted(SUBSET)}). chunk={chunk}.", flush=True)
    if not todo:
        print("[se_fold] nothing to do."); return

    t0 = time.time()
    folded = total_edges = total_pass = 0
    for ci in range(0, len(todo), chunk):
        if shutil.disk_usage(HERE).free < DISK_FLOOR:
            print(f"[se_fold] DISK GUARD (<12GB) — stopping cleanly at chunk start. resume later.", flush=True)
            break
        batch = todo[ci:ci + chunk]
        agg = Counter()                                  # (a,b) -> weight, deduped IN RAM for this chunk
        ex_pid = {}                                      # (a,b) -> one example pid (the compressed-edge example)
        pbuf = []                                        # passages to insert this chunk
        cpass = 0
        for f in batch:
            label = f.name.split("__")[0]
            try:
                passages = split_passages(clean_text(f.read_text(errors="replace")))
            except Exception:
                passages = []
            for p in passages:
                pid = k._npass; k._npass += 1
                pbuf.append((pid, p, label))             # store the science text → quotable by pid
                present = [(m.group(), m.start()) for m in TOK.finditer(p.lower())
                           if m.group() in nodeset][:MAX_NODES]
                seen = set()
                for i in range(len(present)):
                    wa, sa = present[i]
                    for j in range(i + 1, len(present)):
                        wb, sb = present[j]
                        if wa == wb or abs(sa - sb) > WIN or (wa, wb) in seen:
                            continue
                        seen.add((wa, wb))
                        agg[(wa, wb)] += 1; agg[(wb, wa)] += 1
                        if (wa, wb) not in ex_pid:
                            ex_pid[(wa, wb)] = pid; ex_pid[(wb, wa)] = pid
                cpass += 1
            stacked.add(f.name)
            if mem_avail_gb() < RAM_FLOOR_GB:            # safety: flush early if RAM tightens
                break
        # insert this chunk's passages (quotable science text), then flush DEDUPED edges with example pids.
        # SORTED by (a,b) so the ON CONFLICT probes hit the edges(a,b) index in order (sequential page
        # access, not random) — this is what makes the merge fast vs the thrashing unsorted UPSERT.
        k.db.executemany("INSERT OR IGNORE INTO passages VALUES(?,?,?)", pbuf)
        k.db.executemany(
            "INSERT INTO edges(a,b,src,pid,w) VALUES(?,?,'se',?,?) "
            "ON CONFLICT(a,b) DO UPDATE SET w = w + excluded.w",
            [(a, b, ex_pid[(a, b)], w) for (a, b), w in sorted(agg.items())])
        k.db.execute("INSERT OR REPLACE INTO meta VALUES('stacked',?)", (json.dumps(sorted(stacked)),))
        k.db.commit()
        folded += len(batch); total_edges += len(agg); total_pass += cpass
        print(f"[se_fold] chunk {ci//chunk+1}: +{len(batch)} files, {cpass:,} passages, "
              f"{len(agg):,} deduped edges flushed | {folded:,}/{len(todo):,} files | "
              f"{mem_avail_gb():.1f}GB RAM, {shutil.disk_usage(HERE).free/1e9:.0f}GB disk | {time.time()-t0:.0f}s", flush=True)
        agg.clear(); ex_pid.clear(); pbuf.clear()

    print(f"[se_fold] folded {folded:,} files, {total_pass:,} passages, {total_edges:,} edge-flushes in "
          f"{time.time()-t0:.0f}s. rebuilding node_deg ...", flush=True)
    k.db.execute("DROP TABLE IF EXISTS node_deg")
    k.db.execute("CREATE TABLE node_deg AS SELECT a AS node, SUM(w) AS deg FROM edges GROUP BY a")
    k.db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_node_deg ON node_deg(node)")
    k.db.commit()
    for w in ["photosynthesis", "resistor", "voltage", "quantum", "circuit", "entropy"]:
        d = k.db.execute("SELECT deg FROM node_deg WHERE node=?", (w,)).fetchone()
        print(f"[se_fold] '{w}' deg = {d[0] if d else 0}", flush=True)
    np_ = k.db.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
    ne = k.db.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    print(f"[se_fold] DONE. DB now {np_:,} passages, {ne:,} edges.", flush=True)


if __name__ == "__main__":
    main()

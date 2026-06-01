#!/usr/bin/env python3
"""fold_subset.py — fold the SAFE unstacked files (OMC source + code + arxiv/wiki), SKIPPING the SE dumps
(the 11.3M-passage monster that crashed every full fold). RAM-flat (no passage caching), with a PER-PASSAGE
NODE CAP so no dense file explodes O(n²) edges. UPSERT into the live DB — fine for a few thousand small files
(the I/O-thrash only bit at 27k SE-dense files). Marks folded files stacked; refreshes node_deg + stop.
Gives OMC genuine self-knowledge (its own source becomes addressable) without the SE blow-up.

  python fold_subset.py            # fold OMC+code+arxiv+wiki, skip SE
"""
import sys, re, json, time, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from kdb import KnowledgeDB, load_embedding
from nl_ground import clean_text, split_passages

HERE = Path(__file__).parent
LIB = HERE / "library"
TOK = re.compile(r"[a-z][a-z']+")
WIN = 14 * 6
MAX_NODES_PER_PASSAGE = 60     # cap O(n^2) edge blow-up on dense passages
RAM_FLOOR_GB = 3.0
SE_DOMS = {"electronics", "math", "signals", "computing", "engineering", "robotics", "music",
           "design", "physics", "scientific_computing"}


def mem_gb():
    try:
        for l in open("/proc/meminfo"):
            if l.startswith("MemAvailable:"):
                return int(l.split()[1]) / 1_048_576
    except Exception:
        return 99.0
    return 99.0


def is_se(name):
    pre = name.split("__")[0]
    return pre in SE_DOMS or "__se" in name


def main():
    if shutil.disk_usage(HERE).free < 2_000_000_000:
        sys.exit("[fold] disk<2GB — abort")
    stoi, E, nodes, entries = load_embedding()
    k = KnowledgeDB(str(HERE / "knowledge.db"), stoi, E, nodes, entries)
    k.db.execute("PRAGMA busy_timeout=120000")
    row = k.db.execute("SELECT val FROM meta WHERE key='stacked'").fetchone()
    stacked = set(json.loads(row[0])) if row else set()
    now = time.time()
    new = [f for f in sorted(LIB.glob("*.txt"))
           if f.name not in stacked and not is_se(f.name) and now - f.stat().st_mtime > 5]
    print(f"[fold] {len(new):,} safe files (OMC+code+arxiv+wiki; SE skipped). per-passage cap {MAX_NODES_PER_PASSAGE}.", flush=True)
    if not new:
        print("[fold] nothing"); return
    nodeset = k.nodeset; cur = k.db.cursor()
    t0 = time.time(); npass = 0; nedge = 0
    for fi, f in enumerate(new):
        label = f.stem.split("__")[0]
        try:
            ps = split_passages(clean_text(f.read_text(errors="replace")))
        except Exception:
            ps = []
        for p in ps:
            pid = k._npass; k._npass += 1
            cur.execute("INSERT INTO passages VALUES(?,?,?)", (pid, p, label)); npass += 1
            present = [(m.group(), m.start()) for m in TOK.finditer(p.lower()) if m.group() in nodeset][:MAX_NODES_PER_PASSAGE]
            seen = set()
            for ai in range(len(present)):
                wa, sa = present[ai]
                for bi in range(ai + 1, len(present)):
                    wb, sb = present[bi]
                    if wa == wb or abs(sa - sb) > WIN or (wa, wb) in seen:
                        continue
                    seen.add((wa, wb))
                    cur.executemany("INSERT INTO edges(a,b,src,pid,w) VALUES(?,?,?,?,1) "
                                    "ON CONFLICT(a,b) DO UPDATE SET w=w+1",
                                    [(wa, wb, label, pid), (wb, wa, label, pid)])
                    nedge += 2
        stacked.add(f.name)
        if fi % 500 == 0 and fi:
            k.db.commit()
            print(f"[fold] {fi}/{len(new)} files, {npass:,} passages, {nedge:,} edge-ops, {mem_gb():.1f}GB free", flush=True)
            if mem_gb() < RAM_FLOOR_GB:
                print("[fold] RAM low — committing, continuing");
    k.db.execute("INSERT OR REPLACE INTO meta VALUES('stacked',?)", (json.dumps(sorted(stacked)),))
    k.db.commit()
    print(f"[fold] folded {npass:,} passages, {nedge:,} edge-ops in {time.time()-t0:.0f}s. refreshing node_deg + stop ...", flush=True)
    k.db.execute("DROP TABLE IF EXISTS node_deg")
    k.db.execute("CREATE TABLE node_deg AS SELECT a AS node, SUM(w) AS deg FROM edges GROUP BY a")
    k.db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_node_deg ON node_deg(node)")
    k.db.commit()
    print(f"[fold] DONE. OMC source + code now folded & addressable.", flush=True)


if __name__ == "__main__":
    main()

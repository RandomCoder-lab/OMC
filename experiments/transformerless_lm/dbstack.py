#!/usr/bin/env python3
"""dbstack.py — RAM-FLAT incremental grower for the disk-backed knowledge store (kdb).

Replaces stack.py (which loaded the whole web into RAM). Streams new library texts into knowledge.db via
KnowledgeDB.add_field (INSERTs, no whole-web load). Tracks incorporated files in the DB meta('stacked').
RAM stays flat (only the embedding + SQLite buffers) no matter how large the web grows → disk-bound, unbounded.
"""
import sys, json, time, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from kdb import KnowledgeDB, load_embedding

HERE = Path(__file__).parent
DB = HERE / "knowledge.db"
LIB = HERE / "library"


def main():
    if shutil.disk_usage(HERE).free < 1_500_000_000:
        print(f"[dbstack] DISK GUARD: {shutil.disk_usage(HERE).free/1e9:.1f}GB free — skip"); return
    stoi, E, nodes, entries = load_embedding()
    kdb = KnowledgeDB(DB, stoi, E, nodes, entries)
    row = kdb.db.execute("SELECT val FROM meta WHERE key='stacked'").fetchone()
    stacked = set(json.loads(row[0])) if row else set()
    now = time.time()
    new = [f for f in sorted(LIB.glob("*.txt"))
           if f.name not in stacked and now - f.stat().st_mtime > 5]   # skip files still being written
    if not new:
        print(f"[dbstack] nothing new — {kdb.stats()}"); return
    t0 = time.time(); tot = 0
    for f in new:
        label = f.stem.split("__")[0]
        try:
            added, _ = kdb.add_field(label, f.read_text(errors="replace"))
            stacked.add(f.name); tot += added
        except Exception as e:
            print(f"[dbstack] FAIL {f.name}: {e}")
    kdb.db.execute("INSERT OR REPLACE INTO meta VALUES('stacked',?)", (json.dumps(sorted(stacked)),))
    kdb.db.commit(); kdb.index()
    np_, ne, nd = kdb.stats()
    print(f"[dbstack] +{len(new)} texts, +{tot:,} edges in {time.time()-t0:.0f}s (RAM-flat). "
          f"DB now {np_:,} passages, {ne:,} edges, {nd} fields.", flush=True)


if __name__ == "__main__":
    main()

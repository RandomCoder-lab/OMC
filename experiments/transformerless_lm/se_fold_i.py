#!/usr/bin/env python3
"""se_fold_i.py — fold the focused-science SE subset into the COMPRESSED (interned) live knowledge.db.

The DB is now interned: edges_i(a_id,b_id,src_id,pid,w), passages_z(pid,ztext,dom), nodes(id,word),
srcs(id,name), node_deg_i(node_id,deg) — with TEXT-schema VIEWS for readers. The old se_fold wrote TEXT
`edges(a,b,...)` which is now a (non-writable) VIEW, so we write the interned tables directly:
  - load word→id from nodes; new science terms get fresh ids (extend the dict).
  - load src→id from srcs; new domains (physics/electronics/...) get fresh ids.
  - passages → passages_z with zlib; edges → edges_i upsert (a_id,b_id), sorted for index locality.
WHY THIS IS FAST NOW: the (a_id,b_id) INT index is ~1.5GB (vs 3-4GB TEXT) → fits the 4GB cache → the
merge that thrashed on the 9GB TEXT table runs in RAM. Per-chunk in-RAM dedupe + per-chunk WAL truncate
keep memory/WAL bounded. Resumable via meta('stacked'). node_deg_i rebuilt once at the end.

  python se_fold_i.py [files_per_chunk]      # default 300
"""
import sys, re, json, time, shutil, zlib
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from kdb import KnowledgeDB, load_embedding
from nl_ground import clean_text, split_passages

HERE = Path(__file__).parent
LIB = HERE / "library"
TOK = re.compile(r"[a-z][a-z']+")
WIN = 14 * 6
MAX_NODES = 50
DISK_FLOOR = 12_000_000_000
RAM_FLOOR_GB = 5.0
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
    chunk = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    k = KnowledgeDB(str(HERE / "knowledge.db"), *load_embedding())
    if not k.compressed:
        sys.exit("[se_fold_i] live db is NOT the compressed schema — use se_fold.py instead")
    db = k.db
    db.execute("PRAGMA busy_timeout=120000")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA cache_size=-4000000")     # 4GB: the small INT index now fits → no thrash
    nodeset = k.nodeset

    w2id = {w: i for i, w in db.execute("SELECT id,word FROM nodes")}
    next_id = (max(w2id.values()) + 1) if w2id else 1
    src2id = {n: i for i, n in db.execute("SELECT id,name FROM srcs")}
    next_src = (max(src2id.values()) + 1) if src2id else 1
    npass = db.execute("SELECT COALESCE(MAX(pid),-1)+1 FROM passages_z").fetchone()[0]
    row = db.execute("SELECT val FROM meta WHERE key='stacked'").fetchone()
    stacked = set(json.loads(row[0])) if row else set()

    todo = [f for f in sorted(LIB.glob("*.txt"))
            if f.name not in stacked and f.name.split("__")[0] in SUBSET]
    print(f"[se_fold_i] {len(todo):,} focused-science files (interned write). chunk={chunk}. "
          f"dict={len(w2id):,} words, npass starts {npass:,}.", flush=True)
    if not todo:
        print("[se_fold_i] nothing to do."); return

    def wid(w):                                  # get-or-create node id
        nonlocal next_id
        i = w2id.get(w)
        if i is None:
            i = w2id[w] = next_id; next_id += 1
            new_nodes.append((i, w))
        return i

    t0 = time.time(); folded = tot_pass = tot_edges = 0
    for ci in range(0, len(todo), chunk):
        if shutil.disk_usage(HERE).free < DISK_FLOOR:
            print("[se_fold_i] DISK GUARD (<12GB) — stopping cleanly. resume later.", flush=True); break
        batch = todo[ci:ci + chunk]
        agg = Counter(); ex_pid = {}; pbuf = []; new_nodes = []
        cpass = 0
        for f in batch:
            dom = f.name.split("__")[0]
            sid = src2id.get(dom)
            if sid is None:
                sid = src2id[dom] = next_src; next_src += 1
                db.execute("INSERT OR IGNORE INTO srcs VALUES(?,?)", (sid, dom))
            try:
                passages = split_passages(clean_text(f.read_text(errors="replace")))
            except Exception:
                passages = []
            for p in passages:
                pid = npass; npass += 1
                pbuf.append((pid, zlib.compress(p.encode("utf-8"), 6), dom))
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
                        ai, bi = wid(wa), wid(wb)
                        agg[(ai, bi, sid)] += 1
                        agg[(bi, ai, sid)] += 1
                        if (ai, bi) not in ex_pid:
                            ex_pid[(ai, bi)] = pid; ex_pid[(bi, ai)] = pid
                cpass += 1
            stacked.add(f.name)
            if mem_avail_gb() < RAM_FLOOR_GB:
                break
        # flush: new nodes, passages, then deduped edges (sorted by (a_id,b_id) for index locality)
        if new_nodes:
            db.executemany("INSERT OR IGNORE INTO nodes VALUES(?,?)", new_nodes)
        db.executemany("INSERT OR IGNORE INTO passages_z VALUES(?,?,?)", pbuf)
        db.executemany(
            "INSERT INTO edges_i(a_id,b_id,src_id,pid,w) VALUES(?,?,?,?,?) "
            "ON CONFLICT(a_id,b_id) DO UPDATE SET w = w + excluded.w",
            sorted((a, b, s, ex_pid[(a, b)], w) for (a, b, s), w in agg.items()))
        db.execute("INSERT OR REPLACE INTO meta VALUES('stacked',?)", (json.dumps(sorted(stacked)),))
        db.commit()
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")   # keep WAL bounded across chunks
        folded += len(batch); tot_pass += cpass; tot_edges += len(agg)
        print(f"[se_fold_i] chunk {ci//chunk+1}: +{len(batch)} files, {cpass:,} passages, "
              f"{len(agg):,} edges, +{len(new_nodes):,} new terms | {folded:,}/{len(todo):,} files | "
              f"dict {len(w2id):,} | {mem_avail_gb():.1f}GB RAM, {shutil.disk_usage(HERE).free/1e9:.0f}GB disk "
              f"| {time.time()-t0:.0f}s", flush=True)
        agg.clear(); ex_pid.clear(); pbuf.clear(); new_nodes.clear()

    print(f"[se_fold_i] folded {folded:,} files, {tot_pass:,} passages in {time.time()-t0:.0f}s. "
          f"rebuilding node_deg_i ...", flush=True)
    db.execute("DROP TABLE IF EXISTS node_deg_i")
    db.execute("CREATE TABLE node_deg_i AS SELECT a_id AS node_id, SUM(w) AS deg FROM edges_i GROUP BY a_id")
    db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_node_deg_i ON node_deg_i(node_id)")
    db.commit()
    for w in ["photosynthesis", "resistor", "voltage", "circuit", "entropy", "quantum"]:
        d = db.execute("SELECT deg FROM node_deg WHERE node=?", (w,)).fetchone()
        print(f"[se_fold_i] '{w}' deg = {d[0] if d else 0}", flush=True)
    np_ = db.execute("SELECT COUNT(*) FROM passages_z").fetchone()[0]
    ne = db.execute("SELECT COUNT(*) FROM edges_i").fetchone()[0]
    print(f"[se_fold_i] DONE. DB now {np_:,} passages, {ne:,} edges, {len(w2id):,} terms.", flush=True)


if __name__ == "__main__":
    main()

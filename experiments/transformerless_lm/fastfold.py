#!/usr/bin/env python3
"""fastfold.py — FAST, RAM-FLAT fold of the new library files + node expansion, preserving the existing DB
(19-language align edges + parallel-verse passages + code edges all stay).

v2 (post-OOM fix): NEVER caches passages in RAM (the v1 bug — caching all 27k files' split passages OOM'd on
the SE dumps). Both passes STREAM from disk (re-reading is cheap, local). Edges accumulate in a distinct-pair
dict that FLUSHES (bulk-UPSERT) whenever it hits PAIR_CAP or a passage buffer fills — so RAM is bounded
regardless of how dense the corpus is. Hard RAM guard flushes early and aborts clean if memory runs low.

Steps: (1) stream new files → tf/df counts → derive new content nodes (OMC/code/SE/music/art) + stopwords;
persist embedding. (2) stream again in flush-bounded fashion: passages + RAM-deduped edges, bulk-UPSERT.
(3) rebuild node_deg + stop. Dedups edges before touching the 110M-row table = an order of magnitude fewer
index touches than dbstack.
"""
import sys, re, json, time, shutil
from pathlib import Path
import torch

HERE = Path(__file__).parent
LIB = HERE / "library"
CACHE = HERE / ".kwebcache"
DB = HERE / "knowledge.db"
sys.path.insert(0, str(HERE))
from kdb import KnowledgeDB, load_embedding
from nl_ground import clean_text, split_passages

TOK = re.compile(r"[a-z][a-z']+")
WIN = 14 * 6
MIN_DF_NEW = 8
NEW_CAP = 30000
DF_RATIO_STOP = 0.05
PAIR_CAP = 1_200_000      # flush the edge dict when it exceeds this many distinct pairs (bounds RAM)
PASS_BUF = 5000           # flush passage insert buffer
RAM_FLOOR_GB = 4.0


def mem_gb():
    try:
        for l in open("/proc/meminfo"):
            if l.startswith("MemAvailable:"):
                return int(l.split()[1]) / 1_048_576
    except Exception:
        return 99.0
    return 99.0


def iter_passages(f):
    try:
        for p in split_passages(clean_text(f.read_text(errors="replace"))):
            yield p
    except Exception:
        return


def main():
    stoi, E, nodes, entries = load_embedding()
    k = KnowledgeDB(str(DB), stoi, E, nodes, entries)
    k.db.execute("PRAGMA busy_timeout=120000")
    row = k.db.execute("SELECT val FROM meta WHERE key='stacked'").fetchone()
    stacked = set(json.loads(row[0])) if row else set()
    now = time.time()
    new = [f for f in sorted(LIB.glob("*.txt")) if f.name not in stacked and now - f.stat().st_mtime > 5]
    print(f"[fastfold] {len(new):,} new files (RAM-flat, streaming — preserving existing DB)", flush=True)
    if not new:
        print("[fastfold] nothing new"); return
    nodeset = set(nodes); t0 = time.time()

    # ── PASS 1: stream, count tf/df (NO passage caching) ──
    tf = {}; df = {}; npass = 0
    for i, f in enumerate(new):
        for p in iter_passages(f):
            npass += 1
            seen = set()
            for m in TOK.finditer(p.lower()):
                w = m.group()
                if w not in nodeset:
                    tf[w] = tf.get(w, 0) + 1
                if w not in seen:
                    seen.add(w); df[w] = df.get(w, 0) + 1
        if i % 5000 == 0 and i:
            print(f"[fastfold] scan {i}/{len(new)} files, {npass:,} passages, {len(tf):,} new-types, {mem_gb():.1f}GB free", flush=True)
            if mem_gb() < RAM_FLOOR_GB:
                sys.exit("[fastfold] RAM low during scan — abort clean")
    new_words = [w for _, w in sorted(((tf[w], w) for w in tf if df.get(w, 0) >= MIN_DF_NEW
                 and df.get(w, 0) / max(npass, 1) <= DF_RATIO_STOP), reverse=True)[:NEW_CAP]]
    print(f"[fastfold] {npass:,} passages; +{len(new_words):,} new nodes. examples: {new_words[:18]}", flush=True)
    if new_words:
        base = len(nodes)
        E = torch.cat([E, torch.zeros((len(new_words), E.shape[1]), dtype=E.dtype)], 0)
        for j, w in enumerate(new_words):
            stoi[w] = base + j; nodes.append(w); nodeset.add(w)
        shutil.copy(CACHE / "E.pt", CACHE / "E.pt.prefast"); shutil.copy(CACHE / "web.json", CACHE / "web.json.prefast")
        torch.save(E, CACHE / "E.pt")
        (CACHE / "web.json").write_text(json.dumps({"stoi": stoi, "nodes": nodes, "entries": entries}))
        k.stoi = stoi; k.nodes = nodes; k.nodeset = nodeset
    del tf

    # ── PASS 2: stream again, flush-bounded passages + deduped edges ──
    cur = k.db.cursor(); pairs = {}; pbuf = []; tot_edges = 0; tot_pass = 0; flushes = 0

    def flush():
        nonlocal pairs, pbuf, tot_edges, tot_pass, flushes
        if pbuf:
            cur.executemany("INSERT INTO passages VALUES(?,?,?)", pbuf); tot_pass += len(pbuf); pbuf = []
        if pairs:
            cur.executemany("INSERT INTO edges(a,b,src,pid,w) VALUES(?,?,?,?,?) "
                            "ON CONFLICT(a,b) DO UPDATE SET w=w+excluded.w",
                            [(a, b, e[2], e[1], e[0]) for (a, b), e in pairs.items()])
            tot_edges += len(pairs); pairs = {}
        k.db.commit(); flushes += 1

    for fi, f in enumerate(new):
        label = f.stem.split("__")[0]
        for p in iter_passages(f):
            pid = k._npass; k._npass += 1
            pbuf.append((pid, p, label))
            present = [(m.group(), m.start()) for m in TOK.finditer(p.lower()) if m.group() in nodeset]
            seen = set()
            for ai in range(len(present)):
                wa, sa = present[ai]
                for bi in range(ai + 1, len(present)):
                    wb, sb = present[bi]
                    if wa == wb or abs(sa - sb) > WIN:
                        continue
                    for key in ((wa, wb), (wb, wa)):
                        if key in seen:
                            continue
                        seen.add(key)
                        e = pairs.get(key)
                        if e:
                            e[0] += 1
                        else:
                            pairs[key] = [1, pid, label]
            if len(pairs) >= PAIR_CAP or len(pbuf) >= PASS_BUF:
                flush()
                if mem_gb() < RAM_FLOOR_GB:
                    print(f"[fastfold] RAM {mem_gb():.1f}GB after flush — continuing flushed", flush=True)
        if fi % 5000 == 0 and fi:
            stacked.update(x.name for x in new[max(0, fi-5000):fi])
            print(f"[fastfold] fold {fi}/{len(new)} files, {tot_pass:,} passages, {tot_edges:,} edge-merges, "
                  f"{flushes} flushes, {mem_gb():.1f}GB free", flush=True)
    flush()
    stacked.update(f.name for f in new)
    k.db.execute("INSERT OR REPLACE INTO meta VALUES('stacked',?)", (json.dumps(sorted(stacked)),))
    k.db.commit()

    print("[fastfold] rebuilding node_deg + stop ...", flush=True)
    k.db.execute("DROP TABLE IF EXISTS node_deg")
    k.db.execute("CREATE TABLE node_deg AS SELECT a AS node, SUM(w) AS deg FROM edges GROUP BY a")
    k.db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_node_deg ON node_deg(node)")
    stops = [w for w in df if w in nodeset and df[w] / max(npass, 1) > DF_RATIO_STOP]
    k.db.execute("CREATE TABLE IF NOT EXISTS stop(node TEXT PRIMARY KEY)")
    k.db.execute("DELETE FROM stop")
    k.db.executemany("INSERT OR IGNORE INTO stop VALUES(?)", ((w,) for w in stops))
    k.db.commit()
    np_, ne, nd = k.stats()
    print(f"[fastfold] DONE in {time.time()-t0:.0f}s. {np_:,} passages, {ne:,} edges, {nd} fields, "
          f"{len(nodes):,} nodes, {len(stops):,} stopwords, {flushes} flushes.", flush=True)


if __name__ == "__main__":
    main()

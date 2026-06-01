#!/usr/bin/env python3
"""reindex.py — the MEMORY-SAFE integrate: make ALL of the accumulated knowledge addressable.

The web grew to ~930k passages / 48M edges, but its addressable CONCEPTS were frozen at the 5,999
Webster-1913 dictionary headwords. Modern terms (qubit, gene, neuron, dna, transistor, entropy) were not
nodes, so the modern corpus mostly produced edges that got DROPPED, and the skip-gram meaning-vectors were
noisy (gravity~chiefly). This rebuilds the addressable layer from the corpus itself:

  Stage A (--derive):  expand the node set from CORPUS FREQUENCY across all fields (agnostic — frequency &
                       document-frequency are derived from data, not a hand list). Keep all Webster nodes;
                       add the top content terms; drop function words by their (derived) document-frequency
                       ratio, not by a stoplist.
  Stage B (--embed):   re-learn meaning by PPMI + truncated SVD of the co-occurrence matrix (classic
                       distributional semantics — faster AND cleaner than the undertrained skip-gram).
  Stage C (--edge):    rebuild the edge graph with the expanded node set so modern terms actually connect.

Everything STREAMS from knowledge.db (passages already on disk). RAM stays bounded by the node embedding +
the sparse co-occurrence matrix — never the whole web. A hard RAM guard aborts if MemAvailable drops low
(the full in-RAM rebuild is what crashed the box; this must not).

Usage:  python reindex.py --derive [--limit N]      # A: write nodes_new.json
        python reindex.py --embed                    # B: write .kwebcache/E.pt + web.json (small)
        python reindex.py --edge                     # C: rebuild edges table in knowledge.db
        python reindex.py --all                      # A→B→C
"""
import sys, os, re, json, time, sqlite3
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
DB = HERE / "knowledge.db"
CACHE = HERE / ".kwebcache"
TOK = re.compile(r"[a-z][a-z']+")
NODES_NEW = HERE / "nodes_new.json"

NEW_MAX = 22000          # corpus terms to add on top of the Webster nodes
MIN_DF = 8               # a new term must appear in >= this many passages (drop junk/typos)
DF_RATIO_MAX = 0.05      # df/N cut in the function-word/content gap (of..do >=0.06; content <=0.04) — DERIVED
WIN_CHARS = 14 * 6       # co-occurrence window (matches kdb.add_field)
RAM_FLOOR_GB = 3.0


def mem_available_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1_048_576
    except Exception:
        return 99.0
    return 99.0


def ram_guard(where):
    g = mem_available_gb()
    if g < RAM_FLOOR_GB:
        sys.exit(f"[reindex] RAM GUARD at {where}: only {g:.1f}GB available — aborting (no crash).")
    return g


def stream_passages(limit=None, batch=2000):
    """Yield (pid, text, dom) from disk in batches — RAM-flat (never loads all passages)."""
    db = sqlite3.connect(str(DB))
    db.execute("PRAGMA query_only=1")
    cur = db.execute("SELECT pid,text,dom FROM passages" + (f" LIMIT {limit}" if limit else ""))
    while True:
        rows = cur.fetchmany(batch)
        if not rows:
            break
        for r in rows:
            yield r
    db.close()


def load_old_meaning():
    """Old stoi/nodes/entries via the existing cache (entries = Webster definitions, preserved)."""
    d = json.loads((CACHE / "web.json").read_text())
    return d["stoi"], d["nodes"], d.get("entries", {})


# ─────────────────────────────── Stage A: derive nodes ───────────────────────────────
def derive(limit=None):
    ram_guard("derive:start")
    print("[A] loading Webster node set (grounded headwords, always kept)...", flush=True)
    _, webster_nodes, entries = load_old_meaning()
    webster = set(webster_nodes)
    print(f"[A]   {len(webster):,} Webster nodes; {len(entries):,} have definitions", flush=True)

    tf = {}            # term -> total frequency
    df = {}            # term -> document (passage) frequency
    npass = 0
    t0 = time.time()
    for pid, text, dom in stream_passages(limit):
        npass += 1
        seen = set()
        for m in TOK.finditer(text.lower()):
            w = m.group()
            tf[w] = tf.get(w, 0) + 1
            seen.add(w)
        for w in seen:
            df[w] = df.get(w, 0) + 1
        if npass % 100000 == 0:
            ram_guard("derive:scan")
            print(f"[A]   scanned {npass:,} passages, {len(tf):,} types, "
                  f"{mem_available_gb():.1f}GB free", flush=True)
    print(f"[A] scanned {npass:,} passages -> {len(tf):,} term types in {time.time()-t0:.0f}s", flush=True)

    # AGNOSTIC function-word separation by DOCUMENT-FREQUENCY RATIO (df/N), read off the data's own
    # distribution — not a stoplist, not a length rule. Empirically the closed-class function words form a
    # dense high-df_ratio cluster (of=0.66 … do=0.08, up=0.06) with a clear GAP before the content words
    # (greater=0.03, labour=0.02, called=0.03, years=0.03). DF_RATIO_MAX sits in that gap, so ubiquitous
    # words are dropped by how the corpus uses them. Survivors are ranked by frequency = the most-used
    # content terms. MIN_DF drops singletons/typos (a count, not a list). This is "function words via
    # frequency" applied as a derived cut.
    cands = []
    dropped_fw = []
    for w, c in tf.items():
        if w in webster:
            continue
        d = df.get(w, 0)
        if d < MIN_DF:
            continue
        if d / npass > DF_RATIO_MAX:          # ubiquitous -> closed-class function word, drop
            dropped_fw.append((c, w)); continue
        cands.append((c, w))
    cands.sort(reverse=True)
    dropped_fw.sort(reverse=True)
    new_terms = [w for _, w in cands[:NEW_MAX]]
    nodes = sorted(webster | set(new_terms))
    NODES_NEW.write_text(json.dumps({"nodes": nodes, "new_terms": new_terms,
                                     "npass": npass}, ))
    print(f"[A] node set: {len(webster):,} Webster + {len(new_terms):,} corpus-derived = {len(nodes):,} total")
    print(f"[A]   sample new nodes: {', '.join(new_terms[:30])}")
    print(f"[A]   closed-class dropped (derived, df/N>{DF_RATIO_MAX}): "
          f"{', '.join(w for _,w in dropped_fw[:15])}")
    for probe in ["qubit", "gene", "neuron", "dna", "transistor", "entropy", "algorithm", "quantum"]:
        print(f"[A]   '{probe}': {'NODE now' if probe in set(nodes) else 'still absent (rare in corpus)'}"
              f"  (tf={tf.get(probe,0)}, df={df.get(probe,0)})")
    return nodes


# ─────────────────────────────── Stage B: re-embed (PPMI + SVD) ───────────────────────────────
def embed(dim=64, limit=None):
    import scipy.sparse as sp
    from scipy.sparse.linalg import svds
    ram_guard("embed:start")
    info = json.loads(NODES_NEW.read_text())
    nodes = info["nodes"]
    stoi = {w: i for i, w in enumerate(nodes)}
    V = len(nodes)
    print(f"[B] {V:,} nodes -> {dim}-dim via PPMI+SVD (streaming co-occurrence)", flush=True)
    _, _, entries = load_old_meaning()

    # accumulate symmetric co-occurrence counts in batched COO -> CSR (sparse, RAM-bounded)
    rows, cols, data = [], [], []
    M = sp.csr_matrix((V, V), dtype=np.float32)

    def flush():
        nonlocal rows, cols, data, M
        if not rows:
            return
        B = sp.coo_matrix((np.asarray(data, dtype=np.float32),
                           (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
                          shape=(V, V)).tocsr()
        M = M + B
        rows, cols, data = [], [], []

    npass = 0; t0 = time.time()
    for pid, text, dom in stream_passages(limit):
        npass += 1
        present = [(stoi[m.group()], m.start()) for m in TOK.finditer(text.lower()) if m.group() in stoi]
        seen = set()
        for ai in range(len(present)):
            ia, sa = present[ai]
            for bi in range(ai + 1, len(present)):
                ib, sb = present[bi]
                if ia == ib or abs(sa - sb) > WIN_CHARS or (ia, ib) in seen:
                    continue
                seen.add((ia, ib))
                rows.append(ia); cols.append(ib); data.append(1.0)
                rows.append(ib); cols.append(ia); data.append(1.0)
        if len(rows) > 4_000_000:
            flush(); ram_guard("embed:cooc")
        if npass % 100000 == 0:
            print(f"[B]   co-occurrence {npass:,} passages, nnz={M.nnz:,}, "
                  f"{mem_available_gb():.1f}GB free", flush=True)
    flush()
    print(f"[B] co-occurrence built: nnz={M.nnz:,} in {time.time()-t0:.0f}s", flush=True)
    ram_guard("embed:ppmi")

    # PPMI on nonzeros:  pmi = log( c_ij * total / (c_i * c_j) ),  PPMI = max(0, pmi)
    M = M.tocoo()
    row_sum = np.asarray(M.sum(axis=1)).ravel() + 1e-9
    total = float(M.data.sum()) + 1e-9
    pmi = np.log(M.data * total / (row_sum[M.row] * row_sum[M.col]) + 1e-12)
    ppmi = np.maximum(pmi, 0.0).astype(np.float32)
    P = sp.coo_matrix((ppmi, (M.row, M.col)), shape=(V, V)).tocsr()
    P.eliminate_zeros()
    print(f"[B] PPMI nnz={P.nnz:,}; running truncated SVD (k={dim})...", flush=True)
    ram_guard("embed:svd")
    # svds needs k < min(shape); fine for V>>dim
    U, S, Vt = svds(P, k=dim)
    emb = U * np.sqrt(np.maximum(S, 0.0))           # standard PPMI-SVD embedding
    # L2-normalize rows so kdb.relate()'s dot product == cosine
    norm = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    emb = (emb / norm).astype(np.float32)

    import torch
    E = torch.from_numpy(emb)
    # back up the old cache, then write the NEW (small) cache
    if (CACHE / "E.pt").exists():
        os.replace(CACHE / "E.pt", CACHE / "E.pt.bak")
    if (CACHE / "web.json").exists():
        os.replace(CACHE / "web.json", CACHE / "web.json.bak")
    torch.save(E, CACHE / "E.pt")
    # keep only Webster definitions that are still nodes
    nodeset = set(nodes)
    entries = {w: d for w, d in entries.items() if w in nodeset}
    (CACHE / "web.json").write_text(json.dumps({"stoi": stoi, "nodes": nodes, "entries": entries}))
    print(f"[B] WROTE .kwebcache/E.pt ({V}x{dim}) + small web.json. old cache -> *.bak", flush=True)

    # sanity: clean neighbors now?
    def like(x, k=6):
        if x not in stoi:
            return f"{x}: absent"
        v = emb[stoi[x]]
        sims = emb @ v
        idx = np.argsort(-sims)[:k + 1]
        return f"{x} ~ " + "  ".join(f"{nodes[i]}·{sims[i]:.2f}" for i in idx if i != stoi[x])[:k]
    for w in ["gravity", "quantum", "light", "force", "number"]:
        print("[B]   " + like(w))
    return E


# ─────────────────────────────── Stage C: re-edge ───────────────────────────────
def reedge(limit=None):
    ram_guard("edge:start")
    info = json.loads(NODES_NEW.read_text())
    nodeset = set(info["nodes"])
    print(f"[C] rebuilding edges with {len(nodeset):,}-node set (streaming, RAM-flat)...", flush=True)
    db = sqlite3.connect(str(DB))
    db.execute("PRAGMA journal_mode=WAL"); db.execute("PRAGMA synchronous=NORMAL")
    db.execute("DROP TABLE IF EXISTS edges_new")
    db.execute("CREATE TABLE edges_new(a TEXT, b TEXT, src TEXT, pid INTEGER)")
    cur = db.cursor()
    npass = 0; nedge = 0; t0 = time.time(); buf = []
    for pid, text, dom in stream_passages(limit):
        npass += 1
        present = [(m.group(), m.start()) for m in TOK.finditer(text.lower()) if m.group() in nodeset]
        seen = set()
        for ai in range(len(present)):
            wa, sa = present[ai]
            for bi in range(ai + 1, len(present)):
                wb, sb = present[bi]
                if wa == wb or abs(sa - sb) > WIN_CHARS or (wa, wb) in seen:
                    continue
                seen.add((wa, wb))
                buf.append((wa, wb, dom, pid)); buf.append((wb, wa, dom, pid)); nedge += 2
        if len(buf) > 200000:
            cur.executemany("INSERT INTO edges_new VALUES(?,?,?,?)", buf); buf = []; db.commit()
            ram_guard("edge:write")
        if npass % 100000 == 0:
            print(f"[C]   {npass:,} passages, {nedge:,} edges, {mem_available_gb():.1f}GB free", flush=True)
    if buf:
        cur.executemany("INSERT INTO edges_new VALUES(?,?,?,?)", buf); db.commit()
    print(f"[C] swapping edges table ({nedge:,} edges)...", flush=True)
    db.execute("DROP TABLE edges")
    db.execute("ALTER TABLE edges_new RENAME TO edges")
    db.execute("CREATE INDEX idx_edges_a ON edges(a)")
    db.commit()
    ne = db.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    db.close()
    print(f"[C] DONE: {ne:,} edges in {time.time()-t0:.0f}s (RAM-flat)", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--derive", action="store_true")
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--edge", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dim", type=int, default=64)
    a = ap.parse_args()
    if a.all or a.derive:
        derive(a.limit)
    if a.all or a.embed:
        embed(a.dim, a.limit)
    if a.all or a.edge:
        reedge(a.limit)
    if not (a.all or a.derive or a.embed or a.edge):
        print(__doc__)

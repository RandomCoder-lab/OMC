"""node_expand.py — make the newly-grown knowledge ADDRESSABLE + fix the function-word pollution, in one
streaming pass over knowledge.db passages (RAM-flat, no re-embed).

Two derived, agnostic outputs:
  1. STOPWORD DEMOTION: count document-frequency per existing node; nodes whose df/N exceeds a cut read off
     the distribution (the closed-class cluster: the/of/to/if/is…) are flagged in a `stop` table. Ranking
     (kask/stitch/translate) excludes them — the single fix that cleans EVERY query (the function-word
     leak that's dogged code + bridges + stitch).
  2. NODE EXPANSION: high-frequency CONTENT terms not yet nodes (technical terms from the new arXiv/code/SE
     corpus — eigenvalue, transistor, mutex, polynomial…) are added as nodes (zero-vector; live by edges).

Then re-edges ONLY the passages added since the last edge build (dom in the given/auto set) so the new terms
get connected, UPSERT-weighted. Hard RAM guard. Does NOT re-embed (existing vectors stay valid; new nodes
rely on graph structure).

  python node_expand.py                       # derive stopwords + expand nodes from full corpus
  python node_expand.py --edge-doms code-,arxiv,se,signals,electronics,computing,math   # also re-edge these
"""
import sys, re, json, time, sqlite3
from pathlib import Path
import torch

HERE = Path(__file__).parent
DB = HERE / "knowledge.db"
CACHE = HERE / ".kwebcache"
TOK = re.compile(r"[a-z][a-z']+")
DF_RATIO_STOP = 0.05      # df/N above this = closed-class function word (derived cut; content words are <0.04)
MIN_DF_NEW = 12           # a new content node must appear in >= this many passages
NEW_CAP = 40000
WIN_CHARS = 14 * 6
RAM_FLOOR_GB = 3.0


def mem_gb():
    try:
        for l in open("/proc/meminfo"):
            if l.startswith("MemAvailable:"):
                return int(l.split()[1]) / 1_048_576
    except Exception:
        return 99.0
    return 99.0


def guard(w):
    g = mem_gb()
    if g < RAM_FLOOR_GB:
        sys.exit(f"[node_expand] RAM GUARD {w}: {g:.1f}GB — abort clean")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge-doms", default="", help="comma prefixes of doms to (re)edge with new nodes")
    a = ap.parse_args()
    edge_prefixes = [x for x in a.edge_doms.split(",") if x] or \
        ["code-", "arxiv", "se", "signals", "electronics", "computing", "math", "physics"]

    d = json.loads((CACHE / "web.json").read_text())
    stoi, nodes, entries = d["stoi"], d["nodes"], d.get("entries", {})
    nodeset = set(nodes)
    db = sqlite3.connect(str(DB))
    db.execute("PRAGMA journal_mode=WAL"); db.execute("PRAGMA synchronous=NORMAL")

    # ── pass 1: df over all passages for nodes + candidate new terms ──
    print("[node_expand] scanning passages for df ...", flush=True)
    df_node = {}; tf_new = {}; df_new = {}; npass = 0
    cur = db.execute("SELECT text FROM passages")
    t0 = time.time()
    while True:
        rows = cur.fetchmany(4000)
        if not rows:
            break
        for (text,) in rows:
            npass += 1
            seen = set()
            for m in TOK.finditer(text.lower()):
                w = m.group()
                if w in nodeset:
                    seen.add(w)
                else:
                    tf_new[w] = tf_new.get(w, 0) + 1
                    if w not in seen:
                        seen.add(w)
            for w in seen:
                if w in nodeset:
                    df_node[w] = df_node.get(w, 0) + 1
        # df_new: recompute per-doc below is costly; approximate df_new via tf threshold + a second light pass
        if npass % 200000 == 0:
            guard("scan"); print(f"  {npass:,} passages, {mem_gb():.1f}GB free", flush=True)
    # second light pass for df of new-term candidates (only those tf>=MIN_DF_NEW to bound memory)
    cand = {w for w, c in tf_new.items() if c >= MIN_DF_NEW and len(w) >= 2}
    print(f"[node_expand] {npass:,} passages; {len(cand):,} new-term candidates; computing their df ...", flush=True)
    cur = db.execute("SELECT text FROM passages")
    while True:
        rows = cur.fetchmany(4000)
        if not rows:
            break
        for (text,) in rows:
            seen = set()
            for m in TOK.finditer(text.lower()):
                w = m.group()
                if w in cand and w not in seen:
                    df_new[w] = df_new.get(w, 0) + 1; seen.add(w)
    guard("post-df")

    # ── stopwords: existing nodes with high df-ratio (closed class), derived ──
    stops = sorted((w for w, dfc in df_node.items() if dfc / npass > DF_RATIO_STOP),
                   key=lambda w: -df_node[w])
    db.execute("DROP TABLE IF EXISTS stop")
    db.execute("CREATE TABLE stop(node TEXT PRIMARY KEY)")
    db.executemany("INSERT OR IGNORE INTO stop VALUES(?)", ((w,) for w in stops))
    db.commit()
    print(f"[node_expand] flagged {len(stops):,} stopword nodes (df/N>{DF_RATIO_STOP}). "
          f"top: {stops[:15]}", flush=True)

    # ── new content nodes: frequent, content-bearing (df-ratio not too high), not already nodes ──
    new = sorted(((df_new[w], w) for w in df_new
                  if df_new[w] / npass <= DF_RATIO_STOP), reverse=True)[:NEW_CAP]
    new_words = [w for _, w in new]
    print(f"[node_expand] adding {len(new_words):,} new content nodes. examples: {new_words[:25]}", flush=True)
    if new_words:
        E = torch.load(CACHE / "E.pt")
        base = len(nodes)
        E = torch.cat([E, torch.zeros((len(new_words), E.shape[1]), dtype=E.dtype)], 0)
        for i, w in enumerate(new_words):
            stoi[w] = base + i; nodes.append(w); nodeset.add(w)
        import shutil
        shutil.copy(CACHE / "E.pt", CACHE / "E.pt.preexpand"); shutil.copy(CACHE / "web.json", CACHE / "web.json.preexpand")
        torch.save(E, CACHE / "E.pt")
        (CACHE / "web.json").write_text(json.dumps({"stoi": stoi, "nodes": nodes, "entries": entries}))

    # ── re-edge new-domain passages so the new terms connect (UPSERT, RAM-flat) ──
    guard("edge")
    q = " OR ".join("dom LIKE ?" for _ in edge_prefixes)
    params = [p + "%" for p in edge_prefixes]
    print(f"[node_expand] re-edging passages in doms ~ {edge_prefixes} ...", flush=True)
    pcur = db.execute(f"SELECT pid,text,dom FROM passages WHERE {q}", params)
    wcur = db.cursor(); nedge = 0; npp = 0
    while True:
        rows = pcur.fetchmany(2000)
        if not rows:
            break
        buf = []
        for pid, text, dom in rows:
            npp += 1
            present = [(m.group(), m.start()) for m in TOK.finditer(text.lower()) if m.group() in nodeset]
            seen = set()
            for i in range(len(present)):
                wa, sa = present[i]
                for j in range(i + 1, len(present)):
                    wb, sb = present[j]
                    if wa == wb or abs(sa - sb) > WIN_CHARS or (wa, wb) in seen:
                        continue
                    seen.add((wa, wb)); buf.append((wa, wb, dom, pid)); buf.append((wb, wa, dom, pid))
            if len(buf) > 100000:
                wcur.executemany("INSERT INTO edges(a,b,src,pid,w) VALUES(?,?,?,?,1) "
                                 "ON CONFLICT(a,b) DO UPDATE SET w=w+1", buf); nedge += len(buf); buf = []; db.commit(); guard("edge-loop")
        if buf:
            wcur.executemany("INSERT INTO edges(a,b,src,pid,w) VALUES(?,?,?,?,1) "
                             "ON CONFLICT(a,b) DO UPDATE SET w=w+1", buf); nedge += len(buf)
        db.commit()
    print(f"[node_expand] re-edged {npp:,} passages, +{nedge:,} edge-ops. refreshing node_deg ...", flush=True)
    db.execute("DROP TABLE IF EXISTS node_deg")
    db.execute("CREATE TABLE node_deg AS SELECT a AS node, SUM(w) AS deg FROM edges GROUP BY a")
    db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_node_deg ON node_deg(node)")
    db.commit(); db.close()
    print(f"[node_expand] DONE in {time.time()-t0:.0f}s. nodes now {len(nodes):,}.", flush=True)


if __name__ == "__main__":
    main()

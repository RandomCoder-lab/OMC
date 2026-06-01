#!/usr/bin/env python3
"""finalize_compressed.py — make knowledge_compressed.db a DROP-IN for the TEXT-schema tooling, via VIEWS.

compress_web.py produced interned tables (edges a_id/b_id, passages ztext BLOB, node_deg node_id). Every
reader (kdb.neighbors/_passage/assoc, langexec.edge_w, engine/create direct SQL) expects the OLD TEXT
schema: edges(a,b,src,pid,w), passages(pid,text,dom), node_deg(node,deg). Rather than rewrite all of
them, we present that schema with VIEWS over the interned tables:
  - rename real tables → edges_i / passages_z / node_deg_i
  - VIEW edges     = join edges_i to nodes(word) + srcs(name)          → (a,b,src,pid,w)
  - VIEW passages  = unzip(ztext)  (unzip = per-connection Python zlib fn, registered in kdb)
  - VIEW node_deg  = join node_deg_i to nodes(word)                    → (node,deg)
  - copy the `stop` table (compress_web omitted it) + index nodes(word) for fast view lookups.

After this + the kdb.__init__ patch (register unzip, detect compressed schema), the compressed DB is a
drop-in. NON-DESTRUCTIVE: edits only knowledge_compressed.db.
"""
import sqlite3, zlib, sys
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "knowledge.db"            # for copying the stop table
DST = HERE / "knowledge_compressed.db"


def main():
    if not DST.exists():
        sys.exit("no knowledge_compressed.db — run compress_web.py first")
    d = sqlite3.connect(str(DST))
    d.create_function("unzip", 1, lambda b: zlib.decompress(b).decode("utf-8", "replace") if b else "")

    # already finalized? (idempotent)
    have_view = d.execute("SELECT 1 FROM sqlite_master WHERE type='view' AND name='edges'").fetchone()
    if have_view:
        print("[finalize] already finalized (views present).")
    else:
        # 1. copy stop table from the original (compress_web didn't carry it; ~130 hub words)
        d.execute("CREATE TABLE IF NOT EXISTS stop(node TEXT PRIMARY KEY)")
        s = sqlite3.connect(f"file:{SRC}?mode=ro", uri=True)
        try:
            rows = s.execute("SELECT node FROM stop").fetchall()
            d.executemany("INSERT OR IGNORE INTO stop VALUES(?)", rows)
            print(f"[finalize] copied {len(rows)} stop words")
        except sqlite3.OperationalError:
            print("[finalize] (source has no stop table — skipping)")
        s.close()

        # 2. index nodes(word) so the view's word→id lookups are fast
        print("[finalize] indexing nodes(word) ...", flush=True)
        d.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_nodes_word ON nodes(word)")

        # 3. rename interned tables, create TEXT-presenting views
        print("[finalize] renaming tables + creating views ...", flush=True)
        d.execute("ALTER TABLE edges RENAME TO edges_i")
        d.execute("ALTER TABLE passages RENAME TO passages_z")
        d.execute("ALTER TABLE node_deg RENAME TO node_deg_i")
        d.execute("""CREATE VIEW edges AS
            SELECT na.word AS a, nb.word AS b, s.name AS src, e.pid AS pid, e.w AS w
            FROM edges_i e JOIN nodes na ON e.a_id=na.id JOIN nodes nb ON e.b_id=nb.id
            LEFT JOIN srcs s ON e.src_id=s.id""")
        d.execute("CREATE VIEW passages AS SELECT pid, unzip(ztext) AS text, dom FROM passages_z")
        d.execute("""CREATE VIEW node_deg AS
            SELECT na.word AS node, nd.deg AS deg FROM node_deg_i nd JOIN nodes na ON nd.node_id=na.id""")
        d.commit()
        print("[finalize] views created.")

    # 4. verify the views reproduce real data
    print("\n[finalize] verifying views ...", flush=True)
    # a known edge round-trips through the view
    row = d.execute("SELECT a,b,w FROM edges WHERE a='bacteria' AND b='resistant'").fetchone()
    print(f"  edges view: bacteria->resistant = {row}")
    nd = d.execute("SELECT node,deg FROM node_deg WHERE node='quantum'").fetchone()
    print(f"  node_deg view: quantum deg = {nd}")
    pv = d.execute("SELECT pid, substr(text,1,60) FROM passages WHERE pid=(SELECT pid FROM passages_z LIMIT 1)").fetchone()
    print(f"  passages view (decompressed): pid {pv[0]} -> {pv[1]!r}")
    nstop = d.execute("SELECT COUNT(*) FROM stop").fetchone()[0]
    print(f"  stop words: {nstop}")
    d.close()
    print("\n[finalize] DONE. knowledge_compressed.db is now TEXT-schema-compatible (via views).")


if __name__ == "__main__":
    main()

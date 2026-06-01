#!/usr/bin/env python3
"""memory.py — a write-don't-train MEMORY layer over the web. The corpus is what humanity wrote; memory is
what THIS system was told, derived, or did. Same principle as the whole project: knowledge (now experience)
enters as addressable DATA — provenance-marked, salience-weighted, reinforceable, and FORGETTABLE — never as
retrained weights.

HONESTY RULE (non-negotiable): memory is provenance-separated from corpus. A told/derived fact is NEVER
corpus evidence — it surfaces tagged ⟨memory:kind @ time⟩, and factcheck excludes src='memory' from grounding.
"you told me X" must never masquerade as "the sources say X".

Kinds: told (user stated) · derived (system discovered via bridge/infer/connect) · episodic (a query+result) ·
note. Dynamics: salience (reinforced on re-encounter, like the edge-weight UPSERT), recency decay at recall,
explicit forget (reversible — unlike a fine-tuned weight).

  python memory.py remember "gravity is not the sole cause light bends" --kind told
  python memory.py recall "why does light bend"
  python memory.py list
  python memory.py forget 3
"""
import sys, re, time, json, sqlite3
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from kdb import KnowledgeDB, load_embedding

DB = Path(__file__).parent / "knowledge.db"
TOK = re.compile(r"[a-z][a-z']+")
DECAY_PER_DAY = 0.97   # gentle recency decay applied at recall


class Memory:
    def __init__(self, k):
        self.k = k
        self.k.db.execute("PRAGMA busy_timeout=120000")
        self.k.db.execute("""CREATE TABLE IF NOT EXISTS memory(
            id INTEGER PRIMARY KEY, ts REAL, kind TEXT, text TEXT, concepts TEXT, salience REAL)""")
        # multiple derived form-tags per memory (relation/source/scope/salience) — agnostic, from data signals
        cols = [r[1] for r in self.k.db.execute("PRAGMA table_info(memory)")]
        if "forms" not in cols:
            self.k.db.execute("ALTER TABLE memory ADD COLUMN forms TEXT DEFAULT ''")
        self.k.db.commit()

    def concepts(self, text):
        out, seen = [], set()
        for w in TOK.findall(text.lower()):
            if w in self.k.stoi and w not in self.k.stop and w not in seen:
                out.append(w); seen.add(w)
        return out

    def remember(self, text, kind="note", salience=1.0):
        cs = self.concepts(text)
        cur = self.k.db.cursor()
        cur.execute("INSERT INTO memory(ts,kind,text,concepts,salience) VALUES(?,?,?,?,?)",
                    (time.time(), kind, text, json.dumps(cs), salience))
        mid = cur.lastrowid
        # addressable memory edges among the concepts — src='memory' so corpus-grounding never sees them
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                cur.executemany("INSERT INTO edges(a,b,src,pid,w) VALUES(?,?,?,?,1) "
                                "ON CONFLICT(a,b) DO UPDATE SET w=w+1",
                                [(cs[i], cs[j], "memory", mid), (cs[j], cs[i], "memory", mid)])
        self.k.db.commit()
        return mid, cs

    def remember_discovery(self, key_concepts, text, kind="derived", forms=None):
        """Auto-memory for reasoning acts (bridge/infer/navigate). Dedup by concept-set: rediscovering the
        same connection REINFORCES it (salience++) rather than duplicating. `forms` = a set of derived
        form-tags (relation/source/scope) so one memory carries multiple tagged forms of information.
        Returns (id, 'new'|'reinforced')."""
        key = json.dumps(sorted(set(key_concepts)))
        fstr = ",".join(sorted(forms)) if forms else kind
        row = self.k.db.execute("SELECT id,forms FROM memory WHERE concepts=? AND kind IN (?, 'derived','conceptual','institutional')",
                                (key, kind)).fetchone()
        if row:
            self.reinforce(row[0])
            # merge any new form-tags seen on rediscovery
            merged = ",".join(sorted(set((row[1] or "").split(",")) | set(fstr.split(",")) - {""}))
            self.k.db.execute("UPDATE memory SET forms=? WHERE id=?", (merged, row[0])); self.k.db.commit()
            return row[0], "reinforced"
        cur = self.k.db.cursor()
        cur.execute("INSERT INTO memory(ts,kind,text,concepts,salience,forms) VALUES(?,?,?,?,?,?)",
                    (time.time(), kind, text, key, 1.0, fstr))
        mid = cur.lastrowid
        ks = sorted(set(key_concepts))
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                cur.executemany("INSERT INTO edges(a,b,src,pid,w) VALUES(?,?,?,?,1) "
                                "ON CONFLICT(a,b) DO UPDATE SET w=w+1",
                                [(ks[i], ks[j], "memory", mid), (ks[j], ks[i], "memory", mid)])
        self.k.db.commit()
        return mid, "new"

    def reinforce(self, mid, by=1.0):
        self.k.db.execute("UPDATE memory SET salience=salience+?, ts=? WHERE id=?", (by, time.time(), mid))
        self.k.db.commit()

    def forget(self, mid):
        self.k.db.execute("DELETE FROM memory WHERE id=?", (mid,)); self.k.db.commit()

    def recall(self, query, topk=8):
        q = set(self.concepts(query)); now = time.time()
        scored = []
        for mid, ts, kind, text, cj, sal in self.k.db.execute(
                "SELECT id,ts,kind,text,concepts,salience FROM memory"):
            cs = set(json.loads(cj))
            overlap = len(q & cs)
            if overlap == 0 and q:
                continue
            age_days = max(0.0, (now - ts) / 86400.0)
            score = (salience := sal) * (DECAY_PER_DAY ** age_days) * (1 + overlap)
            scored.append((score, mid, kind, text, sal, age_days))
        scored.sort(reverse=True)
        return scored[:topk]


def _fmt_age(d):
    if d < 1/24: return f"{int(d*1440)}m ago"
    if d < 1: return f"{int(d*24)}h ago"
    return f"{int(d)}d ago"


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    s, E, n, e = load_embedding()
    m = Memory(KnowledgeDB(str(DB), s, E, n, e))
    cmd = sys.argv[1]
    if cmd == "remember":
        kind = "note"
        args = sys.argv[2:]
        if "--kind" in args:
            i = args.index("--kind"); kind = args[i+1]; args = args[:i] + args[i+2:]
        mid, cs = m.remember(" ".join(args), kind)
        print(f"remembered #{mid} ⟨{kind}⟩ — concepts: {cs}")
    elif cmd == "recall":
        for score, mid, kind, text, sal, age in m.recall(" ".join(sys.argv[2:])):
            print(f"  #{mid} ⟨memory:{kind} · sal {sal:.1f} · {_fmt_age(age)}⟩  {text}")
    elif cmd == "list":
        rows = m.k.db.execute("SELECT id,kind,salience,text FROM memory ORDER BY ts DESC LIMIT 20").fetchall()
        for mid, kind, sal, text in rows:
            print(f"  #{mid} ⟨{kind} sal{sal:.1f}⟩ {text[:90]}")
        if not rows: print("  (no memories yet)")
    elif cmd == "reinforce":
        m.reinforce(int(sys.argv[2])); print(f"reinforced #{sys.argv[2]}")
    elif cmd == "forget":
        m.forget(int(sys.argv[2])); print(f"forgot #{sys.argv[2]}")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()

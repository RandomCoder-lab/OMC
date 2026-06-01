#!/usr/bin/env python3
"""code_nodes.py — make CODE addressable. The web's nodes came from natural-language corpora, so code
keywords/identifiers (fn, async, malloc, impl, defn, struct, await) aren't addresses and the fetched source
only links the few tokens that happen to be English words. This does for code what add_language did for a
foreign language: derive the frequent code tokens from the fetched source files as NEW nodes (zero-vector;
they live by edges, not embedding), then create windowed co-occurrence edges among them (src='code') so
async↔await, malloc↔free, fn↔rust emerge. Agnostic: tokens + frequency derived from real source, no
hand-listed keywords.

  python code_nodes.py                 # scan library/<lang>__*.txt, expand nodes, edge them
"""
import sys, re, json, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
from kdb import KnowledgeDB, load_embedding

HERE = Path(__file__).parent
LIB = HERE / "library"
CACHE = HERE / ".kwebcache"
CODE_TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")   # identifiers / keywords
WIN = 8            # co-occurrence window in tokens
MIN_DF = 5         # a code token must appear in >= this many files to become a node
NEW_CAP = 16000    # cap new code nodes (top by frequency) to keep the node set bounded
# the languages we fetched as code (file prefix <language>__)
CODE_LANGS = {"python", "rust", "go", "c", "cpp", "javascript", "typescript", "java", "ruby", "swift",
              "kotlin", "haskell", "julia", "csharp", "lua", "elixir", "clojure", "zig", "ocaml", "perl",
              "fortran", "erlang", "dart", "nim", "d", "shell", "verilog", "systemverilog", "vhdl", "scala",
              "groovy", "elm", "objc", "assembly"}


def code_files():
    return [f for f in LIB.glob("*.txt") if f.stem.split("__")[0] in CODE_LANGS]


def main():
    files = code_files()
    print(f"[code-nodes] {len(files)} code source files", flush=True)
    if not files:
        sys.exit("no code files (run code_corpus.py first)")
    stoi, E, nodes, entries = load_embedding()
    k = KnowledgeDB(str(__import__('pathlib').Path(HERE / 'knowledge.db')), stoi, E, nodes, entries)
    nodeset = set(nodes)

    # pass 1: token frequency + document frequency over code
    tf, df = {}, {}
    t0 = time.time()
    toks_cache = {}
    for i, f in enumerate(files):
        try:
            txt = f.read_text(errors="replace")
        except Exception:
            continue
        toks = [t.lower() for t in CODE_TOK.findall(txt)]
        toks_cache[f.name] = (f.stem.split("__")[0], toks)
        seen = set()
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
            seen.add(t)
        for t in seen:
            df[t] = df.get(t, 0) + 1
    # new code nodes: frequent enough, not already a node; cap by frequency
    cands = sorted(((tf[t], t) for t in tf if df.get(t, 0) >= MIN_DF and t not in nodeset), reverse=True)
    new_words = [t for _, t in cands[:NEW_CAP]]
    print(f"[code-nodes] {len(tf):,} code token types; adding {len(new_words):,} new nodes "
          f"(df>={MIN_DF}, cap {NEW_CAP}). examples: {new_words[:25]}", flush=True)
    if new_words:
        base = len(nodes)
        E = torch.cat([E, torch.zeros((len(new_words), E.shape[1]), dtype=E.dtype)], 0)
        for j, w in enumerate(new_words):
            stoi[w] = base + j; nodes.append(w); nodeset.add(w)
        k.stoi = stoi; k.E = E; k.nodes = nodes; k.nodeset = nodeset

    # pass 2: windowed co-occurrence edges among code nodes (src=language), UPSERT-weighted
    cur = k.db.cursor(); nedge = 0; pid_base = k._npass
    for fname, (lang, toks) in toks_cache.items():
        present = [t for t in toks if t in nodeset]
        if len(present) < 2:
            continue
        pid = k._npass; k._npass += 1
        # store a small provenance passage (first 400 tokens joined) so edges trace to source
        cur.execute("INSERT INTO passages VALUES(?,?,?)", (pid, " ".join(toks[:400]), f"code-{lang}"))
        seen = set()
        for a in range(len(present)):
            for b in range(a + 1, min(a + 1 + WIN, len(present))):
                wa, wb = present[a], present[b]
                if wa == wb or (wa, wb) in seen:
                    continue
                seen.add((wa, wb))
                cur.executemany("INSERT INTO edges(a,b,src,pid,w) VALUES(?,?,?,?,1) "
                                "ON CONFLICT(a,b) DO UPDATE SET w=w+1",
                                [(wa, wb, f"code-{lang}", pid), (wb, wa, f"code-{lang}", pid)])
                nedge += 2
    k.db.commit()
    print(f"[code-nodes] +{nedge:,} code edges. persisting + refreshing PMI ...", flush=True)
    import shutil
    if (CACHE / "E.pt").exists():
        shutil.copy(CACHE / "E.pt", CACHE / "E.pt.precode"); shutil.copy(CACHE / "web.json", CACHE / "web.json.precode")
    torch.save(E, CACHE / "E.pt")
    (CACHE / "web.json").write_text(json.dumps({"stoi": stoi, "nodes": nodes, "entries": entries}))
    k.index()
    print(f"[code-nodes] DONE in {time.time()-t0:.0f}s. nodes now {len(nodes):,}.", flush=True)


if __name__ == "__main__":
    main()

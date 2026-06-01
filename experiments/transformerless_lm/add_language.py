#!/usr/bin/env python3
"""add_language.py — bridge a THIRD language into the meaning-geoform (the interlingua proof / reverse-Babel).

Thesis (omc_addressed_cognition_thesis): the parallel-verse is a language-INDEPENDENT MEANING ADDRESS. A new
language attaches to it the same way Latin did — its words co-occur, per aligned verse, with the English words
that share the meaning. Bridge enough verses and the new language's word for a concept becomes a neighbor of
the English word, WITH NO bilingual dictionary ever given. Do it for N languages and they all meet at the same
addresses → translation between any pair, by addressing.

What's new vs bible_bridge.py: the third language's words are NOT yet nodes (the node set came from the
English/Latin corpus). So we ADD them as nodes — appended to stoi/nodes with placeholder (zero) embedding
rows. Translation rides on EDGES + PMI (node_deg), not the embedding, so a meaning-less vector is fine; the
word is addressable and its neighbors are its translations. Only words seen >= MIN_DF aligned verses become
nodes (drops junk). Latin-script languages are diacritic-normalized to ASCII (agnostic Unicode NFKD, not a
word list); CJK/Cyrillic need script-aware tokenization (noted, not handled here).

  python add_language.py --trans almeida --name portuguese --pivot dra --books GEN,JHN,EST,1KI,2KI,DAN,1SA,2SA
"""
import sys, time, json, urllib.request
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
from kdb import KnowledgeDB, load_embedding
import langtok
from apicache import get

HERE = Path(__file__).parent
DB = HERE / "knowledge.db"
CACHE = HERE / ".kwebcache"
API = "https://bible-api.com/data"
FUNC_DEG = 200_000
MIN_DF = 3
CHAPTERS = {"GEN": 50, "EXO": 40, "JOS": 24, "1SA": 31, "2SA": 24, "1KI": 22, "2KI": 25, "EST": 10,
            "DAN": 14, "PSA": 150, "JHN": 21, "MAT": 28, "LUK": 24, "REV": 22, "1CH": 29, "2CH": 36}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="language name in langtok.REGISTRY (e.g. russian, chinese)")
    ap.add_argument("--trans", default="", help="override bible-api translation id (else from REGISTRY)")
    ap.add_argument("--mode", default="", help="override tokenizer mode latin|unicode|cjk (else from REGISTRY)")
    ap.add_argument("--pivot", default="dra", help="English pivot translation id")
    ap.add_argument("--books", default="GEN,JHN,EST,1KI,2KI,DAN,1SA,2SA")
    args = ap.parse_args()
    books = [b.strip().upper() for b in args.books.split(",") if b.strip()]
    args.trans = args.trans or langtok.trans_for(args.name)
    mode = args.mode or langtok.mode_for(args.name)
    if not args.trans:
        sys.exit(f"unknown language '{args.name}' — add it to langtok.REGISTRY or pass --trans/--mode")
    print(f"[lang:{args.name}] trans={args.trans} mode={mode} pivot={args.pivot}", flush=True)

    stoi, E, nodes, entries = load_embedding()
    k = KnowledgeDB(str(DB), stoi, E, nodes, entries)
    nodeset = set(nodes)

    # ── fetch all verse pairs into memory (bounded), count new-language token doc-freq ──
    verses = []          # (bk,ch,vno, Ltext, Etext)
    df = {}
    t0 = time.time()
    for bk in books:
        for ch in range(1, CHAPTERS.get(bk, 1) + 1):
            la = get(f"{API}/{args.trans}/{bk}/{ch}"); en = get(f"{API}/{args.pivot}/{bk}/{ch}")
            if not la or not en:
                time.sleep(0.3); continue
            lav = {v["verse"]: v["text"] for v in la.get("verses", [])}
            env = {v["verse"]: v["text"] for v in en.get("verses", [])}
            for vno in lav.keys() & env.keys():
                verses.append((bk, ch, vno, lav[vno], env[vno]))
                for w in set(langtok.tokenize(lav[vno], mode)):
                    df[w] = df.get(w, 0) + 1
            time.sleep(0.4)
        print(f"[lang:{args.name}] fetched {bk}: {len(verses):,} verse-pairs, {len(df):,} L-types, "
              f"{time.time()-t0:.0f}s", flush=True)

    # ── new nodes: new-language words seen in >= MIN_DF verses and not already a node ──
    new_words = sorted(w for w, c in df.items() if c >= MIN_DF and w not in nodeset)
    print(f"[lang:{args.name}] adding {len(new_words):,} new nodes (df>={MIN_DF}). examples: {new_words[:20]}", flush=True)
    if new_words:
        base = len(nodes)
        addE = torch.zeros((len(new_words), E.shape[1]), dtype=E.dtype)
        E = torch.cat([E, addE], 0)
        for i, w in enumerate(new_words):
            stoi[w] = base + i; nodes.append(w); nodeset.add(w)
        k.stoi = stoi; k.E = E; k.nodes = nodes; k.nodeset = nodeset

    # ── bridge: per verse, edges between new-language nodes and English nodes (UPSERT-weighted) ──
    cur = k.db.cursor()
    nbridge = 0
    for bk, ch, vno, Lraw, Et in verses:
        Ln = [w for w in dict.fromkeys(langtok.tokenize(Lraw, mode)) if w in nodeset]
        En = [w for w in dict.fromkeys(langtok.tokenize_en(Et)) if w in nodeset and k.deg.get(w, 0) < FUNC_DEG]
        if not Ln or not En:
            continue
        pid = k._npass; k._npass += 1
        cur.execute("INSERT INTO passages VALUES(?,?,?)",
                    (pid, f"[{bk} {ch}:{vno}] {args.name.upper()}: {Lraw.strip()} || ENG: {Et.strip()}",
                     f"parallel-{args.name}"))
        for a in Ln:
            if k.deg.get(a, 0) >= FUNC_DEG:    # skip new-lang function-word hubs once they have degree
                continue
            for b in En:
                if a == b:
                    continue
                cur.executemany(
                    "INSERT INTO edges(a,b,src,pid,w) VALUES(?,?,?,?,1) "
                    "ON CONFLICT(a,b) DO UPDATE SET w=w+1",
                    [(a, b, f"parallel-{args.name}", pid), (b, a, f"parallel-{args.name}", pid)])
                nbridge += 2
    k.db.commit()
    print(f"[lang:{args.name}] {nbridge:,} bridge-edges. persisting embedding + refreshing PMI ...", flush=True)

    # persist extended embedding (backup first), refresh node_deg
    if (CACHE / "E.pt").exists():
        import shutil; shutil.copy(CACHE / "E.pt", CACHE / "E.pt.prelang"); shutil.copy(CACHE / "web.json", CACHE / "web.json.prelang")
    torch.save(E, CACHE / "E.pt")
    (CACHE / "web.json").write_text(json.dumps({"stoi": stoi, "nodes": nodes, "entries": entries}))
    k.index()
    print(f"[lang:{args.name}] DONE in {time.time()-t0:.0f}s. nodes now {len(nodes):,}.", flush=True)


if __name__ == "__main__":
    main()

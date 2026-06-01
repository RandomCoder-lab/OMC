#!/usr/bin/env python3
"""bible_corpus.py — bulk parallel-Bible bridge (the engine for ALL languages). Instead of ~120 throttled
per-chapter API calls, download ONE verse-per-line file per language from the BibleNLP/ebible corpus, where
line N is the same verse across every translation (aligned by metadata/vref.txt). Align foreign↔English by
line index, add the foreign words as nodes, and bridge them (same as add_language). Then `align.py --lang
<name> --write` learns clean IBM-Model-1 translations. Files are cached to library/_corpus so re-runs are
instant. Agnostic: parallel text is data; alignment is the verse index.

  python bible_corpus.py --name russian              # uses langtok.CORPUS[russian] ↔ eng-engDRA
  python bible_corpus.py --name chinese --foreign cmn-cmn_cu89t
"""
import sys, json, time, urllib.request
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
from kdb import KnowledgeDB, load_embedding
import langtok

HERE = Path(__file__).parent
DB = HERE / "knowledge.db"
CACHE = HERE / ".kwebcache"
CORPUS_DIR = HERE / "library" / "_corpus"
RAW = "https://raw.githubusercontent.com/BibleNLP/ebible/main"
UA = {"User-Agent": "OMC-knowledge-web/1.0 (research; local)"}
FUNC_DEG = 200_000
MIN_DF = 3


def fetch_lines(name):
    """Download a corpus file (or vref) once; cache to disk; return list of lines."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    f = CORPUS_DIR / (name.replace("/", "_") + ".txt")
    if f.exists():
        return f.read_text(encoding="utf-8").splitlines()
    url = f"{RAW}/metadata/vref.txt" if name == "vref" else f"{RAW}/corpus/{name}.txt"
    for attempt in range(3):
        try:
            data = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read().decode("utf-8", "replace")
            f.write_text(data, encoding="utf-8")
            return data.splitlines()
        except Exception as e:
            print(f"  fetch {name} attempt {attempt+1}: {e}", flush=True); time.sleep(3 * (attempt + 1))
    return []


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--foreign", default="", help="corpus file id (else from langtok.CORPUS)")
    ap.add_argument("--eng", default=langtok.ENG_CORPUS)
    ap.add_argument("--mode", default="")
    args = ap.parse_args()
    foreign_id = args.foreign or langtok.CORPUS.get(args.name, "")
    mode = args.mode or langtok.mode_for(args.name)
    if not foreign_id:
        sys.exit(f"no corpus file for '{args.name}' — pass --foreign or add to langtok.CORPUS")
    print(f"[corpus:{args.name}] foreign={foreign_id} eng={args.eng} mode={mode}", flush=True)

    vref = fetch_lines("vref")
    flines = fetch_lines(foreign_id)
    elines = fetch_lines(args.eng)
    if not (len(vref) == len(flines) == len(elines)):
        sys.exit(f"line-count mismatch vref={len(vref)} foreign={len(flines)} eng={len(elines)}")
    verses = [(vref[i], flines[i].strip(), elines[i].strip())
              for i in range(len(vref)) if flines[i].strip() and elines[i].strip()]
    print(f"[corpus:{args.name}] {len(verses):,} aligned verse pairs", flush=True)
    if not verses:
        sys.exit("no aligned verses (foreign file may not cover the same books)")

    stoi, E, nodes, entries = load_embedding()
    k = KnowledgeDB(str(DB), stoi, E, nodes, entries)
    nodeset = set(nodes)

    # new foreign nodes: words seen in >= MIN_DF verses and not already a node
    df = {}
    for _, f, _e in verses:
        for w in set(langtok.tokenize(f, mode)):
            df[w] = df.get(w, 0) + 1
    new_words = sorted(w for w, c in df.items() if c >= MIN_DF and w not in nodeset)
    print(f"[corpus:{args.name}] adding {len(new_words):,} new nodes (df>={MIN_DF}). examples: {new_words[:15]}", flush=True)
    if new_words:
        base = len(nodes)
        E = torch.cat([E, torch.zeros((len(new_words), E.shape[1]), dtype=E.dtype)], 0)
        for i, w in enumerate(new_words):
            stoi[w] = base + i; nodes.append(w); nodeset.add(w)
        k.stoi = stoi; k.E = E; k.nodes = nodes; k.nodeset = nodeset

    cur = k.db.cursor(); nbridge = 0
    TAG = args.name.upper()
    for ref, fraw, eraw in verses:
        Fn = [w for w in dict.fromkeys(langtok.tokenize(fraw, mode)) if w in nodeset]
        En = [w for w in dict.fromkeys(langtok.tokenize_en(eraw)) if w in nodeset and k.deg.get(w, 0) < FUNC_DEG]
        if not Fn or not En:
            continue
        pid = k._npass; k._npass += 1
        cur.execute("INSERT INTO passages VALUES(?,?,?)",
                    (pid, f"[{ref}] {TAG}: {fraw} || ENG: {eraw}", f"parallel-{args.name}"))
        for a in Fn:
            if k.deg.get(a, 0) >= FUNC_DEG:
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
    print(f"[corpus:{args.name}] {nbridge:,} bridge-edges. persisting + refreshing PMI ...", flush=True)
    import shutil
    if (CACHE / "E.pt").exists():
        shutil.copy(CACHE / "E.pt", CACHE / "E.pt.prelang"); shutil.copy(CACHE / "web.json", CACHE / "web.json.prelang")
    torch.save(E, CACHE / "E.pt")
    (CACHE / "web.json").write_text(json.dumps({"stoi": stoi, "nodes": nodes, "entries": entries}))
    k.index()
    print(f"[corpus:{args.name}] DONE. nodes now {len(nodes):,}. next: python align.py --lang {args.name} --write", flush=True)


if __name__ == "__main__":
    main()

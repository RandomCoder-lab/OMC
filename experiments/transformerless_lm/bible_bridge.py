#!/usr/bin/env python3
"""bible_bridge.py — TRANSLATION BY ADDRESSING. The web learned to *speak* Latin (a coherent Latin island)
but couldn't *translate* (Latin↔English cosine ≈0, no bridge edges) because the corpus had ~no parallel
text. Fix, pure thesis: feed verse-aligned parallel scripture and let the bridge edges form. Then translation
is navigation — look up the Latin address, hop the cross-language edge, read the English neighbor.

Source: bible-api.com — Clementine Latin Vulgate (`clementine`) ↔ Douay-Rheims (`dra`, the English
translation OF the Vulgate = tightest alignment), structured endpoint /data/<trans>/<BOOK>/<chapter>,
verse-for-verse aligned. For each aligned verse we add CROSS-LANGUAGE edges between the Latin nodes and the
English nodes in that verse (bipartite, UPSERT-weighted exactly like growth's add_field). A true translation
pair (rex–king, aqua–water, lux–light, deus–god) co-occurs across MANY verses → high weight → high PMI;
spurious pairs co-occur rarely → demoted. Function-word hubs are dropped by node degree (agnostic, the same
signal kdb uses) so bridges aren't drowned by 'the'/'et'/'unto'. The verse pair is stored as a grounded,
pid-traceable passage (provenance ⟨bible-parallel⟩).

Agnostic: parallel text is DATA; alignment is given by verse number; bridges are co-occurrence. No translation
model, no bilingual dictionary in code.

  python bible_bridge.py            # default king/light/water-rich book set
  python bible_bridge.py --books EST,DAN,1KI,2KI,GEN,JHN
"""
import sys, re, time, json, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from kdb import KnowledgeDB, load_embedding

HERE = Path(__file__).parent
DB = HERE / "knowledge.db"
API = "https://bible-api.com/data"
UA = {"User-Agent": "OMC-knowledge-web/1.0 (research; local)"}
TOK = re.compile(r"[a-z][a-z']+")
FUNC_DEG = 200_000          # drop nodes whose total edge-weight degree exceeds this (function-word hubs)
CURSOR = HERE / "library" / "_bible_cursor.json"

# books rich in rex/king (EST/Kings/DAN/CH), lux/light + aqua/water + deus/god (GEN/JHN/PSA), war (Kings/JOS)
DEFAULT_BOOKS = ["GEN", "EXO", "JOS", "1SA", "2SA", "1KI", "2KI", "1CH", "2CH", "EST", "DAN",
                 "PSA", "PRO", "ISA", "JHN", "MAT", "LUK", "REV"]
CHAPTERS = {"GEN": 50, "EXO": 40, "JOS": 24, "1SA": 31, "2SA": 24, "1KI": 22, "2KI": 25, "1CH": 29,
            "2CH": 36, "EST": 10, "DAN": 14, "PSA": 150, "PRO": 31, "ISA": 66, "JHN": 21, "MAT": 28,
            "LUK": 24, "REV": 22}


from apicache import get


def nodes_in(text, stoi, deg):
    out, seen = [], set()
    for w in TOK.findall(text.lower()):
        if w in stoi and w not in seen and deg.get(w, 0) < FUNC_DEG:
            out.append(w); seen.add(w)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", default="")
    ap.add_argument("--max-ch", type=int, default=0, help="cap chapters per book (0=all)")
    args = ap.parse_args()
    books = [b.strip().upper() for b in args.books.split(",") if b.strip()] or DEFAULT_BOOKS

    stoi, E, nodes, entries = load_embedding()
    k = KnowledgeDB(str(DB), stoi, E, nodes, entries)
    done = set()
    if CURSOR.exists():
        done = set(json.loads(CURSOR.read_text()))

    cur = k.db.cursor()
    npair = 0; nbridge = 0; nverse = 0; t0 = time.time()
    for bk in books:
        nch = CHAPTERS.get(bk, 1)
        if args.max_ch:
            nch = min(nch, args.max_ch)
        for ch in range(1, nch + 1):
            key = f"{bk}/{ch}"
            if key in done:
                continue
            la = get(f"{API}/clementine/{bk}/{ch}")
            en = get(f"{API}/dra/{bk}/{ch}")
            done.add(key)
            if not la or not en:
                time.sleep(0.4); continue
            lav = {v["verse"]: v["text"] for v in la.get("verses", [])}
            env = {v["verse"]: v["text"] for v in en.get("verses", [])}
            for vno in lav.keys() & env.keys():
                Ln = nodes_in(lav[vno], stoi, k.deg)
                En = nodes_in(env[vno], stoi, k.deg)
                if not Ln or not En:
                    continue
                pid = k._npass; k._npass += 1
                cur.execute("INSERT INTO passages VALUES(?,?,?)",
                            (pid, f"[{bk} {ch}:{vno}] LAT: {lav[vno].strip()} || ENG: {env[vno].strip()}",
                             "bible-parallel"))
                nverse += 1
                for a in Ln:
                    for b in En:
                        if a == b:
                            continue
                        cur.executemany(
                            "INSERT INTO edges(a,b,src,pid,w) VALUES(?,?,?,?,1) "
                            "ON CONFLICT(a,b) DO UPDATE SET w=w+1",
                            [(a, b, "bible-parallel", pid), (b, a, "bible-parallel", pid)])
                        nbridge += 2
            npair += 1
            if npair % 20 == 0:
                k.db.commit(); CURSOR.write_text(json.dumps(sorted(done)))
                print(f"[bridge] {key}: {nverse:,} verse-pairs, {nbridge:,} bridge-edges, "
                      f"{time.time()-t0:.0f}s", flush=True)
            time.sleep(0.5)
    k.db.commit(); CURSOR.write_text(json.dumps(sorted(done)))
    print(f"[bridge] refreshing node degrees / PMI ...", flush=True)
    k.index()
    print(f"[bridge] DONE: {nverse:,} verse-pairs, {nbridge:,} bridge-edges added in {time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()

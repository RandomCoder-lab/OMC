#!/usr/bin/env python3
"""make_stops.py — derive the stopword (function-word) set NOW, without the full node_expand pass, so the
fact-check / agnostic tools work immediately. Scans a sample of passages, flags existing nodes whose
document-frequency ratio exceeds a derived cut (closed-class function words appear in a huge fraction of
passages), writes the `stop` table. Read-heavy + tiny write (busy_timeout so it waits politely if growth is
mid-commit). node_expand later rebuilds this identically (idempotent). Agnostic: df-ratio derived from data.
"""
import sys, re, json, sqlite3
from pathlib import Path

HERE = Path(__file__).parent
DB = HERE / "knowledge.db"
TOK = re.compile(r"[a-z][a-z']+")
DF_RATIO_STOP = 0.05
SAMPLE = 250_000


def main():
    nodes = set(json.loads((HERE / ".kwebcache" / "web.json").read_text())["nodes"])
    db = sqlite3.connect(str(DB), timeout=120)
    db.execute("PRAGMA busy_timeout=120000")
    df = {}; n = 0
    cur = db.execute(f"SELECT text FROM passages LIMIT {SAMPLE}")
    while True:
        rows = cur.fetchmany(4000)
        if not rows:
            break
        for (text,) in rows:
            n += 1
            seen = set()
            for m in TOK.finditer(text.lower()):
                w = m.group()
                if w in nodes:
                    seen.add(w)
            for w in seen:
                df[w] = df.get(w, 0) + 1
    stops = sorted((w for w, c in df.items() if c / n > DF_RATIO_STOP), key=lambda w: -df[w])
    db.execute("CREATE TABLE IF NOT EXISTS stop(node TEXT PRIMARY KEY)")
    db.execute("DELETE FROM stop")
    db.executemany("INSERT OR IGNORE INTO stop VALUES(?)", ((w,) for w in stops))
    db.commit(); db.close()
    print(f"[make_stops] scanned {n:,} passages → {len(stops):,} stopwords (df/N>{DF_RATIO_STOP}). "
          f"top: {stops[:20]}")


if __name__ == "__main__":
    main()

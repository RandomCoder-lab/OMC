#!/usr/bin/env python3
"""wiki_ingest.py — Wikipedia into the knowledge web (the broadest single modern source: "everything").

Fetches plaintext article extracts via the MediaWiki API and drops them in library/ for the disk-backed
grower to fold into knowledge.db. Two modes:
  --random N : N random articles (broad coverage across all of human knowledge)
  --topics "a,b,c" : specific titles (target a field/concept)
Field label = the article's first content category if available, else "wikipedia". Polite (UA + delay),
resumable (skips already-fetched pageids).
"""
import sys, re, json, time, urllib.request, urllib.parse
from pathlib import Path

HERE = Path(__file__).parent
LIB = HERE / "library"
API = "https://en.wikipedia.org/w/api.php"
UA = "OMC-knowledge-web/1.0 (research; local)"


def api(params):
    params["format"] = "json"
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read().decode("utf-8", errors="replace"))
        except Exception:
            time.sleep(5 * (attempt + 1))
    return {}


def field_for(cats):
    # crude field from categories (agnostic: Wikipedia's own metadata); else 'wikipedia'
    for c in cats:
        t = c.lower()
        for f in ("physic", "chemis", "biolog", "mathemat", "astronom", "comput", "philosoph",
                  "histor", "economic", "psycholog", "medicin", "art", "music", "linguist", "geograph"):
            if f in t:
                return re.sub(r"[^a-z]", "", f)[:9] or "wikipedia"
    return "wikipedia"


def save_page(pageid, title, extract, cats, existing):
    name = f"{field_for(cats)}__wiki{pageid}.txt"
    if name in existing or len(extract) < 300:
        return 0
    (LIB / name).write_text(f"{title}. {extract}")
    existing.add(name)
    return len(extract)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", type=int, default=0)
    ap.add_argument("--topics", default="")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    LIB.mkdir(exist_ok=True)
    existing = {f.name for f in LIB.glob("*wiki*.txt")}
    total = 0

    def fetch_pages(extra):
        nonlocal total
        d = api({**extra, "prop": "extracts|categories", "explaintext": 1,
                 "exlimit": "max", "cllimit": 5, "clshow": "!hidden"})
        for p in d.get("query", {}).get("pages", {}).values():
            ex = p.get("extract", "")
            cats = [c["title"].replace("Category:", "") for c in p.get("categories", [])]
            n = save_page(p.get("pageid", 0), p.get("title", "?"), ex, cats, existing)
            if n:
                total += 1

    if args.topics:
        titles = [t.strip() for t in args.topics.split(",") if t.strip()]
        for i in range(0, len(titles), args.batch):
            fetch_pages({"action": "query", "titles": "|".join(titles[i:i + args.batch])})
            print(f"[wiki] topics {i+args.batch}/{len(titles)} -> {total} saved", flush=True); time.sleep(1.5)
    if args.random:
        done = 0
        while done < args.random:
            fetch_pages({"action": "query", "generator": "random", "grnnamespace": 0,
                         "grnlimit": min(args.batch, args.random - done)})
            done += args.batch
            print(f"[wiki] random {done}/{args.random} -> {total} saved", flush=True); time.sleep(1.5)
    print(f"\n[wiki] saved {total} articles into library/", flush=True)


if __name__ == "__main__":
    main()

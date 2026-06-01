#!/usr/bin/env python3
"""se_dump.py — Stack Exchange data dumps into the web (downloadable bulk, no API, no auth). archive.org
hosts each SE site as a .7z; Posts.xml inside is the full Q&A (human-explained, concept-dense — ideal for
"basics → PhD" in exactly the technical domains). We stream-download the 7z, extract Posts.xml, iterparse it
(RAM-flat), strip HTML, and write question+answer text to library/ for the disk grower to fold. Big files are
deleted after parsing to bound disk. Agnostic: public Q&A as data.

  python se_dump.py                       # default: signals, computing, electronics, math
  python se_dump.py --sites electronics.stackexchange.com,dsp.stackexchange.com
"""
import sys, re, subprocess, urllib.request, html
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).parent
LIB = HERE / "library"
DUMPS = LIB / "_sedumps"
BASE = "https://archive.org/download/stackexchange"
UA = {"User-Agent": "OMC-knowledge-web/1.0 (research; local)"}
TAG = re.compile(r"<[^>]+>")
WS = re.compile(r"\s+")
# site -> field label
SITES = {"dsp.stackexchange.com": "signals", "cs.stackexchange.com": "computing",
         "electronics.stackexchange.com": "electronics", "math.stackexchange.com": "math",
         "engineering.stackexchange.com": "engineering", "robotics.stackexchange.com": "robotics",
         "music.stackexchange.com": "music", "graphicdesign.stackexchange.com": "design",
         "physics.stackexchange.com": "physics", "scicomp.stackexchange.com": "scientific_computing"}
DONE = DUMPS / "_done.txt"


def strip_html(s):
    return WS.sub(" ", html.unescape(TAG.sub(" ", s or ""))).strip()


def download(site):
    DUMPS.mkdir(parents=True, exist_ok=True)
    f = DUMPS / f"{site}.7z"
    if f.exists() and f.stat().st_size > 1000:
        return f
    url = f"{BASE}/{site}.7z"
    print(f"  downloading {site} ...", flush=True)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, open(f, "wb") as out:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        return f
    except Exception as e:
        print(f"  download FAIL {site}: {e}", flush=True); return None


def parse_posts(site, field, sevenz):
    # extract Posts.xml to the dumps dir, iterparse, write chunked passages, then clean up
    subprocess.run(["7z", "x", "-y", f"-o{DUMPS}", str(sevenz), "Posts.xml"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    px = DUMPS / "Posts.xml"
    if not px.exists():
        print(f"  no Posts.xml for {site}", flush=True); return 0
    short = field
    existing = {f.name for f in LIB.glob(f"{short}__se*.txt")}
    buf, fileno, posts = [], len(existing), 0
    for _, el in ET.iterparse(str(px), events=("end",)):
        if el.tag != "row":
            el.clear(); continue
        pt = el.get("PostTypeId")
        if pt in ("1", "2"):
            title = strip_html(el.get("Title", ""))
            body = strip_html(el.get("Body", ""))
            text = (title + ". " + body) if title else body
            if len(text) > 80:
                buf.append(text); posts += 1
        el.clear()
        if len(buf) >= 250:
            (LIB / f"{short}__se{fileno}.txt").write_text("\n\n".join(buf)); fileno += 1; buf = []
    if buf:
        (LIB / f"{short}__se{fileno}.txt").write_text("\n\n".join(buf))
    px.unlink(missing_ok=True); sevenz.unlink(missing_ok=True)   # reclaim disk
    print(f"  {site}: {posts:,} posts -> library/{short}__se*.txt", flush=True)
    return posts


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default="")
    args = ap.parse_args()
    sites = [s.strip() for s in args.sites.split(",") if s.strip()] or list(SITES)
    done = set(DONE.read_text().split()) if DONE.exists() else set()
    DUMPS.mkdir(parents=True, exist_ok=True)
    for site in sites:
        if site in done:
            print(f"[se] {site} already done", flush=True); continue
        field = SITES.get(site, site.split(".")[0])
        print(f"[se] {site} ({field})", flush=True)
        f = download(site)
        if not f:
            continue
        parse_posts(site, field, f)
        done.add(site); DONE.write_text("\n".join(sorted(done)))
    print("[se] DONE", flush=True)


if __name__ == "__main__":
    main()

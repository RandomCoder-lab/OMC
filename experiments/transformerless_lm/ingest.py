#!/usr/bin/env python3
"""ingest.py — autonomous, polite, RESUMABLE ingestion of public-domain knowledge across every field.

Grows the knowledge library piece by piece (the user's vision: "every bit of information we can").
Field-labeled Gutenberg texts -> library/<field>__<id>.txt. Polite (sequential, delay, UA), resumable
(skips already-fetched via manifest), size-capped (so kweb rebuilds stay tractable), best-effort (tries
multiple URL patterns, skips failures). Agnostic: everything is DATA dropped into the library; kweb
addresses it uniformly. Run repeatedly — it only fetches what's missing.
"""
import sys, time, json, re, shutil, urllib.request
from pathlib import Path

HERE = Path(__file__).parent
LIB = HERE / "library"
MANIFEST = LIB / "_manifest.json"
CAP_BYTES = 3_000_000          # cap each text (~3MB) — breadth over depth; keeps rebuild tractable
DELAY = 2.5                    # politeness between fetches (seconds)
DISK_FLOOR = 1_200_000_000     # hard disk guard: never fetch below ~1.2GB free (protect the box)


def disk_free():
    return shutil.disk_usage(HERE).free
UA = "Mozilla/5.0 (research; OMC knowledge-web ingest; contact local)"

# field -> [Gutenberg ids]. Best-effort; wrong/old ids just skip. Broad coverage of human knowledge.
FIELDS = {
    "philosophy": [1497, 8438, 6762, 4280, 59, 3800, 2680, 34901, 11224, 1998, 3207, 10615, 9662, 4705, 1080],
    "science":    [944, 2300, 33504, 1228, 2009, 37729, 14744, 28820],
    "math":       [15114, 21076, 16653, 13700, 33283, 17384],
    "astronomy":  [51265, 28994, 29782, 37383],
    "history":    [2707, 7142, 674, 10657, 2848, 1404, 18, 731],
    "economics":  [3300, 33310, 30107, 833, 1232],
    "psychology": [621, 15489, 57724, 35880],
    "religion":   [10, 3434, 2388, 216, 3623, 4928, 3327, 45631],
    "geography":  [10636, 2055, 944, 3456],
    "literature": [100, 2701, 1727, 6130, 8800, 20, 996, 2591, 11339, 1322, 76],
    "law":        [25831, 1404],
    "psych_phil": [4363, 5740, 1656],
    "language":   [10681, 37134, 29765],
}


def load_manifest():
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {"done": {}, "failed": []}


def fetch_text(gid):
    """Return (text, subject) or (None, None). Subject auto-classified from Gutenberg's own metadata."""
    for url in (f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
                f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
                f"https://www.gutenberg.org/files/{gid}/{gid}-8.txt",
                f"https://www.gutenberg.org/files/{gid}/{gid}.txt"):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=45) as r:
                data = r.read(CAP_BYTES)
            if len(data) > 15000:
                txt = data.decode("utf-8", errors="replace")
                m = re.search(r"Subject:\s*([A-Za-z][\w &'-]+)", txt[:6000])
                subj = re.sub(r"[^a-z]", "", m.group(1).split()[0].lower()) if m else "general"
                return txt, (subj or "general")
        except Exception:
            continue
    return None, None


def fetch(gid, dest, log):
    txt, _ = fetch_text(gid)
    if txt:
        dest.write_text(txt); return len(txt)
    return 0


def crawl_ids(want):
    """Pull more IDs from Gutenberg's top-downloads (continuous breadth for the hours-long loop)."""
    ids = []
    for page in ("https://www.gutenberg.org/browse/scores/top",):
        try:
            req = urllib.request.Request(page, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                html = r.read().decode("utf-8", errors="replace")
            for m in re.finditer(r"/ebooks/(\d+)", html):
                gid = int(m.group(1))
                if gid not in ids:
                    ids.append(gid)
        except Exception:
            pass
    return ids[:want]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--crawl", type=int, default=0, help="also fetch N popular Gutenberg texts")
    ap.add_argument("--seq", type=int, default=0, help="fetch N sequential Gutenberg IDs (UNLIMITED source; subject auto-labeled)")
    ap.add_argument("--max-lib", type=int, default=10**9, help="library cap (default effectively none)")
    args = ap.parse_args()
    LIB.mkdir(exist_ok=True)
    man = load_manifest()
    man.setdefault("cursor", 0)
    total_new = total_bytes = 0
    t0 = time.time()
    have = {int(k.split("__")[1]) for k in man["done"] if "__" in k and k.split("__")[1].isdigit()}
    plan = [(f, gid) for f, ids in FIELDS.items() for gid in ids if gid not in have]
    if args.crawl:
        plan += [("general", gid) for gid in crawl_ids(args.crawl * 3) if gid not in have][:args.crawl]
    # --seq: walk sequential Gutenberg IDs (cursor persisted) — the unlimited self-classifying source.
    seq_left = args.seq
    for field, gid in plan:
            if disk_free() < DISK_FLOOR:
                print(f"[ingest] DISK GUARD: {disk_free()/1e9:.1f}GB free < floor — stopping fetch", flush=True); break
            if len(list(LIB.glob("*.txt"))) >= args.max_lib:
                print(f"[ingest] library cap {args.max_lib} reached — stopping fetch", flush=True); break
            key = f"{field}__{gid}"
            dest = LIB / f"{key}.txt"
            if key in man["done"] and dest.exists():
                continue
            if dest.exists() and dest.stat().st_size > 15000:
                man["done"][key] = dest.stat().st_size; continue
            n = fetch(gid, dest, print)
            if n:
                man["done"][key] = n; total_new += 1; total_bytes += n
                print(f"[ingest] +{field:11} #{gid:<6} {n/1e6:.1f}MB  (total new {total_new})", flush=True)
            else:
                if gid not in man["failed"]:
                    man["failed"].append(gid)
                print(f"[ingest]  {field:11} #{gid:<6} FAILED/skip", flush=True)
            MANIFEST.write_text(json.dumps(man, indent=2))
            time.sleep(DELAY)
    # ── UNLIMITED sequential crawl: walk Gutenberg IDs, auto-label by Subject metadata ──
    attempts = 0
    while seq_left > 0 and attempts < args.seq * 6 and len(list(LIB.glob("*.txt"))) < args.max_lib:
        if disk_free() < DISK_FLOOR:
            print(f"[ingest] DISK GUARD: {disk_free()/1e9:.1f}GB free — stopping seq crawl", flush=True); break
        man["cursor"] += 1; attempts += 1
        gid = man["cursor"]
        if gid in have:
            continue
        txt, subj = fetch_text(gid)
        if txt:
            dest = LIB / f"{subj}__{gid}.txt"
            dest.write_text(txt)
            man["done"][f"{subj}__{gid}"] = len(txt); have.add(gid)
            total_new += 1; total_bytes += len(txt); seq_left -= 1
            print(f"[ingest] seq +{subj:14} #{gid:<6} {len(txt)/1e6:.1f}MB  (new {total_new})", flush=True)
        MANIFEST.write_text(json.dumps(man, indent=2))
        time.sleep(DELAY)
    lib_mb = sum(f.stat().st_size for f in LIB.glob("*.txt")) / 1e6
    print(f"\n[ingest] done this pass: +{total_new} texts ({total_bytes/1e6:.1f}MB) in {time.time()-t0:.0f}s",
          flush=True)
    print(f"[ingest] LIBRARY now: {len(list(LIB.glob('*.txt')))} texts, {lib_mb:.0f}MB across "
          f"{len(set(f.stem.split('__')[0] for f in LIB.glob('*.txt')))} fields", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""arxiv_ingest.py — MODERN science into the knowledge web, via arXiv's open API.

Brings the present-day canon the public-domain (pre-1928) corpus can't: quantum (quant-ph), CERN/particle
(hep-ex, hep-ph), astrophysics (astro-ph), CS/coding (cs.*), biology (q-bio), math, condensed matter,
electromagnetism (physics.*). Fetches title+abstract per paper (real modern terminology + concepts),
saves to library/arxiv-<cat>__<id>.txt labeled by category. Polite (≥3s, arXiv API guideline). Resumable
(skips already-fetched). The disk-backed grower (growloop2/dbstack) folds these into knowledge.db RAM-flat.
"""
import sys, re, time, urllib.request
from pathlib import Path

HERE = Path(__file__).parent
LIB = HERE / "library"
UA = "Mozilla/5.0 (research; OMC knowledge-web; local)"
# arXiv categories spanning the user's "all of it": atoms→quantum→CERN→cosmos→life→computation→math
CATS = [
    # COMPUTING (incl. architecture cs.AR)
    "cs.AR", "cs.PL", "cs.DS", "cs.DC", "cs.SE", "cs.LO", "cs.CC", "cs.DB", "cs.NI", "cs.CR",
    "cs.OS", "cs.PF", "cs.SC", "cs.GR", "cs.LG", "cs.AI", "cs.CL", "cs.FL", "cs.MS", "cs.ET",
    # ELECTRICAL ENGINEERING / CIRCUITRY / SIGNALS / CONTROL
    "eess.SY", "eess.SP", "eess.IV", "physics.app-ph", "cond-mat.mes-hall",
    # MATHEMATICS (broad)
    "math.NT", "math.AG", "math.LO", "math.CO", "math.AT", "math.CA", "math.PR", "math.ST",
    "math.NA", "math.DG", "math.GR", "math.OC", "math.RA", "math.CT",
    # ENGINEERING subfields: robotics, computational eng, fluids, materials, instrumentation, mechanics, audio
    "cs.RO", "cs.CE", "physics.flu-dyn", "cond-mat.mtrl-sci", "physics.ins-det", "physics.class-ph",
    "cs.SD", "eess.AS", "cs.MM", "cs.HC",
    # foundational physics kept for breadth
    "quant-ph", "gr-qc", "physics.optics", "nlin.CD",
]


def fetch_cat(cat, n, start=0):
    url = (f"http://export.arxiv.org/api/query?search_query=cat:{cat}"
           f"&start={start}&max_results={n}&sortBy=submittedDate&sortOrder=descending")
    for attempt in range(3):                                  # polite back-off on 429/timeout
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                xml = r.read().decode("utf-8", errors="replace")
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(20 * (attempt + 1)); continue
            print(f"[arxiv] {cat} fetch err: {e}", flush=True); return []
    out = []
    for entry in re.findall(r"<entry>(.*?)</entry>", xml, re.S):
        idm = re.search(r"<id>.*?abs/([^<]+)</id>", entry)
        tm = re.search(r"<title>(.*?)</title>", entry, re.S)
        sm = re.search(r"<summary>(.*?)</summary>", entry, re.S)
        if idm and tm and sm:
            aid = re.sub(r"[^\w.]", "_", idm.group(1))
            title = re.sub(r"\s+", " ", tm.group(1)).strip()
            summ = re.sub(r"\s+", " ", sm.group(1)).strip()
            out.append((aid, f"{title}. {summ}"))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cat", type=int, default=40)
    args = ap.parse_args()
    LIB.mkdir(exist_ok=True)
    existing = {f.name for f in LIB.glob("arxiv-*.txt")}
    total = 0
    for cat in CATS:
        got = fetch_cat(cat, args.per_cat)
        field = cat.split(".")[0]   # quant, hep, astro, cs, q, math, physics, cond, gr, nlin
        new = 0
        for aid, text in got:
            name = f"arxiv-{field}__{aid}.txt"
            if name in existing or len(text) < 200:
                continue
            (LIB / name).write_text(text); existing.add(name); new += 1; total += 1
        print(f"[arxiv] {cat:18} +{new} papers", flush=True)
        time.sleep(12)   # arXiv politeness (strict rate limit)
    print(f"\n[arxiv] fetched {total} modern papers across {len(CATS)} categories into library/", flush=True)


if __name__ == "__main__":
    main()

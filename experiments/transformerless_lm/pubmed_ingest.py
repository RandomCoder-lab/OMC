#!/usr/bin/env python3
"""pubmed_ingest.py — biomedical literature into the knowledge web (microbiology, genetics, neuroscience…).

NCBI E-utilities: esearch (recent PMIDs for a term) -> efetch (abstracts as text). Saves one file per
term-batch to library/<term>__pubmed<n>.txt (field label = the term); the disk grower splits + folds it.
Polite (NCBI guideline ≤3 req/s without key; tool+email params, delay), resumable.
"""
import sys, re, time, json, urllib.request, urllib.parse
from pathlib import Path

HERE = Path(__file__).parent
LIB = HERE / "library"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
UA = "OMC-knowledge-web/1.0"
TERMS = ["microbiology", "immunology", "genetics", "neuroscience", "virology", "biochemistry",
         "molecular+biology", "epidemiology", "pharmacology", "physiology", "oncology", "ecology"]


def get(url):
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception:
            time.sleep(8 * (attempt + 1))
    return ""


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-term", type=int, default=60)
    args = ap.parse_args()
    LIB.mkdir(exist_ok=True)
    total = 0
    for term in TERMS:
        field = term.split("+")[0]
        es = get(f"{EUTILS}/esearch.fcgi?db=pubmed&term={term}&retmax={args.per_term}"
                 f"&sort=date&retmode=json&tool=omcweb&email=research@local")
        try:
            ids = json.loads(es).get("esearchresult", {}).get("idlist", [])
        except Exception:
            ids = []
        if not ids:
            print(f"[pubmed] {field:14} no ids", flush=True); time.sleep(1); continue
        time.sleep(0.5)
        txt = get(f"{EUTILS}/efetch.fcgi?db=pubmed&id={','.join(ids)}"
                  f"&rettype=abstract&retmode=text&tool=omcweb&email=research@local")
        if len(txt) > 500:
            name = LIB / f"{field}__pubmed_{ids[0]}.txt"
            if not name.exists():
                name.write_text(txt); total += 1
            print(f"[pubmed] {field:14} +{len(ids)} abstracts ({len(txt)//1000}KB)", flush=True)
        time.sleep(1.5)
    print(f"\n[pubmed] saved {total} term-batches into library/", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""apicache.py — disk-cached HTTP GET-JSON for the bible-api fetches. bible-api latency (~6s/call) is the
bottleneck for adding languages, and every language re-fetches the SAME English pivot chapters. Cache each
URL's JSON response (and 404s, negatively) under library/_apicache/ so re-runs and shared pivot fetches are
instant. Agnostic: just memoizes network responses; the data is unchanged.
"""
import urllib.request, urllib.error, json, hashlib, time
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "library" / "_apicache"
UA = {"User-Agent": "OMC-knowledge-web/1.0 (research; local)"}


def get(url, retries=3):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = CACHE_DIR / (hashlib.md5(url.encode()).hexdigest() + ".json")
    if f.exists():
        try:
            return json.loads(f.read_text())          # may be a dict, or null (cached 404)
        except Exception:
            pass
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            data = json.loads(urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace"))
            f.write_text(json.dumps(data))
            return data
        except urllib.error.HTTPError as he:
            if he.code == 404:
                f.write_text("null")                  # negative-cache missing books → never refetch
                return None
            time.sleep(3 * (attempt + 1))
        except Exception:
            time.sleep(3 * (attempt + 1))
    return None

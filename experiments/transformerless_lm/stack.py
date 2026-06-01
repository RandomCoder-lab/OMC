#!/usr/bin/env python3
"""stack.py — incremental "stack then integrate": grow the saved knowledge web WITHOUT a full rebuild.

Loads the .kwebcache checkpoint, finds library/ texts not yet incorporated (tracked in
.kwebcache/stacked.json), STACKS each via KnowledgeWeb.add_field (append passages + co-occurrence edges,
O(new text) — no embedding retrain), and re-saves. Growth cost is now O(new text), not O(total corpus).

The embedding (6,000 fixed dict-concept vectors) is NOT retouched here — those vectors stay valid as
fields stack; a full `kweb.py --rebuild` is only needed occasionally to refresh cross-verification
(INTEGRATE). Run stack.py every cycle (cheap); rebuild rarely.
"""
import sys, json, time, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from kweb import KnowledgeWeb

HERE = Path(__file__).parent
CACHE = HERE / ".kwebcache"
LIB = HERE / "library"
STACKED = CACHE / "stacked.json"


def main():
    if not (CACHE / "web.json").exists():
        print("[stack] no .kwebcache — run kweb.py --rebuild once first"); return
    if shutil.disk_usage(HERE).free < 1_500_000_000:
        print(f"[stack] DISK GUARD: {shutil.disk_usage(HERE).free/1e9:.1f}GB free — skipping"); return
    t0 = time.time()
    web = KnowledgeWeb.load(CACHE)
    stacked = set(json.loads(STACKED.read_text())) if STACKED.exists() else set()
    # if no manifest yet (e.g. after a full rebuild), seed it with the current dom coverage's files:
    # treat every library file as already-in if the web was just fully built — caller writes stacked.json
    # after a rebuild. Here we only add files NOT in the manifest.
    new_files = [f for f in sorted(LIB.glob("*.txt")) if f.name not in stacked]
    if not new_files:
        print(f"[stack] nothing new (web has {len(web.passages):,} passages, "
              f"{sum(len(v) for v in web.adj.values()):,} edges)"); return
    total_added = 0
    for f in new_files:
        label = f.stem.split("__")[0]
        try:
            added = web.add_field(label, f.read_text(errors="replace"))
            total_added += added; stacked.add(f.name)
            print(f"[stack] +{label:14} {f.name:30} +{added:,} edges", flush=True)
        except Exception as e:
            print(f"[stack] FAIL {f.name}: {e}", flush=True)
    web.save(CACHE)
    STACKED.write_text(json.dumps(sorted(stacked)))
    print(f"\n[stack] STACKED {len(new_files)} texts (+{total_added:,} edges) in {time.time()-t0:.0f}s "
          f"— NO retrain. web now {len(web.passages):,} passages, "
          f"{sum(len(v) for v in web.adj.values()):,} edges, {len(set(web.dom))} fields. saved.", flush=True)


if __name__ == "__main__":
    main()

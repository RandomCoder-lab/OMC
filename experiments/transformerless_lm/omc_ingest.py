#!/usr/bin/env python3
"""omc_ingest.py — ingest OMC's OWN source + docs into the web, so it knows itself: the VM, the substrate,
the addressing primitives, the protocol — and even these very tools — answerable on demand with citations to
the actual files. Local (no download). Carefully filtered: source + docs only; skips build artifacts
(target/), weights (*.pt), VCS, caches, and the web's OWN data (library/, knowledge.db, .kwebcache) so it
doesn't eat itself. Writes to library/ for the pending fold + node_expand to pick up.

  python omc_ingest.py
"""
import hashlib
from pathlib import Path

ROOT = Path("/home/thearchitect/OMC")
LIB = Path(__file__).parent / "library"
EXT = {".md": "omc-docs", ".rs": "omc-rust", ".py": "omc-python", ".omc": "omc-lang", ".toml": "omc-docs"}
SKIP_DIR = {"target", ".git", "node_modules", "__pycache__", ".auto-claude", ".kwebcache",
            "library", "_sedumps", "_corpus", "_apicache", "artifacts", "src.deleted.bak", ".cargo",
            "build", "dist", ".pytest_cache", "trails"}
SKIP_NAME = {"Cargo.lock", "knowledge.db"}
CAP = 220_000


def main():
    LIB.mkdir(exist_ok=True)
    existing = {f.name for f in LIB.glob("omc-*__*.txt")}
    got = {}
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIR for part in p.parts):
            continue
        if p.name in SKIP_NAME or p.name.startswith("knowledge.db"):
            continue
        label = EXT.get(p.suffix.lower())
        if not label:
            continue
        try:
            if p.stat().st_size > CAP or p.stat().st_size < 40:
                continue
            txt = p.read_text(errors="replace")
        except Exception:
            continue
        rel = str(p.relative_to(ROOT))
        h = hashlib.md5(rel.encode()).hexdigest()[:8]
        name = f"{label}__omc_{h}.txt"
        if name in existing:
            continue
        # prepend the path so the web knows where the knowledge lives (provenance in the text too)
        (LIB / name).write_text(f"[OMC file: {rel}]\n{txt}")
        existing.add(name)
        got[label] = got.get(label, 0) + 1
    total = sum(got.values())
    print(f"[omc] ingested {total} OMC source/doc files into library/: {got}", flush=True)


if __name__ == "__main__":
    main()

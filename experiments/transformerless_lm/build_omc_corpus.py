"""Collect all .omc files from /home/thearchitect/OMC/ and concatenate into omc_corpus.txt.

Excludes target/ and __pycache__ directories. Skips files that fail UTF-8 decode.
"""

import os
from pathlib import Path

ROOT = Path("/home/thearchitect/OMC")
OUT  = Path("/home/thearchitect/OMC/experiments/transformerless_lm/omc_corpus.txt")

EXCLUDE_DIRS = {"target", "__pycache__"}

def collect_omc_files(root: Path):
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded directories in-place
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fname in filenames:
            if fname.endswith(".omc"):
                results.append(Path(dirpath) / fname)
    results.sort()
    return results

def main():
    files = collect_omc_files(ROOT)
    print(f"Found {len(files)} .omc files")

    skipped = 0
    total_chars = 0
    chunks = []

    for i, fpath in enumerate(files):
        try:
            text = fpath.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, OSError):
            skipped += 1
            continue
        chunks.append(text)
        total_chars += len(text)
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(files)}] processed so far ...")

    print(f"Skipped {skipped} files (non-UTF-8)")
    print(f"Total chars from .omc files: {total_chars:,}")

    corpus = "\n".join(chunks)
    print(f"Corpus total chars (with separators): {len(corpus):,}")

    OUT.write_text(corpus, encoding="utf-8")
    print(f"Written to {OUT}")

if __name__ == "__main__":
    main()

"""Stage 2 of the substrate-tokenizer pipeline.

Read the canonical-hash index from stage 1; pick the top-N
hashes (most-frequently occurring); emit a token table that
maps reserved token IDs to canonical hashes.

The output table assigns token IDs in a range that most BPE
tokenizers reserve for fine-tune extensions:
  - Llama / Mistral: [128000..128255] (256 reserved special tokens)
  - GPT-2: [50257..50337] (similar range)
  - StarCoder: configurable

The mapping is:
  token_id = base_token_id + index_in_top_N

so the first popular canonical hash gets `base + 0`, the second gets
`base + 1`, etc.

Usage:
    python3 build_vocab.py --top N [--base 128000] < canonical_hash_index.jsonl > hash_token_table.json
"""

from __future__ import annotations

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=64,
                        help="Number of canonical hashes to assign reserved tokens (default 64)")
    parser.add_argument("--base", type=int, default=128000,
                        help="First reserved token ID (default 128000 for Llama-style)")
    parser.add_argument("--min-count", type=int, default=2,
                        help="Skip hashes with fewer than this many occurrences (default 2)")
    args = parser.parse_args()

    entries: list[dict] = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("count", 0) < args.min_count:
            continue
        entries.append(rec)
    # Index is already sorted by count desc in corpus_collect.py output,
    # but defensively re-sort.
    entries.sort(key=lambda r: (-r["count"], r["canonical_hash"]))
    top = entries[: args.top]

    table = {
        "base_token_id": args.base,
        "vocab_size": len(top),
        "source": "substrate_canonical_hashes",
        "tokens": [
            {
                "token_id": args.base + i,
                "canonical_hash": rec["canonical_hash"],
                "fn_name": rec.get("fn_name", ""),
                "count": rec.get("count", 0),
                "size_bytes": rec.get("size_bytes", 0),
                "origin_file": rec.get("first_origin_file", ""),
            }
            for i, rec in enumerate(top)
        ],
    }

    json.dump(table, sys.stdout, indent=2)
    sys.stdout.write("\n")

    total_count_covered = sum(rec["count"] for rec in top)
    total_count_all = sum(rec["count"] for rec in entries)
    total_bytes_covered = sum(rec["size_bytes"] * rec["count"] for rec in top)
    print(
        f"build_vocab: assigned {len(top)} tokens "
        f"[{args.base}..{args.base + len(top) - 1}] "
        f"covering {total_count_covered}/{total_count_all} fn occurrences "
        f"({100 * total_count_covered / max(total_count_all, 1):.1f}%, "
        f"{total_bytes_covered:,} bytes of repeated source)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

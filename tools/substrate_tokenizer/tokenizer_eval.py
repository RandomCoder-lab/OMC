"""Stage 4 of the substrate-tokenizer pipeline.

Measure the actual context-compression win of substrate-aware
tokenization on a given input file. Runs without a trained model
— this is the BEFORE / projected-AFTER comparison that tells you
whether the fine-tune is worth the GPU spend.

For an input text:
  1. Count tokens with a naive BPE tokenizer (tiktoken `cl100k_base`
     as the proxy for "what a typical modern LLM sees")
  2. Substitute any OMC fn-body that matches a canonical hash in the
     vocab table with the single-token `<omc:N>` reference
  3. Re-tokenize and count
  4. Report compression ratio

Usage:
    python3 tokenizer_eval.py --table hash_token_table.json INPUT.txt

If tiktoken isn't installed, falls back to a character-count
approximation (~4 chars / token) for a rough projection.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def naive_token_count(text: str) -> int:
    """Best-available token count. Prefer tiktoken cl100k_base."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        # Rough char/token ratio for BPE on English / code.
        return max(1, len(text) // 4)


def extract_top_level_fns(src: str):
    """Pure-Python port of the canonical extractor."""
    n = len(src)
    i = 0
    while i < n:
        if src[i] == "#":
            while i < n and src[i] != "\n":
                i += 1
            continue
        if src[i] in ('"', "'"):
            q = src[i]
            i += 1
            while i < n and src[i] != q:
                if src[i] == "\\" and i + 1 < n:
                    i += 2
                else:
                    i += 1
            if i < n:
                i += 1
            continue
        at_boundary = i == 0 or src[i - 1].isspace()
        if at_boundary and i + 3 < n and src[i : i + 3] == "fn ":
            fn_start = i
            j = i
            while j < n and src[j] != "{":
                j += 1
            if j >= n:
                break
            depth = 0
            k = j
            while k < n:
                c = src[k]
                if c == "#":
                    while k < n and src[k] != "\n":
                        k += 1
                    continue
                if c in ('"', "'"):
                    q = c
                    k += 1
                    while k < n and src[k] != q:
                        if src[k] == "\\" and k + 1 < n:
                            k += 2
                        else:
                            k += 1
                    if k < n:
                        k += 1
                    continue
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        k += 1
                        break
                k += 1
            if depth == 0 and k > fn_start:
                yield src[fn_start:k]
                i = k
                continue
        i += 1


def shell_hash(fn_src: str, kernel_bin: str, tmp_store: str) -> str | None:
    """Compute canonical hash for one fn via omc-kernel `put --kind omc_fn`.
    Returns hex hash or None on canonicalization failure.

    Uses the kernel (not omc_canonical_hash via omnimcode-standalone) so
    the hashing path is IDENTICAL to what corpus_collect.py produced —
    same binary, same canonicalizer, same fnv1a call. Guarantees hashes
    line up between stage 1 (collect) and stage 4 (eval).
    """
    import os
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".omc", delete=False, dir=tempfile.gettempdir()
    ) as f:
        f.write(fn_src)
        src_path = f.name
    try:
        r = subprocess.run(
            [kernel_bin, "put", src_path, "--kind", "omc_fn"],
            capture_output=True, text=True, check=False,
            env={**os.environ, "OMC_KERNEL_ROOT": tmp_store},
        )
        if r.returncode != 0:
            return None
        return r.stdout.strip()
    finally:
        try:
            os.unlink(src_path)
        except OSError:
            pass


def find_kernel_binary() -> str | None:
    import os, shutil
    explicit = os.environ.get("OMC_KERNEL_BIN")
    if explicit and Path(explicit).is_file():
        return explicit
    found = shutil.which("omc-kernel")
    if found:
        return found
    cwd = Path.cwd() / "target" / "release" / "omc-kernel"
    return str(cwd) if cwd.is_file() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True, type=Path,
                        help="hash_token_table.json from build_vocab.py")
    parser.add_argument("input", type=Path,
                        help="Input file to measure")
    args = parser.parse_args()

    table = json.loads(args.table.read_text())
    # hash -> token_id
    hash_to_tok = {tok["canonical_hash"]: tok["token_id"] for tok in table["tokens"]}

    src = args.input.read_text(encoding="utf-8", errors="replace")
    print(f"input: {args.input}  ({len(src):,} chars)", file=sys.stderr)

    naive_tokens = naive_token_count(src)
    print(f"  naive BPE tokens: {naive_tokens:,}")

    # Now rewrite fn bodies to <omc:N> if they match the vocab table.
    omc_bin = find_kernel_binary()
    if not omc_bin:
        print("  (substrate rewriting needs omc-kernel; skipping)", file=sys.stderr)
        sys.exit(0)

    rewritten = src
    n_replaced = 0
    n_total = 0
    bytes_replaced = 0
    import tempfile, shutil as _shutil
    tmp_store = tempfile.mkdtemp(prefix="omc_tokenizer_eval_")
    try:
        # Iterate fns and replace any that match the vocab.
        for fn_src in extract_top_level_fns(src):
            n_total += 1
            h = shell_hash(fn_src, omc_bin, tmp_store)
            if h and h in hash_to_tok:
                tok_id = hash_to_tok[h]
                replacement = f"<omc:{tok_id}>"
                rewritten = rewritten.replace(fn_src, replacement, 1)
                n_replaced += 1
                bytes_replaced += len(fn_src)
    finally:
        try:
            _shutil.rmtree(tmp_store)
        except OSError:
            pass

    substrate_tokens = naive_token_count(rewritten)
    ratio = naive_tokens / max(substrate_tokens, 1)

    print(f"  fns in input:               {n_total}")
    print(f"  fns matching vocab table:   {n_replaced}")
    print(f"  bytes replaced by tokens:   {bytes_replaced:,}")
    print(f"  substrate-tokens:           {substrate_tokens:,}")
    print(f"  compression ratio:          {ratio:.2f}x")
    if ratio > 1.0:
        savings = naive_tokens - substrate_tokens
        print(f"  → {savings:,} tokens saved ({100*savings/naive_tokens:.1f}%)")
    else:
        print("  → no compression (no vocab matches)")


if __name__ == "__main__":
    main()

"""omc_tools.py — Python interface to OMC (Omnimcode) for use as agent tools.

Three entry points:

    omc_eval(code)             — execute OMC source, return stdout as str
    omc_predict(context, n)   — predict next n tokens via MCP omc_predict
    omc_search(query, top_k)  — search the dual address index by φ-fingerprint

Implementation strategy
-----------------------
omc_eval
    Primary:   subprocess to omnimcode-mcp, which runs the program via
               an in-process Interpreter and keeps state across calls in
               one session.  MCP JSON-RPC 2.0 over stdin/stdout.
    Fallback:  subprocess to omnimcode-standalone (write code to a temp
               file, run it, capture stdout).  Stateless.
    Stub:      if neither binary is found, returns a descriptive error.

omc_predict
    Calls MCP omc_predict with the paths= corpus (omc_codebase.txt in
    the experiment dir, or any .omc glob the caller provides).  Returns
    a formatted string of fn_name + source lines.

omc_search
    Pure Python.  Loads omc_dual_index.pt lazily and uses the
    substrate_fingerprint + retrieve_dual logic from corpus_address_index.py.
    No binary required.

Binary locations (searched in order)
-------------------------------------
    1. OMC_MCP_BIN env var
    2. /home/thearchitect/OMC/target/release/omnimcode-mcp
    3. `omnimcode-mcp` on PATH

    1. OMC_STANDALONE_BIN env var
    2. /home/thearchitect/OMC/target/release/omnimcode-standalone
    3. `omnimcode-standalone` on PATH
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

# ── Binary discovery ──────────────────────────────────────────────────────────

_REPO_RELEASE = Path("/home/thearchitect/OMC/target/release")

def _find_bin(env_key: str, name: str) -> Optional[str]:
    explicit = os.environ.get(env_key)
    if explicit and Path(explicit).is_file():
        return explicit
    repo_path = _REPO_RELEASE / name
    if repo_path.is_file():
        return str(repo_path)
    return shutil.which(name)

_MCP_BIN        = _find_bin("OMC_MCP_BIN",        "omnimcode-mcp")
_STANDALONE_BIN = _find_bin("OMC_STANDALONE_BIN", "omnimcode-standalone")

# Default corpus for predict — small hand-curated OMC functions file
_DEFAULT_CORPUS = str(
    Path(__file__).parent / "omc_codebase.txt"
)
# Default dual-scale address index for search
_DEFAULT_DUAL_INDEX = str(
    Path(__file__).parent / "omc_dual_index.pt"
)

# ── MCP JSON-RPC helper ───────────────────────────────────────────────────────

def _mcp_call(tool: str, args: dict, timeout: int = 30) -> str:
    """Send one tools/call to the MCP binary and return the text result.

    Raises RuntimeError if the MCP binary is not found or the call fails.
    """
    if not _MCP_BIN:
        raise RuntimeError(
            "omnimcode-mcp binary not found. "
            f"Set OMC_MCP_BIN or build the binary at {_REPO_RELEASE}/omnimcode-mcp"
        )

    init_msg = json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "omc_tools.py", "version": "1.0"},
        },
    })
    call_msg = json.dumps({
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": tool, "arguments": args},
    })
    stdin_data = (init_msg + "\n" + call_msg + "\n").encode()

    proc = subprocess.run(
        [_MCP_BIN],
        input=stdin_data,
        capture_output=True,
        timeout=timeout,
    )
    # Parse every line as JSON-RPC; find id=2 response
    for raw in proc.stdout.decode(errors="replace").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if msg.get("id") != 2:
            continue
        if "error" in msg:
            raise RuntimeError(f"MCP error: {msg['error']}")
        # result.content[0].text is the payload
        content = msg.get("result", {}).get("content", [])
        if content:
            return content[0].get("text", "")
        return ""
    raise RuntimeError(
        f"No id=2 response from MCP binary.\n"
        f"stderr: {proc.stderr.decode(errors='replace')[:500]}"
    )


# ── omc_eval ──────────────────────────────────────────────────────────────────

def omc_eval(code: str, timeout: int = 30) -> str:
    """Execute OMC source code and return stdout as a string.

    Tries (in order):
      1. omnimcode-mcp  (stateful in-process interpreter, fastest)
      2. omnimcode-standalone  (write to temp file, stateless)
      3. Stub error message
    """
    # Strategy 1: MCP binary (preferred — stateful interpreter, no tempfile)
    if _MCP_BIN:
        try:
            return _mcp_call("omc_eval", {"code": code}, timeout=timeout)
        except Exception as exc:
            # Fall through to standalone
            _mcp_err = str(exc)
    else:
        _mcp_err = f"MCP binary not found at {_REPO_RELEASE}/omnimcode-mcp"

    # Strategy 2: standalone binary (stateless, writes a temp file)
    if _STANDALONE_BIN:
        with tempfile.NamedTemporaryFile(
            suffix=".omc", mode="w", delete=False
        ) as f:
            f.write(code)
            tmppath = f.name
        try:
            proc = subprocess.run(
                [_STANDALONE_BIN, tmppath],
                capture_output=True,
                timeout=timeout,
            )
            stdout = proc.stdout.decode(errors="replace")
            stderr = proc.stderr.decode(errors="replace")
            if proc.returncode != 0:
                return (stdout + stderr).strip() or f"[exit {proc.returncode}]"
            return stdout
        except subprocess.TimeoutExpired:
            return "[omc_eval timed out]"
        finally:
            os.unlink(tmppath)

    # Strategy 3: stub
    return (
        f"OMC eval not available.\n"
        f"  MCP binary: {_mcp_err}\n"
        f"  Standalone binary: not found at {_REPO_RELEASE}/omnimcode-standalone\n"
        f"  To fix: build with `cargo build --release` in /home/thearchitect/OMC/"
    )


# ── omc_predict ───────────────────────────────────────────────────────────────

def omc_predict(
    context: str,
    n: int = 10,
    corpus_paths: Optional[list[str]] = None,
    timeout: int = 60,
) -> str:
    """Predict the next n function completions after context using the OMC predictor.

    Uses the MCP omc_predict tool, which searches the corpus for functions
    whose symbol stream best matches the given prefix.

    Args:
        context:       OMC code prefix (function prefix, partial statement, etc.)
        n:             Number of suggestions to return (clamped to 1–50 by MCP).
        corpus_paths:  List of .omc / corpus text files to search.
                       Defaults to [omc_codebase.txt in this experiment dir].
        timeout:       Subprocess timeout in seconds (large corpora are slow).

    Returns:
        A formatted string with each suggestion's fn_name and source, or an
        error message if the MCP binary is unavailable.
    """
    paths = corpus_paths or [_DEFAULT_CORPUS]
    # Validate paths exist — give a useful error rather than a cryptic MCP one
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        return f"[omc_predict] corpus path(s) not found: {missing}"

    try:
        raw = _mcp_call(
            "omc_predict",
            {"paths": paths, "prefix": context, "top_k": n, "format": "full"},
            timeout=timeout,
        )
    except RuntimeError as exc:
        return f"[omc_predict] {exc}"

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw  # return as-is if it's already plain text

    suggestions = payload.get("suggestions", [])
    if not suggestions:
        return (
            f"[omc_predict] No suggestions for prefix {context!r} "
            f"(corpus_size={payload.get('corpus_size', '?')})"
        )

    lines = [
        f"omc_predict: {len(suggestions)} suggestion(s) "
        f"(corpus_size={payload.get('corpus_size', '?')})\n"
    ]
    for i, s in enumerate(suggestions, 1):
        fn_name = s.get("fn_name", "?")
        source  = s.get("source", "").strip()
        dist    = s.get("substrate_distance", "?")
        lines.append(f"--- [{i}] {fn_name}  (substrate_distance={dist})")
        lines.append(source)
        lines.append("")

    return "\n".join(lines)


# ── omc_search ────────────────────────────────────────────────────────────────

# Module-level cache so we don't reload the 648 k-entry index on every call.
_dual_index_cache: Optional[dict] = None
_dual_index_path_cache: Optional[str] = None

def _load_dual_index(path: str) -> dict:
    global _dual_index_cache, _dual_index_path_cache
    if _dual_index_cache is not None and _dual_index_path_cache == path:
        return _dual_index_cache
    try:
        import torch  # type: ignore
    except ImportError:
        raise ImportError("torch is required for omc_search. Install with: pip install torch")
    print(f"[omc_search] loading dual index from {path} …", file=sys.stderr)
    _dual_index_cache = torch.load(path, weights_only=False)
    _dual_index_path_cache = path
    print(f"[omc_search] loaded {_dual_index_cache['meta']['n_windows']:,} windows", file=sys.stderr)
    return _dual_index_cache


def omc_search(
    query: str,
    top_k: int = 3,
    index_path: Optional[str] = None,
) -> list[str]:
    """Search the OMC codebase for text windows similar to the query.

    Uses the dual-scale Fibonacci-harmonic address index (omc_dual_index.pt).
    The query string is fingerprinted with the same φ-hash used at index time
    and then retrieved by cosine similarity within the matching dodecahedral face.

    No binary is required — this is pure Python + torch.

    Args:
        query:       Any string (code fragment, natural language, keyword).
        top_k:       Number of results to return.
        index_path:  Path to the .pt dual index file.
                     Defaults to omc_dual_index.pt in this experiment dir.

    Returns:
        List of matching text windows (str), ranked by similarity (best first).
        Returns a list with a single error string if the index cannot be loaded.
    """
    path = index_path or _DEFAULT_DUAL_INDEX
    if not Path(path).exists():
        return [
            f"[omc_search] index not found at {path}. "
            "Run the indexer first (corpus_address_index.py)."
        ]

    try:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore
    except ImportError:
        return ["[omc_search] torch not installed. pip install torch"]

    # Ensure the experiment dir is importable for the helper modules
    _exp_dir = str(Path(__file__).parent)
    if _exp_dir not in sys.path:
        sys.path.insert(0, _exp_dir)

    try:
        from corpus_address_index import substrate_fingerprint, retrieve_dual  # type: ignore
    except ImportError as exc:
        return [f"[omc_search] cannot import corpus_address_index: {exc}"]

    try:
        index = _load_dual_index(path)
    except Exception as exc:
        return [f"[omc_search] failed to load index: {exc}"]

    scales = index["meta"]["scales"]   # e.g. [13, 89]
    d_model = index["meta"]["d_model"]

    # Fingerprint the query at both scales
    # substrate_fingerprint uses a fixed-length window of the text; we use the
    # full query string (truncated / extended naturally by the LCG over ordinals).
    fp_short = substrate_fingerprint(query[:scales[0]], d_model)
    fp_long  = substrate_fingerprint(query[:scales[1]], d_model)

    results = retrieve_dual(fp_short, fp_long, index, top_k=top_k)

    if not results:
        return [f"[omc_search] no results found for query {query!r}"]

    return [r["text"] for r in results]


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import textwrap

    print("=== omc_eval ===")
    out = omc_eval(textwrap.dedent("""\
        fn fib(n) {
            if n < 2 { return n; }
            return fib(n - 1) + fib(n - 2);
        }
        print(fib(10))
    """))
    print(repr(out))

    print("\n=== omc_predict (prefix='fn fib') ===")
    pred = omc_predict("fn fib", n=3)
    print(pred)

    print("\n=== omc_search (query='fibonacci') ===")
    hits = omc_search("fibonacci", top_k=3)
    for i, h in enumerate(hits, 1):
        print(f"[{i}] {h[:120]!r}")

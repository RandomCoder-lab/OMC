"""omc-substrate MCP server — expose the OMC kernel as MCP tools.

Lets any MCP-aware LLM (Claude, Cursor, Cline, etc.) use the
canonical-hash content-addressed store as a memory/compression
layer. No retraining required — the LLM just calls these tools.

Tools exposed:

    omc_store(content, kind="prose") -> hex_hash
        Store arbitrary content addressed by canonical hash.
        kind ∈ {omc_fn, json, prose, blob}.

    omc_lookup(hex_hash) -> content | None
        Retrieve stored content by canonical hash.

    omc_canonicalize(content, kind="prose") -> {hash, canonical}
        Compute the canonical hash WITHOUT storing. Useful for
        client-side dedup checks.

    omc_stat(hex_hash) -> metadata dict
        Return the sidecar metadata (kind, attractor, distance,
        bytes, origin_file) for a stored entry.

    omc_list() -> [{hash, fn_name, bytes}, ...]
        Enumerate all stored entries.

    omc_compress(content, every_n=3) -> codec_payload
        Apply the substrate codec (sampled-token compression).
        For OMC code; for prose use omc_store + return hex_hash
        as the reference.

The server shells out to the `omc-kernel` Rust binary so the
backing store is shared with any other process using it (CLI
commands, other agents, etc.).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP


def find_kernel_binary() -> str | None:
    """Locate the omc-kernel binary. Search:
      1. OMC_KERNEL_BIN env (explicit override)
      2. PATH
      3. ./target/release/omc-kernel (when run from repo root)
    """
    explicit = os.environ.get("OMC_KERNEL_BIN")
    if explicit and Path(explicit).is_file():
        return explicit
    found = shutil.which("omc-kernel")
    if found:
        return found
    cwd = Path.cwd() / "target" / "release" / "omc-kernel"
    if cwd.is_file():
        return str(cwd)
    return None


KERNEL = find_kernel_binary()
if not KERNEL:
    print(
        "omc-substrate MCP server: omc-kernel binary not found. "
        "Set OMC_KERNEL_BIN or run from a directory with target/release/omc-kernel.",
        file=sys.stderr,
    )
    sys.exit(1)


def _kernel(args: list[str], stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run the omc-kernel binary with given args. Capture stdout + stderr."""
    return subprocess.run(
        [KERNEL, *args],
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )


mcp = FastMCP("omc-substrate")


# ----- Pure implementations (callable directly for tests) -----


def _impl_store(content: str, kind: str = "prose") -> str:
    """Store arbitrary content in the substrate-keyed kernel.
    Returns the canonical hex hash that addresses the stored entry.

    kind selects the canonicalizer:
      omc_fn  — alpha-rename-invariant OMC canonical form
      json    — recursive key-sort
      prose   — raw bytes (default)
      blob    — alias for prose
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".tmp", delete=False, dir=tempfile.gettempdir()
    ) as f:
        f.write(content)
        tmp_path = f.name
    try:
        r = _kernel(["put", tmp_path, "--kind", kind])
        if r.returncode != 0:
            raise RuntimeError(
                f"omc-kernel put failed (rc={r.returncode}): {r.stderr.strip()}"
            )
        # Kernel writes the hex hash to stdout on success.
        return r.stdout.strip()
    finally:
        os.unlink(tmp_path)


def _impl_lookup(hex_hash: str) -> str | None:
    """Retrieve stored content by canonical hex hash.
    Returns the content string, or None if no entry exists.
    """
    r = _kernel(["fetch", hex_hash])
    if r.returncode != 0:
        return None
    return r.stdout


def _impl_stat(hex_hash: str) -> dict[str, Any]:
    """Return sidecar metadata for a stored entry: kind, attractor,
    attractor_distance, source_bytes, canonical_bytes, origin_file.
    """
    r = _kernel(["stat", hex_hash])
    if r.returncode != 0:
        return {"error": r.stderr.strip(), "found": False}
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError as e:
        return {"error": f"could not parse stat output: {e}", "raw": r.stdout}


def _impl_list() -> list[dict[str, Any]]:
    """List all stored entries: their canonical hash, fn name (or
    first-line summary for non-fn content), and byte size.
    """
    r = _kernel(["ls"])
    if r.returncode != 0:
        return [{"error": r.stderr.strip()}]
    # Parse `omc-kernel ls` output. Format:
    #   N fn(s) in store at /path
    #   canonical-hash        bytes  fn
    #   <hash>                <bytes>  fn <name>
    lines = r.stdout.splitlines()
    out: list[dict[str, Any]] = []
    for ln in lines[2:]:  # skip "N fn(s)..." header + column header
        parts = ln.split(None, 2)
        if len(parts) < 3:
            continue
        hash_hex, bytes_s, rest = parts[0], parts[1], parts[2]
        try:
            n_bytes = int(bytes_s)
        except ValueError:
            continue
        # rest is "fn NAME" — strip the leading "fn ".
        name = rest[3:] if rest.startswith("fn ") else rest
        out.append({"hash": hash_hex, "bytes": n_bytes, "name": name})
    return out


def _impl_canonicalize(content: str, kind: str = "prose") -> dict[str, Any]:
    """Compute the canonical hash WITHOUT storing.
    Useful when a client wants to check 'do I already have this?'
    before paying the store cost. Returns {hash, kind, addressing}.
    """
    # The kernel doesn't have a `hash-only` mode yet, so we cheat: put,
    # then check whether the entry already existed via the stderr line.
    # The hash is the same whether the entry is new or pre-existing.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".tmp", delete=False, dir=tempfile.gettempdir()
    ) as f:
        f.write(content)
        tmp_path = f.name
    try:
        r = _kernel(["put", tmp_path, "--kind", kind])
        hash_hex = r.stdout.strip() if r.returncode == 0 else None
        was_new = "stored" in (r.stderr or "")
        return {
            "hash": hash_hex,
            "kind": kind,
            "was_new": was_new,
            "ok": r.returncode == 0,
        }
    finally:
        os.unlink(tmp_path)


def _impl_compress(content: str, every_n: int = 3) -> dict[str, Any]:
    """Apply the substrate codec (sampled-token compression).
    Returns a dict with the codec payload + canonical hash for
    library-lookup recovery on the receiver side.

    Best for OMC source code; for arbitrary prose, the wire-byte
    win only appears at payloads >~500 B with every_n >= 8.
    """
    # The kernel binary doesn't expose codec_encode directly; for now
    # the cleanest path is to ask the OMC interpreter via stdin. If
    # we hit OMC_KERNEL_BIN's sibling binary, use it.
    omc = (
        shutil.which("omnimcode-standalone")
        or (Path(KERNEL).parent / "omnimcode-standalone").as_posix()
    )
    if not Path(omc).is_file():
        return {
            "error": "omnimcode-standalone binary not found; cannot run codec",
            "hint": "build with `cargo build --release -p omnimcode-cli`",
        }
    program = f"""
fn main() {{
    h content = read_file("{0}");
    h codec = omc_codec_encode(content, {every_n});
    print(json_stringify(codec));
}}
main();
""".strip()
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".tmp", delete=False, dir=tempfile.gettempdir()
    ) as f:
        f.write(content)
        content_tmp = f.name
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".omc", delete=False, dir=tempfile.gettempdir()
    ) as f:
        f.write(program.format(content_tmp))
        prog_tmp = f.name
    try:
        r = subprocess.run(
            [omc, prog_tmp],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PYO3_USE_ABI3_FORWARD_COMPATIBILITY": "1"},
        )
        if r.returncode != 0:
            return {"error": r.stderr.strip(), "rc": r.returncode}
        try:
            return json.loads(r.stdout.strip())
        except json.JSONDecodeError as e:
            return {"error": f"parse failed: {e}", "raw": r.stdout}
    finally:
        for p in (content_tmp, prog_tmp):
            try:
                os.unlink(p)
            except OSError:
                pass


# ----- MCP tool registrations (thin wrappers over _impl_*) -----


@mcp.tool()
def omc_store(content: str, kind: str = "prose") -> str:
    """Store arbitrary content in the substrate-keyed kernel.
    Returns the canonical hex hash that addresses the stored entry.
    kind ∈ {omc_fn, json, prose, blob}.
    """
    return _impl_store(content, kind)


@mcp.tool()
def omc_lookup(hex_hash: str) -> str | None:
    """Retrieve stored content by canonical hex hash. None on miss."""
    return _impl_lookup(hex_hash)


@mcp.tool()
def omc_stat(hex_hash: str) -> dict[str, Any]:
    """Sidecar metadata: kind, attractor, distance, bytes, origin."""
    return _impl_stat(hex_hash)


@mcp.tool()
def omc_list() -> list[dict[str, Any]]:
    """Enumerate all stored entries."""
    return _impl_list()


@mcp.tool()
def omc_canonicalize(content: str, kind: str = "prose") -> dict[str, Any]:
    """Compute the canonical hash without storing — dedup-check."""
    return _impl_canonicalize(content, kind)


@mcp.tool()
def omc_compress(content: str, every_n: int = 3) -> dict[str, Any]:
    """Apply substrate codec for OMC source code."""
    return _impl_compress(content, every_n)


if __name__ == "__main__":
    mcp.run()

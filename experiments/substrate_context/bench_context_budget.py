"""End-to-end LLM context-budget benchmark for v0.4-substrate-context.

Simulates a realistic LLM agent workflow against the OMC MCP server:

  1. Agent queries the corpus to find candidate functions matching a prefix.
  2. Agent picks the best candidate.
  3. Agent fetches the full body to read / adapt.

We compare two strategies:

  - v0.3 baseline ("full"): agent gets full source for every candidate
    on every query. The reasoning-then-fetch loop doesn't exist; the
    agent has to read all candidates' bodies up front.
  - v0.4 compressed ("hash" + on-demand fetch): agent browses cheaply
    (hash format), reasons over substrate metadata, fetches only the
    one or two candidates it commits to using.

Reports byte counts for each strategy across N representative tasks
and the resulting compression ratio.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
MCP = REPO / "target" / "release" / "omnimcode-mcp"

# Representative task prefixes — what an LLM might query in a typical
# OMC authoring session.
TASKS = [
    "fn prom_linear_",
    "fn prom_attention_",
    "fn fibtier_",
    "fn tape_",
    "fn _prom_",
    "fn arr_",
    "fn harmonic_anomaly",
    "fn substrate_search",
    "fn dict_get",
    "fn str_split",
]

# Per task, how many fetches the agent actually makes after browsing
# (the "I picked this one and want to read it" step). v0.4 wins when
# fetches < top_k.
FETCHES_PER_TASK = 1


def rpc_call(method: str, params: dict) -> dict:
    """Send one JSON-RPC call to the MCP server and return the result."""
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": method, "params": params},
    ]
    stdin = "\n".join(json.dumps(r) for r in requests).encode()
    out = subprocess.run([str(MCP)], input=stdin, capture_output=True,
                         cwd=REPO, check=True)
    last = out.stdout.decode().strip().split("\n")[-1]
    return json.loads(last)


def predict(paths: list[str], prefix: str, top_k: int, fmt: str) -> dict:
    """Call omc_predict and return the parsed payload dict."""
    r = rpc_call("tools/call", {
        "name": "omc_predict",
        "arguments": {"paths": paths, "prefix": prefix, "top_k": top_k, "format": fmt},
    })
    return json.loads(r["result"]["content"][0]["text"])


def fetch(paths: list[str], canonical_hash: int) -> dict:
    """Call omc_fetch_by_hash and return the parsed payload dict."""
    r = rpc_call("tools/call", {
        "name": "omc_fetch_by_hash",
        "arguments": {"paths": paths, "canonical_hash": canonical_hash},
    })
    return json.loads(r["result"]["content"][0]["text"])


def bytes_of(payload: dict) -> int:
    """Estimate the LLM context cost of receiving `payload`. Use the
    serialized JSON length — that's exactly what would end up in the
    conversation context window."""
    return len(json.dumps(payload, separators=(",", ":")))


def main():
    if not MCP.exists():
        sys.exit(f"build MCP first: cargo build --release -p omnimcode-mcp\nlooked at {MCP}")
    paths = ["examples/lib"]
    top_k = 5

    rows = []
    baseline_total = 0
    v04_total = 0
    for task in TASKS:
        # Baseline: v0.3 behavior — get everything inline.
        baseline_payload = predict(paths, task, top_k, "full")
        baseline_bytes = bytes_of(baseline_payload)

        # v0.4: browse cheaply, fetch only what you commit to.
        v04_browse = predict(paths, task, top_k, "hash")
        browse_bytes = bytes_of(v04_browse)
        # The fetch step — pretend the agent picks the top suggestion.
        fetch_bytes = 0
        if v04_browse["suggestions"]:
            for s in v04_browse["suggestions"][:FETCHES_PER_TASK]:
                fetch_payload = fetch(paths, s["canonical_hash"])
                fetch_bytes += bytes_of(fetch_payload)
        v04_bytes = browse_bytes + fetch_bytes

        ratio = v04_bytes / baseline_bytes if baseline_bytes else 0.0
        baseline_total += baseline_bytes
        v04_total += v04_bytes
        rows.append((task, baseline_bytes, browse_bytes, fetch_bytes, v04_bytes, ratio))

    print(f"{'task':35} {'v0.3 full':>10} {'v0.4 browse':>12} {'v0.4 fetch':>11} {'v0.4 total':>11} {'ratio':>7}")
    print("-" * 90)
    for (task, full, browse, fetch_b, v04, ratio) in rows:
        print(f"{task:35} {full:>10} {browse:>12} {fetch_b:>11} {v04:>11} {ratio:>6.1%}")
    print("-" * 90)
    overall = v04_total / baseline_total if baseline_total else 0.0
    print(f"{'TOTAL':35} {baseline_total:>10} {'':>12} {'':>11} {v04_total:>11} {overall:>6.1%}")
    print()
    print(f"v0.4 compression vs v0.3 baseline: {1/overall:.2f}x smaller "
          f"({(1-overall)*100:.1f}% reduction)")
    print(f"Strategy: hash-browse + {FETCHES_PER_TASK} fetch per task at top_k={top_k}")

    # Write JSON for the FINDING writeup.
    out = {
        "config": {"top_k": top_k, "fetches_per_task": FETCHES_PER_TASK, "paths": paths},
        "tasks": [
            {"task": t, "baseline_bytes": b, "v04_browse": br,
             "v04_fetch": f, "v04_total": v, "ratio": r}
            for (t, b, br, f, v, r) in rows
        ],
        "totals": {
            "baseline_bytes": baseline_total,
            "v04_bytes": v04_total,
            "ratio": overall,
            "compression_factor": 1 / overall if overall else 0.0,
        },
    }
    json_path = Path(__file__).parent / "results_context_budget.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults written to {json_path}")


if __name__ == "__main__":
    main()

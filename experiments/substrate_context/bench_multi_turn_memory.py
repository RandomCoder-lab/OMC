"""Multi-turn conversation benchmark for v0.5-substrate-memory.

Simulates a 20-turn LLM agent task. Compares three context strategies:

  1. baseline: agent keeps the FULL transcript in context every turn
     (this is the default ChatGPT/Claude conversation pattern)
  2. v0.4 only: agent uses omc_predict format=hash + omc_fetch_by_hash
     for code, but still keeps the full transcript inline
  3. v0.5 full: agent uses memory hashes for prior turns AND
     compressed predict output. Only recalls a turn when reasoning
     needs it (the cited-papers pattern: "as discussed earlier in
     turn 4: <recall>")

Each turn is a realistic mix of:
  - some prose reasoning
  - one omc_predict call against a corpus
  - one chosen fn the agent commits to using

The "recall budget" is how many prior turns the agent revisits per
turn (default 1: agent peeks at the most relevant prior turn).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
MCP = REPO / "target" / "release" / "omnimcode-mcp"


def rpc_call(method: str, params: dict, memory_root: Path) -> dict:
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": method, "params": params},
    ]
    stdin = "\n".join(json.dumps(r) for r in requests).encode()
    out = subprocess.run(
        [str(MCP)], input=stdin, capture_output=True,
        cwd=REPO, env={"OMC_MEMORY_ROOT": str(memory_root), "HOME": str(memory_root)},
        check=True,
    )
    last = out.stdout.decode().strip().split("\n")[-1]
    return json.loads(last)


def predict(paths, prefix, top_k, fmt, memory_root):
    r = rpc_call("tools/call", {
        "name": "omc_predict",
        "arguments": {"paths": paths, "prefix": prefix, "top_k": top_k, "format": fmt},
    }, memory_root)
    return json.loads(r["result"]["content"][0]["text"])


def fetch(paths, h, memory_root):
    r = rpc_call("tools/call", {
        "name": "omc_fetch_by_hash",
        "arguments": {"paths": paths, "canonical_hash": h},
    }, memory_root)
    return json.loads(r["result"]["content"][0]["text"])


def memory_store(text, namespace, memory_root):
    r = rpc_call("tools/call", {
        "name": "omc_memory_store",
        "arguments": {"text": text, "namespace": namespace},
    }, memory_root)
    return json.loads(r["result"]["content"][0]["text"])


def memory_recall(content_hash, namespace, memory_root):
    r = rpc_call("tools/call", {
        "name": "omc_memory_recall",
        "arguments": {"content_hash": content_hash, "namespace": namespace},
    }, memory_root)
    return json.loads(r["result"]["content"][0]["text"])


def jbytes(payload) -> int:
    return len(json.dumps(payload, separators=(",", ":")))


def simulate_turn_reasoning(turn_num: int) -> str:
    """A realistic LLM reasoning blurb per turn. Mix of prose + decisions."""
    return (
        f"Turn {turn_num}: examining the prom_attention_substrate_k_forward fn. "
        f"It composes tape_matmul + prom_substrate_softmax + tape_const "
        f"+ tape_transpose. Need to verify backward gradients still flow "
        f"through Q and V when smod_alpha=1.0. Plan: write a test asserting "
        f"_grad_nonzero(tape_grad(Q)) and same for V after a forward+backward "
        f"pass. The K_const path is correctly severed (no gradient through "
        f"the substrate-K table by design)."
    )


def main():
    if not MCP.exists():
        raise SystemExit(f"build MCP first: cargo build --release -p omnimcode-mcp")

    paths = ["examples/lib"]
    top_k = 10
    n_turns = 20
    recalls_per_turn = 1  # agent peeks at 1 prior turn per current turn
    namespace = "bench_v05"

    with tempfile.TemporaryDirectory(prefix="omc-bench-v05-") as tmpdir:
        memory_root = Path(tmpdir)
        prefixes = [
            "fn prom_linear_", "fn prom_attention_", "fn fibtier_",
            "fn tape_", "fn _prom_", "fn arr_", "fn harmonic_anomaly",
            "fn substrate_search", "fn dict_get", "fn str_split",
            "fn prom_substrate", "fn fibtier_persistent_", "fn _fibtier_",
            "fn prom_softmax", "fn prom_relu", "fn prom_sgd_step",
            "fn prom_adamw_step", "fn prom_one_hot", "fn prom_mse_loss",
            "fn prom_argmax_row",
        ]

        # Each strategy produces:
        #   per_turn_prompt_bytes[t]  — the size of the INPUT PROMPT
        #     the LLM receives on turn t (what an API would charge for)
        #   cumulative_prompt_bytes[t]  — running sum of prompt sizes
        #     across the whole conversation (total tokens processed
        #     across N turns).
        #
        # LLMs charge per turn for the full prompt → that's what we
        # care about. Baseline grows quadratically because each turn
        # re-sends the whole transcript inline. v0.5 grows linearly:
        # each turn sends current content + tiny hash references for
        # prior turns + optional recalls.

        # ============================================================
        # Strategy 1: baseline (full transcript + format=full)
        # ============================================================
        # Prompt on turn t = full conversation history through turn t
        # (every prior turn's content inline) + this turn's content.
        baseline_per_turn_prompt = []
        baseline_per_turn_content = []  # cost of just this turn's NEW content
        for t in range(n_turns):
            reasoning = simulate_turn_reasoning(t)
            pred = predict(paths, prefixes[t % len(prefixes)], top_k, "full", memory_root)
            this_turn = jbytes({"reasoning": reasoning, "predict": pred})
            baseline_per_turn_content.append(this_turn)
            # Prompt at turn t = sum of contents 0..t (transcript carried forward)
            baseline_per_turn_prompt.append(sum(baseline_per_turn_content))

        # ============================================================
        # Strategy 2: v0.4 only (compressed predict, full transcript)
        # ============================================================
        v04_per_turn_prompt = []
        v04_per_turn_content = []
        for t in range(n_turns):
            reasoning = simulate_turn_reasoning(t)
            browse = predict(paths, prefixes[t % len(prefixes)], top_k, "hash", memory_root)
            picked_hash = browse["suggestions"][0]["canonical_hash"] if browse["suggestions"] else None
            fetch_payload = fetch(paths, picked_hash, memory_root) if picked_hash else {}
            this_turn = jbytes({
                "reasoning": reasoning, "browse": browse, "fetched": fetch_payload,
            })
            v04_per_turn_content.append(this_turn)
            v04_per_turn_prompt.append(sum(v04_per_turn_content))

        # ============================================================
        # Strategy 3: v0.5 full (memory hashes + compressed predict)
        # ============================================================
        # Prompt at turn t = this turn's content + hash REFS to every
        # prior turn (cheap, ~20 bytes per hash) + recalled prior-turn
        # bodies (only `recalls_per_turn` of them, not the whole transcript).
        v05_per_turn_prompt = []
        stored_hashes = []
        for t in range(n_turns):
            reasoning = simulate_turn_reasoning(t)
            browse = predict(paths, prefixes[t % len(prefixes)], top_k, "hash", memory_root)
            picked_hash = browse["suggestions"][0]["canonical_hash"] if browse["suggestions"] else None
            fetch_payload = fetch(paths, picked_hash, memory_root) if picked_hash else {}

            # Store this turn's full content for future recall.
            turn_content = json.dumps({
                "reasoning": reasoning, "browse": browse, "fetched": fetch_payload,
            }, separators=(",", ":"))
            store_resp = memory_store(turn_content, namespace, memory_root)
            stored_hashes.append(store_resp["content_hash"])

            # Recall a few prior turns by hash (the agent's "I want to
            # see what I decided in turn N-1" move).
            recalled = []
            recall_targets = stored_hashes[-1 - recalls_per_turn:-1][:recalls_per_turn]
            for rh in recall_targets:
                recalled.append(memory_recall(rh, namespace, memory_root))

            # Prompt at turn t:
            #   - this turn's reasoning + browse + fetched (the work)
            #   - all prior turn HASH REFS (cheap pointers)
            #   - the recalled prior-turn bodies (full text)
            prompt_bytes = jbytes({
                "reasoning": reasoning,
                "browse": browse,
                "fetched": fetch_payload,
                "prior_turn_refs": stored_hashes[:-1],
                "recalled": recalled,
            })
            v05_per_turn_prompt.append(prompt_bytes)

        # Build cumulative (sum of per-turn prompts).
        def cumulative(lst):
            out = []
            s = 0
            for x in lst:
                s += x
                out.append(s)
            return out
        baseline_per_turn_costs = cumulative(baseline_per_turn_prompt)
        v04_per_turn_costs = cumulative(v04_per_turn_prompt)
        v05_per_turn_costs = cumulative(v05_per_turn_prompt)

        # ============================================================
        # Report
        # ============================================================
        print(f"\nv0.5 substrate-memory benchmark — {n_turns} turns, top_k={top_k}, "
              f"recalls_per_turn={recalls_per_turn}")
        print(f"corpus: {paths}\n")
        print(f"{'turn':>4} {'baseline':>10} {'v0.4':>10} {'v0.5':>10} "
              f"{'v0.4/base':>10} {'v0.5/base':>10}")
        print("-" * 64)
        for t in range(n_turns):
            b = baseline_per_turn_costs[t]
            v4 = v04_per_turn_costs[t]
            v5 = v05_per_turn_costs[t]
            print(f"{t+1:>4} {b:>10} {v4:>10} {v5:>10} "
                  f"{v4/b:>9.1%} {v5/b:>9.1%}")
        print("-" * 64)
        final_b = baseline_per_turn_costs[-1]
        final_v4 = v04_per_turn_costs[-1]
        final_v5 = v05_per_turn_costs[-1]
        print(f"{'FINAL':>4} {final_b:>10} {final_v4:>10} {final_v5:>10} "
              f"{final_v4/final_b:>9.1%} {final_v5/final_b:>9.1%}")
        print()
        v4_factor = final_b / final_v4
        v5_factor = final_b / final_v5
        print(f"v0.4 vs baseline:  {v4_factor:.2f}× smaller ({(1-final_v4/final_b)*100:.1f}% reduction)")
        print(f"v0.5 vs baseline:  {v5_factor:.2f}× smaller ({(1-final_v5/final_b)*100:.1f}% reduction)")
        print(f"v0.5 vs v0.4:      {final_v4/final_v5:.2f}× smaller "
              f"({(1-final_v5/final_v4)*100:.1f}% additional reduction)")

        # Write JSON for the writeup.
        result = {
            "config": {
                "n_turns": n_turns, "top_k": top_k,
                "recalls_per_turn": recalls_per_turn, "paths": paths,
            },
            "per_turn": [
                {"turn": t+1, "baseline": baseline_per_turn_costs[t],
                 "v04": v04_per_turn_costs[t], "v05": v05_per_turn_costs[t]}
                for t in range(n_turns)
            ],
            "final": {
                "baseline_bytes": final_b,
                "v04_bytes": final_v4, "v04_factor": v4_factor,
                "v05_bytes": final_v5, "v05_factor": v5_factor,
            },
        }
        out_path = Path(__file__).parent / "results_multi_turn_memory.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()

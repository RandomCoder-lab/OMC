---
name: substrate-synth
description: Generate code that is VALID BY CONSTRUCTION and content-grounded, using the substrate address→generate→VERIFY→accept loop. Use when asked to synthesize or improve code with an execution guarantee, to retrieve content-relevant functions from a codebase, or to run a self-improvement loop. Pairs an agnostic scaffold (content-addressing + grammar/execution verification) with you as the generator — nothing invalid is ever accepted.
---

# Substrate-Synth — verified, address-grounded code synthesis

**The principle:** the substrate is an agnostic *scaffold* — content-addressing + grammar-validity +
execution-eval — **not** a generator. Pair it with a strong generator (YOU, the model) and a verify
gate, and nothing invalid gets through. Validity is guaranteed by construction; correctness is
guaranteed by execution. It works on any codebase with a grammar + interpreter + corpus.

Reference implementation + the full math: **OMNIcode (OMC)**, MIT, public at
<https://github.com/RandomCoder-lab/OMC> — see [`SUBSTRATE.md`](https://github.com/RandomCoder-lab/OMC/blob/master/SUBSTRATE.md)
and `examples/harmonic_mind.omc`.

## When to use
- Synthesize code with an **execution/validity guarantee** (not "looks right" — *verified*).
- Retrieve **content-relevant** functions from a codebase by similarity.
- Run a **self-improvement loop**: address the target's own code → generate an improvement → verify
  by execution before accepting.

## The loop — always: address → generate → VERIFY → accept
1. **Address** the need: find the content-relevant existing code (by similarity), or the slot a new
   function belongs at (by content address). This grounds generation in real, working parts.
2. **Generate** the candidate (you are the generator) using that grounding.
3. **VERIFY** by execution — parse + run + check against tests. Accept ONLY if it passes; on failure,
   revise using the error and retry. Never return code that hasn't passed the gate.

## In OMC, the loop is in the language (v1.8+) — fastest path
If your target is OMC, build the public repo (`cargo build -p omnimcode-cli --release` →
`omnimcode-standalone`) and use these core builtins directly — no external scaffold needed:
- `gen_omc([seed])` / `gen_at(addr)` — valid-by-construction OMC (parse/run 1.000 over 300 seeds).
- `code_parse_check(src)` → `{ok, error}` and `eval_omc(src)` — the verify gate.
- `fn_swap_verified(name, new_src, test_src)` → `{accepted, error, result}` — the WHOLE loop in one
  call: install a candidate, test it in a sandbox, keep it only if it passes, else roll back.
- `@memo` — transparent, persistent, cross-run memoization (`@memo fn fib(n)...` makes `fib(90)`
  instant). Pure-only; impure functions are refused at definition.
- `haddr`/`haddr_face` (uniform content keys), `locality_sim`/`locality_nearest`/`nearest_fn`
  (content-similarity retrieval + dispatch), `cas_put`/`cas_get`/`same_value` (content-addressed heap
  + O(1) semantic equality), `value_addr`/`value_hash`.
- Dual-band coherence: `phi_shadow(v)`, `bands(v)` → `[α, β]`, `value_divergence(v)`, `@dualband`.

```omc
fn target(n) { return 0 - 1; }                   // a stub to improve
h cand = "fn target(n) { return n * n; }";
h r = fn_swap_verified("target", cand, "target(5) == 25");
print(r["accepted"]);                            // true — verified, installed; else rolled back
```

## For non-OMC targets (e.g. Python) — the agnostic scaffold
The same loop runs on any language: (1) a validity function (parse/typecheck), (2) an executor for
correctness, (3) a corpus to address against. The OMC repo's `experiments/transformerless_lm/`
includes a worked Python instantiation (`super_loop.py`, `py_substrate.py`, `locality_fp.py`,
`exec_eval.py`) and a learned NL→code retriever (`desc_encoder.pt`, held-out recall@5 0.89). Clone the
repo to reuse them, or re-implement the three hooks for your stack — everything else is unchanged.

## Hard rules (earned the hard way — these are load-bearing)
- **VERIFY before accept.** Never return code that hasn't passed the gate. The gate is the whole
  point: it lets even an imperfect generator be safe.
- **Two fingerprints, two jobs.** Use *content-similarity* (locality/byte-histogram) for "find similar";
  use *uniform content-addressing* (`haddr`) only for exact keys/buckets. A uniform hash has **no**
  content locality — φ-cosine similarity retrieval measured ≈ random; do not use it for similarity.
- **Similarity ≠ semantics.** Character/locality similarity is typo/variant-tolerant (`"quicksrt"` →
  `quicksort`) but does NOT map a natural-language description to code (`"greatest common divisor"`
  will not find `gcd`) — that needs a learned encoder.
- **valid ≠ correct.** The grammar/exec gate guarantees the code parses and runs; *correctness* needs
  test cases (derive them by running reference implementations). Always state which you verified.
- **Substrate is a detector/prior, not a computation path.** It helps on identity/addressing/position
  (attenuable); it does not belong on the learned-float scoring path (measured: it loses there).

## Honest scope
Capability scales by adding addressed content at flat per-query CPU cost (measured: correctness rises
with coverage; exact-key lookup stays O(1); verify is constant). The open frontier is generalizing
*beyond* stored content — bounded by generator quality, not by GPU.

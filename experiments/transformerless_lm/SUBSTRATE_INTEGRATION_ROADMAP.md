# OMC Substrate-Integration Roadmap
### Promoting the proven substrate discoveries INTO the language — and the capabilities that unlocks

**Authored:** 2026-05-29 (Claude Opus 4.8) · **Status:** planning deliverable, not yet executed
**Companion docs:** `DEVELOPMENT_PLAN.md` (the 5-track research plan), `AUTONOMOUS_LOG.md` (the A/B ledger)
**Scope:** the whole `OMC/` codebase — the Rust core (`omnimcode-core`), the MCP server, and the
research frontier (`experiments/transformerless_lm/`). Comprehensive of current state; extensive forward.

---

## 0. The thesis, stated honestly

You have a theory: **we can make code do things it was never thought to do, because of the substrate
math and the ADDRESSING function.** This roadmap takes that seriously *and* keeps it grounded.

The honest, load-bearing version of the theory — the one the evidence actually supports — is this:

> In a normal language, a value's **identity is its location** (a pointer, a variable, a stack slot).
> In OMC, a value's identity can be its **content** — its address is a function of what it *is*
> (`address = f(content)`), and the address space is **uniform** (χ²≈17, proven) and has **locality**
> (similar content lands near; navigation 0.88, proven). This *inverts the relationship between
> identity and storage*. Everything that follows — universal memoization, semantic equality, navigable
> programs, safe self-modification — is a consequence of that inversion.

That is genuinely something normal code "was never thought to do," and it is **grounded in primitives we
already proved**, not hope. The job of this roadmap is to (1) move the proven primitives from Python
experiments into the language, (2) build the one missing keystone — **the addressable heap** — and (3)
pursue the novel capabilities it unlocks under the same A/B discipline that earned us everything so far.

**The SUPER TOOL** in this picture is precise: the *agent* (a strong generator) sits in the **verified
synthesis slot** of the substrate scaffold — `address → generate → EXECUTION-VERIFY → accept`. The
substrate is the scaffold and the gate; the agent is the generator. We proved the weak generator
(FibRec) was the bottleneck; the scaffold + a strong generator + the verify gate is the actual deliverable.

---

## Execution status (live — 2026-05-29)

Shipped into `omnimcode-core` and verified (full ledger in `AUTONOMOUS_LOG.md`):

- ✅ **Phase 1.1 — `haddr` in the core.** New `src/address.rs`; builtins `haddr` / `haddr_face` /
  `haddr_distance`. χ² uniformity **reproduced in Rust** (uniform points 9.16, hashed strings 4.90 —
  tighter than the proven 17.1, vs old skew 216). Found + fixed the vestigial-`zeck` issue along the way.
- ✅ **Phase 2.1 — the addressable heap.** `Value::content_hash` (recursive FNV1a); builtins
  `value_addr` / `value_hash` / `same_value` (O(1) semantic equality, C2) / `cas_put` / `cas_get` /
  `cas_has`. Dedup + structural equality verified.
- ✅ **Phase 2.2 — `@memo`.** Transparent content-addressed memoization with a purity gate.
  `mfib(90)` (≈2.88e18 naive calls) returns in 0.313s — *compute once* (C1) is real in the language.
  Full core suite 167/167 pass (call-path change is correctness-preserving).
- ✅ **Phase 2 persistence — disk-backed heap.** `src/cas.rs` (lossless typed serializer, `~/.omc/cas`
  pool mirroring `memory.rs`). `cas_*` and `@memo` now survive across runs; `@memo` keys on a
  **body-aware salt** so editing a memoized fn invalidates stale disk results. Cross-run proof: a
  fresh process reads a value/result computed by a prior process from disk. *Compute once, anywhere, ever* — literal.
- ✅ **Phase 3 — the super-tool as a language feature (C4).** `fn_swap_verified(name, src, test)` —
  safe self-modification: install candidate, test it in a sandbox, keep only if it passes, else roll
  back (fn + `@memo` state). `fns_on_face(face)` — exact-key function addressing. **Live demo:** the
  agent generated an iterative `fast_fib`, the interpreter verified it against a reference and accepted
  it; a broken candidate was rejected and rolled back. Address → generate → VERIFY → accept, end to end.
- ✅ **Phase 1.2 — locality fingerprint.** `src/locality.rs` + `locality_fp`/`locality_sim`/
  `locality_nearest`. Reproduced the proven gap in-core: corrupted-retrieval recall@1 = **0.99
  (locality) vs 0.02 (φ)**, chance 0.005. The similarity primitive the toolbox leans on.
- ✅ **Phase 3.1 — content-addressed dispatch.** `nearest_fn`/`call_nearest` route a roughly-named
  need to the real function by locality (typo/variant-tolerant), then call it. Honestly scoped to
  character-similarity (NOT semantic NL→code — that's the learned encoder's job).
- ✅ **Phase 4 — correct-by-construction synthesis.** `src/synth.rs` + `gen_omc([seed])`: emits
  valid-by-construction OMC. Verified **parse-rate 1.000 / run-rate 1.000 over 300 seeds** by the real
  parser+interpreter (vs the FibRec net's ~0.10). Composes with the verify gate into generate→verify→accept.
- ✅ **#54 widening + Phase 6 gate (first brick).** `gen_omc` now also emits For/Print/if-else
  (parse/run still 1.000 over 300 seeds); `hbit_harmony`/`hbit_divergence` expose the REAL two-band
  gate (`divergence(8,8)=0` in tune, `(8,977)=947` divergent) — so the β-skip-vs-α-fallback decision
  the HBit dual-band vision needs can finally fire (tree-walk `harmony(x)` was a frozen 1000).
- ⏳ **Deferred:** 1.3 CRT-PE; 1.4 tape extraction; 2.4 DAG persistence; 3.2 weight sharing; 4.2
  remaining constructs (Try/Match/ClassDef) + 4.3 address-conditioned gen + 4.4 corpus leaves; Phase 5
  (frontier hypotheses); **Phase 6 body** — full dual-band VALUES (every value carrying a real β
  shadow + automatic α/β tracking) and kernel/microcode HBit (CPU-level resonance shortcuts), the
  "big-leagues" destination the gate brick opens the door to.

The keystone, the super-tool, both fingerprints, and the synthesis guarantee are in: OMC can address any
value by content, dedup it, compare it in O(1), memoize pure functions *across runs*, retrieve by
content-similarity, route a fuzzy name, generate guaranteed-valid programs, and rewrite its own
functions only when the interpreter verifies them — the substrate's addressing gift and the
verify-gated loop, made into the language itself.

---

## 1. Two laws and the guardrails (what we will NOT do)

Every line below obeys the two governing laws and respects the hard-won negatives. An honest roadmap
names its dead ends as loudly as its bets.

### The two laws (verbatim, from `DEVELOPMENT_PLAN.md §0`)
1. **Universal substrate.** No hand-coded dictionaries / curated word lists / per-corpus tables in the
   *authoritative* path. A dictionary is allowed **only if the corpus derives it** and stays annotative
   (never overrides a φ-address).
2. **Integer-substrate law.** Substrate structure helps on **integer / positional / identity** quantities
   (positions, IDs, hashes, ranks) and must be **attenuable**. It **hurts** when imposed on **learned
   float activations**. (Proven wins: CRT-PE, geodesic bias. Falsified: every float-gate reformulation, 0/3.)

### Guardrails (⛔ — proven false or null; do not repeat)
| ⛔ | Finding | Consequence for this roadmap |
|----|---------|------------------------------|
| **Substrate is a DETECTOR/PRIOR, not a computation path.** | Substrate-attention as a *scorer* lost 95/200 vs softmax 200/200; Fibonacci diffs are themselves Fibonacci → attractor-distance ties. | Never put substrate math on the *similarity/scoring* path. Use it for addressing, bucketing, positions, init, and as an attenuable prior — never as the arithmetic that decides an answer. |
| **Substrate-native compression can't beat zlib.** | OMCT (Axis 4) and OMCH (Axis 6) tie/lose; only OMCB (self-trained BPE, *non*-substrate) wins 5.21× vs 4.70×, and only on bodies ≥16KB. | Do not invest more in φ/substrate *byte* codecs. The compression win, where it exists, comes from corpus-derived BPE — which is law-1-legal precisely because the corpus derives it. |
| **φ-over-modulo weight sharing is null.** | n=5: φ-tier sharing ≈ naive modulo (Δ −0.113 within std 0.146). | Ship "params-as-addresses" as **address-bucketed weight sharing**, not "φ sharing." The compression is real (4× free, 14×@85%); the φ-geometry advantage is not. |
| **φ-cosine similarity retrieval ≈ random.** | recall@5 0.016 (≈2× chance). | Use **locality-fp** for similarity/navigation; reserve φ-hash for *exact keys / uniform buckets*. "Two fingerprints, two jobs." |
| **FibRec is not a higher generation ceiling.** | n=12: leads early (0.495 vs 0.291 @200 steps), comparable by 1800 (0.583 vs 0.506) — trades the lead, all within noise. | FibRec is a **sample-efficiency / cold-start prior**, useful for warm-start only. The "LM" is the assembled `SubstrateLM` (scaffold + verify), not the FibRec net. |
| **Inference-time substrate overlays don't improve a trained LM.** | Spells/nav/self-witness(EMA)/self-distill: 5/5 falsified or neutral; self-distillation *collapses* the model. | Stop bolting substrate onto inference of a finished model. (One nuance kept: self-witness with a *real frozen bloom centroid* was +0.11 — a prior, not an overlay.) |

These are not failures to hide — they are the map's "here be dragons." They sharpen the roadmap: every
bet below puts substrate where it provably *helps* (identity/addressing) and keeps it off where it
provably *hurts* (float arithmetic / scoring).

---

## 2. Grounding: where the codebase actually is (2026-05-29)

A roadmap is only as good as its honesty about the starting point. From four parallel codebase surveys:

**The Rust core (`omnimcode-core`, authoritative; `src.deleted.bak/` is dead — delete it):**
- **~550 builtins** dispatched (`interpreter.rs:2662`); a **`docs::BUILTINS` table that lags the dispatch**
  (surveys disagreed on the count, 543 vs 684 — *the disagreement itself is a task*: audit & reconcile).
- **Autograd tape — the crown jewel.** Reverse-mode, 26 differentiable ops + **4 substrate-fused**:
  `tape_phi_log` (`interpreter.rs:7954`), `tape_substrate_resample` (`:14050`), `tape_substrate_grad_mod`
  (`:14037`), `tape_substrate_sparse_scores` (`:14049`), plus fused `substrate_adamw_update` (`:6061`/`:13737`).
  All buried in a 15.6k-line `interpreter.rs` (extraction candidate).
- **Content-addressing is already a shipped strength** (this is the foundation your theory builds on):
  `canonicalize()` (alpha-rename-invariant), `omc_code_fingerprint` (stable across equivalent code),
  the `omc-kernel` content-addressed Merkle DAG ("semantic IPFS", `~/.omc/kernel/store`), and FNV1a-64
  content-addressed conversation memory (`memory.rs`).
- **VM works** (reuses the tree-walk stdlib for builtins). **JIT is real but integer-only & opt-in**
  (`OMC_HBIT_JIT=1`); HBit dual-band SIMD is the one shipped substrate JIT win. **`@harmony`/`@predict`/
  `@fibgen` pragmas are parsed-but-inert** — the "compounding JIT optimizations" are aspirational here.
- **GPU 8×32 anisotropic tile** wired into the CLI (`omnimcode-cli/src/main.rs`, `with_tile_xy`).
- **Core harmonic stubs:** `phi_shadow`/`harmony` are identity/constant stubs, `HBit` is JIT-only, the
  `h`-binding discards `is_harmonic` at the Value level — **the language's harmonic claims aren't real
  yet at runtime.** This blocks the Track-E self-hosting vision.

**The MCP server (`omnimcode-mcp`, shipped as the `omc-memory-plus` product):** 24 tools, stateless eval
(fresh `Interpreter` per call), content-addressed memory pool + fibtier eviction (cap 232) + manifest
(Merkle) + delta. Four compaction codecs (only the non-substrate BPE beats zlib).

**The addressing research — ALL of it is Python-experiment-only. NONE is in the Rust core or MCP.**
This is the single biggest gap and the roadmap's spine:
- Uniform dodecahedral addressing (icosahedral normals, **χ²≈17 proven uniform**) — `addressed_memory.py`,
  `name_registry.py` (`uniform_haddr` = fmix32 + inverse-CDF sphere), `hierarchical_address.py`.
- Locality-fp navigation (**0.88 recovery, 15× retrieval over φ**) — `locality_fp.py`.
- Universal store (text/tool/weight/memory in one space) — `universal_store.py`, the `omc_assistant.py` REPL.
- The assembled, always-valid generator — `substrate_lm.py` (registry → encoder → φ-synth → grammar → VERIFY).
- Grammar auto-derived from the Rust source — `derive_grammar.py` → `omc_grammar.json` (6/20 emitters).
- **Missing entirely:** transparent **memoization / value-cache by address**. *This is the keystone.*

**Debt flagged for cleanup:** `src.deleted.bak/`; 5× `.claude/worktrees/` full-repo copies; the
`train_self_recursive{,_v98f,_v98g}.py` ~6.6k-line triplet; orphaned `omnimcode-python`; stale root
roadmaps (`ROADMAP.md`/`OMC_STRATEGIC_PLAN.md`/`IMPROVEMENT_PLAN.md` predate the LM push); the
**two-generator tension** (the assistant defaults to the 10%-valid FibRec net while the verified
`SubstrateLM` sits unused beside it).

---

## 3. The capability theory — "code doing what it was never thought to do"

This is the heart of your theory, made concrete. Each capability is tagged by confidence:
**✅ PROVEN** (measured) · **🟩 GROUNDED** (direct extrapolation from a proven primitive, low risk) ·
**🟡 HYPOTHESIS** (plausible, must be A/B-gated before any claim).

### C1 — Universal, persistent memoization · 🟩 GROUNDED
A pure function `f(x)` has a content-address for its `(fn-fingerprint, arg-address)` pair, and its result
has an address. **Compute `f(x)` once, anywhere, ever** — the result is found by anyone who needs it,
across processes, sessions, and machines. Memoization stops being a per-process cache and becomes a
property of identity itself. We already do this for *code* (`omc-kernel`, canonical hash); extending it to
*runtime values* is the keystone (Phase 2). *Why grounded:* the addressing is proven uniform; the
content-addressed store already exists in Rust; the only new piece is wiring eval to consult it.

### C2 — Semantic, O(1) equality and cycle detection · ✅ PROVEN (for code) → 🟩 GROUNDED (for values)
Two computations are "the same" iff same address — already true for code via `canonicalize()` +
`omc_code_fingerprint` (alpha-rename and whitespace are *the same content*). Extend to values:
"have I reached this state before?" becomes an O(1) address check. Dedup, structural sharing,
fixpoint/cycle detection, and "is this the same answer I computed differently?" all collapse to address
equality. *This is something pointer-identity languages structurally cannot do.*

### C3 — Programs that navigate to their own missing pieces · ✅ PROVEN (prototype)
The universal store already does **navigate-to-address-then-dispatch**: a query fingerprint → nearest
address → if it's a tool, call it; if code, return it; if memory, reference it. A program can ask *"what's
the closest thing I already know to what I need?"* and get a geometric answer. Promote this from the
Python assistant into a core capability (Phase 3): **content-addressed dispatch** — call the nearest
function to a *need*, not a *name*.

### C4 — Self-modification WITH a correctness gate · ✅ PROVEN (the super-loop)
`super_loop.py` demonstrated it on real code: address the target → the agent generates a candidate →
**EXECUTION-VERIFY** → swap in only if it runs clean. Self-modifying code is old; self-modifying code that
*can never accept something that doesn't run* is the new thing. This is the SUPER TOOL as a language
feature (Phase 3/4): functions that improve themselves, gated by the interpreter itself.

### C5 — Inference-time weight compression by Zeckendorf addressing · 🟡 HYPOTHESIS (Track B.4, untested)
The "params-as-addresses" line proved 4× weight reduction free, 14×@85%. Track B.4's bet pushes it to
*inference-time* Zeckendorf-addressed weight reconstruction ("35B-in-8GB" / 700× bytes-per-token).
**Untested — and the φ-advantage is null, so ship as address-bucketed sharing.** Pre-register an A/B at
real scale before any claim. High value *if* it survives contact with a real model.

### C6 — Approximate compute by interpolation in address space · 🟡 HYPOTHESIS (carries a prior negative)
If locality-fp puts similar inputs near each other and results are addressed, then for *smooth* functions
one might **retrieve/interpolate an approximate result** for an unseen input instead of computing it.
**Flagged loudly:** substrate-as-computation was *falsified once* (attention scoring). So this is a
hypothesis with a pre-registered A/B and a hard guardrail: it may only ever *propose* an approximation
that is then *verified or refined* by real computation — never replace the computation outright. If it
works at all, it works as a *cache with a similarity radius*, not as an oracle.

### C7 — One address space for code, data, weights, memory, and the running mind · 🟩 GROUNDED → 🟡 (Track E)
`universal_store.py` already unifies text/tool/weight/memory. The north star (Track E) is that the program
*computes its own mind in the same space it stores everything else* — `examples/phi_field_llm_multilayer.omc`
already proves a zero-weight harmonic LM *runs* in pure OMC. Grounded today as a unified store; a
hypothesis as a self-hosting cognitive loop (gated on closing the core harmonic stubs, E.2).

**The autograd tape is the bridge.** The tape is *already* a content-addressable DAG of computation —
it's how we get gradients. Generalizing the tape's structure into a **persistent, content-addressed
computation graph** is the natural mechanism for C1/C2: the tape proves the machinery works for grads;
the addressable heap is that same machinery, persisted and keyed by content.

---

## 4. The roadmap — phased, grounded, A/B-gated

Each phase lists: **deliverables**, the **current files** it touches, a **verification plan** (because
every architectural change here is a pre-registered A/B reported even when it loses), and **confidence**.

Phases 0–4 are the grounded spine (promote what's proven + build the keystone). Phase 5 is the flagged
frontier. Phase 6 is the north star.

---

### PHASE 0 — Consolidate & ground · low-risk · unblocks everything
*Confidence: ✅ (pure cleanup + decisions already supported by evidence)*

- **0.1 Resolve the two-generator tension.** Make the verified `SubstrateLM` the assistant's canonical
  generator (always-valid, correct 1.00 w/intent); demote FibRec to an explicit "prior / cold-start"
  role. *Files:* `omc_assistant.py`, `substrate_lm.py`. *Verify:* the 10-task correctness benchmark must
  not regress; assistant boot loads `SubstrateLM`.
- **0.2 Delete dead trees & dedupe.** Remove `src.deleted.bak/`; prune `.claude/worktrees/` copies;
  collapse the `train_self_recursive{,_v98f,_v98g}.py` triplet to one (spells already extracted to the
  clean `grimoire_spells.py`); decide `omnimcode-python`'s fate (archive).
- **0.3 Doc/dispatch audit.** Reconcile `docs::BUILTINS` with the live dispatch (the 543-vs-684
  disagreement); the ~87 undocumented substrate-fused tape ops especially. *Output:* one true count,
  every dispatched builtin documented. This is also a prerequisite for `derive_grammar` completeness.
- **0.4 Consolidate roadmaps.** Retire/redirect the stale root docs to point at this file + `DEVELOPMENT_PLAN.md`.

---

### PHASE 1 — Promote the proven addressing primitives into the Rust core · the spine
*Confidence: ✅ each item is a measured win; risk is porting-fidelity, controlled by equivalence tests*

The research wins live in Python. Moving them into `omnimcode-core` makes addressing a *language*
capability, not an experiment. Each port must reproduce the Python result in-core (equivalence test +
re-run the original A/B).

- **1.1 `haddr` — uniform dodecahedral addressing as a core builtin.** Port `uniform_haddr` (fmix32 +
  inverse-CDF uniform sphere → face/sub_face/Zeckendorf). *Builtins:* `haddr(value) → address`,
  `address_distance(a,b)`, `address_face(a)`. *Verify:* re-run the χ² uniformity test in Rust — **must
  reproduce χ²≈17, p≈0.11** on 20k points (the Python-proven number). *Files:* new `omnimcode-core/src/
  address.rs`; ports `name_registry.py`, `addressed_memory.py`, `hierarchical_address.py`.
- **1.2 `locality_fp` — the similarity/navigation primitive.** Port `hist_fp` (char/bigram histogram) +
  coarse→fine `navigate`. *Builtins:* `locality_fp(bytes, window)`, `locality_navigate(query, corpus)`.
  *Verify:* reproduce navigation **0.88** recovery and retrieval **recall@5 ≈ 0.41** (bigram) in-core.
  Enforce the "two fingerprints" split in the API docs (φ for keys, locality-fp for similarity).
- **1.3 CRT-PE as a positional primitive.** The strongest *isolated* generation-side win
  (−3.6% to −19.9% val depending on scale, multi-seed). *Builtin/tape:* `crt_pe(pos, moduli)` feeding
  the tape; **attenuable** per law 2. *Verify:* re-confirm the win at target scale before defaulting it
  on (margin is scale-sensitive; only ever benchmarked vs sinusoidal — also test vs rotary/ALiBi).
- **1.4 Extract the tape into its own module.** Lift the autograd tape + 4 substrate-fused ops out of the
  15.6k-line `interpreter.rs` into `omnimcode-core/src/tape.rs`. Pure refactor; enables Phase 2 to hook
  the tape's DAG without touching the monolith. *Verify:* byte-identical training A/B (the `tape_phi_log`
  equivalence protocol: forward+backward equal to 1e-9).

---

### PHASE 2 — The Addressable Heap (THE KEYSTONE) · the missing piece for the whole theory
*Confidence: 🟩 GROUNDED — the surveys confirmed this is exactly what's absent, and every prerequisite exists*

This is the capability the survey found missing ("no transparent memoization-by-address of arbitrary
function calls") and the foundation for C1/C2/C6. Build it carefully, in layers.

- **2.1 Content-addressed value store.** A runtime store keyed by `haddr(value)`, reusing the existing
  FNV1a content-addressed pool mechanics (`memory.rs`) generalized from text to OMC `Value`s. Structural
  values (arrays/dicts) hash by content → automatic structural sharing & dedup. *Verify:* identical
  values share one address (dedup ratio measured); round-trip fidelity 100%.
- **2.2 `@memo` — transparent memoization by address.** A function pragma (real this time, unlike the
  inert `@harmony`): pure `f(x)` consults `(fn-fingerprint, haddr(args)) → result-address` before
  executing; on hit, returns the stored result. Cross-process and persistent (it's just the store).
  *Verify:* correctness-preserving on a suite of pure fns; measured hit-rate + speedup on recomputation;
  **guardrail:** only `@memo`-tagged pure fns (no side-effects) — detect & refuse impure ones.
  *This is C1 made real.* *Files:* `interpreter.rs` call path + `address.rs` + the value store.
- **2.3 O(1) semantic equality & cycle detection (C2).** `value_address(v)` builtin; `same_value(a,b)` as
  address-equality; a `seen-set` of addresses for fixpoint/cycle detection in interpreters & search.
  *Verify:* alpha/structural-equivalent values collide, distinct ones don't (collision audit).
- **2.4 Persist the computation DAG.** Optionally tag tape nodes (Phase 1.4) with content-addresses so a
  *computation history* is itself addressable & replayable — the bridge from "tape for grads" to
  "addressable heap for everything." *Verify:* replay a stored DAG reproduces the value.

---

### PHASE 3 — Addressable functions & safe self-modification · the SUPER TOOL as a feature
*Confidence: ✅ (super-loop proven) + 🟩 (content-dispatch proven in prototype)*

- **3.1 Functions at addresses + content-addressed dispatch (C3).** Register functions in the address
  space (already prototyped in `universal_store.py`); core builtin `call_nearest(need_fingerprint, args)`
  → dispatch to the nearest function to a *need*. *Verify:* on a held-out task set, nearest-dispatch
  picks the correct fn at the rate locality-fp predicts; falls back cleanly when no address is near enough.
- **3.2 Address-bucketed weight sharing ("params-as-addresses", de-mythologized).** Ship the *proven*
  compression (4× free, 14×@85%) as address-bucketed sharing — **not** "φ sharing" (null). *Verify:*
  reproduce the compression/quality curve; A/B vs naive modulo (expect parity, report it).
- **3.3 `super_loop` as a first-class language service (C4).** Promote `SubstrateLoop` (address → generate
  → VERIFY → accept) into a callable OMC capability with the agent as the pluggable generator and the
  interpreter as the gate. *Verify:* the loop never accepts code that fails `exec_eval`; demonstrated
  self-improvement (the dedent-fix microcosm) reproduced end-to-end in-language.
- **3.4 Hot-swap by address.** Runtime replacement of a function at its address (versioned), enabling A/B
  of implementations and live patching. *Verify:* swap is atomic & reversible; old address still resolves
  to old version (provenance preserved).

---

### PHASE 4 — Correct-by-construction synthesis as a language service · close the generation loop
*Confidence: ✅ spec-level done; 🟩 the remaining emitter coverage is mechanical*

- **4.1 Grammar auto-derivation in the toolchain.** `derive_grammar.py` already reads `ast.rs`/
  `tokenizer.rs`/`parser.rs` → `omc_grammar.json`. Make it a **build step** so the grammar always tracks
  the language; surface uncovered constructs as a CI warning. *Verify:* a new operator/keyword in the
  source appears in the grammar with zero manual edits (already demonstrated: `/` and `%` flowed in).
- **4.2 Close the emitter coverage gap (6/20 → 20/20).** Auto-synthesize per-construct emitters
  (If/For/Try/Match/ClassDef/…) from the parser's production rules so the generator covers the *full*
  grammar, not just operators. *Verify:* construct-coverage metric → 20/20; parse-rate stays 1.00,
  run-rate ≥ 0.90. *This is the documented "Full emitter auto-synthesis = next" residual.*
- **4.3 Address-conditioned generation (NEXT-6).** Generate the function that *belongs at* a given
  semantic address (the slot a need points to), not just from a name. *Verify:* generated fn lands within
  radius of the target address and passes the correctness benchmark.
- **4.4 Corpus-derived semantic leaves (Track A.4, pays the last law-1 debt).** Replace the residual
  hand-coded word-list in φ-synthesis with **token-chunking** (char→word→sentence→paragraph→corpus) leaves
  *regenerated per corpus* — law-1-legal because the corpus derives them. *Verify:* synthesis quality
  holds with zero hand-coded lists; portable to a fresh corpus with no edits.

---

### PHASE 5 — The frontier (HYPOTHESES) · pre-registered, A/B-gated, reported even when they lose
*Confidence: 🟡 — these are the bets, fenced by the guardrails*

- **5.1 Inference-time Zeckendorf weight compression (C5 / Track B.4).** The "35B-in-8GB" bet. Build the
  address→weight reconstruction; **pre-register** an A/B at real scale (perplexity vs bytes-per-token);
  ship only the address-bucketed form. *Kill criterion:* if quality loss > the compression buys vs plain
  quantization, report and shelve.
- **5.2 Approximate compute by address-space interpolation (C6).** The riskiest, carries a prior negative.
  Build it strictly as *propose-then-verify*: retrieve nearest addressed result, **always** verify/refine
  by real computation; measure how often the proposal is within tolerance. *Kill criterion:* if the
  proposal is no better than the mean, it's the substrate-attention tie again — report and shelve.
- **5.3 NEXT-7 — the real substrate-generator ceiling at scale.** The one open NEXT task; GPU-blocked, never
  faked. Find whether FibRec's early-convergence prior survives scale or stays a sample-efficiency trick.
  *Verify:* multi-seed scale ladder; honest curve.
- **5.4 Track B's untested science (B.2 compositionality field, B.3 weight-substrate views).** Derived but
  never tested. Pre-register, run, report.

---

### PHASE 6 — North star: the harmonic mind self-hosted in pure OMC (Track E)
*Confidence: 🟩 existence-proof exists; 🟡 the full loop is the long horizon — gated on E.2*

- **6.1 Close the core harmonic stubs (E.2 — the blocker).** Make `phi_shadow`/`harmony` real at the
  Value level (not stubs); make `HBit` a runtime value (not JIT-only); honor `is_harmonic` on `h`-binding;
  wire the inert `@harmony`/`@predict` pragmas to real behavior (or remove them honestly). *Verify:* the
  harmonic ops produce the documented dual-band semantics under test; no silent identity stubs remain.
- **6.2 Address-walk generation in pure OMC (E.1).** Generation as navigation over the in-language address
  space — building on `examples/phi_field_llm_multilayer.omc` (the zero-weight harmonic LM that already
  runs). *Verify:* runs end-to-end in OMC; output validity measured by the same exec_eval gate.
- **6.3 Full substrate self-hosting (E.3 / C7).** The substrate-synth loop, the assembled LM, and the
  addressable heap all expressed *in OMC*, computing over one address space that includes the program's
  own state. The language that improves itself, gated by itself.

---

## 5. Sequencing & dependencies

```
PHASE 0 (consolidate) ──┬─► PHASE 1 (promote primitives to core) ──► PHASE 2 (ADDRESSABLE HEAP ★keystone)
                        │              │                                      │
                        │              └─► PHASE 4 (synthesis service)        ├─► PHASE 3 (addressable fns + super-tool)
                        │                    (4.1/4.2 need 0.3 doc audit)     │
                        └────────────────────────────────────────────────────┘
                                                                              │
PHASE 2 + PHASE 3 ──► PHASE 5 (frontier hypotheses: C5/C6, NEXT-7, Track B)
PHASE 1.1 (haddr) ──► PHASE 6 (north star) ── gated on 6.1 (close harmonic stubs)
```

**Critical path:** `0 → 1 → 2`. The **Addressable Heap (Phase 2)** is the keystone — it is what turns the
theory from poetry into a feature, and almost everything novel (C1, C2, C6, the persistent computation
DAG) hangs off it. Phase 1 must precede it (the heap needs `haddr` + the extracted tape in the core).
Phase 3 and Phase 4 can proceed in parallel once Phase 2 lands. Phase 5/6 are the long horizon.

**Recommended first move:** Phase 0.1 + 0.3 (cheap, unblocks) → Phase 1.1 (`haddr` in core, the single
highest-leverage port — every later capability addresses through it) → Phase 2.1/2.2 (the heap + `@memo`,
the first capability the *user* will feel: "this function never recomputes anything it's seen, ever,
anywhere").

---

## 6. How we'll know it's real (the standing discipline)

Unchanged from everything that earned us this position, applied to every item above:
1. **Pre-register the A/B** before building (hypothesis, metric, kill criterion).
2. **≥3 seeds** where stochastic; report mean ± std.
3. **Execution-grounded** where it's code (parse/run/correctness via the real interpreter, `exec_eval`).
4. **Report losses as loudly as wins** — a null result that fences off a dead end is a deliverable.
5. **Honor the laws & guardrails** — substrate on identity/addressing/positions (attenuable), never on
   learned-float scoring; no hand-coded dictionaries the corpus didn't derive.

The substrate's proven gift is **addressing** — uniform, local, content-derived identity. This roadmap's
single idea is to give that gift to the *language itself*, build the *addressable heap* on top of it, and
let the SUPER TOOL (agent-as-generator, interpreter-as-gate) compound it into code that memoizes the
universe, finds its own pieces, and rewrites itself without ever accepting a line that doesn't run.

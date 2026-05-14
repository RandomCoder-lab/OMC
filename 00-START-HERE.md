# OMNIcode — Start Here

You've just opened the OMC repository. This file orients you in about 5 minutes.

**Current state (2026-05-14):** OMC is a self-hosting harmonic computing language with a self-healing compiler. The architectural bootstrap is closed (Phase V.9b) and the self-healing compiler works across token and AST stages with user-declared runtime opt-in (Phase H.4). The supporting circuit-evolution engine from the v1.0.0 release is still here too — see *Two arms of the project* below.

---

## If this is your first time here

Read these three files in order, in about 25 minutes total:

1. **`README.md`** (10 min) — what OMC is, what's proven, the V→H phase arc.
2. **`CHANGELOG.md`** (10 min, skim — pin the Phase V.6 through H.4 entries) — the design history with concrete demo files at every milestone.
3. **`ARCHITECTURE.md`** (5 min, skim) — type system, interpreter, VM internals.

Then run any demo from the "What's proven right now" table in the README. If `examples/self_hosting_v9b.omc` prints `✓✓✓ ALL THREE FIXPOINTS REACHED`, you have a working build.

---

## Two arms of the project

OMC has been two different research artifacts at different times. Both are still in the repo. Keep them straight:

### Arm 1 — The language (Phase V + H, 2026-05)

This is what the current README leads with and what gets active development. Self-hosting compiler, self-healing diagnostics, φ-math substrate. Lives in `omnimcode-core/src/{parser,ast,interpreter,vm,value}.rs` and the `examples/self_hosting_*.omc` / `examples/self_healing_*.omc` files. See `CHANGELOG.md` for the milestone-by-milestone account.

### Arm 2 — Circuit evolution (v1.0.0, 2026-04)

The original release. Genetic algorithms over Boolean and float logic circuits, with FFI bindings to Python, Unity, and Unreal. Lives in `omnimcode-core/src/{circuits,evolution,circuit_dsl}.rs` and the `examples/agent-decision-evolution/`, `examples/game-ai-demo/` directories. This arm is **stable**, **functional**, and **mostly frozen** — the circuit engine works as documented. See `RELEASE_BODY_v1.0.0.md` for what shipped.

**Why both?** The circuits / GA work proved out the φ-math primitives (resonance scoring, Fibonacci attractors, harmony values) on a concrete substrate. Those same primitives are what the Phase H self-healing compiler now uses to detect and repair bugs. The line from "evolve a circuit by selecting for harmony" to "heal a program by rewriting toward harmony" is short and real.

---

## Recommended reading paths

### For language designers / PL researchers

1. `README.md` — both tracks
2. `CHANGELOG.md` → Phase V.6 → V.9b → H.1 → H.4 entries
3. `examples/self_hosting_v9b.omc` — the gen2==gen3 fixpoint
4. `examples/self_healing_h4.omc` — the `safe` keyword and runtime healing
5. `PHI_PI_FIB_ALGORITHM.md` — math foundation
6. `ARCHITECTURE.md` — type system internals

### For developers and engineers

1. `README.md` → "Quick start" + "Try the language"
2. `BUILD.md` — build flags, cross-compilation, optimization
3. `examples/` — runnable programs covering most features
4. `DEVELOPER.md` — extending the language host-side
5. `BENCHMARKS.md` — performance numbers (tree-walk vs VM vs VM+opt)

### For circuit / GA work

1. `README.md` (the language sections are skippable for this lane)
2. `RELEASE_BODY_v1.0.0.md` — what the GA arm delivered
3. `omnimcode-core/src/circuits.rs` and `evolution.rs` — implementations
4. `examples/agent-decision-evolution/` — the demo
5. `HBIT_INTEGRATION.md` — the dual-band α/β/harmony programming model

### For LLM-generated-code researchers

The Phase H self-healing compiler is the relevant lane. Specifically:

1. `README.md` → "Implications" section
2. `CHANGELOG.md` → Phase H.1 through H.4
3. `examples/self_healing_h3.omc` — 5 bugs healed in one source across two stages
4. `examples/self_healing_h4.omc` — `safe` keyword for dynamic singularities

---

## What's *not* in this repo

Honest list:

- **No production-grade bytecode runtime.** The OMC-written bytecode VM in `examples/self_hosting_v7c.omc` is *correct* (byte-identical to tree-walk) but runs on the tree-walker, which makes it slow. A native bytecode VM in Rust is future work.
- **No LSP, formatter, debugger, or package manager.** OMC is a research codebase, not a deployment target.
- **No external review.** Single-developer experiment. There are bugs we don't know about.
- **The healer doesn't handle every error class.** What it handles is documented in the README "What this doesn't do yet" section. `stuck` and `exhausted` outcomes are designed but unexercised.

---

## Where to file work

- Issues, observations, and PRs on the GitHub repo.
- For OMC programs that don't behave as expected: include the source and the output. The interpreter at `target/release/omnimcode-standalone` is the reference behavior.
- For research questions about the φ-math substrate or the self-healing approach: open a discussion / issue rather than a PR; the design space is still moving.

---

## Index of top-level docs

| Document | Purpose |
|---|---|
| `README.md` | Landing page, headline claims, arc, quick start |
| `CHANGELOG.md` | Phase-by-phase design history (V.6 → H.4) |
| `ARCHITECTURE.md` | Type system, interpreter, VM internals |
| `BUILD.md` | Build instructions, optimization flags, cross-compilation |
| `BENCHMARKS.md` | Criterion benchmarks: tree-walk vs VM vs VM+optimizer |
| `DEVELOPER.md` | Extending the host language |
| `READING_ORDER.md` | Navigation guide, multiple paths through the docs |
| `INDEX.md` | Detailed deliverable index (v1.0.0 era; partial relevance) |
| `RELEASE_BODY_v1.0.0.md` | v1.0.0 release notes (circuit-evolution arm) |
| `PHI_PI_FIB_ALGORITHM.md` | Mathematical foundation |
| `OMC_STRATEGIC_PLAN.md` | Direction-of-travel for future phases |
| `HBIT_INTEGRATION.md` | Dual-band α/β/harmony programming model |
| `IMPROVEMENT_PLAN.md` | Concrete next-step items |
| `PHI_DISK.md` | Storage-layer experiment notes |
| `TIER_4_HONEST_REVISION.md` | Honest write-up of the Fibonacci-search / LRU sub-component |
| `CODE_SIGNING.md` | Release signing process |
| `BUILD_TARGETS.md` | Supported build targets |

---

**Built around φ (1.618…). The substrate is the architecture.**

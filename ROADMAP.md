# OMC Roadmap

Current release: **v1.8.5** (2026-05-30). The **substrate-into-core** arc has landed: the proven
φ-substrate discoveries are now first-class language primitives (content-addressing, an addressable
heap, persistent `@memo`, locality similarity, verify-gated self-modification, correct-by-construction
synthesis, and HBit dual-band real at the Value level). See [SUBSTRATE.md](SUBSTRATE.md) for the full
reference with verified numbers, and `experiments/transformerless_lm/SUBSTRATE_INTEGRATION_ROADMAP.md`
for the phase plan + evidence ledger.

See [CHANGELOG.md](CHANGELOG.md) and [GitHub Releases](https://github.com/RandomCoder-lab/OMC/releases)
for how OMC got here. This file describes what's on the path forward.

---

## Shipped (v1.8.x) — substrate into the core language

| Area | Primitives | Verified |
|---|---|---|
| Content-addressing | `haddr`, `haddr_face`, `haddr_distance` | face χ² ≈ 9 (uniform) |
| Addressable heap | `cas_put/get/has`, `value_addr`, `value_hash`, `same_value` | dedup + O(1) semantic equality |
| Memoization | `@memo` (transparent, persistent across runs, body-aware) | `fib(90)` instant; cross-process |
| Similarity | `locality_fp/sim/nearest`, `nearest_fn`, `call_nearest` | recall 0.99 vs φ 0.02 |
| Self-modification | `fn_swap_verified`, `fns_on_face` | verify-gated accept/rollback |
| Synthesis | `gen_omc`, `gen_at` | parse/run 1.000 over 300 seeds |
| Dual-band (Value-level) | `phi_shadow`, `bands`, `harmony`, `value_divergence`, `@dualband`, `hbit_*`, `band_*` | β rides through arithmetic; α always exact |

Scaling result (NEXT-7, on CPU): capability rises with addressed content while per-query cost stays
flat (exact-key O(1) + constant verify). The substrate's scaling axis is content + verify (CPU), not
parameters (GPU). 267 tests pass (172 lib + 95 integration, incl. `tests/substrate_v18.rs`).

---

## Next — grounded, no new compute required

- **Synthesis coverage (Phase 4.2):** extend `gen_omc` to the remaining run-safe constructs
  (`for` over expressions, `try`/`match`, nested blocks). The valid-by-construction guarantee is in;
  this widens what it can emit.
- **In-core grammar derivation (Phase 4.1):** derive the generator's operator/keyword/construct set
  from the live AST/parser at build time (the Python `derive_grammar.py` already does this at the
  toolchain level; bring it in-core so the generator can't drift from the language).
- **Assistant unification (Phase 0.1):** make the verified `SubstrateLM` (always-valid, correct-with-
  intent) the assistant's canonical generator; demote the FibRec neural net to an explicit cold-start prior.

## Next — needs compute (model / GPU / training)

- **Inference-time weight compression at scale (Phase 5.1):** the address-bucketed / Zeckendorf bet
  ("big model in small memory"). Small-scale is settled (address-bucketed sharing is real ~4× free /
  14× @ 85%; the φ-vs-modulo advantage is null). Needs a real model to test at scale.
- **Track-B science (Phase 5.4):** compositionality-coherence field and weight-substrate views —
  derived, not yet trained.
- **Substrate-generator generalization:** raising held-out (compose-beyond-coverage) correctness —
  CPU-scalable in principle (grammar-gen + verify), bounded by generator quality.

## Horizon — the dual-band dream

- **Value-granular skip:** today the dual band is a coherence monitor + exact-memo router (α is always
  computed as ground truth). A strict speedup *from the gate* is safe only on smooth domains
  (measured: interpolation works for smooth functions, not discrete ones) — pursue it opt-in there.
- **Kernel / microcode HBit:** the long-horizon goal — α/β as a hardware-level dual band, short-cutting
  computation through resonance/dissonance at the CPU level. The JIT already packs both bands into
  SSE2 `<2 x i64>` and elides branches by harmony; pushing that into a true microcoded skip is the
  frontier this whole substrate program is paving toward.

---

## Method (unchanged)

Every architectural change is a **pre-registered A/B**, reported even when it loses. Substrate goes
where it provably helps — identity / addressing / positions, attenuable — and stays off the
learned-float scoring path, where it was falsified. Validity is guaranteed by construction; correctness
is verified by execution. Honest limits travel with every claim.

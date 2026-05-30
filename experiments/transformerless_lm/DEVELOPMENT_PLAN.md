# OMC Substrate / LM — Development Plan

*Author: Claude (Opus 4.8), 2026-05-28. A sequenced plan I would execute to carry
the substrate-math + transformerless-LM work forward, grounded in the project's own
proven/falsified ledger. Written to be picked up across sessions.*

---

## ADDR BACKLOG RESULTS (2026-05-29 night) — the two-fingerprint correction

Went down the whole ADDR list. The headline: **φ-fp is for KEYS, not SIMILARITY.**
- **ADDR-3 (biggest):** φ-cosine *similarity retrieval* ≈ RANDOM (recall@5 0.016 vs 0.008 random;
  content-Jaccard gold). The dual-index / `omc_search` retrieval returns content-irrelevant
  windows. "Addressing is proven-real" now precise: **uniform buckets (χ²17) ✓ + exact-key
  registry ✓ are real; φ similarity-retrieval is NOT.**
- **ADDR-1 (fix):** a LOCALITY fingerprint (char histogram) — overlapping windows share features —
  gives navigation 0.29→0.88 and retrieval recall 15× φ. Module: `locality_fp.py`. Principle:
  **two fingerprints — φ for keys/buckets, locality-fp for similarity/navigation.**
- **ADDR-7:** char-BIGRAM histogram > unigram for relevance (0.41 vs 0.24) — best nav/retrieval fp.
- **ADDR-2:** execution parse-rate is the standard eval (`exec_eval.py`): registry/φ-synth 1.00,
  bloom_best LM 0.10 — exposed what 4-gram (0.52) hid.
- **ADDR-6:** locality-fp cross-type retrieval works — one query → relevant code+tool+memory.
- **ADDR-5 (null):** overlapping nodes HURT (0.82→0.67); locality-fp already offset-robust.
- **ADDR-4 (skip, reasoned):** φ-bucket uniformity migration deprioritized — φ similarity-retrieval
  is being replaced by locality-fp, so it has ~zero remaining value.

**Net:** the path to better addressing is the two-fingerprint split + execution-grounded eval,
not more φ. Next: wire `locality_fp` retrieval (bigram) into `omc_assistant` (replace φ-cosine),
and build the retrieval-relevance benchmark into CI.

---

## PHASE 4 ADDENDUM (2026-05-29 eve) — develop the two proven threads

- **Addressing (proven-real, developed further):** hierarchical multi-scale index
  char→word→sentence→paragraph (T13/34/89/233) → **full 12/12 dodecahedral coverage** (no
  single scale exceeds 8/12); scales occupy **complementary face regions** (chunk scale
  determines where content lands). Clean positive — the user's chunking idea on the retrieval side.
- **New LM (bloom_best) — built, integrated, honest null on stacking:** first model COMBINING
  the wins (chunk233 + depth6 + long1800 + lr3e-4, 333K params). Guarded validity **0.524**
  (distinct 0.23, real/non-degenerate) — did NOT beat chunk233-alone (0.572); all decent
  configs cluster ~0.47–0.57 within n=12 noise. Stacking the levers did not yield a new champion.
  BUT bloom_best is trained, functional, and **now the assistant's default LM** (verified: loads
  seq233 + 256 retrieval index, generates). A usable artifact even if not a quality jump.
- **Compression (params-as-addresses), settled at n=5:** ~14× FFN reduction retains ~85%
  validity (good trade); φ-structured sharing ≈ naive modulo (no substrate edge — null).

---

## GRAND SYNTHESIS — two autonomous nights (2026-05-28 → 05-29)

Full ledger in `AUTONOMOUS_LOG.md`. ~20 pre-registered experiments, ≥1 control each,
corpus-4-gram gen-validity (degeneracy-guarded after EXP-6). The honest bottom line:

**WHAT'S REAL (proven, positive):**
1. **Addressing is the deliverable.** φ-addressing made provably uniform (χ² 216→17 via
   canonical icosahedral normals + fmix avalanche). Threshold-free name registry. Address =
   corpus-derived intersection. This is the solid foundation.
2. **Parameters-as-addresses (user's idea) — strongest generation-side win.** Weights
   shared by address give **4× param reduction at no quality cost** (0.526→0.552 validity),
   16× at modest cost. Verified non-degenerate. This is the most promising lead for future work.
3. **Token-chunk scale (user's idea) helps, monotone.** word(0.337) < sentence(0.531) <
   paragraph(0.572) — larger-form context improves generation, diminishing past sentence scale.
4. **Depth helps and is ~free** in FibRec (recurrence): n2→n6 lifts validity.
5. **Substrate = sample-efficiency prior.** FibRec beats a transformer early (0.495 vs 0.291
   @200 steps); they're **comparable** by 1000+ steps (trade leads, n=12 noisy). Not a
   higher ceiling, but a real fast-convergence advantage.

**WHAT'S NOT (falsified or fixed-but-unhelpful):**
6. **Inference-time substrate overlays don't help.** Spells (C.1), Platonic nav (C.2),
   bloom self-distillation (D.1 → model collapse), naive Self-Witness (B.1). Revisions:
   attenuated spells (R2) and real-bloom Self-Witness (R1, +0.11 gen-validity — partial
   win, the user's "it was the proxy" reframe was partly right).
7. **K-shrink was a NO-OP** (a real bug — `set_K_active` never reached the model). FIXED &
   verified — but **applying it collapses the model** (shrinking to K=13 in few steps).
8. **K_init=144 capacity doesn't raise the ceiling** (0.489); d_model/n_blocks barely move
   params in FibRec — capacity is fixed by K_init.

**METHOD CATCHES (the discipline that held):** raw n-gram validity is gameable by
repetition (EXP-6's fake 0.821 = collapsed whitespace) → added a degeneracy guard; caught a
LayerNorm-cancellation no-op in B.1; caught the K-shrink no-op. Negatives reported straight.

**WHERE TO GO NEXT:** parameters-as-addresses (#2) is the lead worth pursuing — extend from
the n-gram testbed to the full FibRec LM; combine with chunk-scale (#3) and depth (#4); keep
addressing (#1) as the proven core. Stop bolting substrate onto inference (#6).

---

## CONCLUSION — substrate for generation (2026-05-29, REVISED after the full step-ladder)

**FibRec and a vanilla transformer are COMPARABLE generators at ~330K params; FibRec is
more sample-efficient early.** The fuller evidence (corpus 4-gram gen-validity, from scratch,
n=12 each) overturns the mid-session "transformer overtakes" reading, which rested on a
single 1000-step point:

| steps | FibRec | Transformer |
|---|---|---|
| 200 | **0.495** | 0.291 |
| 1000 | 0.484 | **0.538** |
| 1800 / 2000 | **0.583** | 0.506 |

- They **trade the lead** — FibRec early, transformer mid, FibRec late — all within n=12
  noise (±0.05). Neither dominates. The substrate is a genuinely *competitive* generation
  architecture, not the dead-end the inference-overlay failures suggested, nor a clear champ.
- **FibRec is clearly more sample-efficient very early** (200 steps: 0.495 vs 0.291).
- The earlier "plateau at ~0.49" was a SHORT-TRAINING artifact: at 1800 steps FibRec reaches
  0.583 (char fell 2.48→2.25, and validity followed). Extended training breaks it.
- **Discipline note:** both the 200-step "substrate wins" AND the 1000-step "transformer
  overtakes" were over-reads of single noisy points. The honest claim is "comparable, n=12
  noisy, FibRec more sample-efficient early." Bigger n / more seeds would sharpen this.

**What the φ-substrate is genuinely good for (the honest scope):**
1. **Addressing/retrieval — REAL and proven.** Uniform content-addressing (χ²=17),
   threshold-free name registry, corpus-derived address intersections. This is the deliverable.
2. **Fast convergence from a cold start** — useful for low-step/few-shot regimes, warm-starts,
   and as an init prior — but it hands the lead back to a transformer given enough steps.
3. **NOT for** high-fidelity generation at scale (the transformer's learned weights have higher
   ceiling), nor as inference-time overlays (5/5 falsified: spells, nav, witness, self-distill).

**Why (mechanism — now CONFIRMED 2026-05-29):** FibRec weights are *generated* from a frozen
Fibonacci basis + seeds of size K²·4 per projection. The autonomous EXP-2 revealed the
decisive fact: **K-shrink is a NO-OP in the production model.** `FibRecLMSubsim` computes via
`stateless_fibgen_forward(x, seed, basis, self.K)` with FIXED K; the lazy-K-subsampling lives
only in `FibGenLinear` (unused by this model). So K_min=8/13/21 gave bit-identical results,
and the model always runs at full K_init=89. **Capacity is therefore HARD-FIXED by K_init**,
and K-shrink/d_model/n_blocks barely move it — that IS the ~0.49 plateau mechanism. The
"linguistic-abstraction K-shrink curriculum" narrative does not apply to the deployed LM.
Corrected capacity test (queued): K_init sweep — is the plateau set by K_init=89, and does
K_init=144 raise it (or NaN, hard-capping the architecture)?

---

## 0. The governing law (do not violate)

Two hard constraints sit above every step here:

1. **Universal substrate.** No hand-coded dictionaries, no curated word lists, no
   per-corpus tables in the *authoritative* path. Anything that stops the model
   moving from one corpus to another universally is forbidden. A dictionary is
   allowed **only if the corpus derives it** (token-chunking / frequency-rank
   tiers / k-means leaves regenerated per corpus) and it stays *annotative*, never
   overriding a φ-address.

2. **The integer-substrate law** (empirically earned — see ledger §6 of the review):
   > Substrate structure helps when applied to **integer / positional / identity**
   > quantities (positions, IDs, hashes, ranks) and is **attenuable**.
   > It *hurts* when imposed on **learned float activations**.
   > PROVEN wins: CRT-PE, geodesic attention bias. FALSIFIED: every float-gate
   > reformulation (0/3), the naive "stack all substrate primitives" (token_crt +4.1%).

Every new idea below must answer: *what integer quantity does this act on, and is the
injection attenuable?* If it touches learned floats directly, it must be gated to fade
to identity — or it will be falsified.

The honest open question the whole project orbits:
**can a substrate-native architecture train competitively at scale?** That is the
target. These steps are the path to a defensible yes/no.

---

## 1. Current state (done)

- **seq_len=256 pipeline** complete & loading: `bloom_256_model.pt` (base, char 1.95),
  `omc_dual_index_256.pt` (225k windows, all 12 faces), `multiskill_navigator_omc_corpus.pt`,
  `bloom_256_curriculum_model.pt` (val 1.926).
- **Name registry → pure φ(name)** (`name_registry.py`): face/sub_face/zeck all from
  `assign_address(substrate_fingerprint(name))`. Zero dictionary, threshold-free,
  faces equidistribute 90–334. Underscore-collapse lookup (quicksort↔quick_sort) is a
  universal string op, not a list.
- **φ-template synthesis** for novel functions (`phi_synthesis.py`) — but it still uses
  anchor word-lists internally (a constraint debt, see Track A.4).

---

## 2. Track A — Substrate Math (the foundation)

**A.1 — Quantify the equidistribution claim. [DONE 2026-05-28]** χ² over the 12 registry
faces. *Result — falsified my "skew is gone" overclaim and found two real defects:*
  - **Defect #1 (FIXED): wrong face normals.** `addressed_memory.py` used
    `(±1,±1,±1)×8 + (0,±φ,±1/φ)×4` — unequal solid angles (χ²=1570 on 20k *uniform*
    points). Replaced with the canonical icosahedral-vertex normals `(0,±1,±φ)`&perms
    (χ²=17 on uniform = equal-area) + programmatic 3-nearest neighbors. Registry χ²
    dropped **216 → 149**, max/min ratio **3.16 → 2.09**. Requires dual-index rebuild
    (done) since face buckets changed.
  - **Defect #2 (OPEN → A.1b): the φ-LCG hash is itself non-uniform** on real identifier
    strings — χ²≈125–149 even with equal-area faces and a uniform-S² projection (vs 17
    baseline). The golden-ratio LCG's equidistribution is asymptotic; it does not
    decorrelate short, structured names (shared prefixes `arr_`, `is_`, `_sort`).

**A.1b — Improve φ-hash uniformity. [DONE 2026-05-28]** Root cause: (i) sin/cos
interleaving puts `vec[:3]=(sinθ0,cosθ0,sinθ1)` on a band, not S²; (ii) the LCG output
isn't avalanched. Fix = murmur3 `fmix32` finalizer on each hash + inverse-CDF
uniform-sphere projection (`z=2u₀−1, θ=2πu₁`). Registry χ² **216 → 17.1 (p=0.106,
ratio 1.30)** — statistically uniform. Implemented as `name_registry.uniform_haddr`
(registry-local, so the shared fingerprint / index / navigator are untouched). Lookups
verified intact.

**A.1c — Migrate the SHARED fingerprint to uniform (deliberate). [SPECIFIED]** Make
`corpus_address_index.substrate_fingerprint` (scalar AND the overflow-sensitive batch
path) use fmix+uniform-sphere so the index gets equal-area buckets too. Requires:
overflow-safe `fmix32` in torch int64 (split 32-bit mul), rebuild dual index, retrain the
navigator `addr_head` (its φ-hash target shifts), retrieval-quality A/B before/after. Held
out of this wave to avoid destabilizing the working assistant. *Effort: M.* *Depends: A.1b.*

**A.2 — The co-location prize. [DONE 2026-05-28 — IMPOSSIBILITY PROVEN, design vindicated]**
Tested whether `face(φ(name)) == face(φ(body))` is achievable without a table.
  - *Content-hash transforms* (raw body, signature line, identifier set, call set,
    body-minus-name): all ≈ **8% face-match = chance**. Even the signature line (which
    contains `fn NAME(`) gets 8.4% — φ("fn fibonacci(n){") is unrelated to φ("fibonacci").
  - *Learned linear map* φ(body)→φ(name) (ridge, 90/10 split): **9.5%**, cos −0.006 — no
    better than chance.
  - *Why (the real finding):* φ(name) is an **arbitrary hash of a label**; it shares no
    recoverable information with φ(body), so NO map — content or learned — can connect
    them. (The navigator's `addr_head` only works because it predicts φ of the next
    *sequential* window, which shares actual character context; a name is not sequentially
    continuous with its body.)
  - *Conclusion:* the prize as literally stated is impossible by derivation — and that
    **vindicates the registry**: you don't make the body hash to the name's location, you
    *store* the body AT φ(name), keyed by identity. That is "multiple points of info at one
    address" by construction. Key-by-identity is correct; derive-identity-from-content is
    a category error. No further work; the registry already embodies the answer.

**A.3 — Multi-scale Fibonacci ladder. [DONE 2026-05-28]** Forced by the normals fix:
canonical normals at the old scales (21,256) covered only 10/12. Probed coverage per
scale on a 3M-char corpus sample → **(13, 89) covers all 12 faces** with canonical
normals (each single scale covers 6–8/12; the pair tiles the solid). Rebuilt
`omc_dual_index_256.pt` at scales (13,89): combined coverage = ALL 12, verified. Remaining
(optional): retrieval precision@k vs index-size curve for 3-scale sets. *Effort: M.*

**A.4 — Corpus-derived semantic leaves (pay the constraint debt).** Build the
`SubstrateFingerprint` 4×3 = 12-leaf k-means clustering over the OMC corpus as an
**optional, annotative** layer. It may *label* a φ-face with an emergent cluster, but
must never override the φ address. Then refactor `phi_synthesis.py` to derive its
template routing from these corpus leaves + frequency-rank tiers instead of the
hand-coded anchor words. *Why:* removes the last word-list in the synthesis path; honors
"dictionary is OK if the corpus makes it." *Success:* synthesis routing identical-or-better
with zero hardcoded anchors; regenerates on a different corpus. *Effort: L.* *Depends: A.1.*

---

## 3. Track B — The LM (test the ready hypotheses)

The derivation docs contain **four fully-derived, never-tested** architectural ideas.
Testing them is the highest-value science available — each is a clean A/B with a
pre-registered prediction, in the project's own falsification style. All four must be
**attenuable** (integer-substrate law).

**B.1 — Self-Witness (bloom heartbeat). [DONE 2026-05-28 — FALSIFIED]** Implemented as an
attenuable flag on `FibRecLMSubsim` (`apply_self_witness`); A/B harness
`train_self_witness_ab.py` (seq_len 89, 2 blk, 500 steps, 3 seeds). *Methodology catch:*
first run was Δ=0.0000 bit-identical across seeds → exposed that placing the scalar
modulation **before `ln_f`** is a no-op (LayerNorm is per-position scale-invariant). Moved
it **after** ln_f (scales logits = on-manifold confidence). *Corrected result:* witnessed
2.6357 vs baseline 2.5658, **Δ=+2.72%, 0/3 — falsified.** Why: b = EMA-of-mean-hidden is a
poor proxy for the bloom centroid; distance-to-centroid doesn't track next-token quality,
so the logit rescale just distorts a calibrated softmax (integer-substrate law: injection
on learned floats loses). *Unfalsified variant:* a centroid from a REAL trained bloom
(Track D) rather than the mean-hidden proxy. *Effort: M.* (`SELF_WITNESS_DERIVATION.md`)

**B.2 — Compositionality coherence field.** `W_n^composed = W·φ⁻¹·(1 + ρ_n·CF_pos)`,
`CF_pos = Π T[leaf]^{φ^(−i/φ)}`. No new params. A/B. *Effort: M.* (`COMPOSITIONALITY_DERIVATION.md`)

**B.3 — Weight-Substrate, two steps (sequenced, falsifiable).**
  - Step 1: express Q/K/V as Fibonacci **cyclic-shift views** of one shared W (acts on
    weight *indices* — integer quantity, legal).
  - Step 2: Fibonacci-tier **quantization** of weights.
  Run as separate A/Bs; if either fails cleanly, we learn which principle is wrong.
  *Why:* directly probes "is natural-language weight structure Fibonacci-tier-quantizable"
  — the core compression bet. *Effort: L.* (`WEIGHT_SUBSTRATE_REFORMULATION.md`)

**B.4 — Inference-First context compression.** Zeckendorf context-state + trie LM-head.
This is the "~700× bytes-per-token" bet. Prototype the Zeckendorf context encoder, measure
actual bytes-fetched-per-token vs the baseline. *Why:* it's the load-bearing claim for
"35B-in-8GB"; either it holds or the bet is called. *Effort: L.* (`INFERENCE_FIRST_DERIVATION.md`)

**B.5 — Scale ladder.** Continue the chained-gen seq_len ladder 256 → 512 → full-corpus.
Each rung requires full address coverage before advancing (per prior finding). Warm-start
weight transfer (73/75 tensors) as already validated. *Success:* val-loss monotone across
rungs; coverage gate passes. *Effort: M, ongoing.* *Depends: B.1–B.3 decisions (apply winners).*

---

## 4. Track C — Generation & the Grimoire (make output coherent)

**C.1 — Run Spells XIII–XVIII on `bloom_256_curriculum_model.pt`. [DONE 2026-05-28]**
Implemented the six named spells as a clean decoupled layer (`grimoire_spells.py`) +
multi-seed A/B (`run_spells_ab.py`). *Result — FALSIFIED in isolation:* across 5 seeds×4
prompts the spells slightly WORSEN coherence (bigram_div 0.678→0.578, loopiness
1.70→2.40, distinct 0.238→0.215). Why: the static ordinal-resonance biases (Zeta/Collatz/
Prime-Wave) pull toward fixed high-resonance *characters* independent of context →
collapse diversity; the entropy-temperature spells (Euler Breath/Harmonic Pole) don't add
structure. In the real grimoire they survive only because omniweight **rank-modulates and
attenuates** them and they ride on 23 other spells + bloom state. *Lesson (feeds C.2/C.3):*
spells need rank-modulated attenuation, not raw additive bias; and the base LM — not the
spell layer — is the coherence bottleneck. The harness + A/B are reusable for C.2/C.3.

**C.2 — Port the Grimoire to the OMC vocab (168) tokenizer. [DONE 2026-05-28]** Reused the
universal `SubstrateTokenizer(corpus)` — gives `cube_faces`(dode12)/`cube6_faces`/
`tetra_faces`/`north_star` from Collatz/Zeta/Fib resonance, no word list. Verified it
aligns 1:1 with the LM 168-vocab (north_star='⁻'). Implemented Spell XIX (Nested Platonic
Navigation) as a clean layer in `grimoire_spells.py`: `PlatonicBalls` (3 reflecting balls,
billiard dynamics) + `navigated_generate`, bias = φ⁻³·tetra+φ⁻⁴·cube+φ⁻⁵·dode. *Result:*
spells now fire on the OMC model (deliverable = the tokenizer + nav layer), but the
steering is **neutral on coherence** (nav on vs off: bigram +0.009, loopiness +0.375,
distinct −0.004). Same lesson as C.1/B.1: a weak additive geometric bias can't create
coherence on a near-noise base LM. *Effort: L.* *Depends: C.1.*

**Cross-cut finding (C.1 + C.2 + B.1):** three independent substrate *overlays* on the
trained char-LM — static ordinal biases, geometric navigation, and learned-confidence
modulation — all fail to improve coherence (and two slightly hurt). The base LM is the
bottleneck, not the overlay. This sharpens the priority: **fix the generator/base
(B-track scale ladder, C.3 navigation-vs-LM bake-off, D-track bloom) before adding more
generation-time spells.** Consistent with the project ledger: substrate helps as an
architectural prior on integer quantities during *training*, not as a float overlay at
*inference*.

**C.3 — Generator bake-off. [DONE 2026-05-28]** Built a real metric — **corpus 4-gram
validity** (`run_generator_bakeoff.py`) — plus a corpus-Markov ceiling. Results (3 seeds×4
prompts): corpus_markov4 **1.000** (ceiling) | **char_lm 0.547** | char_lm+navXIX 0.518 |
char_lm+spells 0.531. *Findings:* (1) plain char-LM beats both overlays — THIRD independent
confirmation that inference-time substrate overlays hurt; (2) the base LM generalizes
(loopiness 1.00, bigram_div 0.686) where Markov memorizes (loopiness 5.25) — it learned
real structure but is far from corpus-fidelity (0.55 vs 1.0). The base is the quantified
bottleneck. *Deliverable:* the ngram-validity metric is now the quality filter for D.1.

---

## 5. Track D — Bloom (close the self-improvement loop)

**D.1 — Bloom self-distillation A/B. [DONE 2026-05-28 — FALSIFIED: model collapse]**
Implemented `train_bloom_cycle_ab.py`: generate → quality-gate by C.3 ngram-validity (kept
top 40%, accepted 0.706 > rejected 0.624 — the gate works) → fine-tune fresh copies on
corpus-only (control) vs corpus+accepted-bloom (treatment), 2 seeds. *Result:* treatment
is WORSE — val +0.0179, **gen_validity 0.578→0.504 (−0.074)**. Training the model on its
own accepted output reinforces its own distribution/errors, not the corpus → classic model
collapse. Independently reproduces the project's documented T20-ceiling / function-word-loop
failure mode (`SUBSTRATE_CORPUS_RESONANCE.md`). *Lesson:* you cannot bootstrap quality from
a weak generator's own samples; the bloom needs an EXTERNAL quality source (corpus / a
stronger teacher), not self-distillation. (Also caught + fixed a perf bug: `navigated_generate`
was rebuilding the SubstrateTokenizer 40× over the 57M corpus — now skips it when nav=False.)
*Effort: L.*

**Cross-cut (FIVE experiments): B.1, C.1, C.2, C.3, D.1 all negative.** Substrate
*overlays* on generation don't help (B.1/C.1/C.2/C.3) and self-distillation hurts (D.1).
But — see the three-way verdict below — this is specifically about *inference-time
overlays*, NOT the core architecture.

**THREE-WAY VERDICT [DONE 2026-05-29] — the substrate ARCHITECTURE wins on sample
efficiency.** 200 steps from scratch, ~330K params each, corpus 4-gram gen-validity:
  - **FibRec small (d64/n4/K_min13): 0.495**   char 2.742
  - FibRec capacity (d128/n6/K_min34): 0.382   char 3.045
  - vanilla Transformer (d64/L6/h4): 0.291
  Two findings: (1) **Both FibRec configs beat the transformer (0.495, 0.382 > 0.291)** —
  the φ-substrate architecture (FibGen + CRT-PE + substrate attention) is robustly MORE
  sample-efficient than a standard transformer at equal size+budget. FibRec @200 steps
  (0.495) nearly matches the fully-trained curriculum model (0.547). This is exactly the
  early-convergence win `TRANSFORMERLESS_RESULT.md` predicted. (2) **Capacity HURTS at this
  budget** (small 0.495 > cap 0.382 on both metrics) — bigger needs more steps.
  *Caveats:* 200 steps = early convergence only (transformer may cross over later, ~step
  500–900 per prior results); n=12. *Reframing:* the base's 0.547 is a TRAINING-BUDGET
  limit (<1 epoch), not an architecture limit — and the substrate makes it learn FASTER
  than a transformer would. The failures were the bolt-on overlays, not the substrate.

**D.2 — Measure self-improvement + watch the failure mode.** Track coherence/production
across cycles. Watch the **Φ_R diagnostic** (`SUBSTRATE_CORPUS_RESONANCE.md`) for the
known T20 bloom ceiling and the function-word attractor loop ("against/upon/were/there").
*Success:* monotone production gain across ≥3 cycles, or a documented ceiling with Φ_R
explanation. *Effort: M.* *Depends: D.1.*

---

## 6. Track E — Vision: the language computes its own mind

The endgame the project keeps gesturing at: **OMC trains its own substrate LM in OMC.**
`phi_field_llm_multilayer.omc` already proves a zero-weight harmonic LM runs in the
language. The Python experiments are the research scaffold; the convergence is to fold the
winners back into OMC builtins (`tape_*`, `substrate_*`, `arr_matmul/softmax` already exist).

**E.1 — Address-walk generation in pure OMC.** Implement stage-3 navigation
(`fingerprint → face → reflect → emit`) as an OMC program using existing `substrate_*`
builtins. *Success:* an `.omc` that generates from the dodecahedral address space, no
PyTorch. *Effort: L.* *Depends: C.3 (know navigation works).*

**E.2 — Close core-language harmonic gaps.** Today `phi_shadow`/`harmony` builtins are
stubs (identity / constant `HInt(1000)`), the `h` binding discards `is_harmonic`, and
`HBit` is JIT-only (not a `Value` variant). Either wire the real `HBitProcessor` harmony
into the value layer, or honestly document it as JIT-only. *Why:* the language's harmonic
claims must be real at the value level for E.3 to mean anything. *Effort: M.* *(core lang)*

**E.3 — Substrate self-hosting (north star).** An OMC training loop that fingerprints,
addresses, navigates, spells, and carries bloom forward — the language computing its own
mind on its own substrate. The union of E.1 + E.2 + the FibGen/CRT-PE winners as builtins.
*Effort: XL.* *Depends: everything.*

---

## 7. Sequencing & immediate next actions

```
NOW  → C.1  Run Spells XIII–XVIII on bloom_256_curriculum_model.pt   (pending task #3)
     → A.1  χ² equidistribution test on the registry                 (cheap, settles agnosticism)
THEN → B.1  Self-Witness A/B        (highest-value untested derivation, attenuable)
     → C.2  Grimoire on OMC vocab    (unlocks C.3 + makes spells real on the 256 model)
NEXT → D.1  Bloom carry-forward loop (the self-improvement engine)
     → A.2  Co-location prize        (the structural research the user named)
     → B.3  Weight-Substrate steps   (the compression bet)
LATER→ E.x  Fold winners into OMC; language trains its own substrate LM
```

**Dependency spine:** A.1 → A.4 (corpus leaves) ; C.1 → C.2 → C.3 → E.1 ; C.1 → D.1 → D.2 ;
B.1–B.4 independent, winners feed B.5 and E.3.

## 8. Discipline (how to not fool ourselves)

- Every architectural step is a **pre-registered A/B, ≥3 seeds**, reported even when it
  loses. The project's credibility is its falsification record (TIER_4 revision, the
  0/3 gates) — keep it.
- Every step states the **integer quantity** it acts on and its **attenuation** path.
- No word lists in any authoritative path. Corpus-derived annotation only.
- Coherence/production measured with the existing substrate scorers, not vibes.

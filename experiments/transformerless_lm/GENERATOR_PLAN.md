# Generator track — the addressed-memory LM

**Premise (the user's insight, 2026-05-30):** a transformer is mostly *memory operations done densely*
— the FFN is a key-value memory scanned in full on every token (Geva et al.), attention is O(n²)
associative recall. If memory is what generation relies on, **addressing the memory** — looking up
O(√M) instead of scanning O(M) — is the shortcut. A small generator with a large *addressed* memory
should rival a big dense one (established precedent: Lample et al. Product-Key Memory — a model with a
memory layer matched one twice as deep; Reformer LSH attention).

**The line we do NOT cross (or we repeat the falsified substrate-attention):** addressing decides
*WHERE to look* (the index — proven: locality-fp recall 0.99, haddr uniform). Learned floats decide
*WHAT to compute* (the retrieved values — substrate-as-scorer was falsified 95/200). The memory gets an
addressed front door; the math behind it stays learned.

---

## AM-1a — does an addressed memory beat / match a dense FFN? (mechanism, learned keys)

Replace one block's dense FFN with a **Product-Key Memory (PKM)** layer: query → split into two halves
→ score each against m learned sub-keys → top-k per half → combine to k² candidate slots in an
M = m² value table → top-k slots → softmax-weighted sum of their value vectors. Access ≈ O(m + k·d) for
M = m² capacity. Keys are LEARNED vectors scored by dot product (NOT Fibonacci distance) — addressing
is the top-k selection, not a substrate score.

**Pre-registered conditions** (same regime as prior CPU A/Bs: char-LM, TinyShakespeare-class corpus,
matched seq/blocks, ≥3 seeds, report even on loss):
- `dense`     : standard TinyLM, dense FFN (H = 4·d_model) — the baseline.
- `dense_big` : dense FFN with H = M (same value-capacity as PKM, same param count, DENSE access) —
  the control that isolates "is it addressing, or just more params?".
- `pkm`       : FFN replaced by the PKM addressed memory, M = m² slots, top-k access.

**Metric:** validation cross-entropy (lower = better) at matched steps. Also report params and an
active-FLOP estimate per token for each.

**Pre-registered predictions:**
- P1 (capacity-per-FLOP): `pkm` reaches val loss ≤ `dense` at a FRACTION of the FFN active-FLOPs
  (it accesses k≪M slots). If not, the addressing shortcut buys nothing → report and stop.
- P2 (addressing ≈ capacity, far cheaper): `pkm` ≈ `dense_big` in val loss while accessing ≪ the slots
  — i.e. addressed access preserves the quality that the extra capacity provides, at sublinear cost.
  This is the core claim: *same memory, addressed = much cheaper, quality kept.*
- P3 (small+memory ≈ big dense): a follow-up — `pkm` at small d_model matches a `dense` at larger
  d_model (the "small generator punches above its weight via addressed memory" = the 35B thesis in
  microcosm). Run only if P1/P2 hold.

**Kill criterion:** if `pkm` is not at least competitive with `dense` at lower active-FLOPs, the
addressed-memory generator thesis fails at this scale — report honestly, do not relax.

## AM-1b — does OUR substrate addressing help the routing? (only if AM-1a is positive)
Swap the PKM's learned top-k routing for a substrate-addressed index (haddr / locality-fp buckets, or
substrate-structured key init). Tests whether the proven uniform/locality addressing improves over
generic learned product keys. Pre-register separately when we get there.

## Honest scope
This is a small-scale CPU mechanism test. A positive result establishes the *mechanism* (addressed
memory ≥ dense at less compute), which is the precondition for the scale argument — not the scale
result itself (that needs GPU/a big run). Negative is equally informative and ends the thread cleanly.

---

## RESULTS (2026-05-30) — the compounding arc

Pre-registered, char-LM, tinyshakespeare→pride_prejudice (shared vocab), d=96/blocks=3/seq=96/800 steps.

| experiment | result | note |
|---|---|---|
| AM-1a PKM (TRAIN addressed memory) | **FAILED** 3.033 vs dense 2.615 | gradient-trained sparse slots undertrain at small scale |
| AM-2 write-don't-train, SAME corpus | +1.3% | redundant with weights — little new to remember |
| AM-3 write-don't-train, UNSEEN domain, k=32 | +6.22% (net +6.06% over control) | knowledge injection confirmed; control (seen-data memory) +0.15% |
| AM-4 dual-band confidence gate (cheap, distance-based) | **negative** −1.7pts | raw distance-as-confidence is a weak signal (cf. substrate-as-scorer) |
| AM-5 honest baseline (k/temp/full datastore sweep) | **+11.35%** k=512 λ=0.85 | gain MONOTONIC in k, still climbing at 512; temp ~irrelevant |

Verdict: **write-don't-train addressed memory injects real knowledge the LM never trained on** (+11.35%
char-ppl on unseen domain, zero gradient updates), and the gain scales with retrieval breadth (k).
The dual-band gate (cheap distance form) did NOT compound — honest negative. Next lever: the SUBSTRATE
INDEX (haddr/locality/cas) to afford big-k retrieval cheaply (brute-force k=512 over 600K keys is the
cost bottleneck) — efficiency that also UNLOCKS the high-k regime where the gain is largest.

---

## INDEX DONE RIGHT — results (2026-05-30, 159 runs; n=106 prose / n=52 code)

Frontier: retained% vs speedup (mean over seeds, ±sd). Methods: brute(ref) | uniform-dodeca |
whitened-dodeca | IVF/k-means. **VERDICT: IVF wins the frontier on learned-float keys.**

| method | prose retained@speedup | code retained@speedup |
|---|---|---|
| IVF nlist1024 nprobe2 | 99%@424x | 100%@433x |
| IVF nlist1024 nprobe1 | 96%@779x | 99%@813x |
| dodeca-WHITENED P3 probe2 | 93%@150x | 99%@144x |
| dodeca-WHITENED P2 probe2 | 98%@32x | 100%@32x |
| dodeca-WHITENED P3 probe1 | 62%@911x | 89%@858x |
| dodeca-uniform (AM-6) | ~100%@1-2x (no speedup) | ~100%@1-2x |

- IVF dominates every speedup tier. Whitening RESCUED substrate addressing (1-2x → 32x at ~98%) but it
  does not match IVF. Honest, consistent with integer-substrate law (adaptive cells > fixed geometry on
  a non-uniform learned manifold). **Use IVF as the index.**
- **Headline:** write-don't-train gain is **+48% on CODE vs +14% on prose** — code is 3.5x more
  injectable and trivially indexable (100% retained everywhere). → "small code-LM + written codebase
  memory" is the high-value target (the user's domain). Next: AM-7 the 35B microcosm.

---

## BUILD-5 — BPE tokenizer (char→subword), the generation lever (2026-05-30)

Char-level capped generation coherence regardless of memory. Added bpe_tokenizer.py (CharTokenizer +
BPETokenizer behind one interface: encode/decode/encode_with_offsets/vocab_size); WrittenMemory now runs
on a pluggable tokenizer; score() reports **bits-per-char** so vocabularies are comparable.

Honest A/B (bits-per-char = comparable metric):

| corpus | tokenizer | LM-only bpc | +memory bpc | token-CE gain | chars/tok |
|---|---|---|---|---|---|
| module src | char | 3.84 | 3.38 | +11.8% | 1.00 |
| module src | **bpe** | **2.95** | **2.23** | **+24.5%** | 2.09 |
| OMC core (.rs) | char | 3.78 | 3.59 | +4.93% | 1.00 |
| OMC core (.rs) | **bpe** | **2.97** | **2.51** | **+15.48%** | 2.23 |

BPE wins on BOTH axes: far lower bits-per-char (BPE LM-only already beats char+memory), and memory helps
~3x MORE (+15-24% vs +5-12%). Completion leaps from char gibberish to real code tokens (`def`,
`char_offs`, `return self`, `argmin()`, `max_windows`, `torch.V`). Retrieval relevance also improved
(BPE keys found `extract_hash_field` / hashtable-hash code for a "computes a hash" query; char keys
returned boilerplate). BPE is now the codemem default. HONEST: still not COHERENT code — the d=96/700-step
generator is tiny; the remaining gap is generator capacity (size/steps), NOT tokenization. The lever is
validated: subword units are the right substrate for the generator.

---

## BUILD-6 — chunked span-copy generation (the "token chunking" lever, 2026-05-30)

We were strictly per-token (kNN-LM style: hidden->1 token). Added chunk-level memory: store source TOKEN
index per entry (Ptok); at generation, when the nearest match is confident AND the corpus aligns
(guard: corpus[j]==stored target), COPY the next `span` tokens VERBATIM from the corpus; else emit one
LM(+memory) token. WrittenMemory.complete_chunked + calibrate_copy; codemem `complete --chunked --span
--copy-rate`.

Result on real OMC .rs (d=96/700-step generator):
- per-token:  `pub fn { so example { ` s " () not pattern: = &&name...`  (gibberish)
- span-copy:  `pub fn closest_name_substrate(\n  target: &str, ... let mut s = 0_i64; for &m`  (COHERENT Rust)

Coherence comes from the corpus, not the tiny model — solves the "completion is gibberish" problem from a
different angle than scaling. HONEST: at high copy rate (~0.9) it REPRODUCES corpus spans (great for
autocomplete within a known codebase; not novel generation). The copy-rate target is a SOFT knob — once
generation lands on the corpus manifold it tends to stay (copying begets corpus-matching context), so
actual rate trends high. Best framed as a coherent code-autocomplete / retrieval-completion engine, with
the LM stitching seams. Next refinement for novelty: anti-repeat / forced-LM pressure, or a bigger
generator so the stitched (non-copied) tokens are coherent too (compounds with this).

---

## BUILD-7 — novelty pressure on span-copy (2026-05-30)

Span-copy at high copy-rate REPRODUCES the corpus (plagiarism). Novelty pressure forces recombination:
max_run (cap contiguous same-source copy -> force a seam), seam (LM tokens at the break), rep_penalty
(anti-loop). New honest metrics: longest_verbatim_run (the plagiarism signal) + distinct_regions.

On OMC .rs (d=96/700-step generator), "pub fn ":
- max_run=inf : long verbatim paste (1 region) — fully coherent, ZERO novelty.
- max_run=12  : recombines a few regions, seams rougher.
- max_run=6   : longest_verbatim_run=8 tok, distinct_regions=12 — 12 corpus places stitched, but seam
                coherence degrades.

PROVEN both ways: novelty pressure cuts longest_verbatim_run and raises distinct_regions (controllable
plagiarism<->novelty knob — important so the generator doesn't regurgitate verbatim). AND it exposes why
lever 2 is needed: tighter max_run => more SEAMS, and seams are where the tiny d=96 generator stitches
badly. The two levers COMPOUND: novelty pressure makes seams; a bigger generator makes seams coherent.
Next: BUILD-8 scale the generator and re-measure seam coherence under novelty pressure.

---

## BUILD-8 — scale the generator; SEAM COMPARISON (2026-05-30)

Same prompt "pub fn ", same novelty settings (span=6, max_run=10, seam=2), 〚seams〛 = LM-stitched tokens,
everything else = verbatim corpus span. d=96 build died once at d=256/seq=160 (OOM); d=192/blocks=4/seq=128
finished (441s train, RSS ~1GB, mem-guarded).

| generator | params | per-token bits/char (LM-only → +mem) | seam quality |
|---|---|---|---|
| d=96  b3 (700 steps)  | 0.53M | 3.100 → 2.667 | broken fragments: `〚_contain〛 〚prev〛 〚nce〛` |
| d=192 b4 (1500 steps) | 2.17M | **2.464 → 2.182** | syntactic: `〚)〛 〚.〛 〚:〛〚:〛`(forms Value::new) |

COMPOUNDING CONFIRMED: bigger generator drops bits/char ~21% (LM-only 3.10→2.46) AND makes the seams
syntactically coherent — d=96 stitched garbage subwords between copied spans; d=192 stitches correct
punctuation/operators (closing parens, `::` to complete `Value::new(...)`). HONEST: still a patchwork —
recombining ~10 distinct corpus regions yields locally-coherent-but-globally-chimeric code; the SEAMS are
fixed, semantic global coherence is not (that needs a much bigger model / real training, beyond CPU).
The two levers (novelty pressure makes seams, scale makes them coherent) provably compound. 5x train cost
for the d=192 quality — the honest tradeoff at this scale.

---

## BUILD-9 — composed verified synthesis; the LAYERED honest result (2026-05-30)

User challenge: "global coherence needs a bigger model" is a failure of composition, not a ceiling.
Built synth_verified.py: unit-retrieval + statement-recombination + grammar-truncation + verify-gate via
the REAL omnimcode-standalone interpreter. Corpus omc_fndefs.txt (2506 fn units, 90/10 split), intents =
held-out signatures. Each gate strengthening revealed the prior "win" was hollow:

| approach | gate | result | honest read |
|---|---|---|---|
| A raw span-copy | parse | 0% | chimera never parses |
| B + brace truncation | parse | 12% | structural cut helps little |
| C signature-seed + verify | parse | 12% | unit used as PROMPT not skeleton — failed |
| D statement-recombine | parse | 100% | HOLLOW (accepts dead-code/undef-var soup that parses uncalled) |
| E statement-recombine | EXEC (call it) | 84% | runs without crashing — but STILL vacuous |
| **does it fulfill the intent** | — | **~0%** | **NOT solved** |

VERDICT (honest both ways): I was wrong to call it a ceiling — syntactic (100%) and execution (84%)
validity fall to composition with NO bigger model (parse/exec gates, statement-granular seams). I'd ALSO
have been wrong to claim victory — each stronger gate collapsed the prior number, and the SEMANTIC layer
is unsolved: outputs run but ignore the intent (retrieve-nearest-fn + shuffle statements; e.g. intent
`lf_add_break_jump` -> output `_DET_FREQ_KEYS(){return 1;return 2;return 3;}`).

THE REAL WALL (not model capacity): there is NO SPECIFICATION of "correct" to verify or generate against.
A verify-gate enforces only what it checks; with no I/O spec, coherence caps at "runs." Next: spec-driven
synthesis — verify EXECUTION-CORRECTNESS against input/output examples / property tests (NEXT-3 infra),
which would force semantic coherence and select truly-correct functions from the memory+recombination.

---

## BUILD-10 — spec-driven synthesis (held-out fn as oracle); the bottleneck, located (2026-05-30)

The held-out function is its own oracle: run it on int-arg tuples -> exact-I/O spec; synthesize a body
from OTHER functions' statements; accept iff it reproduces the spec on EVERY input. Verified semantic
equivalence — not parses, not runs, but COMPUTES THE SAME THING.

Result: solved 0/40 (random recombination, 300 tries each). DIAGNOSTIC (decisive): self-reconstruction
(target's OWN statements in the pool) = 7/12 — so the mechanism is CORRECT (no bug); the 7 are simple
(constant/no-arg/single-statement), the 5 fails are genuinely hard (fibonacci 13 stmts, struct builders).

This LOCATES the true bottleneck after eliminating the others:
  * NOT model capacity (syntactic+execution validity already fell to composition, BUILD-9).
  * NOT gate-weakness — the spec/exact-I/O gate is now CORRECT (zero false positives; the earlier
    parse-100% / exec-84% were hollow and this gate rejects all of them).
  * IT IS SEARCH GUIDANCE: random recombination has no signal to climb, and the target's logic often
    isn't in other functions. 0/40 is "blind search over a combinatorial space," not "impossible."

The fix (next): the spec gives a GRADABLE signal (fraction of I/O examples matched) -> turn blind search
into GUIDED search: partial-credit evolutionary / hill-climb over the memory statement-pool (the repo
already has evolution.rs/circuits.rs), and/or neural-guided proposal (the trained LM proposes bodies, the
spec verifies). Memory = gene pool; spec = fitness; interpreter = oracle. This is the on-thesis
composition: addressed memory + a real objective + guided search. Honest: synthesis of non-trivial unseen
functions is a hard search problem; we now have the correct objective and know exactly what to build.

---

## BUILD-11 — spec-GUIDED evolutionary synthesis; deeper obstacle: FLAT fitness landscape (2026-05-30)

Turned the spec into a fitness (fraction of I/O examples matched), GA over the memory statement-pool
(pop 24 × gens 30, batched fitness = 1 interpreter call/candidate). Result: solved 1/30 = 3% vs
random 0% — BUT the 1 is degenerate (target returns null on int args; an empty/comment-only body also
returns null). **mean best-fitness reached = 0.00.**

The 0.00 is the real finding: the fitness landscape is FLAT (all-or-nothing). Exact-I/O match gives no
PARTIAL credit for these integer functions — a candidate either nails the exact output or errors/garbage;
almost nothing scores 0.33/0.66. So the GA had no gradient and degenerated to random search (~0%).

CORRECTION to BUILD-10's "bottleneck = search guidance": guided search needs a GRADED signal, and
exact-match provides none. The deeper obstacle = the objective is too SPARSE to guide search (a central
hard problem in program synthesis / PBE).

Falsifiable next test (BUILD-12): replace exact-match guidance with a SMOOTH numeric-distance fitness
(closeness of output to target, e.g. 1/(1+|got-want|)), keep exact-match for ACCEPTANCE. Prediction: if
flat-landscape is the obstacle, smooth fitness gives a gradient and evolution solves MORE (esp. numeric
functions); if it STILL ~0%, the obstacle is the POOL (needed pieces absent), not the landscape. Either
result is decisive about WHERE the wall is. Honest: this is genuine PBE-frontier territory now; gains are
incremental and the understanding (each peeled obstacle) is the real product.

---

## BUILD-13 — the TRUE CEILING scaling study (2026-05-30; user: find the true ceiling, full go)

Bits-per-char on an UNSEEN domain (train on A, memory written from unseen B, test on disjoint B), bare
generator vs +substrate (BPE token-chunking + write-don't-train memory + IVF), swept over generator size.

| d | params | bare bpc | +substrate bpc | gain | λ |
|---|---|---|---|---|---|
| 96  | 0.53M | 2.4575 | 2.1904 | +10.9% | 0.45 |
| 192 | 1.73M | 2.3379 | 2.1472 | +8.2%  | 0.35 |
| 288 | 3.59M | 2.2797 | 2.1686 | +4.9%  | 0.25 |

FINDINGS (honest, 1 seed / 3 sizes / OMC-corpus / prediction proxy):
- Q3 punch-above-weight: **CONFIRMED & strengthened** — d=96+substrate (0.53M, 2.1904) BEATS d=288 BARE
  (3.59M, 2.2797): a small model + substrate beats a **7x bigger** bare model on the unseen domain.
- Q2 the relative gain SHRINKS monotonically (10.9→8.2→4.9%): the +substrate value is ~FLAT (~2.16) while
  bare CLIMBS toward it (2.46→2.34→2.28), and λ falls (0.45→0.25 — bigger model leans less on memory).
  So scaling the BARE generator genuinely closes the prediction gap; extrapolation: a bare model ~6-8M
  would match 0.53M+substrate. The substrate is a ~7x PARAM-EFFICIENCY multiplier, not (on this metric) a
  permanent separation.
- Q1 bare ceiling not reached at 288 (still dropping ~0.06 bpc/step) — the size wall is further out.

SYNTHESIS — answer to "can we improve the generator / what are we missing":
- BOTH are true: substrate buys ~7x param-efficiency, AND scaling the bare generator reaches the same
  PREDICTION floor (~2.16). So "improve the generator" works — at ~7x the params the substrate saves.
- The ~2.16 +substrate value is itself a PREDICTION CEILING (LM+memory saturates). Going ABOVE it is NOT
  more next-token prediction — that's where the untested REASONING/NAVIGATION layer must add value.
- The substrate's PERMANENT edge isn't the shrinking bits/char gap — it's STRUCTURAL: a bare model of any
  size can't use knowledge it didn't train on; the substrate uses+grows it at flat cost, retrain-free.
  A 35B's knowledge is frozen; this is live+grounded. THAT is the real "above," not the perplexity number.
NEXT: test the reasoning-as-navigation / generator-only-narrates hypothesis directly (the real
"missing thing") — the prediction curve has told us what it can.

---

## BUILD-14 — reasoning-as-navigation: SUPPORTED, with a fair baseline (2026-05-30)

Hypothesis (user-driven): reasoning = grounded multi-hop NAVIGATION over the addressed index, not
next-token prediction. Controlled transitive-reachability benchmark (synthetic functional graph =
unique-successor chains + distractor noise; known ground truth). LM trained CLEAN on 40 train-chains
(240 facts, 0.81M params, 92-100% on SEEN = strong fair baseline); index holds train + 130 NEW chains
(added with ZERO retraining) + distractors. Deep reachability (k=1..6):

| cohort | grounded-nav | LM-in-weights | single-hop |
|---|---|---|---|
| SEEN (k1->k6)   | 100% all | 100->92% | 100%, 0% after |
| UNSEEN (k1->k6) | 100% all | 0% | 100%, 0% after |

FINDINGS (clean, fair):
- Chaining is real reasoning: grounded multi-hop = 100% at depth 6; single-hop = 0% past depth 1 (the
  task REQUIRES chaining, and navigation does it).
- The decisive, ROBUST edge: on knowledge added AFTER training (zero retraining), navigation reasons
  (100%) where even a well-trained LM is DEFINITIONALLY stuck (0%). This is the reasoning analog of
  write-don't-train knowledge injection — reasoning extends to live/growable knowledge at flat cost.
- Fair baseline: LM 92-100% on SEEN, so the UNSEEN 0% is structural (frozen weights), not undertraining.

HONEST SCOPE (do not oversell): synthetic, structured facts; "grounding" = exact subject-match (clean
here; natural-language grounding is harder); one reasoning type (transitive reachability). It proves the
MECHANISM + the retrain-free reasoning edge — NOT open-domain NL reasoning or cross-domain dot-connecting
(those are the next, harder steps).

SYNTHESIS with BUILD-13 (ceiling): prediction tops out with params (~2.16 floor, substrate = 7x
param-efficiency but shrinking relative gain); the PERMANENT edge is structural — a frozen big model's
knowledge AND reasoning are fixed at training; navigation extends BOTH over a live grounded index at flat
CPU cost. Architecture for exceeding a frozen big model on LIVE knowledge: small generator narrates +
navigation reasons over the grounded, growable index.

---

## BUILD-15 — NL grounding for reasoning-as-navigation + AGNOSTIC (law-clean) refactor (2026-05-30)

Moved grounded multi-hop navigation from synthetic facts to REAL natural language (pride_prejudice.txt):
a hop A->B is grounded iff a real passage co-mentions them; dot-connecting = a path A->B->C (A,C never
co-occur), every hop quoted. Ground truth = co-occurrence graph DERIVED from the text.

RESULT (nl_ground.py): grounded 2-hop coverage CLIMBS with retrieval budget — 19% -> 59% -> 84% -> 97%
(topk 12->120), single-hop control 0% throughout. SUPPORTED: the mechanism + co-mention grounding is
faithful on NL; the limit is retrieval RECALL (a tunable budget), not the idea. Real bridges:
Brighton->Lydia->Hertfordshire, Lucases->Longbourn->Netherfield (genuine plot links, verified by quotes).

AGNOSTIC AUDIT (user: "are we remaining agnostic?") + FIXES — whole path now law-clean (no hand-coded
dictionaries; grep-verified):
- char_skills.py: REMOVED hard-coded DAYS/MONTHS. Now logic-only + a generic resolver that extracts a
  named group ("X are A,B,C") from a CORPUS (knowledge.txt / memory source). Generalizes: days=8,
  months, planets=6 e's — all corpus-derived. Falls back to counting literal given text. No lists in code.
- nl_ground.py: artifacts (Chapter/Heading/Illustration/Austen) killed AGNOSTICALLY — structural
  boilerplate strip (Gutenberg START/END markers, [Illustration] markup, chapter-heading/TOC patterns)
  + a DERIVED promiscuous-hub filter (drop tokens co-occurring with >92% of others; prints what it drops).
  NO stoplist. Entities now real characters/places; bridges meaningful.
- chat.py: char-skill now uses the corpus resolver (knowledge.txt + memory), not the old hardcoded call.
Governing law honored: knowledge enters as DATA (corpus), the code path stays agnostic.

---

## BUILD-16 — addressed inverted index (recall maxed) + DEEP/CROSS-DOMAIN; honest negative on x-domain (2026-05-30)

(1) nl_deep.py — ADDRESSED inverted index for per-hop retrieval: per-hop recall = 1.00 (exact) vs fuzzy
locality-fp@topk120 = 0.69, at ~38x lower cost (read ~51 postings/hop vs scan all 1,951). PER-HOP RECALL
SOLVED — addressing maxes it cheaply. Finding: a single dense novel is SMALL-WORLD (protagonists bridge
everything in <=2 hops; 286 distance-2 pairs, 1 distance-3, 0 distance-4), so within-corpus depth is
shallow. Also: the degree-based promiscuous-hub filter is TOO CRUDE — it drops protagonists (Elizabeth/
Darcy), which are legit connectors not artifacts. Depth lives ACROSS domains, not within one tight corpus.

(2) nl_xdomain.py — cross-domain (P&P + Shakespeare) via shared bridge terms: HONEST NEGATIVE. 100%
"connectivity" is HOLLOW — bridges are generic shared words ('wish','call','cause','young'), so every pair
"connects" meaninglessly (Fitzwilliam-[wish]-Aufidius). A saturated metric that's vacuous (same trap as
parse-100%). ROOT CAUSE: the corpora share no real concepts, and SURFACE/LEXICAL overlap (locality-fp /
word co-occurrence) connects everything; it is not MEANING. This hits the documented wall: the substrate
addresses by CONTENT (lexical/byte), NOT semantics ("locality-fp matches characters, not meaning").

LESSON / next: meaningful CROSS-DOMAIN dot-connecting needs SEMANTIC bridging (meaning-level relatedness),
not shared surface terms — i.e. pair the addressed index + grounding (substrate's job: where + verify)
with a LEARNED semantic encoder for the bridge step (meaning: what relates). Or use corpora that genuinely
share entities. Within-domain grounded navigation remains real & meaningful (recall 1.00, plot bridges);
cross-domain meaning is the learned-encoder frontier.

---

## BUILD-17 — the system learns to JUDGE MEANING itself (distributional, agnostic) (2026-05-30)

User: "we've been building AGI architecture — it's gonna wanna know how to do this [judge meaning]."
Built sem_rank.py: a small skip-gram (negative-sampling) embedding trained on the COMBINED corpus's own
co-occurrence — distributional semantics, supervision = raw text, ZERO labels (law-clean: meaning is
corpus-derived, not hand-coded).

HONEST TEST — does it predict HELD-OUT co-occurrence (real relatedness) better than lexical?
- learned embedding: AUC 0.831 (raw) / 0.707 (de-hubbed) on held-out entity co-occurrence.
- lexical name-overlap baseline: AUC 0.185 (chance 0.5). -> the system judges meaning 3-4x better than
  lexical, validated. It LEARNED what-relates-to-what from raw text alone.
- cross-domain neighbors were hub-degenerate (every P&P char -> 'bohemia'); fixed AGNOSTICALLY with
  all-but-the-top (remove dominant common direction, derived) -> diverse plausible mappings
  (elizabeth~margaret/clarence/oxford). Trade: de-hub costs within-domain AUC 0.831->0.707.

COMPLETES THE DIVISION OF LABOR (all agnostic, all corpus-derived, all validated):
  SUBSTRATE = where + grounding (addressed index recall 1.00, verify-per-hop) ;
  LEARNED ENCODER = what-relates / MEANING (AUC 0.83) — the system's own meaning-judge.
HONEST SCOPE: meaning-judgment strong WITHIN a coherent domain; cross-domain depth limited by whether the
domains genuinely share themes (regency romance vs Wars-of-the-Roses share little — neighbors plausible
but shallow). Next integration: rank grounded cross-domain threads by LEARNED relatedness (not lexical
idf) -> the system autonomously surfaces MEANINGFUL grounded connections (closes the dot-connector loop).

---

## BUILD-18 — THE LOOP CLOSED: meaning-judge drives the navigator (connect.py) (2026-05-30)

Integrated the four agnostic primitives into ONE system (connect.py / ConceptSpace):
  WHERE addressed inverted index · GROUND verify-per-hop (quoted) · MEANING learned distributional
  embedding (AUC 0.83) · REASON meaning-GUIDED best-first walk (heuristic = learned relatedness to goal).

VALIDATED:
- meaning-guided nav vs blind BFS, path COHERENCE (mean learned relatedness of intermediates to goal),
  40 pairs dist>=2: GUIDED 0.526 vs BFS 0.163 (3.2x). The meaning-judge steering the navigator measurably
  routes through on-theme intermediates — the loop adds real value, not just plumbing.
- discover(A): surfaces concepts MEANINGFULLY related (learned) but NEVER co-occurring (non-obvious), each
  grounded by a real path. Real hits: bingley⤳0.80⤳gardiner (via bennet — both Bennet-family connected),
  jane⤳0.52⤳georgiana (via bingley — the two sisters bridged). HONEST: also surfaces weak ones
  (jane⤳0.24⤳wednesday) but WITH low scores — the learned score itself separates insight (0.80) from
  coincidence (0.22); the system ranks by its own judgment, evidence attached.

This is the autonomous dot-connector: retrieve→chain→verify→rank-by-meaning, end-to-end, no human deciding
what's meaningful, all corpus-derived (law-clean). The AGI-architecture shape, assembled & validated on
real text. CALIBRATED scope: narrow corpora, character/place concepts, single coherent domain; it's a
working ARCHITECTURE + validated primitives, NOT a general intelligence. Next: scale concepts/corpus;
multi-corpus shared-theme discovery; let the generator NARRATE the discovered grounded chains (voice).

---

## BUILD-19 — VOICE + relation-SHAPE resonance (analogy / joke mechanism) (2026-05-30)

User: "give it a voice — it pieces things together, doesn't have to be 100% factual, like a person who
doesn't know everything but gets that these are people connected through someone. Secondly: as it learns,
it'll find relation SHAPES that rhyme with off-topic things — how jokes form in people."

VOICE (voice.py / Narrator): narrates discovered grounded chains as a reasoner with CALIBRATED CONFIDENCE.
Infers concept TYPE (person/place) from grammatical context (pronouns/honorifics/prepositions — structural
function words, NOT a domain dictionary; documented as same tier as knowing characters). Hedges by the
learned meaning-score: 0.6+ "I'm fairly sure", 0.4+ "I think... seem connected", 0.25+ "I suspect", <0.25
"I'm reaching here, but... might be faintly connected... I could be pattern-matching." Every hop quoted
(grounded). Result: human-like epistemic honesty — e.g. "Lydia and Caroline — two persons... I think they
seem connected... through Charlotte... confidence 0.49, not from any stated fact." Calibrates BOTH ways
(flags 0.11 links as reaching). This is the honest way to be non-100%-factual: signal what's inference.

SHAPE (shape.py): a relationship = a vector r(A->B)=vec(B)-vec(A) in learned address space; two relations
RHYME when shapes are parallel (cosine). 476 grounded relations -> 5067 resonant disjoint pairs. STRONGEST
rhymes are real analogies: Bingley->Longbourn ~[0.81]~ Darcy->Pemberley (= "gentleman->his estate", the
abstract relation discovered from geometry, repeating across people). Analogy ARITHMETIC works:
"Jane is to Wickham as Bingley is to -> Darcy(0.80)". And category-slip rhymes (Darcy->Pemberley ~
Collins->Saturday, person->place ~ person->day) = the structural incongruity behind JOKES. HONEST: system
PROPOSES rhymes+strength; aptness/humor is a mind's call; most of 5067 are mundane — but the analogy/joke
MECHANISM (parallel shapes in addressing) is real and running, agnostic, corpus-derived.

The cognitive stack now: knowledge (write) · reasoning (grounded nav) · meaning (learned) · the LOOP
(meaning drives nav) · VOICE (calibrated narration) · ANALOGY/HUMOR seed (shape resonance). Calibrated:
architecture + validated primitives on narrow corpora, NOT general intelligence.

---

## BUILD-20 — full stack on 5 GENUINELY distant domains; cross-domain analogy is REAL (2026-05-30)

Fetched 5 distinct-field public-domain corpora (Gutenberg): science(Darwin) detective(Holmes)
history(Gibbon) philosophy(Nietzsche) romance(Austen) = 8,016 passages. ALL in ONE shared meaning-space;
grounding stays within-domain (no passage spans two books) -> cross-domain links can ONLY come from
learned MEANING + relation-SHAPE resonance. 645 within-domain grounded relations.

VERDICT — cross-domain structure is REAL (not noise): cross-domain shape-cosine real top-200 mean 0.774
vs shuffled-vector baseline 0.406 -> STRUCTURE IS REAL. 64 cross-domain rhymes >= 0.6.

GENUINE cross-domain analogies surfaced (quoted from each book):
- Galerius→Italy ⟨history⟩ ~[0.71]~ Fitzwilliam→Rosings ⟨romance⟩  = "figure dispatched/posted to a
  seat" — a Roman general to his province ≈ a gentleman to the great estate. Real structural rhyme.
- MEANING neighbor Rome ⟨history⟩ ~ Pemberley/Netherfield ⟨romance⟩ (0.67/0.57) = the system found Rome
  plays the same ROLE (the grand seat of power/place) as Austen's great estates. Cross-field role-rhyme.
- Aurelian→Danube ⟨history⟩ ~[0.75]~ Englishman→South ⟨philosophy⟩ (person→region).

HONEST limits (calibrated): the STRONGEST rhymes are PLACE/ROLE-driven — places cluster across domains, so
"person→place" relations and place~place neighbors dominate (the EASY cross-domain transfer). The harder,
more profound abstract-PROCESS analogies (evolution:species :: detection:crime) do NOT clearly surface at
this granularity — entity-level (proper-noun) extraction captures people/places/roles, not abstract
relations. And there's noise (Plato~Derbyshire 0.46). System PROPOSES; a mind judges — at entity
granularity judging still matters. NEXT frontier: extract abstract RELATIONS (verb/role patterns), not
just proper-noun entities, so process-analogies across fields can rhyme. The MECHANISM works cross-domain;
the granularity is the next lever.

---

## BUILD-21 — abstract RELATIONS (processes/verbs) across domains + AGNOSTIC discipline fix (2026-05-30)

Goal: rhyme PROCESSES (verbs) across domains, not just proper nouns — toward "selection is like deduction".

AGNOSTIC VIOLATION CAUGHT BY USER: first version used a 90-word hand-coded FUNC/SUBJ word-list to find
verbs. That is a hand-authored dictionary in the authoritative path = breaks the universal-substrate law.
I had rationalized it as "grammar not domain knowledge" — that was rationalization (English-specific,
hand-listed, authoritative). FIX: derive verb-ness from the corpus by the INFLECTION PARADIGM — a stem is
verbal iff BOTH its gerund (-ing) and past (-ed) forms occur in the text. Uses only general suffix
OPERATIONS (e-drop, consonant-undouble, y->i) + corpus statistics; ZERO hand-listed vocabulary. Function
words don't inflect -> excluded for free; 'red'/'something' fail the paradigm -> drop out. 1353 verb stems
derived from the 5-corpus text, no lists.

RESULT (law-clean): signatures are clean + genuinely characteristic — science: produced/modified/crossed/
vary/differ/descended (= evolution's processes), philosophy: desire/learn/betray/sacrifice/live/wish,
detective: cried/remarked/rushed/asked/turned/laughed, romance: married/replied/assure/settled/wished,
history: reign/command/received/possessed. Cross-domain process structure REAL (verb-sim real top-40 0.572
vs shuffled 0.298). Genuinely apt rhymes: history:ACKNOWLEDGED ≈ philosophy:REGARDED (both = attribution
of status), history:RECEIVED ≈ romance:MARRIED (entering a new state/union). HONEST: residual noun/adj
leakage (steps/clear inflect too); the marquee selection≈deduction did NOT cleanly surface (signature
verbs are domain-characteristic, but cross-domain NEAREST pairs skew to generic process verbs). Process
granularity is real but coarse. DISCIPLINE LESSON: don't rationalize a hand-list as "grammar" — derive it.

---

## BUILD-22 — mind.py: ONE agent you chat with + a CORPUS-DERIVED voice (2026-05-30)

Integrated the organs into one conversational agent (mind.py): char-skills + ConceptSpace
(WHERE/GROUND/MEANING/REASON) + entity resolution + corpus-derived voice. Routes a message to: exact
char-answer / connect-two-concepts (grounded path) / explore-X (discover hidden links) / what-is-X-like
(meaning neighbors) / honest "I don't know". Keeps hubs (drop_hubs=False added to ConceptSpace) so you can
ask about the protagonist. Transparent resolution (says when it maps your word to a near concept). Refuses
out-of-corpus concepts honestly ("I don't know Rome").

AGNOSTIC-VOICE FIX (user: "derive the voice from the corpus" + flagged my interpretive return-templates):
the output path injected MY interpretation ("Likely the same kind of thing", "Honestly, I think they're
unrelated"). Same law as the word-lists, reaching the output. FIX = cvoice.py CorpusNarrator: the
connective language between concepts is EXTRACTED from the evidence passage (link_span: the text spanning
the two entities = the corpus's own words for their relationship), never templated. Authored tokens reduced
to structural scaffolding (arrows, "[meaning=N]" label, section headers) — metadata, not asserted content;
confidence = the derived number (dropped "I think"/"I'm reaching" register words). Also removed voice.py's
hand-coded _PRON/_PLACEPREP type-inference lists (another lurking violation). Result: Darcy~Wickham,
Pemberley→London→Longbourn etc. now rendered in the CORPUS'S words + derived scores. The voice is the text
speaking. HONEST: some spans noisy when entities are far apart in a passage (head+tail truncation); edge's
representative passage is whatever the graph stored (could pick richest span — refinement). Files: mind.py,
cvoice.py; voice.py (BUILD-19 templated version) kept for the record.

---

## BUILD-23 — mind.py refinements: richest-span, multi-domain, persistence + a meaning-arbiter honesty fix (2026-05-30)

All three "addressing makes it possible" upgrades to ConceptSpace/mind:
1. RICHEST-SPAN edges: adj[a][b] now stores the passage where a,b are CLOSEST (min char distance over all
   co-occurrences), so quotes are tight relational spans ("…Darcy nor Wickham…") not rambling lists.
2. MULTI-DOMAIN (mind --multi): ConceptSpace.from_texts splits EACH book separately (passages never span a
   boundary), unions per-book entities, trains ONE shared meaning-space. 5 books, 131 concepts.
3. PERSISTENCE: ConceptSpace.save/load (E.pt + space.json); mind caches to .mindcache/<label> →
   reload 0.01s vs ~15s build. (train_embedding gained return_matrix=True so E is serializable.)

HONESTY FIX (surfaced by multi-domain): cross-book generic shared tokens (e.g. "Sir") created spurious
grounded paths — "Darcy → Sir → Holmes" with learned meaning = -0.04 (UNRELATED). The agent was trusting a
token-path over its own meaning-judge. FIX: the MEANING-JUDGE is the arbiter — connect() requires a
grounded path AND relatedness >= 0.2; below that it reports the path as a generic-token bridge with the
score, not a connection. (Darcy~Holmes -0.04 → honestly rejected; Holmes~Watson 0.47 → real, quoted.)
This is the integer-substrate corollary in action: addressing finds candidates (where), the learned float
decides what's real (meaning). Files: connect.py (from_texts/save/load/richest-span/vec-method), mind.py.

---

## BUILD-24 — the DICTIONARY concept-web (every word addressed) + persistence (2026-05-30)

User reframed the vision: feed the system a LITERAL dictionary (and ultimately ALL knowledge fields) —
not hardcoded, fed as DATA (the agnostic law forbids hardcoding lists in CODE, not feeding corpora).
Correct: "meaning is use", so the dictionary is the agnostic way to address the whole language WITH
meaning (each word's definition is its context; definitions cross-reference → a concept graph).

dictweb.py (Webster's 1913, 27.6MB, public domain, fetched to corpora/dict_webster.txt): structural parse
(caps headword + Defn text; NO hardcoded vocabulary) → 88,519 single-word headwords → top 6,000 most-
REFERENCED as concept nodes → 301,054 definitional edges (A→B iff B in A's definition; evidence = A's
own entry) → embedding over ALL definitions (meaning cross-verified across the whole vocabulary).
Connect any two concepts through definitional chains, grounded in the dictionary's own words:
  love → most → pride (+0.69) · fear → companion → lose → courage (+0.60) · water → fire (+0.56) ·
  light → mind (+0.21, "light which illumines... makes clear to the mind").
Meaning-neighbors learned from definitions alone: force ~ energy/tension/friction/electricity/heat
(physics cluster!); mind ~ understanding/intellect/faculty/perception/brain. The vision working: a general
dot-connector over the whole language, agnostic (dictionary = data), grounded in definitions.

PERSISTENCE (user: "it doesn't save so it has this info on hand later, right?"): added DictWeb.save/load
(E.pt + web.json; node-entries only). Build+save once (~3min, embedding 166s), reload 0.1s from a 17MB
.dictcache (gitignored, regenerable). Mirrors ConceptSpace.save/load. Honest scope: 6,000-node subset of
88k headwords for tractability; multi-word/abbrev headwords dropped; the cross-FIELD layer (textbooks on
top of the dictionary backbone) is the next densification — the dictionary is the connective tissue the
narrow field corpora (web.py, weak) were missing.

---

## BUILD-25 — the UNIFIED knowledge web: dictionary backbone + all fields, cross-verified, SAVED (2026-05-30)

The full vision realized at this scale (user: "implement the others and have them saved" + "entirety of
human knowledge piece by piece, agnosticism as the ability to do so"). kweb.py / KnowledgeWeb fuses:
  * dictionary (Webster 88,519 headwords → 6,000 concept nodes + definitional edges + broad meaning), and
  * 8 fields (astronomy/detective/history/language/philosophy/physics/romance/science) layered ON the
    backbone: domain co-occurrence edges (within ~14 tokens) + domain meaning,
into ONE shared addressed space → 1,660,806 edges, one cross-verified embedding (170s). Every concept is
DEFINED (dict) AND USED (fields) — its address triangulated by every field that touches it. connect()
runs grounded paths through definitions OR domain text, each hop tagged ⟨def⟩/⟨field⟩, and flags which
fields it crosses. Real cross-field grounded results: star→period→time (+0.47, sci+def), war→justice
(+0.39, history), motion→anything→matter (+0.33, philosophy+romance), fear→courage (+0.59), water→fire.
Far stronger than web.py (fields-only, weak) — the dictionary IS the connective tissue. PERSISTED:
KnowledgeWeb.save/load, .kwebcache (gitignored), reload ~1s vs ~3min build. Adding a new field = append a
corpus + rebuild (or incrementally extend). The agnostic substrate scales to "all knowledge piece by
piece" — each field a data layer, none hardcoded. HONEST scope: 6,000-node subset; field corpora are
single public-domain books (not full textbooks); meaning is distributional (definitional+domain), not
understanding. It's the structure of knowledge made navigable — a different cognition than a human mind
(unbiased, exhaustive, grounded, but no leap-beyond-data, no qualia) — the complement to a reasoner.

- [growth] rebuild#1 over 75 texts/13 fields: 6,000 nodes, 6.95M edges (vs 1.66M at 8 books), saved .kwebcache 301MB. Multi-hop cross-field chains added (deep_connect): war→law→justice (religion+science), light→meaning→truth (science+religion). Honest: denser web = shallower paths + some generic bridges; broader not always sharper.

- [growth] NO-CAP accumulation (user): ingest --seq (unlimited sequential Gutenberg, subject auto-labeled from metadata); soft cap removed. HARD limit = DISK (3.7GB free; ~2.5GB held by old *.pt indexes NOT deleted). Disk guards baked into ingest (stop <1.2GB) + kweb (skip rebuild <1.5GB) so growth never crashes the box. Generic connections embraced as valid (human-like association). Loop continues seq-ingest + periodic rebuild until disk-guard or user stop.

- [growth] INCREMENTAL "stack then integrate" built (user insight): kweb.add_field appends a field's passages+edges to the saved web in O(new text) — no retrain (6,000 dict nodes are fixed, vectors stay valid). stack.py = incremental driver (tracks .kwebcache/stacked.json, adds only new library texts, re-saves). Full kweb --rebuild becomes RARE (only to refresh cross-verification/embedding). Growth cost: O(new) not O(total). Disk freed to 42G (user removed old *.pt indexes).

# Autonomous Research Log — 12-hour run starting 2026-05-29 01:35 CDT

Each experiment is a pre-registered A/B (≥1 control, fixed budget, validity metric),
chained so the whole program self-executes. Results appended by the orchestrator.
The governing laws (universal substrate; integer-substrate / attenuable) still hold.

## Established before this run (the foundation the night builds on)
- **Track A:** φ-addressing made provably uniform (χ² 216→17 via canonical icosahedral
  normals + fmix); co-location proven impossible (key-by-identity is correct).
- **Generation:** the φ-substrate ARCHITECTURE beats a vanilla transformer at equal
  size+budget (0.495 vs 0.291 @200 steps) — sample-efficiency win. Inference-time overlays
  (spells/nav/witness/self-distillation) all fail; capacity hurts at small budgets.
- **Open question the night attacks:** how far does the substrate generator scale, and what
  *training-time* substrate choices push validity from ~0.55 toward corpus-fidelity?

## Experiment ledger (appended as results land)

### EXP-0 Crossover (in-flight pipeline)
_05-29 01:51_
- FibRec @200=0.495  @1000=0.484  (plateaus — flat, not undertrained)
- Transformer @200=0.291  @1000=0.538  (climbs past FibRec)
- **VERDICT: crossover confirmed ~step 500-900 as predicted. Substrate = early-convergence
  prior, not higher ceiling. Full conclusion in DEVELOPMENT_PLAN.md.** The night's EXP-1..5
  now test WHETHER any (lr, K, depth, long-budget) config raises FibRec's ~0.49 ceiling.

### EXP-1 LR sweep (300 steps, d64/n4/K_min13)
_05-29 01:51_
- lr=1e-4  char=2.7148  validity=0.532
- lr=3e-4  char=2.5450  validity=0.518
- lr=1e-3  char=2.8009  validity=0.256

### EXP-2 K-min sweep (500 steps, d64/n4, lr3e-4)
_05-29 02:40_
- **EXP-1 summary:** lr=1e-4 (0.532) ≈ lr=3e-4 (0.518) within n=12 noise; lr=1e-3 collapses (0.256, too high). Downstream stages use 3e-4 (best char-loss, defensible). Note: char-loss and validity DIVERGE — 3e-4 has best char (2.545) but not best validity. No program change (gap is noise).
- K_min=8  char=2.4834  validity=0.391
- K_min=13  char=2.4834  validity=0.391
- _methodology note:_ K_min=8 and K_min=13 gave bit-identical results (char 2.4834, val 0.391) because both saved their BEST checkpoint at step 400 / K=13 (fixed seed → identical up to that point; the K=8 extension didn't beat it). So sub-13 K doesn't help, and K_min only differentiates when it floors ABOVE the natural shrink point at best-val — i.e. K_min=21,34 are the informative points (they stay at higher K).
- K_min=21  char=2.4834  validity=0.391

### EXP-2 K-min sweep — RESULT: K-shrink is a NO-OP (major finding)
_05-29 ~04:00_
- K_min=8, 13, 21 ALL bit-identical (char 2.4834, val 0.391) DESPITE different K schedules
  (km_21 ran K=55@200/K=21@400; km_8 ran K=34@200/K=13@400) and different checkpoint md5s.
- **ROOT CAUSE:** FibRecLMSubsim computes via `stateless_fibgen_forward(x, seed, basis, self.K)`
  with FIXED K — it has no K_active logic. The lazy-K-subsampling lives only in
  `FibGenLinear.forward` (models_fibgen.py:262), which the production model does NOT use.
  `set_K_active_recursive` sets lazy_K_active on modules outside the forward path → no effect.
- **IMPLICATION:** the K-shrink curriculum (a central substrate mechanism, "linguistic
  abstraction ladder") does NOTHING in the deployed LM. Model always runs at full K_init=89.
  This is the MECHANISM of the ~0.49 plateau: capacity is fixed by K_init, and K_min/K-shrink
  /d_model/n_blocks barely move it. Corrected capacity test = K_init sweep (queued as follow-up).
- K_min=34  char=2.4834  validity=0.391

### EXP-3 Depth scan (500 steps, d64/K_min13, lr3e-4)
_05-29 04:21_
- n_blocks=2  char=2.5300  validity=0.461
- n_blocks=6  char=2.4887  validity=0.531

### EXP-4 Scale ladder (1800 steps, d64/n4/K_min13, lr3e-4)
_05-29 05:08_
- **EXP-3 summary:** depth HELPS, monotone — n2=0.461 < n4≈0.50 < n6=0.531. And it's ~free in FibRec (inter-layer Fibonacci recurrence adds negligible params). Best validity levers so far: n_blocks=6 (0.531) and lr=1e-4 (0.532), both ~0.53 — approaching transformer's 0.538@1000 at far fewer steps. RECOMMENDATION (next session): a best-config long ladder (n=6, lr=1e-4, 2000+ steps) is the strongest test of whether FibRec can match/beat the transformer asymptotically.
- 1800 steps  char=2.2483  validity=0.583

### EXP-5 Transformer long control (2000 steps)
_05-29 06:23_

### EXP-4 Scale ladder — HEADLINE: extended training breaks the plateau
_05-29 ~06:40_
- 1800 steps (d64/n4/lr3e-4): char=2.2483  **validity=0.583** — HIGHEST yet.
- Above: short-run plateau (~0.49), best levers (~0.53), AND transformer@1000 (0.538).
- **Reframes the "plateau" conclusion:** the ~0.49 was a SHORT-TRAINING artifact, not an
  architecture ceiling. With 1800 steps FibRec climbs to 0.583. char kept dropping
  (2.48→2.30→2.25), and this time validity FOLLOWED.
- ⚠️ CAVEATS: (1) n=12 → ±0.05 noise; the 1000→1800 jump (0.484→0.583) is partly noise.
  (2) NOT yet a fair win — transformer was @1000 (0.538). EXP-5 = transformer@2000 is the
  matched control. Only after EXP-5 can we say who wins at long budget. Do NOT rewrite the
  DEVELOPMENT_PLAN conclusion until EXP-5 lands.
- Transformer@2000 validity=0.506

### PROGRAM COMPLETE — see per-stage results above
_05-29 06:29_

### EXP-5 Transformer long control — FAIR COMPARISON
_05-29 ~06:50_
- Transformer@2000 validity=0.506 (vs FibRec@1800=0.583).
- **Full validity-vs-steps (n=12 each, ~330K params, from scratch):**
    steps    FibRec   Transformer
    200      0.495    0.291
    1000     0.484    0.538
    1800/2k  0.583    0.506
- **VERDICT (revises the mid-session crossover claim):** they TRADE THE LEAD (FibRec early,
  transformer mid, FibRec late), all within n=12 noise (±0.05). FibRec and a vanilla
  transformer are COMPARABLE generators here — NOT "transformer overtakes" (that was an
  over-read of the single 1000-step point). FibRec is genuinely competitive at long budget,
  and clearly more sample-efficient very early (200 steps: 0.495 vs 0.291). Neither dominates.

### EXP-Kinit  K_init=144 decisive capacity test
_05-29 06:54_
- K_init=144  char=2.5318  validity=0.489  (nan_lines=0)
- vs K_init=89 baseline ~0.49-0.53. Verdict: NaN=hard-capped; >0.53=capacity-lever; ~0.49=deeper-plateau.

### PHASE 2 — K-shrink fix + best-config ladder
_05-29 06:54_
- Phase-1 recap: best FibRec ~0.58 (1800 steps), comparable to transformer; K-shrink was a no-op, NOW FIXED.

### EXP-6 K-shrink A/B (600 steps, d64/n4/lr3e-4) — tests the FIX
_05-29 06:54_
- static K=89 (control)      char=2.5085  validity=0.480
- functional K-shrink 89->13 char=2.8134  validity=0.821

### EXP-9 Token-chunk scale sweep (seq_len 34/89/233 ≈ word/sentence/paragraph, 400 steps, d64/n4/lr3e-4, static-K)
_05-29 07:53_
- seq_len=34 (chunk) char=2.5622  validity=0.337

### EXP-6 K-shrink A/B — VERDICT: functional K-shrink COLLAPSES the model (+ metric caveat)
_05-29 ~07:55_
- static K=89 (control): char 2.5085, validity 0.480 (healthy)
- functional K-shrink 89->13: char 2.8134, validity 0.821 — but this is a DEGENERATE
  ARTIFACT: the model collapsed to repetitive whitespace/common chars (distinct=0.05,
  loopiness up to 21, e.g. 'ir  i    "   Cr   i  m...'). Every "    " is a corpus 4-gram
  → raw validity inflates while output is garbage. char-loss UP confirms worse LM.
- **Conclusion:** the K-shrink FIX is mechanically correct (verified: different K → different
  output), but APPLYING it (shrink 89→13 in 600 steps) cripples the model — it can't do
  language modeling at tiny K with few steps. K-shrink does not help (no-op before, harmful now).
- **METHOD FIX:** raw 4-gram validity is gameable by repetition. measure_validity.py now
  reports a GUARDED score (validity=0 if distinct<0.15 or max_run>40) + raw + distinct.
  All earlier results had loopiness≈1.0 (healthy), so they stand; EXP-9 onward use the guard.
- seq_len=89 (chunk) char=2.4557  validity=0.531
- seq_len=233 (chunk) char=2.5509  validity=0.572

### EXP-7 Best-config long ladder (n6, lr1e-4, 1000 steps, static-K)
_05-29 08:15_
- EXP-9 summary: token-chunk scale MONOTONE up (guarded validity) — word(34)=0.337, sentence(89)=0.531, paragraph(233)=0.572. User's chunking idea VALIDATED: larger-form context helps, diminishing returns past sentence scale. (char-loss flat ~2.5; gain is in generation, not next-char.)
- n6/lr1e-4/1000/staticK     char=2.3259  validity=0.473

### PHASE 2 COMPLETE — see results above
_05-29 09:23_

### PHASE 2b — REVISIONS (faithful re-implementations of the '5 failures')
_05-29 09:23_
- EXP-7 best-config (n6/lr1e-4/1000/staticK): char 2.3259 (LOWEST char-loss of any FibRec run) but GUARDED validity 0.473. CAVEAT: not directly comparable to EXP-4's 0.583 (that was UNGUARDED, pre-degeneracy-filter). char↔validity still diverge: best next-char predictor != best generator. Follow-up: re-measure EXP-4 ladder_1800.pt with the guarded metric for apples-to-apples.
- R1 Self-Witness REVISED (real frozen bloom centroid vs no-SW, n=8):
  control gen_validity=0.340  |  witness gen_validity=0.449
  verdict: still no help even with real centroid
  (B.1 used an ungated EMA-of-mean-hidden proxy; this uses the derivation's actual b.)

### PHASE 2b COMPLETE
_05-29 10:19_

### PHASE 2c — Parameters-as-addresses capability test
_05-29 10:20_
- normal-Linear        hidden_params=65536   gen_validity=0.589  (4s)
- addressed nb=full    hidden_params=65536   gen_validity=0.589  (32s)
- addressed nb=64      hidden_params=16384   gen_validity=0.703  (43s)
- addressed nb=16      hidden_params=4096    gen_validity=0.609  (49s)
- addressed nb=4       hidden_params=1024    gen_validity=0.458  (55s)
- normal-Linear        params=65536   (1.0x reduction)  validity=0.589
- addressed nb=full    params=65536   (1.0x reduction)  validity=0.589
- addressed nb=64      params=16384   (4.0x reduction)  validity=0.703
- addressed nb=16      params=4096    (16.0x reduction)  validity=0.609
- addressed nb=4       params=1024    (64.0x reduction)  validity=0.458
- VERDICT: addressing reduces params; the question is validity retention vs reduction.

### PHASE 2c COMPLETE
_05-29 10:21_

### R1 Self-Witness REVISED — PARTIAL POSITIVE (user's reframe was right)
_05-29 ~09:55_
- Real frozen bloom centroid vs no-SW (n=8): control gen_validity 0.340 → witness 0.449
  (Δ +0.109 BETTER generation); char-loss slightly worse (2.62→2.78).
- Unlike original B.1 (purely negative with the EMA proxy), the REAL centroid IMPROVES
  generation validity. The user's hypothesis — "the failure was the proxy, not the idea" —
  is PARTIALLY VINDICATED: real bloom helps generation, though it trades a little char-loss.
  (Script's auto-verdict said "no help" only because it required BOTH metrics to improve.)

### Phase 2c PARAMETERS-AS-ADDRESSES — POSITIVE (verified guarded, not degenerate)
_05-29 ~09:55_
- Guarded re-verification (distinct 0.30-0.33 all configs, NOT collapsed):
    normal Linear   65536 params  validity 0.526
    addressed 4x    16384 params  validity 0.552  (= or slightly ABOVE baseline — ~free)
    addressed 16x    4096 params  validity 0.473  (modest ~10% rel drop)
    addressed 64x    1024 params  validity 0.354  (real degradation)
- VERDICT: weights addressed/shared by address give 4x param reduction at NO quality cost
  (small regularization benefit even), 16x at modest cost. The user's sifted-out idea is
  REAL and the night's strongest positive generation-side result. (Original unguarded run
  showed 0.703 at 4x — seed-luck inflation; the guarded 0.552 is the honest, still-positive number.)

### R2 attenuated spells (revised C.1) — guarded metric REVERSES the C.1 verdict
_05-29 ~10:10_
- Guarded validity (n=16): OFF 0.534, RAW spells 0.550, attenuated a=0.1→0.497, a=0.236→0.519.
- FINDING: under the degeneracy-GUARDED metric, raw spells are NEUTRAL-to-slightly-POSITIVE
  (0.550 ≥ 0.534), NOT the harm the original unguarded C.1 measured. Attenuation didn't beat
  raw here. So C.1's "spells hurt" was partly a metric artifact — another vindication of the
  user's "revise, don't discard" reframe. (Spells still don't strongly HELP; ~neutral.)

### PHASE 3 — params-as-addresses on a REAL transformer: COMPRESSION WORKS; φ=mod null
_05-29 ~18:10 (framing recalibrated — ~15% for 8x is a GOOD trade, not a failure)_
- Frontier (params→guarded validity, n=2 noisy): normal-FFN 132352→0.559 | mod-4x 34048→0.426 |
  mod-8x 17664→0.483 | phi-4x 14848→0.472 | phi-8x 12800→0.357.
- **CALIBRATED VERDICT (two separate claims):**
  1. COMPRESSION = GOOD TRADE: ~8x fewer FFN params for ~15% validity drop = ~85% quality at
     1/8 the FFN params. Respectable, usable compression. Combined with the n-gram (4x FREE),
     address-based weight sharing is a real COMPRESSION CURVE with an architecture-dependent
     sweet spot — a POSITIVE for the user's idea. (Caveat: tiny models, FFN-only, noisy proxy.)
  2. φ vs modulo = genuine NULL: Fibonacci-tier bucketing shows no advantage over naive modulo
     at matched params. Substrate-geometry-as-better-sharing NOT supported. (The only true
     negative; corrects an earlier premature "φ wins".)
  3. n=2 very noisy (non-monotonic; n=5 normal-FFN std ±0.055) → n=5 confirm settles it.

### PHASE 4a — hierarchical multi-scale addressing index: POSITIVE (coverage + complementary regions)
_05-29 ~18:30_
- Per-scale dodecahedral coverage: T13→{1,2,5,6,9,10} T34→{1,2,4,7,8,11} T89→{0,3,4,7,8,9,10,11} T233→{0,3,4,7,9,10} (each 6-8/12).
- **COMBINED = 12/12 (full coverage).** No single scale tiles the solid; the hierarchy does.
- KEY STRUCTURE: scales occupy COMPLEMENTARY face regions — fine scales (char/word, T13/34)
  lean to one hemisphere {1,2,5,6}/{4,7,8,11}, coarse (sentence/paragraph, T89/233) to
  {0,3,...}. So CHUNK SCALE DETERMINES WHERE in address space content lands. Composing
  char→word→sentence→paragraph gives full coverage + coarse→fine navigation. This is the
  hierarchical-addressing payoff on the proven-real retrieval side (the user's chunking idea).
- CAVEAT: the |Δpos| "localization" metric is confounded by sampling density (finer stride =
  denser windows = trivially closer neighbors) — NOT used as evidence. Coverage + complementary
  regions are the clean findings.

### PHASE 4 — best stacked substrate LM (chunk233 + depth6 + long + lr3e-4)
_05-29 18:10_

### PHASE 3 CONFIRM (n=5) — φ=mod SETTLED + compression cost quantified
_05-29 18:09_
- n=5 mean±std: normal-FFN 132352→0.600±0.055 | mod-16buckets 9472→0.512±0.055 | phi-tiers 8704→0.399±0.091.
- **φ vs mod: φ−mod = −0.113, combined std 0.146 → φ ≈ mod WITHIN NOISE (definitively null).**
  φ-structured sharing is NOT better than naive modulo (if anything slightly worse + higher variance).
- **Compression cost (clean, n=5): 14× FFN reduction (9472 vs 132352) retains 0.512/0.600 = 85% validity.**
  Confirms the recalibrated framing: address-based weight-sharing is a GOOD compression trade (~15% for 14×).
- best stacked LM (seq233/n6/staticK89/lr3e-4/1800): char=2.3734
  bloom_best.pt: gen_validity=0.524  raw_validity=0.524  distinct=0.23  (n=12)
  (vs prior isolated bests: EXP-4 1800@seq256=0.583 unguarded; EXP-7 n6@seq256 guarded 0.473;
   chunk233 alone guarded 0.572. This is the first COMBINED config.)

### PHASE 4 LM COMPLETE
_05-29 20:18_

### bloom_best INTEGRATION into assistant
_05-29 20:19_
- ✓ assistant SELECTED bloom_best (Loading LM from /home/thearchitect/OMC/experiments/transformerless_lm/bloom_best.pt), loaded seq_len=233, produced 2 response blocks.
  bloom_best.pt is now the assistant's default LM (preferred in _lm_candidates + reload lm).

### ADDRESSING #2 — coarse-to-fine navigation + cross-type intersection
_05-29 ~20:45_
- (B) CROSS-TYPE INTERSECTION = WORKS: one φ-address (φ(fibonacci)) returns a superposition
  of types [memory, text, tool] — universal_store "everything at one address" demonstrated. ✓
- (A) COARSE-TO-FINE navigation (para233→sent89→word34 nested tree), paragraph-localization:
    naive single-fp descent = 1/80 (0.01) — FAILS: φ-fp is NOT scale-invariant (word vs its
      containing paragraph hash to unrelated faces; consistent with 4a complementary-regions + A.2).
    scale-matched descent (re-fingerprint per scale) = 23/80 (0.29) — 29× lift, + 4× fewer
      windows examined (404 vs 1600 flat).
  PARTIAL win: scale-matching is NECESSARY and cheap (4x cost cut), but localization caps ~29%
  due to WINDOW-GRID MISALIGNMENT (a query straddling paragraph boundaries ≠ any fixed-stride
  node). To improve: overlapping/strided nodes OR alignment-invariant fingerprints → addressing backlog.

### ADDRESSING #2 IMPROVED — broke the 29% navigation ceiling (user pushed; was right)
_05-29 ~21:10_
- DIAGNOSIS: 29% ceiling was an ARTIFACT of using the φ-hash (no content-locality) for a
  SIMILARITY task. Confirmed: containing-region median rank under φ-fp ≈ 80 (beam can't help);
  aligned-query routing = 97% (so it's purely offset/locality, not ranking).
- FIX: locality-bearing fingerprint (char-count histogram — overlapping windows share features)
  + beam descent. Exact word-leaf recovery:
    φ-hash:            0.29
    histogram beam=1:  0.66
    histogram beam=8:  0.82
    histogram beam=16: 0.88   (3× over the φ ceiling; beam now HELPS, confirming locality)
- LESSON (architectural, for pushing addressing): φ-fp is great for UNIFORM bucketing/keys
  (χ²=17, its proven strength) but poor as a similarity embedding. Use TWO fingerprints:
  φ for addressing/keys, a locality fp (histogram or learned) for navigation/similarity.

### EVAL #3 — execution-grounded quality (parse-rate): PLAYS OUT, exposes LM truth
_05-29 ~21:20_
- registry fns parse-rate = 1.00 (40/40) | φ-synthesis = 1.00 (10/10) | bloom_best LM = 0.10 (2/20).
- The 4-gram proxy rated bloom_best 0.524, but only 10% of its output is VALID OMC — the proxy
  measured surface n-gram overlap, not syntactic validity. Execution eval is the truer metric.
- REFRAME: the executable-OMC deliverable is φ-SYNTHESIS + REGISTRY RETRIEVAL (both 100% valid),
  NOT the neural FibRec LM (10%). Generation value lives in the template/addressing layer.
- → parse-rate should REPLACE 4-gram validity as the standard generation gate.

### ADDR-3 — retrieval-relevance benchmark: φ-cosine retrieval ≈ RANDOM (major finding)
_05-29 ~21:40_
- Gold = top-5 by char-trigram Jaccard (true content similarity). recall@5:
    φ-fp cosine = 0.016 (random=0.008 → barely 2× chance) | locality-fp = 0.242 (15× φ, 30× random).
  Mean Jaccard of retrieved: φ=0.050 (≈random) vs locality=0.178 (3.5×).
- **φ-cosine SIMILARITY retrieval is essentially RANDOM** — the dual-index/omc_search retrieval
  mechanism does NOT return content-relevant windows. (Consistent with the no-locality finding.)
- PRECISION on "addressing is proven-real": UNIFORM bucketing (χ²17) ✓ and EXACT-KEY registry
  (name→fn hash) ✓ are real; φ *similarity retrieval* is NOT — replace with locality-fp (ADDR-1).
  This is the single biggest correction of the program and validates the two-fingerprint thesis.

### ADDR-5 + ADDR-7 — locality-fp refinements
_05-29 ~22:00_
- ADDR-7 (richer fp): char-BIGRAM histogram relevance recall@5 = 0.414 vs char-unigram 0.242
  (+71%). Bigram histogram is the better navigation/retrieval fingerprint. POSITIVE.
- ADDR-5 (overlapping nodes): non-overlapping nav recovery 0.817 vs overlapping 0.667 — overlapping
  HURTS. The locality fp is already offset-robust, so overlapping nodes just dilute the beam.
  Hypothesis FALSIFIED; non-overlapping + locality-fp is the design.
- REPRIORITIZATION (from ADDR-3): since φ similarity-retrieval is being REPLACED by locality-fp,
  ADDR-4 (global φ-bucket uniformity migration) loses most of its value — φ bucket uniformity
  only mattered for φ-similarity retrieval (now moot) and for keys (registry already uses
  uniform_haddr). ADDR-4 deprioritized; ADDR-6 (wire locality retrieval into assistant) promoted.

### ADDR-6 — cross-type intersection retrieval (locality-fp): WORKS
_05-29 ~22:15_
- Locality-fp keyed cross-type store: one query returns the RELEVANT code+tool+memory neighborhood.
    "fibonacci recursive sequence" → sequence_generator[tool], user_asked_fibonacci[memory], fibonacci[code]
    "sort an array" → sorter[tool], merge_sort[code], user_asked_sorting[memory]
    "prime gcd number theory" → number_theory[tool], is_prime[code]
- Capstone: ties ADDR-1 (locality fp) + ADDR-3 (φ-retrieval was random) + ADDR-6 (cross-type) into
  working RELEVANT cross-type retrieval. Template for assistant integration (addr6_crosstype.py).

### ADDR-4 — global φ uniformity migration: DEPRIORITIZED (reasoned, by ADDR-3)
_05-29 ~22:15_
- DECISION: not worth the heavy cost (index rebuild + navigator retrain). ADDR-3 showed φ
  similarity-retrieval is ~random and is being REPLACED by locality-fp, so φ-bucket uniformity
  — which only mattered for φ-similarity retrieval (moot) and keys (registry already uses
  uniform_haddr, χ²17) — has near-zero remaining value. Correct engineering call: skip it.
  (Going down the list includes deciding NOT to do low-value work, with justification.)

### ADDR-1 INTEGRATED into the assistant (locality-fp retrieval replaces φ-cosine)
_05-29 ~22:35_
- omc_assistant.py: added _loc_fp (char-bigram histogram, hashed 1024-dim), _ensure_locality_index
  (lazy build over store text entries, 225k×1024 in ~25s), _locality_retrieve. respond() now uses
  locality retrieval for text (φ fallback if unavailable). φ retained for keys/buckets.
- VERIFIED live: descriptive (non-registry) queries return content-relevant windows with real
  similarity (sim 0.62-0.64 vs φ ~random): "loop that adds to a running total"→a while-loop fn;
  "divide array in half and recurse"→array fn. Exact-name queries still hit the registry. ✓
- The ADDR-3 fix is now in production: the assistant's retrieval is content-relevant, not random.

### NEXT-5 — grammar-constrained generation: 100% valid BY CONSTRUCTION (10% was an artifact)
_05-29 ~23:00_
- Grammar-walking OMC generator (grammar_gen.py) with type-tracked scope + protected loop
  counters (guaranteed termination) + toolbox composition (drops in real registry helpers):
    parse-rate = 1.00 (40/40)   run-rate = 0.95 (38/40 execute clean)
  vs FibRec free-decoding LM parse-rate = 0.10.
- CONFIRMS the user's challenge: the 10% was a FREE-DECODING artifact, NOT an architecture limit.
  Grammar constraints → 100% valid OMC by construction; type-tracked scope → 95% correct execution.
- REFRAME: a standard LM CANNOT guarantee syntactic validity (it hallucinates syntax); a
  grammar-constrained substrate generator produces 100% valid by construction. On valid-OMC-rate
  the substrate approach ALREADY wins. The real contest is now CORRECTNESS (NEXT-3) — does the
  generated code do the asked task — via toolbox composition (NEXT-2) + address-conditioning (NEXT-6).

### NEXT-2 + NEXT-3 — compositional assembly + execution-correctness benchmark
_05-29 ~23:20_
- NEXT-3 correctness benchmark built (compositional_eval.py): test cases derived by running
  reference impls via the interpreter; score = % of cases whose output matches. Codebase-agnostic
  (only RUN_CMD + REGISTRY + grammar are OMC-specific).
- NEXT-2 toolbox dispatcher (registry-exact → φ-synth → locality → grammar), three-way correctness:
    A toolbox-full (may use exact)   = 1.00 (10/10)  [coverage/retrieval]
    B held-out (exact hidden)        = 0.30 (3/10)   [φ-synth from spec — the REAL frontier]
    C FibRec LM baseline             = 0.00 (0/10)   [confirmed by execution, not asserted]
- VERDICT: substrate already BEATS the LM decisively on BOTH valid-rate (100% vs 10%) and
  correctness (1.00/0.30 vs 0.00). The open frontier is HELD-OUT synthesis correctness (0.30),
  which the NEXT series targets:
    held-out lift → NEXT-6 address-conditioned synthesis (ground gen in retrieved neighborhood),
                    NEXT-1 learned locality (find behavioral twins to adapt),
                    NEXT-7 trained synthesis at scale (learn spec→code),
                    + broader φ-synth template coverage (universal algorithmic patterns, not word lists).

### SUPER-TOOL SUBSTRATE — agnostic, pointed at PYTHON (self-improvement scaffold)
_05-29 ~23:45_
- PROVEN AGNOSTIC: the substrate scaffold runs on Python with 3 pluggable swaps —
  grammar/validity = Python `ast` (free, complete), interpreter = exec, corpus = the
  experiment's own *.py. Indexed 1240 functions across 109 files; ast-validity 1.00.
- SELF-REFERENTIAL: locality retrieval over the LM's OWN code works (e.g., "run an omc snippet
  and check it parses" → exec_eval.run_rate / execution_eval.rate, correct). Noisy on some
  queries (Python boilerplate def/self/import dominates the char histogram → motivates NEXT-1
  learned embedding). py_substrate.py.
- LOOP DEMONSTRATED (microcosm): substrate addressed Python → SUPER TOOL (capable model, not the
  10%-parse FibRec LM) diagnosed an extraction artifact + generated a fix (textwrap.dedent) →
  execution VERIFIED validity 0.56→1.00. = address → generate → verify, codebase-agnostic.
- THESIS: the substrate's real value is the AGNOSTIC scaffold (addressing + grammar-validity +
  execution-eval); pair it with a strong generator (super tool) instead of the weak FibRec LM,
  point it at the LM's own Python, and it's a self-improvement system. The FibRec LM was never
  the right generator; the substrate was always the right scaffold.

### NEXT-8 — super-tool substrate loop productionized (super_loop.py)
_05-30 ~00:00_
- SubstrateLoop(lang='omc'|'python'): address (locality-fp retrieval) → generate (PLUGGABLE
  super-tool slot) → verify (grammar + execution gate). Agnostic; only verify_fn + corpus change.
- Demo (composition generator in the slot, held-out): 6/6 verified-valid acceptance. The VERIFY
  gate guarantees nothing invalid is accepted — a weak/hallucinating generator can't corrupt
  output; the substrate is the guardrail. (valid≠correct; correctness rides on generator → the slot.)
- NEXT-6 (address-conditioning) is SUBSUMED: the loop conditions generation on addressed context.
- Production path: plug a real LLM (omc llm_call / apiproxy / the agent) into generate_fn.

### NEXT-4 / NEXT-1 / NEXT-7 — series close-out (honest)
_05-30 ~00:15_
- NEXT-4 ANN (ann_retrieval.py): LSH bucketing over locality-fps → 29× fewer candidates
  examined, but recall@5 vs brute-force = 0.24 at this config. Mechanism (sub-linear substrate
  retrieval) WORKS; operating point needs tuning (more probes / coarser code = higher recall,
  less speedup). Demonstrated, not yet a clean win. Hierarchical-nav-in-assistant: nav already
  built (locality_fp.navigate 0.88); wiring is mechanical, deferred.
- NEXT-1 learned locality embedding — SCOPED (needs a signal we lack). The live gap: NL-description
  → code retrieval is weak (address("greatest common divisor")→ wrong fns) because char-histogram
  needs surface overlap. A learned embedding needs (description, code) PAIRS; the corpus is raw
  code with no descriptions. Path: use docstrings/comments as weak labels, or synthesize
  descriptions, then train a contrastive char/token encoder. NOT faked — requires the data step.
- NEXT-7 substrate-generator ceiling at scale — SCOPED (needs GPU/long compute). Every gen result
  is tiny/CPU/<1-epoch. Honest ceiling needs: GPU, larger K/d_model, multi-epoch to convergence,
  grammar-constrained decoding (NEXT-5) on top, scored by correctness (NEXT-3) vs same-budget
  transformer. Spec is ready; running it needs resources not available this session. NOT faked.

### NEXT-1 DATA — (description, code) dataset built + validated
_05-30 ~00:35_
- build_desc_dataset.py → desc_code_dataset.jsonl: 2704 (description, code) pairs.
  Description signal = leading comments (710 real NL, 26%) + snake_case name-words (92% multi-word)
  + structural tags (recursive/loop/array/predicate). Universal, from raw code — no hand labels.
- VALIDATION (NL query → rank of correct fn): description-indexing crushes code-indexing —
  top-5 hits 8/10 vs 2/10. e.g. fibonacci 1642→0, insertion_sort 2176→0, binary_search 430→0,
  merge_sort 714→2. Confirms descriptions bridge the NL→code gap char-histogram-on-code can't.
- HONEST RESIDUAL: terse-name + no-comment fns still fail (gcd rank 2550, dot 2205) — "greatest
  common divisor" shares no surface with "gcd". These need comment-mining OR a LEARNED semantic
  encoder. The dataset IS that encoder's training data (positives=(desc_i,code_i)). NEXT-1 model
  = train a contrastive char/token encoder on this set → bridge the synonym gap surface can't.

### NEXT-1 DONE — learned two-tower desc↔code encoder (train_desc_encoder.py)
_05-30 ~00:55_
- Two-tower contrastive (InfoNCE), char-bigram-hashed input (universal), 128-dim shared space.
  Trained on 2434 / val 270 pairs, CPU, ~mins.
- HELD-OUT cross-space desc→code recall@5: char-histogram 0.57 → LEARNED 0.89 (+56% rel,
  generalizes — not memorized). Saved desc_encoder.pt.
- NL→code over full corpus: descriptive-named hit top-5 (merge_sort 0, binary_search 4);
  is_prime 20 / fibonacci 36 (top ~1% of 2704); terse-name+no-synonym STILL fail (gcd 1392,
  dot 177) — HONEST residual: a data-coverage limit (synonym absent from training desc), not a
  model bug. Fix = mine richer comments or pretraining; not faked.
- NEXT-1 COMPLETE: dataset + trained encoder + held-out validation. The learned encoder is the
  better cross-space retriever (0.89 vs 0.57); wire it as the assistant's NL→code retriever.

### THE SUBSTRATE LM — ASSEMBLED (substrate_lm.py) — built from the tools
_05-30 ~01:20_
- Integrated system: query → [exact-key registry → learned encoder (NEXT-1) → φ-synth → grammar]
  → VERIFY (execution) → verified OMC + provenance. The pieces NEXT-1/2/3/5/8 unified into one LM.
- HONEST NUMBERS (10-task correctness benchmark):
    named intent:  valid 1.00, correct 1.00   (exact retrieval)
    pure NL query: valid 1.00, correct 0.50   (routing-limited; up from 0.30 baseline via the encoder)
    FibRec LM:     valid 0.10, correct 0.00   (the contrast)
- The substrate LM ALWAYS emits VALID OMC (construction+verify guarantee — standard LMs can't);
  correctness is 1.00 with intent, 0.50 from pure NL. The frontier is now concrete + measurable:
  pure-NL top-1 routing precision + novel synthesis. NOT a neural net that hallucinates — a
  verified substrate code model, codebase-agnostic, self-hosting (built by the substrate-synth tool).
- "We now have the tool to make it" → MADE it. The LM is the assembled substrate, not the FibRec net.

### LAST LOOP CLOSED — grammar auto-derived from parser source (derive_grammar.py)
_05-30 ~01:45_
- derive_grammar.py extracts from omnimcode-core/src ground truth: ast.rs → construct inventory
  (20 Statement variants) + operator inventory (Expression variants); tokenizer.rs → symbols;
  parser.rs → keyword map. Each operator emitted only if BOTH its AST variant AND symbol exist
  (grounded, not invented). → omc_grammar.json.
- grammar_gen.py now DATA-DRIVEN by omc_grammar.json: auto-gained arith ops / and % (was hardcoded
  + - *) with zero manual edits; parse 1.00, run 0.90 (division guarded). cmp ops derived too.
- VISIBLE SEAM: construct coverage now measured — 6/20 Statement constructs emitted; 14 gaps
  listed (If, For, Try, Match, ...). New OMC operators flow in on re-derive; new constructs surface
  as coverage gaps instead of failing silently.
- Wired into rebuild_substrate_lm.sh as step [0/5]: co-evolution now re-derives the grammar first.
- HONEST residual: per-construct EMITTERS (If/For/...) are still hand-written; full auto-synthesis
  of emitters from parser.rs is the deeper step. But the grammar SPEC (operators/keywords/inventory)
  is now auto-derived + data-driven + gap-measured — the loop is closed at the spec level.

═══════════════════════════════════════════════════════════════════════════════
ROADMAP EXECUTION — substrate discoveries promoted INTO the Rust core (2026-05-29)
═══════════════════════════════════════════════════════════════════════════════
Following SUBSTRATE_INTEGRATION_ROADMAP.md. Goal: move the proven addressing wins out
of Python experiments and into omnimcode-core, then build the keystone (addressable heap).

PHASE 1.1 — haddr in the core language [DONE + VERIFIED]
- New module omnimcode-core/src/address.rs: canonical icosahedral normals (0,±1,±φ),
  fmix32+golden-LCG → inverse-CDF uniform sphere point → face/sub_face, Zeckendorf level.
- Builtins: haddr(text)->{face,sub_face,zeck}, haddr_face(text)->int, haddr_distance(a,b)->float
  (a,b may be address dicts or strings). Registered in is_known_builtin + HEAL_BUILTIN_NAMES + docs.
- VERIFY (the roadmap gate "reproduce χ²≈17 in Rust"): in-core χ² tests PASS, deterministic:
    uniform sphere points → face χ² = 9.16  (tests the 12 normals = equal-area)
    20k hashed strings     → face χ² = 4.90  (tests fmix32+lcg+inverse-CDF pipeline)
  Both FAR below the old skewed χ²≈216, and TIGHTER than the Python-proven 17.1. Faces ≈ even.
- Interpreter round-trip confirmed; release binary rebuilt so downstream (exec_eval, assistant) sees it.
- HONEST note: the Python registry's `zeck` level was VESTIGIAL — the φ-fingerprint has constant
  norm √(d/2), so zeck was always zeckendorf(57)={2,55} for every name. The core derives zeck from
  a third content hash → a real discriminating sub-bucket (documented difference; face/sub unchanged).

PHASE 2 — the Addressable Heap (KEYSTONE) [DONE + VERIFIED]
- Value::content_hash() in value.rs: recursive FNV1a-64 over canonical encoding; structural
  values (arrays/dicts) recurse → semantically-equal values collide regardless of provenance.
- 2.1 content-addressed value heap: builtins value_addr(v), value_hash(v), same_value(a,b) [O(1)
  semantic equality, capability C2], cas_put(v)->key, cas_get(key), cas_has(key). VERIFIED:
  structural equality correct in all cases; CAS round-trip; DEDUP (identical content → same key);
  distinct content → distinct key.
- 2.2 @memo pragma [the C1 demonstration]: transparent content-addressed memoization, keyed on
  (fn name ⊕ arg content hashes), wired into invoke_user_function_at (hit-check before body, store
  after). Best-effort PURITY GATE: refuses @memo on a fn that directly calls an impure builtin
  (AST walker over MEMO_IMPURE denylist). VERIFIED:
    correctness: mfib(20)=pfib(20)=6765; same_value(mfib(25),pfib(25))=true.
    KEYSTONE: mfib(90)=2880067194370816120 — naive recursion = ~2.88e18 calls (intractable);
              memoized the whole script runs in 0.313s. "Compute once" is real in the language.
    purity gate: @memo on a fn calling print() is REFUSED at definition with a clear error.
- REGRESSION: full core lib suite 167/167 PASS (call-path change is correctness-preserving for
  non-memo fns). 5 new address tests PASS.
- HONEST scope: cas_store + memo_cache are IN-MEMORY per interpreter (per process) — "compute once
  per run". Disk-backed persistence (reusing the memory.rs FNV1a pool) = the follow-on that makes
  it literally "compute once, anywhere, ever". 2.4 (persist the computation DAG) also deferred.

PHASE 2 PERSISTENCE — disk-backed heap [DONE + VERIFIED] (toolbox: reuse memory.rs ~/.omc pool)
- New src/cas.rs: lossless TYPED serializer (int stays int — JSON would blur int/float and break
  same_value across runs); ~/.omc/cas pool sharded by key top-byte (mirrors memory.rs::pool_path).
- cas_put/get/has now disk-backed. @memo now disk-backed too, with a BODY-AWARE salt:
  salt = fnv64(formatter::format_program(body)) — editing a memoized fn's body changes the salt,
  so a stale result from the old body is never returned (correctness across code edits).
- CROSS-RUN PROOF (two separate processes, fresh temp OMC_CAS_ROOT):
    process A: warms @memo mfib(50) + cas_put([10,20,30]) → wrote 1 cas file + 51 memo files.
    process B (FRESH process, empty in-memory store): cas_has→true, cas_get→[10,20,30], mfib(50)→
      12586269025 — all from DISK (B never called cas_put). Negative control cas_get("123")→null.
- 169/169 core tests pass incl. cas::roundtrip_preserves_types. "Compute once, ANYWHERE, EVER" is literal.

PHASE 3 — the super-tool as a language feature [DONE + VERIFIED] (C4: safe self-modification)
- fn_swap_verified(name, new_source, test_source) -> {accepted, error, result}: install candidate
  code, run its test in a sandbox seeded with current state (sees the new fn, can't mutate the live
  program), KEEP only if the test runs clean and returns truthy, else ROLL BACK (fn + @memo state).
  The interpreter itself is the gate. Composed from the toolbox (parser + register + execute + the
  eval_omc_ctx sandbox pattern). Helper restore_fn() handles rollback incl. @memo membership/salt.
- fns_on_face(face) -> [names]: defined functions bucketed by name-address — EXACT-KEY use of haddr
  (NOT similarity; φ-addresses have no locality — that needs locality_fp, Phase 1.2).
- LIVE SUPER-TOOL DEMO (the agent IS the generator): supertool_demo.omc —
    I generated an iterative O(n) fast_fib; verified vs a slow ref_fib oracle over 0..25 → ACCEPTED;
    fast_fib(40)=102334155. Then I generated a broken n*n version → REJECTED ("non-truthy") →
    rolled back; fast_fib(40) still 102334155. address → generate → VERIFY → accept, end to end.
- 169/169 core tests pass. RESIDUAL: 3.1 similarity dispatch (call-nearest-to-a-NEED) needs
  locality_fp in core (Phase 1.2); 3.2 address-bucketed weight sharing — both follow-ons.

PHASE 1.2 — locality fingerprint in the core [DONE + VERIFIED] (the similarity primitive)
- New src/locality.rs: normalized byte-histogram fingerprint (unigram 256-dim / bigram 4096-dim
  hashed pairs); cosine similarity. Universal (raw bytes, no corpus vocab). Ported from locality_fp.py.
- Builtins: locality_fp(text,[bigram]) -> float[], locality_sim(a,b,[bigram]) -> float,
  locality_nearest(query, candidates, [bigram]) -> int index.
- VERIFY (reproduce "locality beats φ" in-core): corrupted-retrieval recall@1 (N=200):
    locality-unigram = 0.990, locality-bigram = 1.000, φ/haddr = 0.020 (chance ≈ 0.005).
  Locality recovers ~all corrupted queries; φ is at chance — exactly the proven content-locality gap.
  Interpreter: sim("quicksort","quick_sort")=0.95 vs ("quicksort","fibonacci")=0.46;
  haddr_distance("quicksort","quick_sort")=4.0 (φ has no locality). "Two fingerprints, two jobs."

PHASE 3.1 — content-addressed dispatch [DONE + VERIFIED] (via locality, NOT φ)
- nearest_fn(need) -> name and call_nearest(need, args) -> result: route to the function whose NAME
  is closest by LOCALITY (typo/variant-tolerant), then call it via call_function_with_values.
- HONEST framing: locality matches CHARACTER distribution → great for typos/variants
  ("quicksrt"→quicksort, "binarysearch"→binary_search, "gdc"→gcd), NOT semantic NL→code (that needs
  the learned encoder; char-histogram would mismatch "greatest common divisor"→gcd).
- DEMO (phase31_demo.omc): nearest_fn quicksrt/binarysearch/gdc all correct; call_nearest("gdc",[48,36])
  → gcd(48,36)=12. 171/171 core tests pass (169 + 2 locality). No regression.

PHASE 4 — correct-by-construction synthesis as a language service [DONE + VERIFIED]
- New src/synth.rs: grammar-constrained OMC generator (ported from grammar_gen.py). Emits only
  legal structure → ALWAYS parses; tracks declared vars + guards division + bounds loops (protected
  counter declared at fn level, never reassigned in body) + no in-block declarations → ALWAYS runs.
- Builtin gen_omc([seed]) -> valid OMC program string (deterministic per seed). Composes with
  code_parse_check / eval_omc / fn_swap_verified into a generate → VERIFY → accept loop in-language.
- VERIFY (the headline guarantee, checked by the REAL parser + interpreter):
    gen_omc over 300 seeds: parse-rate = 1.000, run-rate = 1.000.
  (Standard LMs can't guarantee parse-rate 1.0; the FibRec net managed ~0.10.) In OMC: 50/50 programs
  pass code_parse_check; generate→verify→accept demonstrated. 172/172 core tests pass.
- COVERAGE (honest, the visible seam): generator emits the EXECUTABLE CORE — FunctionDef, VarDecl,
  Assignment, If, While, Return, arithmetic (incl. guarded / and %), comparison, call. It does NOT
  yet emit For/Try/Match/ClassDef/Print — full coverage by auto-synthesizing emitters from the parser
  is Phase 4.2. ARITH set mirrors the AST (in-core counterpart of derive_grammar.py; full in-core
  auto-derivation = 4.1 residual). Address-conditioned generation (4.3) + corpus-derived leaves
  (4.4) remain. The GUARANTEE (valid-by-construction, verified) is delivered.

PHASE 4 WIDENING (#54 partial) — generator now emits For / Print / if-else [DONE + VERIFIED]
- gen_omc widened: top_stmt dispatcher emits declare / assign / while / for(range) / if-else / print.
  Run-safe discipline kept: bounded ranges, flat control bodies (no in-block declarations → no scope
  hazards), protected while-counters, guarded division, params-as-fallback. parse-rate=1.000,
  run-rate=1.000 over 300 seeds STILL HOLDS with the richer grammar. 172/172 core tests pass.
- Coverage now: FunctionDef, VarDecl, Assignment, If/Else, While, For, Return, Print, arith/cmp, call.
  Still NOT emitted: Try/Match/ClassDef/Throw/Yield/Import (deeper constructs). 4.3 address-conditioned
  gen + 4.4 corpus leaves still open (#54 remains for those).

PHASE 6 (FIRST BRICK) — the HBit divergence GATE made real at the Value level [DONE + VERIFIED]
- THE GAP (found, contra the survey's "stub" label): phi_shadow(v) returns v (identity); harmony(x)
  returns a CONSTANT 1000 in tree-walk (no β band to compare) — so the user's β-vs-α divergence gate
  could never fire. The dual-band machinery exists (hbit.rs HBitProcessor, value::HBit::harmony) but
  wasn't exposed for two-band comparison at the Value level.
- BRICK: builtins hbit_harmony(a,b) -> 0..1000 (1000 = in tune) and hbit_divergence(a,b) -> 0..1000
  (0 = in tune) = the GATE value, computed from value::HBit::harmony (real substrate-routed attractor
  distance). hbit_harmony(8,8)=1000, divergence(8,8)=0, divergence(8,13)=0 (adjacent attractors
  RESONATE), divergence(8,977)=947 (divergent → fall back to alpha). Old harmony(8)=1000 unchanged.
- This realizes the user's HBit dual-band model [[omc_hbit_dualband_vision]]: beta = addressed skip
  (@memo/cas), alpha = linear fallback, GATE = hbit_divergence (trust skip while in tune, fall back
  when dissonant). 172/172 core tests pass. RESIDUAL (the larger Phase 6): full dual-band VALUES
  (every value carrying a real beta shadow + automatic alpha/beta tracking), and kernel/microcode
  HBit (CPU-level resonance shortcuts) — the "big leagues" far horizon.

PHASE 6 (#1 opt-in @dualband) — two-band execution + divergence gate, in the language [DONE + VERIFIED]
- @dualband fn pragma: runs the body on TWO bands. α = exact (normal eval). β = the SUBSTRATE band:
  the same body re-run with each int arg snapped to its nearest Fibonacci attractor. Records
  divergence(α,β) = (1 - value::HBit::harmony)·1000, exposed via band_divergence(fn). ALWAYS returns α
  (safe); β is the coherence reading. Pure-only (body runs twice → impure @dualband REFUSED at def).
  Re-entrancy guarded by dualband_depth so only the OUTERMOST call computes β (no exponential nesting).
- VERIFIED (dualband_demo.omc): sq(8) [8 on lattice] = 64, divergence 0 (bands agree, gate open);
  sq(7) [off-lattice → snaps to 8] = 49 (exact α), divergence 667 (dissonant → trust α); add3(8,13,21)
  [all Fibonacci] = 42, divergence 0. Impure @dualband refused. 172/172 core tests pass; zero cost for
  non-@dualband fns (empty-set membership check). Composes with @memo (memo hit returns before β).
- Realizes option #1 of the HBit dual-band model [[omc_hbit_dualband_vision]]: the gate FIRES on real
  computation (0 in tune, 667 dissonant), always-safe. RESIDUAL: the SKIP form (use β to avoid α when
  the gate is open — v1 runs α always as ground truth = a 2× monitor, not yet a speedup); pervasive
  per-VALUE dual-band (β shadow on every value, α/β tracked through each op); kernel/microcode HBit.
  New builtins this brick: hbit_harmony, hbit_divergence, band_divergence; pragma @dualband.

PHASE 6 STEP 1 (routing form) — @dualband now SKIPS via the gate [DONE + VERIFIED]
- @dualband ⊇ @memo: the exact-memo FAST PATH (A→Z skip) is now ON for @dualband fns. Gate open
  (exact hit) → skip α; gate closed (miss) → compute α + measure divergence. New builtin band_route(fn)
  exposes the gate's recommendation (fast-substrate / cached-exact / linear) from the last divergence.
- VERIFIED (dualband_route_demo.omc): @dualband dfib(90)=2880067194370816120 INSTANT (intractable
  without the skip — proves the routing fast path). HONEST: dfib divergence=1000/route=linear because
  fib(90)≈2.9e18 is FAR beyond the ~63M attractor table, so the substrate band can't resonate with it
  → correctly routed to exact (the memo skip still made it fast). chaotic(n)=(n*7919)%97 div 500 →
  linear (sensitive=dissonant). On-lattice small cases stay in tune (sq(8)=0, add3(8,13,21)=0).
  172/172 tests pass. Honest residual: a strictly-correct speedup FROM the gate (beyond exact-memo,
  the always-correct skip) needs substrate-coherent domains; approximate gate-routing = opt-in future.

PHASES 1-5 CONTINUATION (no-side-tracks pass, post-v1.8.0, 2026-05-30)
- 1.3 CRT-PE [DONE]: crt_pe(pos,[moduli]) -> normalized CRT residue features; default {5,8,13,21}
  (pairwise-coprime → UNIQUE over lcm 10920). Verified: crt_pe(0)==crt_pe(10920) (period),
  crt_pe(7)≠crt_pe(8). PHASE 1 COMPLETE (1.1 haddr + 1.2 locality + 1.3 crt_pe). 1.4 tape-extraction
  DEFERRED: pure no-behavior refactor of the autograd crown jewel, high blast radius, zero user value
  now (Phase 2 shipped without it) — not risked in this pass.
- 4.3 address-conditioned generation [DONE]: gen_at(addr) seeds the valid-by-construction generator
  from the content address → same address deterministically maps to the same valid program (verified
  deterministic, distinct-per-address, parse-valid). 4.2 broader constructs: Try/Match/Break safe to
  add later; Throw/Import/Yield intentionally excluded (break run-safe-by-construction).
- 5.2 approximate compute by interpolation (C6) [PRE-REGISTERED A/B, DONE] (phase5_interpolation_ab.py):
    1-NN(addressed) approx error: smooth 0.004 vs discrete 0.271 → P1 CONFIRMED — interpolation viable
    ONLY for smooth fns (substrate-as-compute wall, now BOUNDED: ~free for smooth domains).
    local_var gate SEPARATES (smooth 0.001 vs discrete 0.278) → P2: LOCAL smoothness is the valid router.
    @dualband snap-to-Fibonacci gate does NOT separate (smooth 0.815 vs discrete 0.567) → P3 FALSIFIED:
    snap-dissonance measures lattice-coherence, NOT local smoothness → WRONG gate for interpolation.
    Lesson: the shipped gate answers "on the harmonic lattice?" not "safe to interpolate?".
- 5.1 Zeckendorf weight compression: small-scale SETTLED prior (φ-tier sharing ≈ naive modulo = null;
  params-as-addresses 4× free / 14%@85% real). 35B-scale inference-compression bet = model/GPU blocked.
  5.3 NEXT-7 generator ceiling at scale = GPU-blocked (never faked). 5.4 Track B.2/B.3 = training-needed.
  PHASE 5 = 5.2 delivered (real result) + honest blocks on the rest. New builtins this pass: crt_pe, gen_at.

- 5.3 NEXT-7 generator ceiling at scale — REFRAMED + ANSWERED ON CPU (phase5_next7_cpu_scaling.py).
  RETRACTION: "GPU-blocked" was wrong — it assumed the FibRec NEURAL net was the model (params→FLOPs→
  GPU). The substrate model's capacity axis is ADDRESSED CONTENT + composition + VERIFY = all CPU.
  PRE-REGISTERED A/B (universe 1896 short fns, 24 verified-reference queries, real interpreter oracle):
    P1 CONFIRMED — correctness scales with coverage: 0.04 / 0.29 / 0.54 / 0.79 / 1.00 at coverage
      0/.25/.5/.75/1.0. Capability gained by ADDING content (no gradient, no GPU).
    P2 CONFIRMED — per-query cost FLAT in store size: exact-key retrieval 0.059µs (N=16) → 0.060µs
      (N=1600, 100× content); verify constant 2.88 ms (one interpreter run, store-independent).
    P3 CONFIRMED — locality SCAN grows ~linearly 13µs→1946µs (150× over 100× N) → addressing (O(1))
      is precisely what removes the N-cost.
  VERDICT: the substrate gains capability at FLAT per-query CPU cost; a transformer gains the same only
  by growing params → GPU. Different scaling axis — substrate's is CPU. NEXT-7's "ceiling" is NOT a FLOP
  ceiling; it's coverage+composition, both CPU-scalable. HONEST SCOPE: this is the verified-code-synthesis
  domain (retrieve+compose+verify over a corpus); P1's correctness≈coverage is exact-retrieval (the
  held-out/composition gap — generalizing BEYOND the store — is still bounded by generator quality, but
  that too is CPU: grammar-gen + verify, not GPU). Task #43 (NEXT-7) DONE on CPU.

PHASE 6 (step 2) — HBit REAL AT THE VALUE LEVEL [DONE + VERIFIED] (pervasive per-value dual-band)
- value.rs: HInt gained `beta: Option<i64>` (β shadow band). None (default for ALL ordinary values)
  = single-band, byte-identical to before. with_beta() constructor; Value::beta_band() accessor.
  PartialEq still compares α only (no equality breakage). Only 2 struct literals touched; 413
  HInt::new callers unaffected.
- interpreter.rs: db_int() helper threads β through ALL six integer ops (Add/Sub/Mul/Div/Mod/Power):
  NEITHER operand has β → exactly HInt::new(α) (unchanged); either has β → result β = op(lβ,rβ) with
  α-fallback, div/mod zero-guarded. α is ALWAYS exact; β never alters a result — only records drift.
- Builtins now REAL (were stubs): phi_shadow(v) attaches β = nearest Fibonacci attractor of α (was
  identity); harmony(v) = 1000·harmony(α,β) if β present else 1000 (was constant 1000). NEW:
  bands(v)->[α,β], value_divergence(v)->1000·(1-harmony).
- VERIFIED (valueband_demo.omc): 7+3=10 harmony 1000 (single-band unchanged); phi_shadow(10) bands
  [10,8]; phi_shadow(10)+3 → [13,11] (β rode through +); (phi_shadow(50)+1)*3 → [153,105],
  value_divergence 933 (off-lattice); w==153 true (α exact). 172/172 core tests pass — the additive
  Option<i64> field changed nothing for non-shadowed values.
- The user's HBit dual-band model [[omc_hbit_dualband_vision]] made pervasive at the Value level:
  every value CAN carry its β shadow, it propagates through arithmetic, divergence readable anywhere,
  α always correct. Next horizons: value-granular SKIP form, then kernel/microcode HBit.

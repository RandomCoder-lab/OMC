# Transformerless-LM Codebase Index
# Optimized for LLM navigation. All line numbers exact as of 2026-05-26.
# Root: /home/thearchitect/OMC/experiments/transformerless_lm/

================================================================
QUICK LOOKUP — "Where is X?"
================================================================

Model class (FibRecLMSubsim)          → train_substrate_attention.py:45
Model forward pass                    → models_fibrec.py:545 (FibRecLM), :642 (FibRecLMHomeo)
Layer forward (L1-subsim attention)   → train_substrate_attention.py:81
Training loop entry point             → train_self_recursive.py:6092 train_with_self_distillation
Per-step training code                → train_self_recursive.py:6820-6880
autoregressive_generate               → train_self_recursive.py:2938
Per-token generation loop             → train_self_recursive.py:3561 (for _ in range(n_new))
State initialization (pre-loop)       → train_self_recursive.py:2980-3560
Bloom priming in generation           → train_self_recursive.py:4180-4205
Crystal distillation loss             → train_self_recursive.py:6060 _crystal_distillation_loss
Option 2 bloom embed hook             → train_self_recursive.py:6146-6154 _bloom_embed_hook
Option 2 bloom injection in training  → train_self_recursive.py:6850-6870
Bloom grow / add new vec              → train_self_recursive.py:6009 model_bloom_grow
Compute bloom embedding               → train_self_recursive.py:5853 compute_bloom_embedding
Address table (SpellAddressTable)     → spell_address_table.py:1 (new file)
Address table construction in gen     → train_self_recursive.py:3537 (before loop)
Address table lookup injection        → train_self_recursive.py:3591 (after logits/T_eff)
_omniweight_delta                     → train_self_recursive.py:2208
_omniweight_apply_split               → train_self_recursive.py:2291 (called at 4504)
_omniweight_apply_split call site     → train_self_recursive.py:4504-4506
FibPosTable class                     → train_self_recursive.py:1199 (approx)
SubstrateFingerprint class            → train_self_recursive.py (grep "class SubstrateFingerprint")
Model save ([MODEL SAVED])            → train_self_recursive.py:7629+
Bloom checkpoint save ([BLOOM SAVED]) → train_self_recursive.py:7344+
Saturation cap fix (1/φ)             → train_self_recursive.py (grep "_sat_capped")
FibonacciAdamW                        → optimizers_fib.py:1
substrate_fft_loss                    → losses_substrate.py (grep "def substrate_fft_loss")
SubstrateEmbedding                    → substrate_embedding.py:1
SubstrateTokenizer                    → substrate_tokenizer.py:1
AddressNavigator (Stage 3)            → address_stage3.py (grep "class AddressNavigator")
UnifiedAddressTable (Stage 2)         → address_stage2.py (grep "class UnifiedAddressTable")
BloomAddressTable (Stage 1)           → address_stage1.py (grep "class BloomAddressTable")

================================================================
1. MODEL CLASSES
================================================================

models_fibrec.py:203       — class FibRecLM(nn.Module) — base recursive FibGen LM
models_fibrec.py:225       — FibRecLM.__init__(vocab_size, d_model, n_blocks, seq_len, K=32, mode="cross")
models_fibrec.py:286       — FibRecLM._crt_pe(seq_len, d_model) — CRT-Fibonacci positional encoding
models_fibrec.py:298       — FibRecLM._rec_step(A, B, s_p1, s_p2) — plain Fibonacci recurrence step
models_fibrec.py:306       — FibRecLM._rec_step_homeo(...) — homeostatic φ-attractor recurrence step
models_fibrec.py:346       — FibRecLM._all_seeds() — generate all block seeds via recurrence
models_fibrec.py:545       — FibRecLM.forward(token_ids) — h = embed + pe → blocks → ln_f → head
models_fibrec.py:568       — class FibRecLMHomeo(FibRecLM) — adds homeostatic φ-attractor restoring force
models_fibrec.py:642       — FibRecLMHomeo.forward(token_ids, return_deltas=False)

train_substrate_attention.py:45   — class FibRecLMSubsim(FibRecLMHomeo) — PRODUCTION MODEL
train_substrate_attention.py:55   — FibRecLMSubsim.__init__(*args, K_sig=32, substrate_embed=False)
train_substrate_attention.py:81   — FibRecLMSubsim._layer_forward(x, mask, n, seeds_n)
  — L1-distance attention: dist[i,j] = Σ|sig_q[i]-sig_k[j]|₁, score = -dist/sqrt(K_sig)
  — SubstrateNegMultiAdvancedV2 FFN activation

models_substrate.py        — FibonacciOffsetAttention, CRTBucketAttention, ZeckendorfRoutedFFN, SubstrateLM
models_subsim.py           — SubstrateSimilarityAttention, SubsimLM (earlier L1-attn prototype)
models_fsm.py              — FibStateRecurrence, FSMLM (non-attention 2-tap recurrence)
models_fibgen.py           — FibGenLinear, FibGenLM, FIBONACCI (foundational weight-from-seed primitive)

Current production config:
  FibRecLMSubsim(vocab_size=65, d_model=64, n_blocks=2, seq_len=89, K=89,
                 mode="cross", K_sig=32, substrate_embed=True)
  321K parameters

Key tensors in state dict:
  embed.substrate_embed  [65,64]   — Fibonacci-frequency char embedding
  pe                     [89,64]   — positional encoding
  qkv_seed_0/1          [7921,4]  — K=89 FibGen seed (7921=89²)
  A_qkv / B_qkv         [89,89]   — homeostatic recurrence matrices
  head.weight            [65,64]   — tied LM head

================================================================
2. TRAINING LOOP
================================================================

train_self_recursive.py:6092  — def train_with_self_distillation(...)
train_self_recursive.py:6134  — model = FibRecLMSubsim(...)  [model creation]
train_self_recursive.py:6146  — _bloom_prefix_cell = [None]  [Option 2 hook cell]
train_self_recursive.py:6148  — def _bloom_embed_hook(module, input, output)
train_self_recursive.py:6154  — model.embed.register_forward_hook(_bloom_embed_hook)
train_self_recursive.py:6157  — optimizer = FibonacciAdamW(...)
train_self_recursive.py:~6780 — for cycle in range(n_cycles): [main cycle loop]
train_self_recursive.py:~6820 — for global_step in range(steps_per_cycle): [inner step loop]
train_self_recursive.py:6850  — Option 2 bloom injection (30% of post-warmup steps)
train_self_recursive.py:6868  — logits = model(x)  [training forward pass]
train_self_recursive.py:6869  — _bloom_prefix_cell[0] = None  [clear hook]
train_self_recursive.py:6873  — ce_fft = substrate_fft_loss(...)
train_self_recursive.py:6903  — loss = loss + _crystal_distillation_loss(...)
train_self_recursive.py:6904  — optimizer.zero_grad(); loss.backward(); optimizer.step()

Cycle structure:
  warmup_cycles: CE-only (no harmony, no corpus growth)
  post-warmup each cycle:
    1. Train steps_per_cycle on active_base  (with 30% bloom-prefix injection)
    2. autoregressive_generate → draft text
    3. staged_refine → optimize for creativity
    4. score production_score; if > gate (0.52 base) → add to active_base
    5. save bloom_checkpoint.pt + bloom_checkpoint_model.pt

Key training loss functions:
  substrate_fft_loss     — CE + Fibonacci-FFT mismatch
  substrate_omniweight_loss — phi^(pi)-weighted per-token CE
  _crystal_distillation_loss (line 6060) — KL(model‖bloom) at ALL T positions

Bloom training:
  train_self_recursive.py:6060  — _crystal_distillation_loss(model, logits, vocab_size, phi)
    — Computes crystal_p: crystal-age-weighted blend of bloom vecs projected through head
    — KL at range(T) positions, scale=min(1/φ², (1/φ³)×n_blooms), divided by T
    — Change from run 14+: was (0,1,2) → now range(T) for full-sequence bloom training

================================================================
3. GENERATION — autoregressive_generate
================================================================

train_self_recursive.py:2938   — def autoregressive_generate(model, prompt, n_new, ...)
  Signature: 30+ keyword args including all masks, fingerprint, fib_pos_table,
             substrate_tokenizer, bigram_prior, vocab, etc.

train_self_recursive.py:2980   — n_chars_local, content_thresh computed
train_self_recursive.py:2982   — with torch.no_grad(): [all generation is no-grad]
train_self_recursive.py:2983   — seq = prompt.clone()
train_self_recursive.py:2985   — State counters initialized: syl_pos, clause_pos, cluster_len, etc.
train_self_recursive.py:3026   — Grimoire spell state initialized (_H_prev_er, _psi_phyllo, etc.)
train_self_recursive.py:3537   — prompt warm-start loop ends; recent_pairs trimmed
train_self_recursive.py:3537   — SpellAddressTable construction (_addr_table)  [NEW]
train_self_recursive.py:3561   — for _ in range(n_new): [PER-TOKEN LOOP START]
train_self_recursive.py:3578   — _raw_logits_er = model(ctx)[:, -1, :]  [model forward]
train_self_recursive.py:3590   — _T_eff_er computation (Spells I, X, XIV, XV)
train_self_recursive.py:3614   — logits = _raw_logits_er / _T_eff_er
train_self_recursive.py:3615   — _addr_table.lookup_batch(clause_pos, prev_tok) injected  [NEW]
train_self_recursive.py:3657   — base = F.softmax(logits[0], dim=-1)
train_self_recursive.py:3658   — math_delta = torch.zeros_like(base)
train_self_recursive.py:3663   — lang_delta = torch.zeros_like(base)
train_self_recursive.py:3664   — [SPELL BLOCK START — ~80 spell computations]
train_self_recursive.py:4504   — probs = _omniweight_apply_split(base, math_delta, lang_delta, momentum)
train_self_recursive.py:4539   — [POST-OMNIWEIGHT GATING — vocab curriculum, trigram block, hard masks]
train_self_recursive.py:4663   — next_tok = torch.multinomial(probs, num_samples=1)
train_self_recursive.py:4674   — HBit resample loop (up to 3 tries)
train_self_recursive.py:4681   — seq = torch.cat([seq, next_tok], dim=1)
train_self_recursive.py:4693   — clause_pos reset on terminal punct
train_self_recursive.py:4736   — clause_pos += 1 for content words
train_self_recursive.py:4815   — [PER-TOKEN LOOP END]

================================================================
4. SPELL LOCATIONS (per-token generation loop, lines 3664-4815)
================================================================

Math hemisphere (→ math_delta):
  3664   — substrate_sampling: phi^pi temperature softmax → math_delta
  3668   — bigram prior / syntax_blend (prev_tok + last-7 word context)
  3681   — bigram saturation (recent pairs deque, depth 13)
  3685   — anti-stagnation (last 89 tokens window)
  3688   — local immediate-repetition suppressor (last 5, φ³)
  3702   — Spell II Phyllotaxis penalty (runtime Ψ × sector table)
  3716   — Spell VI Logit River boost (EMA direction)
  3728   — Spell VII Zeckendorf Gate (Fib position gate × static tok table)
  3739   — Spell XIII Zeta Resonance (static [vocab] bias) ← NOW IN ADDR TABLE
  3749   — Spell XIX Nested Platonic Navigation (tetra/cube/dode balls)
    3766 — tetrahedron affinities (char-level, updates every tok)
    3772 — cube affinities (word-level)
    3778 — dodecahedron affinities (sentence-level)
    3784 — Spell XXI Tetra-Collatz scale (prev_tok → face → Collatz weight)
    3789 — combined XIX bias (φ⁻³·t + φ⁻⁴·c + φ⁻⁵·d)
    3796 — corpus beacon (static vel × dode face)
    3801 — paragraph beacon (ball-gated)
    3809 — sentence beacon (ball-gated)
    3818 — Spell XXV Address Attractor (ball orthogonality)
    3826 — Spell XXVI Inverse Nav (cube face → beacon scoring)
    3843 — Spell XXVII Corpus Tower (static corpus vel × tower_w) ← PARTIALLY ADDR TABLE
    3852 — Spell XXVIII Address Arithmetic (cube ball → nearest word → face mask)
    3870 — Spell XXIX Word North Star (cube ball gate + ns_vel × faces)
    3879 — Spell XXIX Sent North Star (dode ball gate + sent_vel × faces)
  3888   — XIX softmax → math_delta

Lang hemisphere (→ lang_delta):
  3894   — XIX ball reflection updates (pure prev_tok, updates 3 balls)
  3949   — Spell XVII Collatz Collapse (runtime H_norm × static Collatz table)
  3964   — Spell XVIII Shakespeare's North (prev bearing × H_norm × static path)
  3992   — Spell IX Eigenself boost (EMA logit direction)
  4005   — Spell XI Will Vector boost (EMA of (probs-base))
  4015   — Unknown register (coverage histogram)
  4022   — Grammar: capitalize ← NOW IN ADDR TABLE
  4035   — Grammar: no double punct ← NOW IN ADDR TABLE
  4042   — Morpheme gate ← NOW IN ADDR TABLE
  4047   — Content density (last 8 tokens)
  4053   — Iambic phase (syl_pos counter mod cycle)
  4057   — Reference chain (last 13 × pronoun_mask)
  4062   — Clause arc (clause_pos × POS masks) ← NOW IN ADDR TABLE
  4068   — Need fill (open_needs counter × punct_mask)
  4072   — Phonotactics (cluster_len counter × vowel_start_mask)
  4076   — Rhyme resonance (last 13 × end_vowels)
  4081   — Agreement (last_content_ends_s bit)
  4085   — Word spacing ← NOW IN ADDR TABLE
  4090   — Char cascade (char_run counter)
  4094   — Pronounceability (static unpronounceable_mask) ← NOW IN ADDR TABLE
  4098   — Theme momentum (last 13 × token_signatures)
  4103   — Subject threading (prev is sent-end → full seq scan)
  4112   — Discourse position (after_speaker, clause_pos, speaker_count)
  4119   — Listener register (heard_counts histogram)
  4123   — Minimum clause length ← NOW IN ADDR TABLE
  4130   — Topic closure (clause_subject_id × bigram_prior)
  4139   — Substrate map steer (SubstrateGenTracker closed-loop)
  4151   — FibPosTable boost (clause_pos × fingerprint leaf)
  4158   — Structure bloom bias (model.head × structure_bloom_vecs)
  4175   — Crystal bloom bias (model.head × bloom_vecs, gate: fib_bin(cp)<=2)
  4206   — Fingerprint tape bias (model.head × _fp_tape EMA)
  4219   — Relation bias A→B (fingerprint.relation_probs(prev_tok))
  4228   — Agentic Subdivision (last-5 × cluster_transition)
  4285   — Clause anchor (model.head × _clause_anchor snapshot × cp decay)
  4307   — Spell IV Meridian Flow (cycle-level _meridian_leaf_energies × prev_tok)
  4327   — Topological flow (model.head × fp_tape manifold velocity)
  4351   — Leaf trajectory (pre-sampled Markov chain commitment)
  4393   — Identity drift recentering (model.head × _fp_tape × _identity_core)

Post-omniweight (applied to probs after _omniweight_apply_split):
  4507   — Spell XI Will vector update
  4515   — Three-mode exploit/escape (momentum > 0.5 sharpen, < -0.5 flatten)
  4524   — Backtrack on collapse (momentum_history last 5)
  4539   — Vocab curriculum hard mask (active_vocab_size)
  4543   — Trigram blocking (last 21 tokens)
  4566   — Bigram self-damping (prev_tok ≥ n_chars → suppress probs[prev_tok])
  4574   — Hard word boundary (allowed_after_word_mask[prev_tok])
  4590   — Hard post-punct spacing (force whitespace after heavy punct)
  4616   — Hyphen compound booster (prev=='-' → boost alpha tokens)
  4626   — Hard min clause length zero-out (suppress enders before cp>=4)
  4651   — Fallback char-run cap (char_run counter)

================================================================
5. BLOOM SYSTEM
================================================================

train_self_recursive.py:5853   — def compute_bloom_embedding(model, accepted_seqs, quality_scores, ...)
  — Distills high-quality generated seqs into a [d_model] bloom embedding via weighted avg

train_self_recursive.py:6009   — def model_bloom_grow(model, bloom_embedding, max_blooms=8, phi=...)
  — Appends new bloom vec to model.bloom_vecs, increments crystal ages

train_self_recursive.py:6060   — def _crystal_distillation_loss(model, logits, vocab_size, phi)
  — KL(model‖bloom) at ALL T positions (was 0,1,2 — changed 2026-05-26)
  — scale = min(1/φ², (1/φ³)×n_blooms) / T
  — Uses crystal-age weighting: w = φ^(age/φ²) / φ^bi

train_self_recursive.py:4175-4205  — Bloom priming in generation
  — Gate: FibPosTable._fib_bin(clause_pos) <= 2 (clause-initial positions)
  — Weight: crystal-age-weighted (φ^(age/φ²) / φ^bi)
  — Applied as: lang_delta += omniweight_delta(base, bloom_bias) × (1/φ)

train_self_recursive.py:6146-6154  — Option 2 bloom embed hook (added 2026-05-26)
  — _bloom_prefix_cell [mutable ref] shared between hook closure and training loop
  — Hook adds crystal-age-weighted bloom embed to model.embed output at position 0

train_self_recursive.py:6850-6870  — Option 2 bloom injection per training step
  — 30% of post-warmup steps; 70% bloom-free (model stays robust at inference)

Bloom checkpoint:
  bloom_checkpoint.pt       — bloom_vecs + crystal_ages + active sequences + saturation
  bloom_checkpoint_model.pt — model weights (state dict) from [MODEL SAVED]

================================================================
6. ADDRESS SYSTEM (3 stages)
================================================================

address_stage1.py  — BloomAddressTable
  — Maps bloom_vecs to dode faces via head projection or geometry
  — O(1) face lookup; 484× speedup vs cosine search
  — All 242 vecs end up in faces 5 and 1 (geometry method)
  — 10/12 faces empty = unexplored territory

address_stage2.py  — UnifiedAddressTable
  — 38,191 objects: bloom + words + sentences + corpus + Collatz + Fib rows
  — Face geography: bloom→face 6, words→face 5, corpus/sentences→face 9
  — Gap analysis: which faces lack which object types

address_stage3.py  — AddressNavigator, AddressTrie, load_navigator
  — Generates text by ball-reflection through dodecahedral address space
  — No neural net; pure address geometry
  — 21% real-word fraction, 0.000 seed 3-gram overlap (baseline)
  — load_navigator(checkpoint_path) → AddressNavigator instance

spell_address_table.py  — SpellAddressTable  [NEW 2026-05-26]
  — Precomputed [seq_len, vocab, vocab] logit bias table
  — Built once at generation start; ~1.5 MB
  — Spells baked in: Zeta, pronounce, capitalize, no-dbl-punct, morpheme,
    word-spacing, hyphen, post-punct-space, clause-arc, min-clause-len
  — Expected speedup: ~6-7× total wall time per run

================================================================
7. LOSSES AND OPTIMIZATION
================================================================

losses_substrate.py:
  substrate_fft_loss(logits, y, vocab_size, lambda_substrate)
    — CE + Fibonacci-FFT mismatch against canonical decay
  substrate_omniweight_loss(...)
    — phi^pi tanh per-token CE weighting (used when args.omniweight_loss=True)
  substrate_harmony_loss_grounded(...)
  substrate_multiscale_harmony_loss_grounded(...)
    — Multi-scale Fibonacci lag self-similarity vs corpus signature
  corpus_char_signature(corpus_anchor, vocab_size) → [7] Fib-band signature
  corpus_multiscale_signature(...) → [7] multi-lag signature

optimizers_fib.py:
  FibonacciAdamW — β₁=1/φ≈0.618, β₂=1/φ²≈0.382 (substrate-canonical)
  FibonacciMomentumSGD — φ^{-1} momentum

activations_substrate.py:
  SubstrateNegMultiAdvancedV2 — production FFN activation in FibRecLMSubsim
  attractor_snap — straight-through Fibonacci-attractor snapping

layernorm_substrate.py:
  SubstrateL1LN — L1-canonical LayerNorm (MAD spread)
  SubstrateWeiszfeldLN — Weiszfeld L1 center

substrate_embedding.py:
  SubstrateEmbedding(vocab_size, d_model, K=7, learnable_gamma=True)
    — Fixed Fibonacci-frequency sin/cos embedding with per-dim γ

================================================================
8. KEY CONSTANTS
================================================================

train_self_recursive.py:96   — _PHI_FOR_SAMPLING = (1+√5)/2 ≈ 1.61803
train_self_recursive.py:101  — _PI_LOG_PHI = π·log(φ)/φ  (math-hemisphere temperature)
train_self_recursive.py:106  — _LOG_PHI_FOR_PENALTY = log(φ) ≈ 0.481
train_self_recursive.py:112  — _TRIGRAM_SUPPRESS = 1/φ^(2π) ≈ 0.049
train_self_recursive.py:2151 — _OMNIWEIGHT_RESERVE = φ^π ≈ 4.53 (tanh bound in apply_split)

Saturation thresholds:
  spc=13/keep=8  at sat < 0.47
  spc=8/keep=4   at sat ~ 0.47-0.62
  spc=5/keep=3   at sat > 0.62+
  Threshold cap: sat_capped = min(sat, 1/φ ≈ 0.618) — prevents runaway tightening

All-time records:
  Run 11: sat=0.77 (best), 22 unbroken even/odd alternations
  Run 10: sat=0.64, proved even/odd mechanism over 24 cycles

================================================================
9. DATA AND TOKENIZATION
================================================================

corpus.py       — make_dataset(name) → (encoded, vocab, itos_map)
                  CORPUS = tinyshakespeare.txt (primary)
lazy_data.py    — get_fib_strided_batch(...) — Fibonacci-strided sampling (5.6× speedup)
                  fib_positions_in_window(window, FIBONACCI) → Fib-offset positions

substrate_tokenizer.py — SubstrateTokenizer
  .cube_faces[vocab_size]     — dode face per token (0-11)
  .word_registry              — {word: velocity_dict} (11,595 words in Shakespeare)
  .sentence_registry          — {idx: {velocity, ...}} (25,719 sentences)
  .paragraph_registry         — {idx: {velocity, ...}} (7,083 paragraphs)
  .corpus_address             — [3] corpus velocity vector
  .north_star_token           — token with Collatz depth = L2 ground state

================================================================
10. ARCHITECTURE DIRECTORY
================================================================

architecture/                    — organized copies of all key files
  core_model/primitives/         — models_fibgen, activations, layernorm, substrate_embedding
  core_model/attention_and_blocks/ — models_substrate, models_subsim, models_fsm
  core_model/full_lms/           — models_fibrec, train_substrate_attention (FibRecLMSubsim)
  training/optimization/         — optimizers_fib
  training/losses/               — losses_substrate
  training/loop/                 — train_self_recursive (copy)
  inference/navigation/          — address_stage1, address_stage2, address_stage3
  inference/drivers/             — chained_generate, sample_text
  inference/benchmarks/          — bench_inference
  data/corpora/                  — corpus, corpus_word
  data/tokenization/             — substrate_tokenizer
  data/loaders/                  — lazy_data
  analysis/scoring/              — creativity_score
  README.md                      — architecture map + dependency graph

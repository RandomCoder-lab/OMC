# MANIFEST — the addressed-memory → dot-connector → cross-domain-analogy arc (2026-05-30)

What this session built: a small, agnostic, **CPU** cognitive stack that connects dots across a knowledge
space — knowledge written by address (no training), grounded multi-hop reasoning, self-learned meaning,
the closed loop, a calibrated voice, and an analogy/joke mechanism. Full ledger + results numbers live in
**GENERATOR_PLAN.md** (BUILD-1 … BUILD-21). This file is the *map* — what each file is and how they connect.

> Organizing note: these files form ONE interdependent import-component (see the dependency graph below)
> and also depend on the older `corpus.py`/`models.py` that the prior training scripts use. They are
> grouped here *logically*, not moved on disk — moving them would break sibling imports. Older/prior-session
> files (the addressing/substrate-LM/assistant lines) are listed last and are NOT part of this arc.

---

## 1. CORE ENGINE — write-don't-train memory + IVF index (the foundation)
| file | what | build |
|---|---|---|
| `written_memory.py` | **the engine.** `WrittenMemory`: train tiny LM → WRITE (hidden→next-token) datastore (no backprop) → IVF index; pluggable tokenizer; score(bits/char), complete, complete_chunked (span-copy), nearest_contexts (provenance). | 5/6/7 |
| `bpe_tokenizer.py` | `CharTokenizer` + `BPETokenizer` (one interface, offsets for provenance). BPE was the generation lever. | 5 |
| `codemem.py` | CLI over `WrittenMemory`: `build / info / eval / search / complete [--chunked]` on any folder. | — |
| native: `../../omnimcode-core/src/addressed_memory.rs` + `tests/addressed_memory_omc.rs` | in-core kNN+IVF datastore; 5 OMC builtins `amem_new/write/index/search/len`; 2 unit + 4 OMC tests, no-regression. | 4 |

## 2. CEILING & SYNTHESIS experiments (where the walls are)
| file | what | build |
|---|---|---|
| `ceiling.py` | generator's TRUE ceiling: bits/char on unseen domain vs size, bare vs +substrate. Found: +substrate = ~7× param-efficiency; prediction floor ~2.16; permanent edge is *structural* (live knowledge), not perplexity. | 13 |
| `synth_verified.py` | global coherence by composition (unit-retrieval + span-copy + grammar + REAL-interpreter verify-gate). Modes gates/spec/evolve. Located the wall: no spec → "coherence" caps at "runs"; flat fitness landscape. | 9/10/11 |

## 3. DOT-CONNECTOR / REASONING — retrieve → chain → verify → rank-by-meaning
| file | what | build |
|---|---|---|
| `reasoning_nav.py` | reasoning = grounded multi-hop NAVIGATION (scales with index, not params). Fair LM baseline; retrain-free reasoning on knowledge added after training. | 14 |
| `nl_ground.py` | **dot-connector hub.** NL grounding: a hop is real iff a passage co-mentions both. Exports `clean_text / split_passages / extract_entities / fp` (everything downstream imports these). | 15 |
| `nl_deep.py` | addressed inverted index → per-hop recall **1.00** cheaply; deep reachability. Finding: one dense corpus is small-world. | 16 |
| `nl_xdomain.py` | cross-domain via shared bridge term (honest negative: lexical bridges hollow) + agnostic thread-strength ranking. | 16 |
| `sem_rank.py` | **meaning-judge.** Skip-gram embedding learned from corpus co-occurrence (zero labels). Predicts held-out relatedness AUC 0.83 vs lexical 0.19. | 17 |
| `connect.py` | **THE LOOP CLOSED.** `ConceptSpace` = WHERE+GROUND+MEANING+REASON; meaning-guided best-first walk (3.2× more coherent than blind BFS); `discover()`. Exports `ConceptSpace / train_embedding`. | 18 |
| `voice.py` | `Narrator`: narrates discovered chains with calibrated confidence ("I think… I'm reaching here…"), type-inferred, every hop quoted. | 19 |
| `shape.py` | relation-SHAPE resonance: r(A→B)=vec(B)−vec(A); parallel shapes = analogy / joke-pivot; analogy arithmetic. | 19 |
| `multidomain.py` | full stack on 5 distant domains; cross-domain analogy is REAL (shape-cosine 0.774 vs 0.406 shuffled). Strongest = place/role rhymes. | 20 |
| `process.py` | abstract RELATIONS: rhyme PROCESSES (verbs) across domains. Verbs derived by INFLECTION PARADIGM (agnostic, no word lists). | 21 |

## 4. CHAT / SKILLS
| file | what | build |
|---|---|---|
| `chat.py` | router REPL: char-question → exact char-skill; else addressed-memory recall; honest "no match". `--demo`. | — |
| `char_skills.py` | char-level ops token-LLMs fail (D's in days-of-week = 8). LAW-CLEAN: logic only + corpus resolver, no hardcoded lists. | 15 |
| `knowledge.txt` | the corpus the char resolver reads (knowledge as DATA, not code). | 15 |

## 5. AM-SERIES experiments (the write-don't-train ladder → the engine) + deps
| file | what |
|---|---|
| `am1_addressed_memory.py` | AM-1a PKM (failed) — but defines `LM`/`DenseFFN` (CRT char-LM arch) reused by the engine. |
| `am2_write_memory.py` | AM-2 — defines `train_base`/`hidden_and_logits`/`build_datastore`. |
| `am3_unseen_memory.py` | AM-3 knowledge-injection decider (+6.22% net). |
| `am4_gated_memory.py` | AM-4 dual-band confidence gate (negative −1.7pts). |
| `am5_tune.py` | AM-5 honest baseline (+11.35%). |
| `am6_substrate_index.py` | AM-6 dodecahedral big-k (lost to IVF). |
| `am7_microcosm.py` | AM-7 the 35B microcosm. |
| `idx_night.py` | "Index Done Right" 159-run frontier study → IVF wins. |
| `corpus.py`, `models.py` | **older shared deps** (also used by prior training scripts) — `get_batch`, `crt_pe`, `Attention`. Do NOT move. |

---

## Dependency graph (verified) — what imports what
```
corpus.py, models.py        (older shared infra — load-bearing for both arcs; leave in place)
  └─ am1_addressed_memory.py (LM/DenseFFN)
  └─ am2_write_memory.py     (train_base, build_datastore)
bpe_tokenizer.py             (standalone)
written_memory.py  ← am1, bpe_tokenizer, corpus            ◀ ENGINE HUB
  ├─ codemem.py
  ├─ ceiling.py   (← also am2, bpe_tokenizer)
  ├─ chat.py      (← also char_skills)
  └─ synth_verified.py
char_skills.py     (standalone; reads knowledge.txt)
reasoning_nav.py   ← am1, corpus
nl_ground.py       (standalone)                            ◀ DOT-CONNECTOR HUB
  ├─ nl_deep.py
  ├─ nl_xdomain.py
  ├─ sem_rank.py
  └─ connect.py    (exports ConceptSpace / train_embedding) ◀ LOOP HUB
       ├─ voice.py
       ├─ shape.py
       ├─ multidomain.py  (← also nl_ground)
       └─ process.py      (← also nl_ground)
```
Leaf files (safe to run/move alone): codemem, ceiling, chat, synth_verified, reasoning_nav, voice, shape,
multidomain, process. Hubs (moving breaks importers): written_memory, nl_ground, connect, bpe_tokenizer,
am1, am2, corpus, models.

## Results & data
- **Ledgers:** `GENERATOR_PLAN.md` (the BUILD-1..21 narrative + numbers), `INDEX_RESULTS.md` (idx_night 159 runs).
- **Per-step JSON (this session):** `results_am{1..6}.json`, `results_ceiling.json`, `results_synth_{verified,spec,evolve}.json`, `results_reasoning_nav.json`, `results_nl_{ground,deep,xdomain}.json`, `results_sem_rank.json`, `results_connect.json`, `results_shape.json`, `results_multidomain.json`, `results_process.json`, `results_index.json`.
- **Corpora:** `corpora/{science_darwin,detective_holmes,history_gibbon,philosophy_nietzsche,romance_austen}.txt` (5-domain); `pride_prejudice.txt`, `tinyshakespeare.txt`; `omc_corpus.txt`/`omc_codebase.txt`/`omc_fndefs.txt` (code).

## NOT part of this arc (older / prior-session — separate lines)
`addressed_memory.py` (OLD 12-face episodic memory — distinct from `written_memory.py`), `substrate_lm.py`,
`multiskill_navigator.py`, `name_registry.py`, `universal_store.py`, `locality_fp.py`, `omc_assistant.py`,
the `address_*`/`hierarchical_*`/`corpus_address_index.py` addressing line, the `train_*`/`*_substrate.py`
transformer-substrate line, `*.pt` checkpoints, `phi_synthesis.py`, `grammar_gen.py`, etc.

## Governing laws (honored throughout) — see memory [[omc_development_plan]], [[feedback_no_hand_lists_derive]]
1. **Universal substrate** — no hand-coded dictionaries/word-lists in the authoritative path; derive from
   the corpus (verbs by inflection paradigm, knowledge from a corpus file, etc.).
2. **Integer-substrate / attenuable** — addressing decides WHERE (index, proven); learned floats decide
   WHAT (meaning/values). Substrate-as-scorer was falsified; don't recross.

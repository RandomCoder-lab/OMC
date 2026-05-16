# The Context Problem, Reframed via ONN

## The problem (as commonly stated)

LLMs have finite context windows. As a conversation grows past N
messages, older content must be evicted — losing information.

## What people try

- **Long-context models**: throw more tokens at it. Limited by O(N²)
  attention or O(N) memory; gets expensive.
- **RAG / vector retrieval**: embed history, retrieve top-k.
  Requires an embedding model, has recall problems.
- **Summarization**: collapse N messages to a summary. Lossy,
  no quality bound.

## What ONN claims (and OMC now operationalizes)

For any N messages, you only need **M3(N)** "specialist summaries"
to preserve the field-state of the conversation. M3 grows
**sublogarithmically** — even for N = 10⁶, you need ~25 specialists.

```
N           M3(N)    compression
10            3         3×
50            7         7×
100           7        14×
500          11        45×
1,000        12        83×
5,000        16       312×
10,000       18       555×
```

(Reproducible: `./target/release/omnimcode-standalone examples/demos/context_compression.omc`)

## Why this isn't just summarization

A specialist isn't a paraphrase. It carries:

- `mu` — mean φ-resonance of the items it covers
- `sigma` — variance of resonance
- `dominant_attractor` — nearest Fibonacci to the mean content hash
- `fold_index` — its position in the geometric phase-spread
- `wave_amplitude` — its strength in the φ-field
- `item_count` — how much it absorbed
- `summary` — the raw concatenation (callers can swap for a real
  summarizer)

The substrate properties (`mu`, `sigma`, `dominant_attractor`) are
**measurable**, **comparable**, and **fold-back-able**. You can:

- Measure drift between sessions (Δμ, Δσ)
- Retrieve by substrate distance (which specialist is nearest to
  the current query's resonance?)
- Re-fold without information loss in the statistics, even if you
  drop the raw `summary` text

That's the ONN claim: the substrate-derived statistics ARE the
memory. The summary is a courtesy.

## What's implemented

| Builtin | Purpose |
|---------|---------|
| `omc_m3_spawn_count(n)` | M3 optimal subagent count |
| `omc_self_instantiate(items, hint)` | Fold N items → M3(N) specialists |
| `omc_fold_back(mu, sigma, turn, specs)` | Update parent state from children |
| `omc_context_compress(messages)` | Headline: N msgs → ~log_log(N) specs |
| `omc_prompt_agent(target_id, prompt, sender_id)` | Secondary-brain: fire-and-forget |

## What this is honest about

- **Not lossless**. The raw text of individual messages is dropped
  (only the summary truncation survives). What's preserved is the
  *substrate field state*.
- **Quality of the summary depends on the substrate metric**. For
  natural-language conversation, φ-resonance over fnv1a content
  hashes isn't a topical-similarity signal (we already proved this
  in the prime-resonance study). So substrate-distance retrieval
  here is *geometric*, not *semantic*. For real topical retrieval
  you'd layer an embedding model on top.
- **The M3 bound is empirical/heuristic**, not proven. It comes
  from Hermes's wave-interference threshold experiments. The
  sublogarithmic growth IS observed; whether it's the *optimal*
  basis is not formally proven.

## Why this still matters

Even if the substrate-statistics aren't topical, they ARE:
1. **Stable across rebuilds** (deterministic from content)
2. **Verifiable** (recomputable from content)
3. **Bounded above** by M3(N) regardless of how big N gets
4. **Composable** (fold-back is associative: fold(fold(A, B), C) ≈
   fold(A, fold(B, C)))

That's enough to build a working "geometric memory" that an LLM
can reason over without holding the raw bytes. Combined with the
substrate-signed messaging from earlier, two LLMs can also
exchange compressed context: send only the specialist dicts,
verify substrate integrity, and have geometric continuity across
agents and across sessions.

## What's NOT solved

- **Topical retrieval**: still needs embeddings.
- **Reconstruction of individual messages**: lost. Only the summary
  truncation survives.
- **Cross-vocabulary compatibility**: two OMC builds with different
  TOKEN_DICT versions produce different hashes → different
  specialists. Pin the version (use protocol kind=5 handshake).

## Verdict

**Partial solution** — bounds the compression problem, gives
geometric continuity, doesn't replace topical retrieval. Useful as
the *baseline memory layer*; layer specific retrieval on top.

The substrate gives you *structural continuity* (μ, σ, attractor
drift) for free; topical continuity is a separate problem.

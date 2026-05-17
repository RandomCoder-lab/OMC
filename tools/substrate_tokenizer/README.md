# Substrate-aware tokenizer infrastructure

Pipeline to train an LLM where the top-N most-common OMC canonical
hashes get reserved single-token IDs in the vocabulary. The LLM
then writes `<omc:tok_42>` (one token) instead of repeating the
full source body across context.

This is goal 4 of the OMC-as-content-addressed-AI plan. The
infrastructure ships today; the actual fine-tune on a meaningful
base model needs a GPU.

## Pipeline

```
            corpus_collect.py        build_vocab.py           train_fine_tune.py
                  │                        │                          │
   .omc files ───>│                        │                          │
                  ▼                        ▼                          ▼
        canonical_hash_index.jsonl  hash_token_table.json   fine_tuned_model.pt
                  │                        │                          │
                  └────────────────────────┴──────────┐               │
                                                      ▼               │
                                          tokenizer_eval.py ◀─────────┘
```

| Stage | Script | Input | Output |
|---|---|---|---|
| 1 | `corpus_collect.py DIR` | Directory of `.omc` files | `canonical_hash_index.jsonl` — `{canonical_hash, fn_name, source, count}` |
| 2 | `build_vocab.py --top N` | The index | `hash_token_table.json` — `{token_id: canonical_hash}` for the top N |
| 3 | `train_fine_tune.py [args]` | The table + a base model | `fine_tuned_model.pt` |
| 4 | `tokenizer_eval.py model.pt` | Trained model + test corpus | Token-compression metrics + completion quality |

Stages 1–2 are fast (CPU, minutes). Stage 3 is multi-day on a GPU
for a meaningful base model. Stage 4 measures the actual context-
compression win.

## What ships today

**1. Corpus collector (CPU, fast)** — walks a directory, extracts
every OMC fn, computes canonical hash, counts occurrences. Produces
the JSONL index that downstream stages consume.

**2. Vocabulary builder (CPU, fast)** — reads the index, picks the
top-N canonical hashes by count, assigns them reserved token IDs
in a `[unused_0..unused_N]` range that most tokenizers reserve for
fine-tune extensions.

**3. CPU sanity fine-tune** — a tiny GPT-2-shaped model (~10M
params) trained on a synthetic corpus where the top-N hashes are
overrepresented. Demonstrates the training loop works end-to-end
in ~5 min on CPU. Not a useful model; just proves the pipeline.

**4. Tokenizer evaluator (CPU)** — measures, for a given input
text:
  - Naive BPE token count
  - Substrate-aware token count (hash-refs → 1 token each)
  - Compression ratio

Run on real workloads to project the win before committing to GPU.

## What needs GPU

The actual fine-tune on a real base model (Llama-3 8B, Mistral 7B,
or even a smaller code-focused base like StarCoder2-3B) requires
GPU time. Launch instructions for a single-node 1×A100 setup are in
`gpu_fine_tune.md`. Cost estimate: ~$50–200 depending on base
model size + dataset.

## Honest expected wins

For an agentic workload that heavily reuses standard library fns:
- Naive BPE: each fn reference costs ~10–100 tokens
- Substrate tokens: each fn reference costs 1 token
- Realistic context-compression: 3–10× on code-heavy workloads
- Worst case (no fn reuse): ~1× (no harm)

The fine-tune teaches the model to EMIT `<omc:hash>` tokens when
appropriate. Without that training, the LLM treats them as
unfamiliar special tokens.

## Why this is the long-term unlock

If a major code-LLM is fine-tuned with substrate-aware tokens:
- Every agentic system using that LLM gets cost/context savings
  for free
- The kernel becomes the universal back-end for canonical-hash
  resolution
- The transformerless-LM thesis gains its third validated
  substrate component beyond CRT-PE + HBit-OOD + geodesic-attention

This is the infrastructure that makes that fine-tune cheap to
attempt. The hardest engineering (canonicalization, kernel, codec,
geodesic) is done. The remaining work is dataset curation +
hyperparameter sweeps — bounded compute, bounded time.

## Files

| File | Purpose |
|---|---|
| `corpus_collect.py` | Stage 1: walk OMC files, build canonical-hash index |
| `build_vocab.py` | Stage 2: select top-N hashes, emit token table |
| `train_fine_tune.py` | Stage 3: CPU sanity fine-tune (proves pipeline) |
| `tokenizer_eval.py` | Stage 4: measure compression on real text |
| `gpu_fine_tune.md` | Launch instructions for a meaningful GPU run |
| `README.md` | This file |

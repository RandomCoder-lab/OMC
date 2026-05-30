# OMC Transformerless LM — Architecture Map

Copies of all architectural files, organized by layer. The live training runs from
the parent directory; these copies are for study and reference.

```
architecture/
├── core_model/
│   ├── primitives/          — foundational building blocks (no model-level deps)
│   ├── attention_and_blocks/ — attention variants and block compositions
│   └── full_lms/            — complete LM classes ready for training
│
├── training/
│   ├── optimization/        — FibonacciAdamW, phi-momentum SGD
│   ├── losses/              — CE + FFT, harmony, omniweight losses
│   └── loop/                — main self-distillation training loop + autoregressive_generate
│
├── inference/
│   ├── navigation/          — address stages 1-3 (bloom → dodecahedral navigation)
│   ├── drivers/             — chained_generate, sample_text driver scripts
│   └── benchmarks/          — throughput + weight-memory benchmarks
│
├── data/
│   ├── corpora/             — corpus loaders (char + word level)
│   ├── tokenization/        — SubstrateTokenizer (dodecahedral char mapping)
│   └── loaders/             — lazy_data Fibonacci-strided batching
│
└── analysis/
    └── scoring/             — creativity_score composite metric
```

## Dependency order (bottom → top)

```
data/corpora · data/tokenization · data/loaders
       ↓
core_model/primitives
       ↓
core_model/attention_and_blocks
       ↓
core_model/full_lms     ←——— training/optimization + training/losses
       ↓
training/loop           ←——— analysis/scoring
       ↓
inference/navigation    ←——— inference/drivers · inference/benchmarks
```

## Current production model

`FibRecLMSubsim` in `core_model/full_lms/train_substrate_attention.py`

- Extends `FibRecLMHomeo` (homeostatic phi-attractor recurrence)
- L1-distance substrate-similarity attention (replaces Q·K^T)
- SubstrateNegMultiAdvancedV2 FFN activation
- SubstrateEmbedding (Fibonacci-frequency basis, tied head)
- 321K parameters: d=64, vocab=65 chars, seq_len=89, K=89, 2 blocks

## Key runtime objects (in training/loop/train_self_recursive.py)

| Object | Role |
|---|---|
| `FibPosTable` | Fibonacci position → token-sector mapping |
| `SubstrateFingerprint` | semantic clustering via Fibonacci co-occurrence |
| `ParametricSubstrate` | mutable substrate constants |
| `SubstrateGenTracker` | runtime steering compass |
| `model.bloom_vecs` | list[Tensor[d]] — crystal memory vectors |
| `model.bloom_crystal_ages` | list[int] — age of each bloom vector |

## Three-stage address system

| Stage | File | What it does |
|---|---|---|
| 1 | `address_stage1.py` | bloom vecs → dode face, O(1) retrieval (484× faster than cosine) |
| 2 | `address_stage2.py` | unified (face, depth, velocity) table: bloom + words + sentences + corpus |
| 3 | `address_stage3.py` | generate text by ball-reflection through dode address space (no neural net) |

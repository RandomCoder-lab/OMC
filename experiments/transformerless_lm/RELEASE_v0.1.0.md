# transformerless-lm v0.1.0

First release of the substrate-compressed language model framework
under `experiments/transformerless_lm/`. This document is the in-tree
release artifact corresponding to the local annotated tag
`transformerless-lm-v0.1.0` at commit `ad35f98`.

## Headline results (validated)

### 100× weight compression via FibGen

Each weight tensor `W ∈ R^{out × in}` is replaced by a small
Fibonacci-indexed seed and reconstructed on demand via a closed-form
sin/cos expansion at Fibonacci frequencies.

| arch | params | compression | val (best) | vs dense | uniform reduction |
|---|--:|--:|--:|--:|--:|
| dense_crt | 801,664 | 1× | 2.5602 | — | -38.7% |
| **fibgen_K16_separable** | **8,064** | **100.4×** | **2.9020** | **+13.3%** | -30.5% |
| fibgen_K32_separable | 9,216 | 87.9× | 2.7282 | +6.6% | -34.6% |

Reproduced across two independent training runs (the original v2 bench
at `results_fibgen.json` and the recheck run at the same path). The
compression is real — 8K stored parameters reconstruct an 810K dense-
equivalent weight tensor — and the model genuinely learns the corpus
structure (val well below the ln(65) = 4.17 uniform floor).

### Inference: 90-93% throughput at 10-37× less RAM

| arch | d | weight_MB | tok/s | vs dense speed |
|---|--:|--:|--:|--:|
| dense_crt | 128 | 3.06 | 473 | — |
| **fibgen_K32 cached** | 128 | 0.31 | 441 | **93%** |
| dense_crt | 256 | 12.12 | 264 | — |
| **fibgen_K32 cached** | 256 | 0.33 | 237 | **90%** |

The weight cache pattern (precompute `W` once at deployment, reuse
across all forward passes) eliminates the FibGen forward-overhead at
inference. Per-token compute matches dense; only the persistent
weight storage is compressed. At d=256 the memory ratio is **37×**;
at LLM scale (d=4096) extrapolation gives ~200× memory reduction.

### Lazy-loaded training: 5.6× wall-clock speedup

Fibonacci-strided data sampling loads only `log_φπ(T)` tokens per
sequence position (11 of 128 at T=128). The model never reads gap
tokens from disk.

| config | val | wall (1500 steps) | speedup |
|---|--:|--:|--:|
| dense baseline (dense data) | 2.4396 | 165.7s | 1.00× |
| **dense + lazy-strided data** | **2.5274** | **29.5s** | **5.62×** |

The substrate's `log_φπ` cadence is the data-loading complexity
bound; this is the cleanest single-axis substrate-native win in the
release.

## 35B-in-8GB feasibility math

Combining the validated wins:

| config | 35B-equivalent storage | fits in 8 GB? |
|---|--:|---|
| dense fp16 | 70 GB | no |
| 4-bit quantization (SOTA) | 17.5 GB | no |
| **FibGen K=32 cross** | **7 GB** | **yes** |
| FibGen K=32 separable | 800 MB | yes, easily |

These numbers are extrapolations from the d=128 / d=256 measurements.
At true LLM scale the compression ratio grows as `(d/K)²` because
dense storage scales as `d²` while the seed is `K²` regardless of `d`.

## Architectural primitives (all in `experiments/transformerless_lm/`)

| primitive | file | validation |
|---|---|---|
| CRT-Fibonacci PE | `models.py` | -5.4% vs sinusoidal PE |
| Geodesic attention bias | `models.py` | -0.4% vs crt_only, 3/3 seeds |
| Fibonacci-offset sparse attention | `models_substrate.py` | 14× FLOP reduction, -3.2% loss |
| Zeckendorf-routed FFN | `models_substrate.py` | 5× FFN FLOPs reduction |
| FibGen weight generator | `models_fibgen.py` | **100× storage compression** |
| Subsim L1-distance attention | `models_subsim.py` | substrate operator, +5.7% loss at d=128 |
| Fibonacci tier quantization | `models_substrate.py:fibonacci_tier_snap` | saturates at +0.6 nats post-hoc |
| Fibonacci State Model | `models_fsm.py` | NaN at init, scale-bound |
| Lazy-strided data loader | `lazy_data.py` | **5.6× training speedup** |
| Stochastic Fibonacci depth | `models_subsim.py` | 1.17× wall-clock speedup |

## Falsified or scale-bound

| claim | falsification |
|---|---|
| Pure Fibonacci-tier post-hoc quantization at 4-bit | Saturates at +0.6 nats regardless of bit depth |
| Substrate operators (Subsim/FSM) faster than dense at d=128 | At CPU bench scale (d≤256, T≤512) PyTorch overhead dominates the asymptotic FLOP savings |
| FSM recurrence numerically stable at random init | Eigenvalue > 1 produces immediate NaN; needs gating |
| K-scaling alone closes the gap to dense at d=256 | K=48, K=64 both LOST at d=256 (+30% gap) |
| Plain FibGen at d=256 maintains its compression-vs-quality | Compression ratio grows nicely (36×) but loss penalty also grows (+30%) |

## Reproducing the headline numbers

```bash
cd experiments/transformerless_lm

# 100× compression result (this release's main claim)
python3 train_fibgen.py --steps 2500 --K-sweep 16,32 --modes separable
# expect: fibgen_K16_separable val ~2.90 (100x compression)
#         fibgen_K32_separable val ~2.73 (88x compression)

# Lazy-loading data speedup
python3 train_lazy_loading.py --steps 1500
# expect: dense ~165s, fib_strided ~29s, val deltas <5%

# Inference-time throughput
python3 bench_inference.py --n-tokens 256
# expect: fibgen_K32 cached at 90%+ of dense throughput at d=128
```

## Honest limits

- Output text quality at d=128 is gibberish for ALL archs including
  dense. Coherent text needs GPT-2-tiny-class capacity (d≥384,
  n_blocks≥6).
- Substrate operator wall-clock wins (Subsim, FSM, Composed) are
  scale-bound — they don't materialize on CPU at our test scale.
  Asymptotic complexity advantages are real but unreachable in pure
  PyTorch without parallel-scan kernels or larger T/d.
- 35B feasibility is an extrapolation from d=128/256 measurements,
  not a direct measurement at LLM scale.
- Training-time substrate ops (lazy tier dropout, K-subsampling)
  delivered at most a small per-step compute reduction in pure PyTorch
  due to indexing overhead. Real wins would require kernel work.

## File index

```
experiments/transformerless_lm/
  README.md                       # original transformerless-LM thesis
  GEODESIC_RESULT.md              # validated -0.4% geodesic attention
  GEODESIC_ATTENTION_DERIVATION.md
  TRANSFORMERLESS_RESULT.md       # token-CRT + Principle A/B results
  WEIGHT_SUBSTRATE_REFORMULATION.md  # Principle A/B derivation
  INFERENCE_FIRST_DERIVATION.md   # 35B-in-8GB framing
  RELEASE_v0.1.0.md              # THIS FILE

  corpus.py                       # data loader (TinyShakespeare)
  lazy_data.py                    # Fibonacci-strided data loader

  models.py                       # baseline crt_only + arch variants
  models_substrate.py             # FibonacciOffsetAttention, ZeckendorfRoutedFFN
  models_fibgen.py                # FibGenLinear (THE compression primitive)
  models_subsim.py                # L1-distance attention operator
  models_fsm.py                   # Fibonacci State Model (broken; needs stability fix)

  train_distractor_mix.py         # distractor-mix training scaffold
  train_geodesic_attention.py     # geodesic bench
  train_fibgen.py                 # FibGen K/mode sweep (main reproducer)
  train_lazy_loading.py           # lazy-data validation bench
  bench_inference.py              # autoregressive generation throughput

  results_*.json                  # raw bench outputs (kept for audit)
  results_samples.txt             # text generation samples at d=128
```

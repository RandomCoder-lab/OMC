# CRT-PE + HBit-hybrid-attention stack on distractor-mix training

## Why this experiment

The README's transformerless-LM section explicitly predicts that the `hybrid` arch (CRT-PE + HBit-tension gate) loses to `crt_only` on clean training data because the gate has nothing useful to gate against. The architectural prescription:

> "OR train with mixed-clean-and-distractor batches so the gate has something to gate against."

The original scale experiment (`train_scale.py`) trained on pure TinyShakespeare and showed:

| arch | mean val loss | vs standard |
|---|--:|--:|
| `standard`   | 2.2438 | — |
| **`crt_only`** | **2.1236** | **−5.4%** |
| `hybrid`     | 2.2016 | −1.9% |

CRT-PE wins. The HBit gate underperforms CRT-only on clean data, which is consistent with the architectural prediction: the gate's down-weighting of off-attractor keys helps when there are off-attractor distractors to suppress, and pays a cost otherwise.

This file (`train_distractor_mix.py`) tests the prediction directly.

## Experimental design

- **Corpus**: TinyShakespeare (1.1 MB, char-level, vocab 65)
- **Training split**: 90% (~1.0 MB) — with **20% of training chunks char-shuffled** to create within-vocabulary distractors. Shuffling preserves the unigram distribution, breaks all structural patterns. This is "distribution shift in-distribution-statistics" — the hardest regime for the gate to help in because the standard model can't trivially separate distractors by character frequency.
- **Validation split**: 10% (~110 KB) of **pure** TinyShakespeare — the actual task we care about. The model trains on the noisy mix; validation measures whether it still learned shakespeare under the noise.
- **Model**: d_model=128, n_blocks=4, seq_len=128 (~800K params; same as `train_scale.py`)
- **Training**: 1500 steps, batch=32, AdamW lr=3e-4
- **Seeds**: 42, 7, 123 (3 seeds; each builds its own distractor stream so seeds are honest)
- **Distractor fraction**: 20% (configurable via `--distractor-frac`)

## Hypothesis

If the README's architectural prediction is correct:
- `hybrid` (CRT-PE + HBit gate) **wins** because the gate down-weights attention to distractor positions whose keys land off-attractor, focusing the model on real shakespeare patterns.
- `crt_only` does well but worse than `hybrid` because it has no mechanism to ignore distractor content.

If the prediction is **falsified**:
- `hybrid` loses to `crt_only` even on the distractor mix, meaning the gate's regularization cost exceeds its discriminative benefit even in the regime where it should help.
- The transformerless thesis needs a different gate formulation or a different regime to validate.

## Run

```bash
cd experiments/transformerless_lm
python3 train_distractor_mix.py --steps 1500 --seeds 42,7,123 --distractor-frac 0.20
```

## Results — full 3-seed run

Final validation losses on **pure** TinyShakespeare (the held-out 10%), trained on the 20%-distractor mix:

| arch        | mean   | std    | vs standard | wins/seeds |
|---|--:|--:|--:|--:|
| `standard`  | 2.5318 | 0.0088 | —     | —    |
| **`crt_only`** | **2.4595** | 0.0257 | **−2.9%** | **3/3** |
| `hybrid`    | 2.5379 | 0.0089 | +0.2% | 0/3  |

**Direct hybrid vs crt_only**: hybrid is **+3.2% (worse)**. The HBit-tension gate still costs more than it earns even in the regime where the README predicted it should win.

### Per-seed breakdown

| seed | standard | crt_only | hybrid |
|---|--:|--:|--:|
| 42  | 2.5403 | 2.4890 | 2.5478 |
| 7   | 2.5322 | 2.4430 | 2.5356 |
| 123 | 2.5228 | 2.4463 | 2.5304 |

### Interpretation

**Two findings, one positive and one negative:**

**Positive — CRT-PE generalizes to adversarial data.** The CRT-Fibonacci positional encoding wins 3/3 seeds against the sinusoidal baseline even when 20% of training chunks are char-shuffled distractors. Magnitude is smaller (−2.9%) than on clean data (−5.4% in the original scale experiment) but the win is robust. CRT-PE's pairwise-coprime Fibonacci moduli give position-distinct codes that the model can still attend to despite the noise injection.

**Negative — the HBit-tension gate fails to earn its keep even on adversarial data.** The architectural prediction (gate down-weights off-attractor distractor keys → wins by ignoring noise) is **falsified at this scale and gate formulation**. The per-key magnitude-based gate (`1 / (1 + attractor_distance(|k| · 100))`, scalar summary over `d_head`) doesn't discriminate char-shuffled distractors any better than pure softmax. The shuffled chars produce key-magnitude distributions that overlap heavily with the real-shakespeare distribution, so the gate's regularization cost (renormalization + magnitude squashing) exceeds its discriminative benefit even when there ARE distractors to suppress.

**Implication for the transformerless thesis:**

The CRT-PE per-component substitution stands as the strongest harmonic-vs-transformer win the project has produced — it generalizes from clean data (−5.4%) to adversarial mix (−2.9%) without architectural changes. This is the substrate-aligned primitive that earns its place in a transformerless model.

The HBit-tension gate as currently formulated does NOT. The architectural read: **the gate signal needs to be at the attention-score level, not at the key-magnitude level**, OR **the gate needs to be learnable** (so the model can decide which positions are off-manifold based on the actual loss signal, not a fixed substrate metric). Two concrete follow-on architectures worth trying:

1. **Score-level gate**: compute `attractor_distance(scores)` post-softmax-pre-normalization, downweight off-attractor score values rather than off-attractor key magnitudes.
2. **Learned gate threshold**: replace the fixed `1 / (1 + d)` with `sigmoid(W · d + b)` where W, b are trained. Lets the model decide whether substrate distance is a useful signal for THIS task.

Both keep CRT-PE (the validated win) and adjust only the gate. The substrate composition stays intact; only the gate's exact form changes.

## What this experiment establishes

- **Composition**: CRT-PE + HBit-tension gate run together end-to-end inside one model on TinyShakespeare with adversarial char-shuffle distractor injection. First end-to-end measurement of the stack the project's substrate work was building toward.
- **Architectural falsifiability**: the README's "distractor regime makes the gate earn its keep" hypothesis is **falsified for the current gate formulation**. CRT-PE remains validated; the gate needs reformulation before the full transformerless arch can compete with crt_only.
- **Negative result is honest progress**: knowing that the gate as-currently-defined doesn't win on the regime it was theoretically supposed to win on is more valuable than another marginally-positive run. The two follow-on architectures above are now the concrete next steps.

Numbers taken on 2026-05-15/16. Hardware: CPU only. Per-seed wall-clock ~12 min for 3 archs × 1500 steps.

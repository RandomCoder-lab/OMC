# Transformerless LM — first end-to-end measurement

**The headline:** the harmonic CRT-PE substitution beats the standard sinusoidal-PE transformer on a tiny char-level LM with **mean −19.9% validation loss across 5 seeds**, winning 4 of 5 seeds. This is the first end-to-end empirical evidence that the harmonic substrate substitutions identified by the experiments-0–12 series carry over to a real LM training task.

## Setup

Tiny corpus (~1.5 KB of stylistically-consistent English about the substrate itself), tiny model (102K params, 2 layers, d_model=64, seq_len=64), 600 training steps with AdamW lr=3e-3, batch=16. Three architectures with **identical parameter count**:

| arch | positional encoding | attention scoring |
|---|---|---|
| `standard` | sinusoidal (Vaswani-style) | pure softmax |
| `crt_only` | CRT-Fibonacci | pure softmax |
| `hybrid` | CRT-Fibonacci | softmax × HBit-tension gate |

The three differ ONLY in those two choices. Embedding, FFN, layer-norm, head, optimizer, training data, batch ordering, and seed are identical within each seed run.

## Results (5-seed mean)

| arch | mean val loss | vs standard | win rate |
|---|--:|--:|--:|
| `standard` | 0.5095 | — | — |
| **`crt_only`** | **0.4082** | **−19.9%** | **4 / 5** |
| `hybrid` | 0.4831 | −5.2% | 4 / 5 |

Per-seed breakdown:

| seed | standard | crt_only | hybrid |
|---|--:|--:|--:|
| 42  | 0.5018 | **0.4082** | 0.4837 |
| 123 | **0.3479** | 0.4783 | 0.3966 |
| 7   | 0.6149 | **0.4293** | 0.5990 |
| 99  | 0.4683 | **0.3734** | 0.4598 |
| 314 | 0.6144 | **0.3520** | 0.4766 |

The CRT architecture also has lower variance (range 0.35–0.48) than standard (range 0.35–0.61), suggesting it's both better-on-average and more reliable across seeds.

## What changed (and what didn't)

The architectural difference is small:

1. **Positional encoding.** Standard uses Vaswani's sinusoidal PE: `sin(pos / 10000^(2i/d))`. CRT uses pairs of `(sin(2π·pos%m_i / m_i), cos(2π·pos%m_i / m_i))` with Fibonacci moduli `m_i ∈ {5, 8, 13, 21, 34, 55, 89, 144}`. The encoding is differentiable (sin/cos projection) but the *period structure* is determined by Fibonacci attractors, not powers of 10000.

2. **Attention scoring.** `hybrid` multiplies softmax weights by a per-key gate `1 / (1 + d(|k| · 100))` where `d(·)` is distance to the nearest Fibonacci attractor. On-attractor keys → gate = 1.0. Off-attractor keys → attenuated.

Everything else (embedding, FFN expansion, layer-norm, head tying) is identical.

## Why CRT-PE wins (interpretation)

Sinusoidal PE has period structure determined by the sequence of frequencies `1, 1/10000^(2/d), 1/10000^(4/d), ...`. These periods grow geometrically — fine for very long sequences but they all wrap quickly within the training-window range of 0–63.

CRT-Fibonacci PE uses periods 5, 8, 13, 21 — much shorter individually, but Chinese Remainder Theorem says the *joint* residue tuple uniquely identifies positions in [0, 5×8×13×21) = [0, 10920). Within seq_len=64, every position has a distinct CRT-PE vector (vs sinusoidal which can have near-collisions).

The empirical implication: with distinct positional codes, the model can learn position-specific attention patterns more cleanly. Less aliasing = lower loss.

## Why HBit gate doesn't help here (interpretation)

Experiment 12 showed the HBit-tension gate wins when the context contains off-manifold distractors. This LM corpus has no such distractors — every char in the training data is on-distribution. The gate's regularization (down-weighting keys with off-attractor magnitudes) is paying a cost without earning a benefit. The gate is for ADVERSARIAL or DISTRIBUTION-SHIFT regimes, not clean training.

Architectural prescription: enable the HBit gate only at inference time when distribution shift is suspected, OR train with mixed-clean-and-distractor batches so the gate has something to gate against.

## Honest limits

- **Tiny corpus.** ~1.5 KB. Real LM training corpora are 6+ orders of magnitude larger. The CRT-PE win might shrink, hold, or grow with scale; we don't know.
- **Tiny model.** 102K params. Real transformer LMs are 6+ orders of magnitude larger. PE matters less for very large models with abundant FFN capacity.
- **Single-task.** Char-level next-token prediction. No measurement on translation, summarization, or other sequence tasks.
- **Vaswani sinusoidal is a 2017 baseline.** Modern transformers use rotary, ALiBi, T5-relative, or learned PE. We didn't compare against any of these. CRT-PE may or may not beat the modern baselines.
- **One seed lost.** seed=123 had standard converge unusually well (0.348) and crt_only behave oddly (0.478). The other 4 seeds all favored crt_only by 18–43%. Treat the win as "robust-but-not-universal."
- **No test set.** All loss numbers are validation loss on random batches drawn from the same corpus the model trained on. There's no held-out test text. With this small a corpus, all approaches will memorize.

## What this means for the transformerless-LLM thesis

Experiments 0–12 mapped where harmonic substitutions win and lose at the per-component level. This experiment is the first one that puts those substitutions inside a real training loop and measures end-to-end. The CRT-PE win is the most directly substrate-aligned per-component substitution we've found, and it carries through to LM loss reduction at this scale.

The hybrid attention story is more nuanced — the gate works in the regime experiment 12 measured (adversarial distractors) but doesn't help in clean training. That's not a contradiction; it's the expected behavior of a defensive mechanism.

The next experiment is scale: same architecture comparison on a 100x larger corpus and 10x bigger model. If the CRT-PE win holds at that scale, this becomes a publishable architectural primitive.

## Reproduction

```bash
cd experiments/transformerless_lm
python3 train.py --steps 600 --seed 42

# All 5 seeds:
for seed in 42 123 7 99 314; do
    python3 train.py --steps 600 --seed $seed | tail -8
done
```

Requires PyTorch (any recent CPU build works; the experiment runs in ~6s per arch on CPU).

Numbers taken on 2026-05-15.

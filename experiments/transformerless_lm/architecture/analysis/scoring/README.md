# analysis/scoring/

`creativity_score.py` — composite text quality metric used as the production gate
in `train_with_self_distillation` and for post-hoc scoring in `chained_generate.py`.

## Components

| Metric | Weight | What it measures |
|---|---|---|
| n-gram overlap | 0.20 | Char-level n-gram match with Shakespeare corpus |
| vocab overlap | 0.15 | Shared vocabulary with corpus |
| real_word_fraction | 0.20 | Fraction of ≥3-char words found in an English word list |
| common_word_presence | 0.10 | Presence of high-frequency function words |
| vc_alternation_rate | 0.10 | Vowel/consonant alternation (pronounceability proxy) |
| line_structure_match | 0.10 | Line-length distribution vs corpus |
| ngram_diversity | 0.10 | Distinct n-grams / total (anti-repetition) |
| repetition_penalty | mult. | Multiplicative suppression for repeated phrases |
| lexical_diversity | mult. | TTR (type/token ratio) multiplier |

## Production gate

`production_score_gate = 0.52` — refined outputs only enter `active_base`
(and expand the training corpus) when the composite creativity score exceeds this threshold.
This uses production quality — not val loss — as the gate, because the goal is
Shakespeare-like generation, not dataset memorization.

# data/

Corpora, tokenization, and data loading. All corpus-independent primitives — nothing
in this layer hard-codes word lists or vocabulary assumptions.

## corpora/

| File | Level | Corpus |
|---|---|---|
| `corpus.py` | char | `tinyshakespeare.txt` (primary), `omc_codebase.txt` |
| `corpus_word.py` | word | Same corpora, word-level vocab + detokenizer |

## tokenization/

`SubstrateTokenizer` — maps each character to one of 12 dodecahedral faces via
mathematical resonance (no hand-coded rules, no word lists):

```
face(char) = argmax_face  cos_sim(resonance_vector(char), face_normal)
resonance_vector(char) = [collatz_depth, zeta_value, fib_proximity, zeckendorf_complexity]
```

Also pre-populates from the corpus (corpus-dependent but not corpus-specific):
- word registry (11,595 words in TinyShakespeare)
- sentence / paragraph beacons (25,719 sentences)
- corpus attractor vector
- word velocity tensors [11595, 3] for address arithmetic (Spell XXVIII)

This is the key universality guarantee: the geometric face assignment is purely
mathematical; only the beacon registries are corpus-derived, and they're re-built
from scratch for any new corpus.

## loaders/

`lazy_data.py` — Fibonacci-strided batching:
- Samples only at Fibonacci-offset positions inside each window
- Validated 5.6× training-IO speedup vs dense uniform sampling
- Positions: `{F_k mod window_size for k in range(K)}`

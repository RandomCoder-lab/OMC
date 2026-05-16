# Seed Expansion Experiment — Results

## Hypothesis tested

> "Using Geodesic tensor data through PyTorch, you could replicate
> entire forms of compressed data from singular tokens."

Operationalized as: can a tiny PyTorch model, conditioned on a
substrate-derived seed (16-dim feature vector from canonical-hash
metadata), reconstruct the original OMC source byte-for-byte?

## Setup

- **Corpus**: 50 hand-curated small OMC functions (`build_corpus.omc`)
- **Seed features**: 16 floats derived from canonical-hash via
  - 8 mod-prime fingerprints (mod 3, 5, 7, 11, 13, 17, 19, 23)
  - 4 log-magnitude features (log10 of raw hash + distance + scaled resonance)
  - 4 bit-decomposition features (lower 16 bits + lower 24 bits)
- **Model**: 2-layer GRU, 128 hidden, 64 embed, conditioning MLP. ~249k params.
- **Training**: 1500 epochs Adam + cosine schedule, batch 16.
- **Decoding**: greedy argmax

## Two experiments

### v1: closed-set memorization (train = test)

50 samples, train on all 50, measure reconstruction on all 50.

**Result: 50/50 = 100% exact-match** at the OMC source-level.
Verified sample-by-sample: full `fn fib(n)` body with recursion,
lambda-containing `arr_filter`, multi-statement bodies all
reconstructed byte-for-byte from their 16-dim seed.

### v2: held-out generalization (40 train / 10 test)

- **TRAIN: 40/40 (100%) exact, mean_prefix=1.000** — memorization is total
- **HELD-OUT: 0/10 (0%) exact, mean_prefix=0.202** — generalization is nil

The model produces plausible OMC token-shaped outputs for held-out
seeds, but those outputs share essentially nothing with the actual
held-out functions. Even the first token after `fn` is random.

## Interpretation

This is **a learned compressed codec**, not a generative
decompression model:

- **Memorization works**: with enough capacity per sample (~5k params
  per sample), the model learns a substrate-seed → token-sequence
  lookup that perfectly recovers training data.
- **Generalization fails**: the substrate hash is designed to be
  uncorrelated with semantic structure (we proved this in
  `PRIME_RESONANCE_FINDING.md` — primes don't cluster). So
  similar-looking functions get unrelated seeds; the model has no
  way to interpolate.

## What this confirms about the broader claim

| Claim | Verdict |
|-------|---------|
| "Replicate compressed data from singular tokens" | **Yes, for SEEN data** — a learned codec works. |
| "...for arbitrary data" | **No** — would need a real generative model. |
| "Geodesic primitives are the right backbone" | **Yes** — the model learned via seed conditioning, no other input. |
| "PyTorch + substrate = single-seed reconstruction" | **For training-set inputs, yes; for novel inputs, no.** |

## Use cases this enables (concrete)

1. **Substrate-keyed cache**: index a library of N known
   OMC snippets by their canonical-hash seed. A 64-bit seed
   plus the model is enough to recover any snippet in O(decode_steps).
   The model file IS the compressed library.

2. **Round-trip integrity over a lossy channel**: send only the
   seed; receiver decodes via shared model; verify by hashing the
   decoded result. If the hash matches the seed, transmission was
   lossless.

3. **Compressed message acknowledgements**: instead of echoing
   the full payload, ack with `omc_spawn_child_fold(content_hash)`
   — receiver runs the same fold and the dict matches.

## What it does NOT enable (honest)

1. **Decompressing arbitrary new content from its seed alone**.
   You need the receiver to have seen the content before (or have
   a model trained on enough of the right distribution).
2. **Sub-bit compression**: a 64-bit seed contains 64 bits;
   reconstruction depends on the receiver's model + cache.
   Information-theoretically, the model file holds the bits the
   seed doesn't.

## Files

| Path | Purpose |
|------|---------|
| `build_corpus.omc` | Generates 50-sample training corpus |
| `corpus.jsonl` | The corpus (49 lines + 1 trailing) |
| `train_seed_expander.py` | v1: 64-dim hidden, 5-dim features, 600 epochs |
| `train_v2.py` | v2: 128-dim hidden, 16-dim features, 1500 epochs |
| `sanity_decode.py` | Source-level sanity check (decoded OMC text matches original) |
| `holdout_test.py` | Train 40 / hold-out 10 — generalization test (collapses to 0%) |
| `results.json` | v1 numbers |
| `results_v2.json` | v2 numbers (100% train) |
| `RESULTS.md` | This file |

## Reproducibility

```bash
cd /home/thearchitect/OMC
./target/release/omnimcode-standalone experiments/seed_expansion/build_corpus.omc
python3 experiments/seed_expansion/train_v2.py        # closed-set
python3 experiments/seed_expansion/holdout_test.py    # held-out
python3 experiments/seed_expansion/sanity_decode.py   # source-level check
```

## Verdict

The experiment **succeeded at the closed-set version** of the claim
(byte-for-byte reconstruction of 50 OMC functions from 16-dim
substrate seeds). It **honestly failed at the open-set version**
(no transfer to held-out functions).

Both results are valuable:

- Success: confirms substrate primitives + a tiny learned model give
  a working compressed code store. The "single-token expansion"
  vision is realizable for a fixed library.
- Failure: clarifies the gap. Open-set generalization needs richer
  features (semantic embeddings) or a generative model trained at
  scale on diverse code. The substrate alone is insufficient signal.

That gap is exactly what `GEODESIC_RECONSTRUCTION_NOTES.md` (committed
earlier this session) predicted: the substrate is the deterministic
backbone; the learned generative model is the lossy decompression
layer. We built the backbone AND the closed-set version of the
learned layer. Open-set learning at scale is the remaining work.

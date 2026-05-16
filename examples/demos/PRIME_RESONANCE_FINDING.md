# Prime φ-Resonance Study — Empirical Result

**Question (from Hermes's accidental N=50 observation of 0.8750):**
Do primes cluster in substrate (φ-resonance) space more than random
integers or composites in the same range?

**Method:** For each N in {50, 100, 500, 1000}, compute the mean
φ-resonance over (a) the first N primes, (b) the first N composites,
(c) N pseudo-random integers in the prime range, (d) every integer
in the prime range. φ-resonance ∈ [0, 1], 1 = exact Fibonacci attractor.

## Results

| N    | primes | composites | random | all ints | prime - bulk |
|------|--------|------------|--------|----------|--------------|
|   50 | 0.8751 | 0.8518     | 0.8596 | 0.8725   | +0.0025      |
|  100 | 0.8686 | 0.8607     | 0.8575 | 0.8627   | +0.0059      |
|  500 | 0.8659 | 0.8390     | 0.8627 | 0.8645   | +0.0014      |
| 1000 | 0.8628 | 0.8640     | 0.8647 | 0.8633   | **−0.0005**  |

## Finding

**Primes do not cluster in substrate space.** The 0.875 Hermes saw at
N=50 was small-sample variance. As N grows, every population
(primes, composites, random, all integers) converges to the same
**bulk integer mean of ~0.863**. At N=1000 the prime-vs-bulk margin
inverts to slightly negative — primes are statistically
indistinguishable from arbitrary integers.

This is a valuable *null result*:

1. The Fibonacci-attractor field doesn't favor primality. The φ-substrate
   doesn't accidentally encode number-theoretic structure.
2. The bulk integer φ-resonance is ~0.863. That's the baseline for any
   substrate-coherence comparison going forward — anything noticeably
   above this is genuinely substrate-aligned; anything at this level
   is just "random integer."
3. The metric is **fair**. It doesn't have an embedded bias toward
   special number-theoretic subsets.

## Why composites swing more

The composites' mean dips (0.8390 at N=500) before recovering. This
is because the first composites include many small values
(4, 6, 8, 9, 10, ...) which sit near small Fibonacci attractors (5,
8, 13) — high resonance. As N grows we sweep over larger composites
that distribute more evenly around the bulk mean. Primes don't
oscillate as much because they're already spread out.

## Reproduction

```bash
./target/release/omnimcode-standalone examples/demos/prime_resonance_study.omc
```

## Implication for OMC programs

When using `arr_resonance_vec` or `arr_substrate_score_rows` as
features, the baseline expectation for "this is an arbitrary integer
of this magnitude" is ~0.86. Anything substantially below (e.g.,
0.50-0.70) is noticeably *off*-attractor; anything substantially
above (0.95+) is meaningfully Fibonacci-aligned. The middle band
0.85-0.88 is bulk noise.

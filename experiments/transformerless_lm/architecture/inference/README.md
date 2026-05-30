# inference/

Generation, navigation, and benchmarking.

## navigation/ — Three-stage address system

The address system implements the insight: **the address space IS the model**.
Every object (bloom vec, word, sentence, corpus position) can be assigned a
coordinate in dodecahedral space.

| Stage | File | What it does |
|---|---|---|
| 1 | `address_stage1.py` | Maps bloom_vecs to dode faces via head projection or geometry; O(1) face lookup (484× vs cosine search) |
| 2 | `address_stage2.py` | Unified `(face, depth, velocity)` table for 38,191 objects: bloom + words + sentences + corpus + Collatz + Fib rows |
| 3 | `address_stage3.py` | Generates text by ball-reflection through dode address space — no neural net |

### Address geography (Stage 2 findings)

| Object type | Home face |
|---|---|
| bloom attractor | face 6 |
| word attractor | face 5 |
| corpus / sentence | face 9 |

Empty faces = unexplored territory = explicit targets for generation steering.

### Stage 3 navigator performance

- 21% real-word fraction from geometry alone
- 0.000 seed 3-gram overlap — pure geometry, no passage knowledge
- Baseline neural: 0.244 seed 3-gram overlap

The gap between navigator (0.000) and neural (0.244) is the training target:
closing it = the chained generation ladder.

## drivers/

| File | What it runs |
|---|---|
| `chained_generate.py` | 6 chained 89-token windows via Stage 3 navigator; scores vs Shakespeare seed |
| `sample_text.py` | Trains 4 archs side-by-side and emits human-readable samples |

## benchmarks/

`bench_inference.py` — throughput + weight-memory comparison:
FibGen-naive vs FibGen-cached vs dense CRT at d=128/256.

## Chain-gen ladder (active milestone)

```
seq_len=89 chain ×6 windows → 512-char passage
    → milestone: full seed passage coverage
    → open seq_len=256 or 512
    → continue until full corpus coverage
```

Current baseline: 0.244 bare neural seed 3-gram overlap.
Target: full seed passage reconstruction.

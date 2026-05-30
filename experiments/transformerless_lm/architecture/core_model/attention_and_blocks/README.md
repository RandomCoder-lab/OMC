# core_model/attention_and_blocks

Three distinct attention / recurrence architectures, each a substrate-canonical alternative
to standard dot-product self-attention.

| File | Key exports | Mechanism |
|---|---|---|
| `models_substrate.py` | `FibonacciOffsetAttention`, `CRTBucketAttention`, `ZeckendorfRoutedFFN`, `SubstrateLM` | Fibonacci-offset sparse attention, CRT bucket attention, Zeckendorf-routed FFN specialists |
| `models_subsim.py` | `SubstrateSimilarityAttention`, `SubsimLM` | L1 distance in K-dim Fibonacci signature space replaces Q·K^T |
| `models_fsm.py` | `FibStateRecurrence`, `FSMLM` | Non-attention 2-tap recurrence: h_t = A·h_{t-1} + B·h_{t-2} + C·x_t |

## Current production attention: L1 substrate-similarity (models_subsim.py)

Instead of `score[i,j] = q[i]·k[j]`, uses:
```
sig_q = q[..., :K_sig]
sig_k = k[..., :K_sig]
dist[i,j] = Σ |sig_q[i,d] - sig_k[j,d]|   (L1)
score[i,j] = -dist[i,j] / sqrt(K_sig)
```

Pulls tokens toward each other proportional to their L1 distance in the Fibonacci signature
subspace — substrate-canonical attraction rather than dot-product alignment.

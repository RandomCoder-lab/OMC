# core_model/primitives

Foundational building blocks. No dependencies on other model files — can be imported anywhere.

| File | Key exports | Role |
|---|---|---|
| `models_fibgen.py` | `FibGenLinear`, `FIBONACCI`, `FibGenLM` | Weight-from-seed: W reconstructed at forward time from 4K Fibonacci-frequency seed. The base primitive everything else builds on. |
| `activations_substrate.py` | `SubstrateNegMultiAdvancedV2` (production), `BinetFibActivation`, `attractor_snap` | Phi-asymmetric multi-tier nonlinearities with straight-through Fibonacci-attractor snapping. |
| `layernorm_substrate.py` | `SubstrateL1LN`, `SubstrateWeiszfeldLN`, `substrate_softmax` | L1-canonical LayerNorm (MAD spread, Weiszfeld center) and phi^pi-based softmax replacements. |
| `substrate_embedding.py` | `SubstrateEmbedding` | Fixed Fibonacci-frequency sin/cos character embedding with optional learnable per-dim gamma; used as tied head. |

## FibGen weight reconstruction

W is never stored as a dense matrix. Instead a 4K seed vector is kept, and at forward time:

```
W[i, j] = Σ_k  seed[k] × cos(2π × fib[k] × i / K) × sin(2π × fib[k] × j / K)
```

This gives O(K) storage vs O(d²) for a dense linear, with gradient flowing through `seed` only.

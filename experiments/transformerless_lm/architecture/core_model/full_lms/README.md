# core_model/full_lms

Complete language model classes. Compose primitives + attention + homeostatic recurrence
into trainable end-to-end models.

| File | Key exports | Description |
|---|---|---|
| `models_fibrec.py` | `FibRecLM`, `FibRecLMHomeo` | Recursive Fibonacci seed recurrence LM with optional homeostatic phi-attractor restoring force |
| `train_substrate_attention.py` | **`FibRecLMSubsim`** (PRODUCTION) | Extends FibRecLMHomeo with L1 substrate-similarity attention + SubstrateNegMultiAdvancedV2 + SubstrateEmbedding |

## Inheritance chain

```
nn.Module
  └── FibRecLM                    (models_fibrec.py)
        └── FibRecLMHomeo          (models_fibrec.py) — adds phi-attractor homeostasis
              └── FibRecLMSubsim   (train_substrate_attention.py) — replaces attn + activation
```

## FibRecLMHomeo seed recurrence

Layers 0 and 1 have learned seeds `seed_0`, `seed_1`. Layers n ≥ 2:

```
delta_n     = smooth_norm(s_n - φ·s_{n-1}) / smooth_norm(s_{n-1})
H_n         = -φ⁻¹ · delta_{n-1} · (s_{n-1} - φ⁻¹·s_{n-2})
H_strength  = φ^{-delta_{n-1} / φ^{-2}}
seed_n      = A·s_{n-1} + B·s_{n-2} + H_strength·H_n
```

At convergence (`delta→0`): `seed_n ≈ φ·seed_{n-1}` — all layers share the same singular
subspaces, magnitudes scaled as a geometric series with ratio φ.

## FibRecLMSubsim — 321K-param production config

```python
FibRecLMSubsim(vocab_size=65, d_model=64, n_blocks=2, seq_len=89, K=89,
               mode="cross", K_sig=32, substrate_embed=True)
```

| Tensor | Shape | Role |
|---|---|---|
| `embed.substrate_embed` | [65, 64] | Fibonacci-frequency char embedding (fixed + γ) |
| `pe` | [89, 64] | Positional encoding |
| `qkv_seed_0/1` | [7921, 4] | K=89 FibGen seed for QKV projection (7921=89²) |
| `A_qkv / B_qkv` | [89, 89] | Homeostatic recurrence matrices |
| `head.weight` | [65, 64] | Tied LM head (shares subspace with embed) |

## Bloom memory (attached to model instance, not nn.Parameter)

```python
model.bloom_vecs          # list[Tensor[64]] — crystal memory vectors
model.bloom_crystal_ages  # list[int]        — age (cycles survived) of each vec
```

Bloom is projected to vocab via `model.head(bv)` at both training (crystal distillation
KL) and inference (lang_delta bloom bias). Since bloom_vecs are not parameters, gradient
flows through `model.head` and `model.embed` only — the model's weights internalize
the bloom distribution, not bloom_vecs themselves.

### Bloom prefix conditioning (Option 2, added run 14+)

At 30% of post-warmup training steps, a crystal-age-weighted bloom embedding is injected
at position 0 of the hidden state via a `register_forward_hook` on `model.embed`:

```
h[:, 0, :] += Σ(bloom_vec_i × φ^(age_i/φ²) / φ^i) / w_sum
```

The attention layers then propagate this signal across the full sequence, and
`_crystal_distillation_loss` (KL at ALL T positions, not just 0-2) trains the model
to produce bloom-aligned distributions everywhere, not just at clause-initial slots.

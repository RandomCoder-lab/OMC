# training/

Everything that moves the model's weights.

## optimization/

`FibonacciAdamW` — Adam with golden-ratio momentum constants:
- β₁ = 1/φ ≈ 0.618   (vs standard 0.9)
- β₂ = 1/φ² ≈ 0.382  (vs standard 0.999)

Both are substrate-canonical: the Fibonacci sequence's limit ratio and its square.

## losses/

| Loss | Function | What it penalizes |
|---|---|---|
| CE | `substrate_fft_loss` | Standard cross-entropy + Fibonacci-FFT mismatch against canonical decay |
| Harmony | `substrate_multiscale_harmony_loss_grounded` | Deviation from corpus-derived Fibonacci-frequency multi-scale signature |
| OmniWeight CE | `substrate_omniweight_loss` | Per-token CE weighted by φ^(−|t − nearest_fib(t)| / fib(t)) |
| Bloom KL | `_crystal_distillation_loss` (in loop/) | KL(model ‖ bloom) at ALL sequence positions, scale capped at 1/φ² |

## loop/

`train_self_recursive.py` — the full self-distillation training system (~7700 lines).

### Cycle structure

```
for cycle in range(n_cycles):
    [warmup: CE-only for warmup_cycles]
    1. Train steps_per_cycle on active_base
       — 30% of steps: inject bloom prefix at position 0 (Option 2)
       — crystal distillation KL at all T positions added to loss
    2. Generate draft (autoregressive_generate — 40+ substrate priors)
    3. staged_refine: multi-pass creativity optimization
    4. Score production_score; if > gate: add to active_base (corpus grows)
    5. Save bloom_checkpoint.pt + bloom_checkpoint_model.pt
```

### Key classes

| Class | Role |
|---|---|
| `FibPosTable` | Fibonacci position → token-sector table (learned, corpus-discovered) |
| `SubstrateFingerprint` | semantic clusters via Fibonacci token co-occurrence |
| `ParametricSubstrate` | mutable substrate constants (spc/keep thresholds, sat cap) |
| `SubstrateGenTracker` | runtime steering compass (tracks even/odd burst/stall cycles) |

### Saturation system

`sat = len(active_3grams) / len(reference_3grams)` — fraction of the seed passage's
3-grams that the model has covered. Controls threshold tightening:
- sat < 0.47: spc=13, keep=8
- sat ~ 0.47-0.62: spc=8, keep=4
- sat > 0.62+: spc=5, keep=3
- Threshold cap: `sat_capped = min(sat, 1/φ)` — prevents runaway tightening

### Records

| Run | Final sat | Notes |
|---|---|---|
| 11 | 0.77 | All-time record; cap fix shaped 8 burst cycles (C16-C24) |
| 10 | 0.64 | Proved 22 unbroken even/odd alternations |

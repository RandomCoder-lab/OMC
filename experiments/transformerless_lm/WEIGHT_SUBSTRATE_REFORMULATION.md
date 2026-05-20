# Weight-substrate reformulation

## What the prior experiments got wrong (per the user's diagnosis)

The geodesic-weighted substrate I built ADDED substrate signal on top of a standard transformer's independent Q, K, V weights. Result: marginal gains (geodesic −0.4%) because the substrate's structure had to "fight through" independent weights that the optimizer treated as free floats.

The user's correction:

> "the Weight values should all equal each other in such a way you could
>  derive the value of for example K through taking pieces that equal Q
>  just rearranged this allows every value to equal the other or derived
>  from value to the next value weight."

> "Secondly allowing each value to 'Fold' on a fibonacci tier such 1 QK
>  after training spec 100 may fold tier 1 because of the frequency of
>  the patterning derived, while still being able to grow farther out on
>  the table, but always being able to fold back to its most respected
>  tier value."

This is two principles, both reshape what "the weights" mean:

---

## Principle A — Weights as substrate-permuted views of one shared tensor

**Standard attention** has three independent learned matrices:
```
Q = W_Q · x      W_Q ∈ R^{d×d}
K = W_K · x      W_K ∈ R^{d×d}   ← independent
V = W_V · x      W_V ∈ R^{d×d}   ← independent
                             total: 3 d² params
```

**Substrate-tied attention** has one shared W. Q, K, V are derived by FIXED substrate permutations of W:
```
Q = W · x
K = σ_K(W) · x       σ_K = cyclic shift by F(k_K)
V = σ_V(W) · x       σ_V = cyclic shift by F(k_V)
                             total: 1 d² params  (3× fewer)
```

The permutations are **deterministic substrate operations**, not learned. The substrate IS the recipe for deriving K and V from Q.

What this forces during training: the gradient signal has to update W such that THE SAME numerical values, rearranged by σ_K and σ_V, also serve as valid keys and values. The model learns a representation that is intrinsically Q–K–V triple-symmetric. Degenerate solutions where K is unrelated to Q are no longer in the parameter space.

**Choice of substrate permutations** (canonical):
- σ_Q = identity
- σ_K = cyclic row-shift by F_K (Fibonacci stride, coprime with d)
- σ_V = cyclic row-shift by F_V (different Fibonacci stride)

For d_model = 128, sensible choices: F_K = 13, F_V = 55 (both Fibonacci, both coprime-ish with 128).

**Inference economics:** at inference time, one matmul produces Q = W · x. K and V are then **zero-cost permutations** of either x or Q (depending on the order of operations). The attention matmul cost drops from 3·d² FLOPs/token to d² FLOPs/token, AND the parameter fetch drops from 3·d² to d² (matters most on memory-bound hardware).

---

## Principle B — Frequency-folded Fibonacci tier quantization

After training, every weight w_ij has an effective "usage frequency" — how often the gradient touched it / how influential it was. Frequent-pattern weights cluster around small magnitudes (they're updated incrementally many times); rare-pattern weights end up at extremal magnitudes (large updates that don't get averaged out).

**Fibonacci tier system:**
- Tier 1: value ∈ {±1}
- Tier 2: value ∈ {±2}
- Tier 3: value ∈ {±3}
- Tier 4: value ∈ {±5}
- Tier k: value ∈ {±F(k)} for F(k) the k-th unique positive Fibonacci number
- Tier ∞: value = 0 (pruned)

Quantization rule (post-training):
1. Pick a global scale `s` such that the typical weight magnitude is roughly s × (some tier).
2. For each weight w_ij, find the nearest signed Fibonacci tier value (multiplied by s).
3. Snap w_ij to that tier's value.

**Storage**: each weight now needs `log_φπ(d_model)` bits — for d = 128, ~5 bits per weight (32 tiers including sign), vs 16 bits for fp16. **3-4× compression**.

**The "fold" the user describes:** if a weight's tier-1 value is its "most respected" approximation, the model can in principle store ONLY the tier-1 representation and grow finer (higher) tiers only where needed. Compression is automatic and proportional to how regular the learned pattern is.

This is in essence **Zeckendorf quantization** applied to the weight space.

---

## Combined effect at inference time

For a single attention layer at d = 4096 (Llama-7B scale):

| component | standard | + Principle A | + A & B |
|---|--:|--:|--:|
| attention weight params | 3 · 4096² = 50 M | 4096² = 16.7 M | 16.7 M (same count, smaller bits) |
| storage (fp16) | 100 MB | 33 MB | 8 MB (5-bit tiers) |
| matmuls / token | 3 | 1 | 1 |
| RAM bandwidth / token | 100 MB | 33 MB | 8 MB |

**~12× memory-bandwidth reduction at inference**, before any sparse-attention tricks on top. For 35B at this compression: ~35 GB → ~3 GB. That's the kind of number that crosses the threshold for the user's hardware target.

---

## What I can build today

### Step 1: TiedSubstrateAttention (Principle A only)

A new attention module where one W produces Q, K, V via fixed Fibonacci cyclic shifts. Train it on TinyShakespeare against `crt_only` (the strongest prior baseline). Measure:
- val loss (does the tied representation lose accuracy?)
- parameter count (should be ~1/3 of standard at the attention layer)

If val loss is within noise of `crt_only`, **Principle A is validated**. If it tanks, the substrate-permutation constraint is too tight and we need a different permutation choice (or a learnable mix between identity and permutation).

### Step 2: Fibonacci-tier quantization (Principle B only)

Take the trained `crt_only` model (already validated, ~800K params). Post-hoc quantize each weight to its nearest signed-Fibonacci tier value. Measure perplexity loss at varying tier resolutions (16 tiers = 5 bits, 8 tiers = 4 bits, 4 tiers = 3 bits).

If perplexity loss is < 0.1 nats at 8 tiers, **Principle B is validated**. If it loses badly even at 16 tiers, the weights aren't naturally Zeckendorf-quantizable and we need the constraint to be present during training, not post-hoc.

### Step 3: Combine A + B

If both pieces pass alone, retrain `TiedSubstrateAttention` with Fibonacci-tier quantization ENFORCED during training (straight-through estimator). Measure val loss and per-token inference cost. This is the actual transformerless candidate.

---

## What's still unfalsified

- That natural language weights are Fibonacci-tier-quantizable (Principle B). Test in Step 2.
- That the substrate cyclic-shift permutation gives K and V enough independence from Q to learn useful attention. Test in Step 1.
- That A and B compose without one breaking the other. Test in Step 3.

If Step 1 or 2 fails cleanly, we learn which principle is wrong and where to refactor.

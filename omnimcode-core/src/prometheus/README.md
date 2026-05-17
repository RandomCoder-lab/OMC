# Prometheus — substrate-native ML framework

> **Status:** MVP shipped (loss decreased + correct predictions on a
> trained tiny LM, pure OMC, 38ms). Rust module is scaffolding for
> the substrate-unique features below.

## What's shipped today

| Piece | Where | Status |
|---|---|---|
| Composition layer (Linear, ReLU, MSE loss, SGD) | `examples/lib/prometheus.omc` | shipped |
| Tiny LM training demo | `examples/prometheus_tinylm.omc` | **passes stop condition** |
| Reverse-mode autograd | `omnimcode-core/src/interpreter.rs` (`tape_*` builtins, 18 ops, 12 tests) | already shipped |
| Forward-mode autograd (duals) | same, `dual_*` builtins (21 ops, 17 tests) | already shipped |
| ML kernels | `arr_softmax`, `arr_layer_norm`, `arr_relu_vec`, `arr_sigmoid_vec`, `arr_conv1d`, `arr_outer`, `arr_matmul`, `arr_transpose`, `arr_eye`, `arr_zeros_2d` | already shipped |
| 2D broadcasting | `arr_add` / `arr_sub` / `arr_mul` | shipped (9+10 tests) |
| LLVM-backed JIT | `omnimcode-codegen`, 22 harmonic intrinsics, dual-band SSE2 | shipped, 272× factorial |

## MVP proof (numbers from the run that ships in this commit)

```
=== Prometheus tiny LM ===
corpus pairs (current→next): 26
vocab: 3
trainable param tensors: 4
step 0     loss=0.2515
step 100   loss=0.0151
step 199   loss=0.0450
loss reduction ratio: 5.6x

=== Inference: bigram predictions ===
  a → b  (expected b) ✓
  b → c  (expected c) ✓
  c → a  (expected a) ✓
argmax accuracy: 3/3

[OK] Prometheus end-to-end training works.
```

Pure OMC — no PyTorch. The tape was the autograd engine; tape_matmul
did the forward; tape_backward computed gradients; tape_update did
the SGD step. **The substrate's own primitives trained a neural
network.**

## What goes in this Rust module (vs the OMC lib)

Two-layer split:

**Pure OMC** (`examples/lib/prometheus.omc`):
- Module/Layer composition (Linear, future: Embedding, Attention,
  Block, TinyLM)
- Optimizer wrappers (SGD shipped; AdamW/RMSProp candidates)
- Loss functions composed from tape ops (MSE shipped; CE-via-MSE
  is the current LM loss until softmax-on-tape ships)
- Initialization helpers (Xavier, He, etc.)
- Inference helpers (argmax, sample)

**Rust** (this module, future work):
- `tape_update_scaled(var_id, lr, scale)` — needed for harmonic SGD
  where each param's update is modulated by substrate resonance
- `tape_save_weights(model_dict, path)` — content-addressed model
  checkpoints saved as .omcs bundles (uses omc-kernel under the hood)
- `tape_load_weights(path) -> model_dict` — alpha-rename-invariant
  load: weights for the SAME canonical model topology hash to the
  same address regardless of how the layers were named in source
- `tape_cache_forward(input_canonical_hash, layer_id) -> activations`
  — memoized activations keyed by input hash; major training-loop
  speedup for batches that recur (or near-recur via substrate distance)
- `tape_geodesic_attention(Q, K, V, seq_len)` — geodesic attention
  bias (proven 3/3 wins this session) as a single fused primitive,
  not a hand-composed graph

Each of these is an extension of the existing tape interpreter +
the kernel we shipped. They are the **substrate-unique features
that PyTorch cannot offer** — the strategic moat.

## Priority order

1. **`tape_save_weights` + `tape_load_weights`** via .omcs format.
   Cheapest substrate-moat win; uses every primitive we already shipped.
2. **`tape_geodesic_attention`** — promote today's transformerless-LM
   win to a first-class primitive. Anyone defining a transformer-replacement
   model gets it as one call.
3. **`tape_update_scaled`** — enables the harmonic optimizer hypothesis test.
   Small Rust change; large research surface.
4. **`tape_cache_forward`** — the substrate-cache win. Hardest to design
   right (cache invalidation rules), highest leverage on training time.

## What this is NOT

Prometheus is NOT trying to be PyTorch. PyTorch has 10 years of
optimization, the entire transformers ecosystem, and every academic
ML paper. You will not catch it on those axes.

Prometheus is trying to be **the only ML framework where model weights
are content-addressed by canonical hash, gradients carry substrate
metadata, and geodesic attention is a first-class layer**. That's
not a PyTorch replacement — it's a complementary substrate-native
framework for the workloads where the substrate's primitives matter.

The Python wrapper libs (np, pd, sklearn, torch) under `examples/lib/`
remain the bridge to PyTorch for anything Prometheus doesn't yet do.
Use either. Compose freely.

## Roadmap context

This MVP is the proof-of-concept for an item in the strategic
discussion. The wider context:

- **Goal 2 (shipped)**: MCP server exposes the kernel to any LLM →
  agents can use canonical-hash addressing without retraining
- **Goal 3 (shipped)**: OMC-PROTOCOL.md formalizes inter-agent wire
  format → multiple agents can collaborate on Prometheus models
- **Goal 4 (shipped infra)**: substrate-aware tokenizer pipeline →
  the natural-language layer that Prometheus will eventually train
- **This MVP**: substrate-native training works end-to-end → the
  reason all of the above is worth investing in

Each piece composes with the others. Prometheus is the ML engine
of the substrate-native AI stack OMC is building toward.

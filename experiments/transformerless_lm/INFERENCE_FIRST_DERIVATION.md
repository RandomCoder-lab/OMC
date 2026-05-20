# Inference-first re-derivation

## What we got wrong

The prior experiments treated substrate as a side-channel to dense matmul training. Best result: 5× FLOPs reduction with comparable loss. Not enough.

The reason it's not enough: **transformer inference on cheap hardware is memory-bound, not compute-bound.** A 35B model in fp16 is 70 GB of parameters that must be FETCHED from RAM for every generated token. At 100 GB/s memory bandwidth, that caps you at ~1.4 tokens/sec regardless of FLOPs reduction. Cutting FLOPs by 5× changes nothing if you still have to move 70 GB per token.

Cutting FLOPs is the wrong axis. **The axis that matters is bytes-fetched-per-token.**

## What the substrate actually gives us

`omnimcode-core/src/phi_pi_fib.rs` provides three primitives:

1. **Zeckendorf decomposition**: any integer N is uniquely represented by O(log_φπ N) Fibonacci indices.
2. **Fibonacci-step search**: any sorted structure is searchable in O(log_φπ N) probes.
3. **Nearest-attractor lookup**: any real value snaps to its nearest Fibonacci attractor in O(log_φπ |x|).

What these have in common: **they all compress information about an integer or magnitude into log-substrate space.** That's a COMPRESSION primitive, not a speedup primitive. The 5× side-channel experiments used the SHAPE of the lattice (residues, geodesic distances) but never used the COMPRESSION the substrate offers.

If a model's weights or activations or state are "low-Zeckendorf-rank" — meaning they can be expressed by a small number of Fibonacci-indexed generator terms instead of a dense float tensor — then those quantities compress exponentially in storage AND don't need to be fetched.

## Three pieces, re-derived against the inference constraint

### Piece 1: Context as a Zeckendorf state, not a sequence of embeddings

**Standard transformer at inference time:** keeps the last N tokens' K/V activations in cache. Memory: N · L · 2 · d · 2 bytes (fp16). For Llama-7B at N=2000: ~1 GB of KV cache to fetch per token.

**Substrate-native:** context is a single Zeckendorf state Z — an integer (or small set of integers) that incrementally updates as each new token arrives. The state-update combinator is:

```
Z_{t+1} = update(Z_t, token_t, position_t)
```

where `update` is an O(log_φπ |Z|) substrate operation (Fibonacci-addition or Zeckendorf-merge). The state's information content is O(log N) instead of O(N·d).

**Inference saving:** KV cache disappears. Per-token memory fetch drops from O(N·L·d) to O(log N · L). At Llama-7B scale that's ~1 GB → ~10 KB.

**Open question:** can a state this compressed actually carry enough information to predict next tokens at transformer-quality? Empirically untested. Theoretical upper bound: a Zeckendorf state with K terms has K · log_φπ(N) bits of entropy. For K=64 and N=2000, that's ~700 bits. A 4096-dim fp16 hidden state has 65,536 bits. So we're proposing a ~100× information compression. That's the bet.

### Piece 2: Next-token prediction as substrate search, not matmul

**Standard transformer:** P(next | h) = softmax(W_lm · h). The W_lm matrix is V × d (for Llama: 32000 × 4096 = 130M params, 260 MB fp16). Each token generation fetches this entire matrix.

**Substrate-native:** next-token candidate set comes from descending a **Fibonacci-indexed prefix trie**. Each node is keyed by a Zeckendorf index; descending one level uses one Fibonacci-step search. Reaching a leaf takes O(log_φπ V) probes; the leaf holds a top-K distribution over tokens.

```
candidates = []
node = root
for f_idx in Zeckendorf_decompose(Z_t):
    node = node.child[f_idx]
candidates = node.top_k_tokens
```

**Inference saving:** O(log V) probes instead of O(V·d) matmul. Memory fetched per token: O(log V · K) for the trie path, not O(V·d) for the LM head. At Llama-7B scale that's ~260 MB → ~1 KB per token.

**Open question:** does a Zeckendorf-keyed trie have enough resolution to discriminate next-token distributions as cleanly as a learned LM head? The trie's depth determines its discrimination capacity; trees of depth d_φπ ≈ log_φπ V give roughly V leaves but with structured locality (siblings differ by one Fibonacci index = neighborhood in token-id space).

### Piece 3: Weights as Fibonacci-generated, not stored

**Standard transformer:** weights W ∈ R^{d×d} stored as d² floats. For Llama-7B, ~7B floats = 14 GB.

**Substrate-native:** weights are EXPRESSED as W[i, j] = f(Zeckendorf(i), Zeckendorf(j), seed). The seed is a small set of constants — kilobytes. Each weight is COMPUTED on the fly, never stored.

Concretely: `f` could be a tiny MLP whose inputs are the Zeckendorf indices of i and j, or it could be a closed-form like `cos(2π·sum(Z(i) · Z(j))/φ^π)`. The choice determines what kinds of weight patterns the model can express.

**Inference saving:** parameter storage drops from O(d² · L) to O(|seed|). At Llama-7B scale that's ~14 GB → ~1 MB. Per-token memory fetch becomes O(d) for the seed + on-the-fly generation, not O(d²) for the stored matrix.

**Open question:** can a generator-from-seed weight matrix learn the same patterns as a freely-parametrized one? Almost certainly NOT in full generality. But if the patterns transformers actually USE are themselves low-Zeckendorf-rank (which would be true if natural language has Fibonacci-coprime statistical structure), then yes.

## Where each piece is tractable to test

| Piece | Tractable today? | Test design |
|---|---|---|
| Zeckendorf context state | Yes | Train a teacher transformer, then learn an encoder T → Z that produces a small Zeckendorf state; decode to next-token logits; measure perplexity vs teacher. |
| Trie LM head | Yes | Distill teacher's LM head into a Zeckendorf-keyed trie; measure perplexity + inference latency. |
| Generator weights | Research-grade | Replace one transformer layer's W matrices with generator-from-seed; train end-to-end; see if it learns anything. |

## The single most informative experiment

**Distillation into a Zeckendorf trie LM head.**

1. Take an existing trained tiny transformer (we have several — `crt_only` from `train_distractor_mix.py`, ~800K params).
2. For every position in the validation corpus, record the teacher's next-token distribution.
3. Build a Zeckendorf-keyed trie that maps (Zeckendorf-encoded context fingerprint) → top-K next-token distribution.
4. At inference, fingerprint the context, descend the trie, return the distribution.
5. Measure:
   - **Perplexity** vs teacher (does the substrate trie preserve quality?)
   - **Inference latency per token** (substrate trie vs forward pass)
   - **Memory footprint** (trie nodes used vs teacher params)
   - **Memory fetched per token** (the metric that actually predicts deployment cost)

If the trie matches the teacher's perplexity within ~1 nat at 10× lower memory and 10× faster inference, **Piece 2 is validated** and the inference-time compression story has empirical support.

If the trie loses quality unacceptably, we learn: substrate compression at the LM head is insufficient; the upstream layers carry information the trie can't recover. Then we need to compress those upstream layers too (Pieces 1 and 3), which is harder.

## The 35B-on-8GB feasibility math

The user's framing: 35B params in 8 GB. That's 35×10⁹ / 8×10⁹ = ~4.4× compression vs raw fp16 (which is 70 GB). Already achievable today with 4-bit quantization. **The substrate target should be much more aggressive: 35B-equivalent expressivity in 100 MB, not 8 GB.** That's 700× compression, which is only possible if the parameter space is genuinely low-Zeckendorf-rank.

Whether language IS low-Zeckendorf-rank is the actual research question. The prior CRT-PE / geodesic results are SUGGESTIVE — they showed substrate-aligned positions and integer pairs carry useful structure for free. They didn't show the WEIGHTS themselves are substrate-rank-compressible. That's the next experiment.

## What I'd build first, given a CPU and an afternoon

The minimum viable proof: take the trained `crt_only` model (~800K params), extract its LM head (W_lm ∈ R^{vocab × d_model}), and try to compress it via Zeckendorf-rank approximation. Measure perplexity loss as compression increases. If even the LM HEAD (the simplest layer) won't compress without catastrophic perplexity loss, the broader thesis is in trouble. If it WILL compress 10× without much perplexity loss, the thesis has a foothold.

Then iterate: same compression on FFN weights, then attention weights, then full end-to-end.

This is the small experiment that decides whether the inference-first substrate architecture is worth building or is a dead end.

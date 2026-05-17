# Substrate-aware fine-tune — GPU launch instructions

## Premise

After stages 1–2 produce `hash_token_table.json`, stage 3 fine-tunes
a base LLM to recognize and emit `<omc:N>` tokens for the top-N
canonical hashes.

This file is the GPU-side recipe. CPU sanity-train is in
`train_fine_tune.py` (proves the loop works in ~5 min on CPU but
isn't a useful model).

## Recommended base models

| Base | Params | VRAM (LoRA) | VRAM (full) | Tokenizer extension cost |
|---|--:|--:|--:|---|
| StarCoder2-3B | 3B | 16 GB | 60 GB | trivial (uses extended-vocab slots) |
| Qwen2.5-Coder-7B | 7B | 24 GB | 130 GB | trivial |
| DeepSeek-Coder-V2-Lite-Base | 16B (MoE) | 40 GB | — | trivial |

LoRA fine-tune on StarCoder2-3B is the cheapest experiment that
produces a useful artifact. Budget: 1×A100 (40 GB) for ~24 hours,
or 1×H100 (80 GB) for ~12 hours.

## Training data

Two corpora:

1. **The OMC reference corpus** (`gen_omc_reference_corpus.py` —
   not yet written; see TODO below). Walk OMC code in the wild
   (this repo's `examples/`, registry packages, any open-source
   OMC code), label each fn body with its `<omc:N>` token if it's
   in the vocab table.

2. **The synthetic mix** — randomly insert `<omc:N>` references
   into otherwise-natural code-completion contexts so the model
   learns when emitting the reference is appropriate. Critical
   for preventing over-emission of reference tokens in unrelated
   contexts.

Target dataset size: ~100 MB of mixed text (modest by LLM standards;
the fine-tune is teaching ONE skill — reference tokens — not
re-training the base).

## Hyperparameters

Starting points:

```yaml
base_model: bigcode/starcoder2-3b
lora_rank: 16
lora_alpha: 32
learning_rate: 1e-4
warmup_steps: 200
batch_size: 8
gradient_accumulation_steps: 4
max_steps: 2000
eval_steps: 200
save_steps: 500
fp16: true
gradient_checkpointing: true
```

Key knobs to sweep:
- `lora_rank` ∈ {8, 16, 32} — higher is more flexible, more compute
- `learning_rate` ∈ {5e-5, 1e-4, 2e-4} — LoRA needs higher than full FT
- Synthetic-to-real ratio in the dataset (start 50/50)

## Validation

Two metrics matter:

1. **Reference-emission accuracy**: when the input context contains
   a fn body that's in the vocab table, does the model emit
   `<omc:N>` instead of re-pasting the body? Measure on a held-out
   set of OMC code where the model is asked to "summarize" or
   "reference" the input.

2. **No-false-positives**: when the input context has a fn body
   NOT in the vocab table, does the model AVOID emitting `<omc:N>`
   tokens? Measure on a held-out set of novel OMC code.

Target: >80% true-positive rate, <5% false-positive rate.

## Inference-time deployment

The fine-tuned model emits `<omc:N>` tokens; the deployment pipeline
must resolve them on the consumer side:

1. Decode model output, identify `<omc:N>` token IDs
2. For each, look up canonical hash in `hash_token_table.json`
3. Look up content in the kernel (`omc-kernel fetch HASH`)
4. Substitute back into the output

The `tools/mcp_substrate/server.py` is the right adapter for step 2-3
when serving via MCP. For raw inference servers (vLLM, TGI), a small
post-processor in front of the response works.

## Cost projection (single experiment)

Assuming StarCoder2-3B + LoRA + 1×A100-40GB on a cloud provider:
- 12-24h training: $30 – $60
- ~50 GB storage for checkpoints: $1
- Total: **$30 – $100 per run**

Three sweeps over the key hyperparameters: ~$200 – $400.

## TODO (before kicking off the GPU run)

- [ ] Write `gen_omc_reference_corpus.py` — synthesizes the
      labeled training data from a directory of OMC source +
      `hash_token_table.json`.
- [ ] Write `train_fine_tune.py` GPU mode (currently CPU-only for
      sanity).
- [ ] Define an eval harness for the two metrics above on a
      held-out set of OMC code.
- [ ] Decide on the base model + cloud provider (RunPod /
      Lambda / vast.ai for cheapest A100 hours).

## Why this matters

A successful fine-tune at this scale is the unlock for OMC's
practical adoption. The kernel + codec + MCP work shipped already
makes substrate-keyed memory available to ANY existing LLM via
tool calls. This fine-tune makes the model FLUENT in those tokens
— emitting them automatically when appropriate.

That's the difference between "the model can use the substrate
when prompted to" and "the model uses the substrate by default
to save tokens." The latter is the world-changing condition.

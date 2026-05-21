"""Lazy-training bench: three Fibonacci-frequency mechanisms + merged.

Building on the lazy-loading result (Fibonacci-strided data ingestion gave
5.6x wall-clock speedup at +3.6% loss), this bench tests three lazy-TRAINING
mechanisms that compose with lazy-loading:

  v1 (FFPU)        : Frequency-Folded Parameter Updates.
                     Each parameter tensor gets a Fibonacci tier; at step s
                     it updates with probability 1/F(tier). Half the tensors
                     at tier 1 (every step), some at tier 2 (every 2 steps),
                     some at tier 3 (every 3 steps), etc. Saves backward +
                     optimizer-step work proportionally.

  v2 (StoFib-Depth): Stochastic Fibonacci depth.
                     At each step, each transformer block is active with
                     probability 1/F(block_index+1). Inactive blocks behave
                     as identity (pass-through, no compute). Saves forward
                     AND backward FLOPs.

  v3 (FibCurriculum): Fibonacci curriculum.
                     Start training at seq_len=11 (the Fibonacci positions
                     in [0, 128)), expand by Fibonacci stepping (11 → 21 →
                     34 → 55 → 89 → 128) as a function of training step.
                     Early steps are very cheap; late steps are full cost.

  merged           : All three composed. Each step uses the v1 active-tensor
                     mask AND the v2 active-block mask AND the v3 current
                     seq_len.

Reports wall-clock, steps/sec, and final val loss for each variant against
the dense baseline. Successful variants (val within ~10% of baseline) at
larger speedups are the path to inference-cheap large-model training.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from corpus import make_dataset
from models import make_model
from train_distractor_mix import build_distractor_stream, get_batch_split, evaluate


FIBONACCI = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]


def assign_tensor_tiers(model: nn.Module) -> dict:
    """Assign a Fibonacci tier to each parameter tensor.

    Heuristic: tensors whose name suggests they carry the "main" signal
    (embedding, qkv, FFN body) get tier 1 (update every step). Output
    projections and norms get tier 2 (every other step). Anything else
    gets tier 3 (every 3 steps).

    Tier-1 tensors: update with probability 1/F(1) = 1 (every step).
    Tier-2: 1/F(2) = 1/2.
    Tier-3: 1/F(3) = 1/3.
    Tier-k: 1/F(k).
    """
    tiers = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        if any(s in lname for s in ("embed", "qkv", "net.0.weight", "head.weight")):
            tiers[name] = 1
        elif "out" in lname or "ln" in lname:
            tiers[name] = 2
        else:
            tiers[name] = 3
    return tiers


def lazy_optimizer_step(optimizer, model, tier_map, step, generator):
    """Stochastic optimizer step with Fibonacci-frequency parameter updates.

    For each parameter, draw a Bernoulli(1/F(tier)) to decide whether to
    apply the optimizer's stored gradient. Skip means zero out the grad
    so the optimizer step is a no-op for that param.

    Returns the fraction of params that were actually updated this step.
    """
    n_active = 0
    n_total = 0
    for name, p in model.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        tier = tier_map.get(name, 1)
        F_tier = FIBONACCI[min(tier - 1, len(FIBONACCI) - 1)]
        active = torch.rand(1, generator=generator).item() < 1.0 / F_tier
        if not active:
            p.grad = None        # cheaper than zeroing; optimizer will skip
        else:
            n_active += p.numel()
        n_total += p.numel()
    optimizer.step()
    return n_active / max(n_total, 1)


class StochasticDepthWrapper(nn.Module):
    """Wraps a TinyLM and replaces its block-loop with stochastic skipping.

    Each block at depth i has activation probability 1/F(i+1). Block 0
    always runs (tier 1 -> F(1)=1 -> p=1.0). Block 1 runs with p=1/2.
    Block 2 with p=1/3. Etc.

    Inactive blocks are pure identity — pass x through without compute.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x, gen=None):
        # Same forward as TinyLM, but skipping blocks.
        B, T = x.shape
        h = self.model.embed(x) + self.model.pe[:T]
        mask = self.model.mask[:T, :T]
        for i, block in enumerate(self.model.blocks):
            F_i = FIBONACCI[min(i, len(FIBONACCI) - 1)]
            p_active = 1.0 / F_i
            if (gen is not None and self.training
                    and torch.rand(1, generator=gen).item() >= p_active):
                continue        # block dropped: identity passthrough
            h = block(h, mask)
        h = self.model.ln_f(h)
        return self.model.head(h)


def make_baseline_model(vocab_size, args):
    return make_model(
        "crt_only", vocab_size=vocab_size, seq_len=args.seq_len,
        d_model=args.d_model, n_blocks=args.n_blocks,
    )


# ----------------------------------------------------------------------------
# Training loops for each variant
# ----------------------------------------------------------------------------


def train_baseline(model, train_split, val_split, args, gen):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    t0 = time.time()
    val_hist = []
    for step in range(args.steps):
        x, y = get_batch_split(train_split, args.batch_size, args.seq_len, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % args.eval_every == 0 or step == args.steps - 1:
            vl = evaluate(model, val_split, args.batch_size, args.seq_len, 16, gen)
            val_hist.append((step, vl, time.time() - t0))
            print(f"    step {step:5d}  val={vl:.4f}  ({time.time()-t0:.1f}s)",
                  flush=True)
    final = evaluate(model, val_split, args.batch_size, args.seq_len, 32, gen)
    return val_hist, final, time.time() - t0


def train_v1_ffpu(model, train_split, val_split, args, gen):
    """Frequency-Folded Parameter Updates."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    tier_map = assign_tensor_tiers(model)
    print("    tier assignments:")
    for name, t in tier_map.items():
        if t > 1:
            print(f"      {name}: tier {t} (~1/{FIBONACCI[t-1]} update prob)")
    t0 = time.time()
    val_hist = []
    update_fracs = []
    for step in range(args.steps):
        x, y = get_batch_split(train_split, args.batch_size, args.seq_len, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad(); loss.backward()
        frac = lazy_optimizer_step(optimizer, model, tier_map, step, gen)
        update_fracs.append(frac)
        if step % args.eval_every == 0 or step == args.steps - 1:
            vl = evaluate(model, val_split, args.batch_size, args.seq_len, 16, gen)
            val_hist.append((step, vl, time.time() - t0))
            avg_frac = sum(update_fracs[-100:]) / max(len(update_fracs[-100:]), 1)
            print(f"    step {step:5d}  val={vl:.4f}  "
                  f"update_frac={avg_frac:.2f}  ({time.time()-t0:.1f}s)", flush=True)
    final = evaluate(model, val_split, args.batch_size, args.seq_len, 32, gen)
    return val_hist, final, time.time() - t0, sum(update_fracs)/len(update_fracs)


def train_v2_stofib_depth(model, train_split, val_split, args, gen):
    """Stochastic Fibonacci depth — block-skip per step."""
    wrapped = StochasticDepthWrapper(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    t0 = time.time()
    val_hist = []
    for step in range(args.steps):
        x, y = get_batch_split(train_split, args.batch_size, args.seq_len, gen)
        logits = wrapped(x, gen=gen)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % args.eval_every == 0 or step == args.steps - 1:
            # Eval with all blocks ACTIVE — eval is not lazy.
            wrapped.eval()
            vl = evaluate(model, val_split, args.batch_size, args.seq_len, 16, gen)
            wrapped.train()
            val_hist.append((step, vl, time.time() - t0))
            print(f"    step {step:5d}  val={vl:.4f}  ({time.time()-t0:.1f}s)",
                  flush=True)
    final = evaluate(model, val_split, args.batch_size, args.seq_len, 32, gen)
    return val_hist, final, time.time() - t0


def train_v3_curriculum(model, train_split, val_split, args, gen):
    """Fibonacci curriculum — seq_len grows {11, 21, 34, 55, 89, 128}.

    Equal steps per stage. Model was constructed with max seq_len, we just
    truncate the input.
    """
    stages = [11, 21, 34, 55, 89, args.seq_len]
    steps_per_stage = args.steps // len(stages)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    t0 = time.time()
    val_hist = []
    for stage_i, stage_len in enumerate(stages):
        start_step = stage_i * steps_per_stage
        end_step = min((stage_i + 1) * steps_per_stage, args.steps)
        print(f"    [stage {stage_i}: seq_len={stage_len}, "
              f"steps {start_step}..{end_step-1}]", flush=True)
        for step in range(start_step, end_step):
            x, y = get_batch_split(train_split, args.batch_size, stage_len, gen)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            if step % args.eval_every == 0 or step == args.steps - 1:
                # Eval at FULL seq_len so val is comparable across variants.
                vl = evaluate(model, val_split, args.batch_size, args.seq_len, 16, gen)
                val_hist.append((step, vl, time.time() - t0))
                print(f"    step {step:5d}  val={vl:.4f}  "
                      f"({time.time()-t0:.1f}s)", flush=True)
    final = evaluate(model, val_split, args.batch_size, args.seq_len, 32, gen)
    return val_hist, final, time.time() - t0


def train_merged(model, train_split, val_split, args, gen):
    """Compose v1 + v2 + v3."""
    wrapped = StochasticDepthWrapper(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    tier_map = assign_tensor_tiers(model)
    stages = [11, 21, 34, 55, 89, args.seq_len]
    steps_per_stage = args.steps // len(stages)
    t0 = time.time()
    val_hist = []
    update_fracs = []
    for stage_i, stage_len in enumerate(stages):
        start_step = stage_i * steps_per_stage
        end_step = min((stage_i + 1) * steps_per_stage, args.steps)
        print(f"    [stage {stage_i}: seq_len={stage_len}, "
              f"steps {start_step}..{end_step-1}]", flush=True)
        for step in range(start_step, end_step):
            x, y = get_batch_split(train_split, args.batch_size, stage_len, gen)
            logits = wrapped(x, gen=gen)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            optimizer.zero_grad(); loss.backward()
            frac = lazy_optimizer_step(optimizer, model, tier_map, step, gen)
            update_fracs.append(frac)
            if step % args.eval_every == 0 or step == args.steps - 1:
                wrapped.eval()
                vl = evaluate(model, val_split, args.batch_size, args.seq_len, 16, gen)
                wrapped.train()
                val_hist.append((step, vl, time.time() - t0))
                avg_frac = sum(update_fracs[-100:]) / max(len(update_fracs[-100:]), 1)
                print(f"    step {step:5d}  val={vl:.4f}  "
                      f"update_frac={avg_frac:.2f}  ({time.time()-t0:.1f}s)",
                      flush=True)
    final = evaluate(model, val_split, args.batch_size, args.seq_len, 32, gen)
    return val_hist, final, time.time() - t0, sum(update_fracs)/len(update_fracs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distractor-frac", type=float, default=0.20)
    parser.add_argument("--out", type=str, default="results_lazy_training.json")
    args = parser.parse_args()

    chars, stoi, itos, encoded = make_dataset(
        seq_len=args.seq_len, source="tinyshakespeare",
    )
    vocab_size = len(chars)
    train_split, val_split = build_distractor_stream(
        encoded, args.distractor_frac, args.seq_len, args.seed,
    )

    print(f"Lazy-training bench")
    print(f"Corpus: TinyShakespeare ({encoded.numel():,} chars, vocab {vocab_size})")
    print(f"Model: d_model={args.d_model}, n_blocks={args.n_blocks}, "
          f"seq_len={args.seq_len}", flush=True)

    results = {}

    def run(name, train_fn):
        print(f"\n--- {name} ---")
        torch.manual_seed(args.seed)
        gen = torch.Generator(); gen.manual_seed(args.seed + 1)
        model = make_baseline_model(vocab_size, args)
        out = train_fn(model, train_split, val_split, args, gen)
        if len(out) == 4:
            hist, final, wall, frac = out
            results[name] = {"final_val": final, "wall": wall,
                              "update_frac": frac,
                              "val_hist": hist}
        else:
            hist, final, wall = out
            results[name] = {"final_val": final, "wall": wall,
                              "val_hist": hist}
        print(f"  ✓ {name}: final_val={final:.4f}, wall={wall:.1f}s, "
              f"steps/sec={args.steps/wall:.1f}")

    run("dense_baseline", train_baseline)
    run("v1_ffpu", train_v1_ffpu)
    run("v2_stofib_depth", train_v2_stofib_depth)
    run("v3_curriculum", train_v3_curriculum)
    run("merged_all", train_merged)

    # Summary
    print()
    print("=" * 92)
    print(f"{'variant':<22} {'val':>10} {'wall':>10} {'speedup':>10} "
          f"{'Δval':>10} {'Δval%':>10}")
    print("-" * 92)
    base = results["dense_baseline"]
    for name in ["dense_baseline", "v1_ffpu", "v2_stofib_depth",
                  "v3_curriculum", "merged_all"]:
        r = results[name]
        speedup = base["wall"] / r["wall"]
        dval = r["final_val"] - base["final_val"]
        dval_pct = dval / base["final_val"] * 100
        print(f"{name:<22} {r['final_val']:>10.4f} {r['wall']:>9.1f}s "
              f"{speedup:>9.2f}x {dval:>+10.4f} {dval_pct:>+9.1f}%")

    # Verdict per variant
    print()
    print("VERDICT (validation = within +10% loss vs baseline):")
    for name in ["v1_ffpu", "v2_stofib_depth", "v3_curriculum", "merged_all"]:
        r = results[name]
        dval_pct = (r["final_val"] - base["final_val"]) / base["final_val"] * 100
        speedup = base["wall"] / r["wall"]
        if dval_pct < 10 and speedup > 1.1:
            verdict = f"VALIDATED ({speedup:.2f}x speedup, +{dval_pct:.1f}% loss)"
        elif dval_pct < 10:
            verdict = f"no speedup ({speedup:.2f}x)"
        elif speedup > 1.5:
            verdict = f"FAST BUT BROKEN ({speedup:.2f}x speedup, +{dval_pct:.1f}% loss)"
        else:
            verdict = f"FAILED ({speedup:.2f}x speedup, +{dval_pct:.1f}% loss)"
        print(f"  {name:<22}: {verdict}")

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

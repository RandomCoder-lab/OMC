"""Phase 2: substrate self-recursion training.

Builds on the locked substrate stack (Subsim attn + V2 activation +
FibGen weights + FibAdamW + ce_fft + K-shrink + FibRecLM depth). Adds
a SELF-HARMONY loss that requires no target -- the substrate's own
canonical Fibonacci-frequency decay pattern serves as the prior.

Three arms (all use the substrate stack, train on a TINY ~1k-char
Shakespeare seed):

  tiny_baseline       CE + ce_fft only (no self-recursion)
  tiny_with_harmony   CE + ce_fft + lambda * substrate_harmony_loss
  tiny_self_recursive Interleave supervised (CE on seed) with
                       self-generation + harmony scoring on model's
                       own output. Model generates, scores its own
                       harmony, backprops -- no external label needed.

Hypothesis: with tiny data, the substrate prior (Fibonacci-tier
decay) fills in what the data can't teach. Harmony regularizer
should reduce held-out val. Self-recursion should match or beat
even the harmony regularizer.

Compare against tiny_baseline_gelu (vanilla transformer block with
same data budget) to measure substrate's data-efficiency gain.
"""

import argparse
import json
import sys
import time
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from corpus import make_dataset
from models_fibrec import FibRecLM, stateless_fibgen_forward
from optimizers_fib import FibonacciAdamW
from lazy_data import fib_positions_in_window, get_fib_strided_batch
from train_K_shrink import K_schedule_tier_walk, set_K_active_recursive
from losses_substrate import (substrate_fft_loss, substrate_harmony_loss,
                                substrate_multiscale_harmony_loss,
                                corpus_char_signature,
                                corpus_multiscale_signature,
                                substrate_harmony_loss_grounded,
                                substrate_multiscale_harmony_loss_grounded)
from activations_substrate import SubstrateNegMultiAdvancedV2
from train_substrate_attention import FibRecLMSubsim


def take_tiny_seed(encoded: torch.Tensor, n_chars: int,
                    seed: int = 42) -> torch.Tensor:
    """Slice an n-char window from encoded data at a deterministic offset."""
    g = torch.Generator(); g.manual_seed(seed)
    max_start = encoded.numel() - n_chars
    start = torch.randint(0, max_start, (1,), generator=g).item()
    return encoded[start: start + n_chars].clone()


def evaluate(model, val_split, batch_size, window, fib_positions, generator,
              n_batches=16):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = get_fib_strided_batch(val_split, batch_size, window,
                                           fib_positions, generator)
            logits = model(x)
            losses.append(F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item())
    model.train()
    return sum(losses) / len(losses)


def sample_tiny_batch(seed: torch.Tensor, batch_size: int, window: int,
                       gen: torch.Generator):
    """Random-stride batch from the tiny seed (cycled if needed)."""
    n = seed.numel()
    if n <= window + 1:
        # Pad by wrapping
        seed = seed.repeat((window + 2) // n + 1)
        n = seed.numel()
    starts = torch.randint(0, n - window - 1, (batch_size,), generator=gen)
    xs = torch.stack([seed[s: s + window] for s in starts])
    ys = torch.stack([seed[s + 1: s + window + 1] for s in starts])
    return xs, ys


def autoregressive_generate(model, prompt: torch.Tensor, n_new: int,
                              vocab_size: int, temperature: float = 1.0):
    """Sample n_new tokens autoregressively from prompt. Returns the full
    sequence (prompt + generated). No gradient tracked."""
    model.eval()
    with torch.no_grad():
        seq = prompt.clone()
        for _ in range(n_new):
            T = seq.shape[1]
            ctx = seq if T <= model.seq_len else seq[:, -model.seq_len:]
            logits = model(ctx)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            seq = torch.cat([seq, next_tok], dim=1)
    model.train()
    return seq


def compute_harmony(logits, vocab_size, kind):
    """kind in {'none', 'char', 'multiscale', 'combined'}."""
    if kind == "none":
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    if kind == "char":
        return substrate_harmony_loss(logits, vocab_size)
    if kind == "multiscale":
        return substrate_multiscale_harmony_loss(logits, vocab_size)
    if kind == "combined":
        return (substrate_harmony_loss(logits, vocab_size)
                + substrate_multiscale_harmony_loss(logits, vocab_size))
    raise ValueError(f"unknown harmony kind: {kind}")


def K_to_K_harmony(K_active: int, K_init: int = 89, K_min: int = 13,
                     K_harmony_max: int = 7,
                     K_harmony_min: int = 2) -> int:
    """Map model's active K to the harmony's active frequency count.

    As the model's basis shrinks (89→13), the harmony's measuring stick
    shrinks proportionally (7→2). Substrate stays congruent with model.
    """
    if K_init <= K_min:
        return K_harmony_max
    frac = (K_active - K_min) / (K_init - K_min)
    K_harmony = round(K_harmony_min + frac * (K_harmony_max - K_harmony_min))
    return max(K_harmony_min, min(K_harmony_max, K_harmony))


def compute_harmony_grounded(logits, vocab_size, kind, sig_char, sig_ms,
                                K_harmony=None):
    """Corpus-grounded harmony. sig_char and sig_ms are pre-computed
    target signatures from the actual corpus. K_harmony shrinks the
    harmony's active frequency count to match model's K."""
    if kind == "none":
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    if kind == "char":
        return substrate_harmony_loss_grounded(logits, vocab_size, sig_char,
                                                  K_harmony=K_harmony)
    if kind == "multiscale":
        return substrate_multiscale_harmony_loss_grounded(
            logits, vocab_size, sig_ms, K_harmony=K_harmony)
    if kind == "combined":
        return (substrate_harmony_loss_grounded(logits, vocab_size, sig_char,
                                                   K_harmony=K_harmony)
                + substrate_multiscale_harmony_loss_grounded(
                    logits, vocab_size, sig_ms, K_harmony=K_harmony))
    raise ValueError(f"unknown harmony kind: {kind}")


_FIB_FREQS_LOCAL = [1, 2, 3, 5, 8, 13, 21]
_FIB_LAGS_LOCAL = [1, 2, 3, 5, 8, 13, 21]
_FIB_NUMS_LOCAL = [1, 1, 2, 3, 5, 8, 13]
PHI_LOCAL = (1.0 + 5.0 ** 0.5) / 2.0


class ParametricSubstrate:
    """Substrate constants (phi, pi_exp, fib_weights) as the ONLY mutable
    parameters. The canonical signature is always F(k)/phi^(pi_exp*k) --
    mutations stay congruent to the substrate formula. Free drift away
    from this family is forbidden by construction.
    """

    def __init__(self, phi=None, pi_exp=None, fib_weights=None):
        self.phi = PHI_LOCAL if phi is None else float(phi)
        self.pi_exp = math.pi if pi_exp is None else float(pi_exp)
        self.fib_weights = (list(_FIB_NUMS_LOCAL) if fib_weights is None
                              else [float(x) for x in fib_weights])

    def get_signature(self, K=None) -> torch.Tensor:
        if K is None:
            K = len(self.fib_weights)
        un = [self.fib_weights[k] / (self.phi ** (self.pi_exp * k))
              for k in range(K)]
        total = sum(un) + 1e-8
        return torch.tensor([u / total for u in un], dtype=torch.float)

    def clone(self):
        return ParametricSubstrate(self.phi, self.pi_exp, self.fib_weights)

    def perturb(self, rng, step_size: float = 0.05,
                  fib_step: float = 0.10) -> "ParametricSubstrate":
        """Joint perturbation: phi, pi_exp, AND fib_weights all change.

        Single-constant mutations get stuck on pi because phi/fib changes
        alone don't help. Joint vector perturbation lets all three drift
        together -- each mutation is a 9-dim move in (phi, pi, fib_0..K).

        Bounds: phi within +/- 20% of golden, pi_exp within +/- 20% of
        math.pi, fib_weights non-negative.
        """
        new = self.clone()
        K = len(self.fib_weights)
        # phi: small step
        d_phi = (rng.random() * 2 - 1) * step_size
        new.phi = max(PHI_LOCAL * 0.8,
                        min(PHI_LOCAL * 1.2, self.phi * (1 + d_phi)))
        # pi_exp: small step
        d_pi = (rng.random() * 2 - 1) * step_size
        new.pi_exp = max(math.pi * 0.8,
                            min(math.pi * 1.2, self.pi_exp * (1 + d_pi)))
        # fib_weights: larger step ("toss" the fibs more aggressively)
        for k in range(K):
            d_fib = (rng.random() * 2 - 1) * fib_step
            new.fib_weights[k] = max(0.1, self.fib_weights[k] * (1 + d_fib))
        return new

    def summary(self) -> str:
        fib_str = ",".join(f"{w:.2f}" for w in self.fib_weights[:5])
        return (f"phi={self.phi:.4f} pi={self.pi_exp:.4f} "
                f"fib=[{fib_str},...]")


def measure_emergent_signatures(model, seed, batch_size, seq_len, vocab_size,
                                  gen, n_batches=4):
    """Measure model's emergent substrate signatures from its training outputs.
    Mirrors corpus_char_signature/corpus_multiscale_signature but on
    model's predicted distributions, not on raw tokens."""
    model.eval()
    fib_freqs = torch.tensor(_FIB_FREQS_LOCAL, dtype=torch.float)
    K = fib_freqs.numel()
    v_idx = torch.arange(vocab_size, dtype=torch.float)
    angles = 2 * math.pi * v_idx.unsqueeze(1) * fib_freqs.unsqueeze(0) / vocab_size
    basis_cos = torch.cos(angles)
    basis_sin = torch.sin(angles)
    energies = []
    sims_per = [[] for _ in range(K)]
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = sample_tiny_batch(seed, batch_size, seq_len, gen)
            logits = model(x)
            pred = F.softmax(logits, dim=-1)
            pred_cos = pred @ basis_cos
            pred_sin = pred @ basis_sin
            energy = (pred_cos ** 2 + pred_sin ** 2).mean(dim=(0, 1))
            energies.append(energy)
            T = pred.shape[1]
            for i, lag in enumerate(_FIB_LAGS_LOCAL):
                if T <= lag:
                    sims_per[i].append(torch.tensor(0.0))
                    continue
                p1 = pred[:, :-lag]
                p2 = pred[:, lag:]
                sim = (p1 * p2).sum(dim=-1).mean()
                sims_per[i].append(sim)
    model.train()
    energy_mean = torch.stack(energies).mean(0)
    energy_mean = energy_mean / (energy_mean.sum() + 1e-8)
    ms_mean = torch.stack([torch.stack(s).mean() for s in sims_per])
    ms_mean = ms_mean / (ms_mean.sum() + 1e-8)
    return energy_mean, ms_mean


def train_mutable_substrate(name, train_seed, corpus_anchor, val_split,
                              vocab_size, args, fib_positions,
                              harmony_kind="char",
                              mutation_every: int = 200,
                              mutation_alpha: float = 0.9):
    """Parametric substrate mutation with best-revert.

    Constants (phi, pi_exp, fib_weights) are the ONLY mutable values.
    Signatures are derived from them via F(k)/phi^(pi_exp*k). Mutations
    stay congruent to the substrate formula by construction.

    Best-revert history: every (state, val) is recorded after evaluation.
    A failed mutation (val degraded over the mutation window) reverts
    NOT to the immediately-previous state, but to the BEST state seen
    historically -- the lowest-val phi_pi_fib triple we've discovered.
    """
    import random as _rng_mod
    rng = _rng_mod.Random(args.seed + 7)
    torch.manual_seed(args.seed)
    gen = torch.Generator(); gen.manual_seed(args.seed + 1)
    model = FibRecLMSubsim(
        vocab_size=vocab_size, d_model=args.d_model, n_blocks=args.n_blocks,
        seq_len=args.seq_len, K=args.K_init, mode="cross", K_sig=args.K_sig,
    )
    optimizer = FibonacciAdamW(model.parameters(), lr=args.lr)
    sched = lambda s, T: K_schedule_tier_walk(s, T, K_init=args.K_init,
                                                 K_min=args.K_min)
    n_params = sum(p.numel() for p in model.parameters())

    # Parametric substrate: starts at canonical phi, pi, F(k) values.
    substrate = ParametricSubstrate()
    K_sig = len(_FIB_FREQS_LOCAL)
    sig_char = substrate.get_signature(K_sig)
    sig_ms = substrate.get_signature(K_sig)
    # History of accepted substrate states. Format: list of (state, val).
    history = [(substrate.clone(), float("inf"))]

    print(f"\n[parametric_mutable {name}]  harmony={harmony_kind}  "
          f"mutate_every={mutation_every}  params={n_params:,}", flush=True)
    print(f"  initial constants: {substrate.summary()}")
    print(f"  initial sig: {[round(x, 4) for x in sig_char.tolist()]}")

    t0 = time.time()
    best_val = float("inf"); best_step = -1
    cur_K = None
    eval_every = max(args.steps // 20, 100)
    n_mutations_tried = 0
    n_mutations_kept = 0
    n_mutations_reverted = 0
    last_mutation_step = -mutation_every   # so first mutation can fire promptly
    pending = None    # state for revert-on-fail mutation

    for step in range(args.steps):
        new_K = sched(step, args.steps)
        if new_K != cur_K:
            set_K_active_recursive(model, new_K)
            cur_K = new_K
        x, y = sample_tiny_batch(train_seed, args.batch_size, args.seq_len, gen)
        logits = model(x)
        ce_fft = substrate_fft_loss(logits, y, vocab_size,
                                      lambda_substrate=args.lambda_sub)
        # K_harmony shrinks with model's K -- substrate measures only what
        # the active basis can express.
        K_harmony = K_to_K_harmony(cur_K or args.K_init,
                                      K_init=args.K_init, K_min=args.K_min)
        harmony = compute_harmony_grounded(logits, vocab_size, harmony_kind,
                                             sig_char, sig_ms,
                                             K_harmony=K_harmony)
        loss = ce_fft + args.lambda_harmony * harmony
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        if step % eval_every == 0 or step == args.steps - 1:
            vl = evaluate(model, val_split, args.batch_size, args.seq_len,
                          fib_positions, gen)
            marker = ""
            if vl < best_val:
                best_val = vl; best_step = step
                marker = " ← BEST"
            print(f"  step {step:5d}  val={vl:.4f}  K={cur_K}  "
                  f"({time.time()-t0:.1f}s){marker}", flush=True)

            # Revert-on-fail with BEST-revert: check pending mutation outcome.
            if pending is not None and step >= pending["eval_step"]:
                kept_this = best_val < pending["baseline_val"]
                if kept_this:
                    # Mutation helped -- accept current substrate, add to history.
                    history.append((substrate.clone(), best_val))
                    n_mutations_kept += 1
                else:
                    # Mutation failed -- revert to BEST historical state, not
                    # just the immediately-previous one. Selection pressure.
                    best_state, _ = min(history, key=lambda x: x[1])
                    substrate = best_state.clone()
                    sig_char = substrate.get_signature(K_sig)
                    sig_ms = substrate.get_signature(K_sig)
                    n_mutations_reverted += 1
                # Print only on KEPT or every 10th mutation.
                if kept_this or n_mutations_tried % 10 == 0:
                    status = "KEPT" if kept_this else "REVERTED to best"
                    print(f"    [mutation {status}]  {substrate.summary()}  "
                          f"(tried={n_mutations_tried} kept={n_mutations_kept} "
                          f"reverted={n_mutations_reverted})", flush=True)
                pending = None

            # Apply a new mutation (no pending one, enough steps elapsed).
            if pending is None and (step - last_mutation_step) >= mutation_every:
                # Start from BEST historical state (selection pressure)
                # then perturb from there. Aggressive exploration with
                # larger steps: many tries -> better chance of finding
                # productive directions in 9-D phi_pi_fib space.
                best_state, _ = min(history, key=lambda x: x[1])
                substrate = best_state.clone().perturb(
                    rng, step_size=0.10, fib_step=0.15)
                sig_char = substrate.get_signature(K_sig)
                sig_ms = substrate.get_signature(K_sig)
                n_mutations_tried += 1
                last_mutation_step = step
                pending = {
                    "baseline_val": best_val,
                    "eval_step": step + mutation_every,
                }
                # Only print every 5th mutation to keep log readable.
                if n_mutations_tried <= 5 or n_mutations_tried % 5 == 0:
                    print(f"    [mutation TRIED]  {substrate.summary()}  "
                          f"baseline_val={best_val:.4f}  "
                          f"(n_tried={n_mutations_tried})", flush=True)

    # Final generation sample.
    prompt = train_seed[:16].unsqueeze(0)
    final_gen = autoregressive_generate(model, prompt, n_new=240,
                                          vocab_size=vocab_size,
                                          temperature=0.8)
    best_state, best_state_val = min(history, key=lambda x: x[1])
    print(f"  best constants: {best_state.summary()}  val={best_state_val:.4f}")
    return {"name": name, "mode": "parametric_mutable", "n_params": n_params,
             "best_val": best_val, "best_step": best_step,
             "wall": time.time() - t0,
             "n_mutations_tried": n_mutations_tried,
             "n_mutations_kept": n_mutations_kept,
             "n_mutations_reverted": n_mutations_reverted,
             "best_constants": {
                 "phi": best_state.phi,
                 "pi_exp": best_state.pi_exp,
                 "fib_weights": best_state.fib_weights,
             },
             "final_sig_char": sig_char.tolist(),
             "final_sig_ms": sig_ms.tolist(),
             "generated_tokens": final_gen[0].tolist()}


def train_multi_cycle(name, train_seed, corpus_anchor, val_split, vocab_size,
                       args, fib_positions, harmony_kind="multiscale",
                       n_cycles: int = 3,
                       samples_per_cycle: int = 8,
                       keep_top_frac: float = 0.5):
    """Multi-cycle self-training with corpus-grounded substrate.

    The corpus_anchor (NOT used for token-level training) provides the
    substrate fingerprint -- char-level + multi-scale signatures the
    model must match. The model trains on the tiny seed which GROWS
    each cycle with the model's own most-harmonious generated samples.

    Anchor against collapse: corpus_anchor's signatures are fixed
    (measured once). The model's harmony loss is L1 distance from
    those signatures. Drift toward gibberish would raise this loss.
    """
    torch.manual_seed(args.seed)
    gen = torch.Generator(); gen.manual_seed(args.seed + 1)
    model = FibRecLMSubsim(
        vocab_size=vocab_size, d_model=args.d_model, n_blocks=args.n_blocks,
        seq_len=args.seq_len, K=args.K_init, mode="cross", K_sig=args.K_sig,
    )
    optimizer = FibonacciAdamW(model.parameters(), lr=args.lr)
    sched = lambda s, T: K_schedule_tier_walk(s, T, K_init=args.K_init,
                                                 K_min=args.K_min)
    n_params = sum(p.numel() for p in model.parameters())

    # Compute corpus signatures (the substrate truth) -- done ONCE.
    sig_char = corpus_char_signature(corpus_anchor, vocab_size)
    sig_ms = corpus_multiscale_signature(corpus_anchor, vocab_size,
                                            seq_len=args.seq_len)
    print(f"\n[multi_cycle {name}]  harmony={harmony_kind}  "
          f"n_cycles={n_cycles}  samples_per_cycle={samples_per_cycle}  "
          f"params={n_params:,}", flush=True)
    print(f"  corpus_anchor: {corpus_anchor.numel()} chars")
    print(f"  sig_char (corpus): {[round(x, 4) for x in sig_char.tolist()]}")
    print(f"  sig_ms (corpus):   {[round(x, 4) for x in sig_ms.tolist()]}")

    t0 = time.time()
    corpus_tokens = train_seed.clone()
    best_val = float("inf"); best_step = -1; global_step = 0
    cur_K = None
    steps_per_cycle = args.steps // n_cycles
    eval_every = max(steps_per_cycle // 6, 100)

    for cycle in range(n_cycles):
        print(f"\n  --- Cycle {cycle+1}/{n_cycles}  "
              f"corpus_size={corpus_tokens.numel()} chars ---", flush=True)

        # Phase A: supervised + grounded harmony on current corpus_tokens.
        for s in range(steps_per_cycle):
            new_K = sched(global_step, args.steps)
            if new_K != cur_K:
                set_K_active_recursive(model, new_K)
                cur_K = new_K
            x, y = sample_tiny_batch(corpus_tokens, args.batch_size,
                                       args.seq_len, gen)
            logits = model(x)
            ce_fft = substrate_fft_loss(logits, y, vocab_size,
                                          lambda_substrate=args.lambda_sub)
            harmony = compute_harmony_grounded(logits, vocab_size,
                                                 harmony_kind, sig_char, sig_ms)
            loss = ce_fft + args.lambda_harmony * harmony
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            if global_step % eval_every == 0:
                vl = evaluate(model, val_split, args.batch_size, args.seq_len,
                              fib_positions, gen)
                marker = ""
                if vl < best_val:
                    best_val = vl; best_step = global_step
                    marker = " ← BEST"
                print(f"    step {global_step:5d}  val={vl:.4f}  "
                      f"K={cur_K}  ({time.time()-t0:.1f}s){marker}",
                      flush=True)
            global_step += 1

        # Phase B: generate, score harmony, keep top, add to corpus.
        print(f"  generating {samples_per_cycle} samples to score...",
              flush=True)
        samples_scored = []
        for s in range(samples_per_cycle):
            prompt_len = 16
            prompt_start = torch.randint(0, corpus_tokens.numel() - prompt_len,
                                          (1,), generator=gen).item()
            prompt = corpus_tokens[prompt_start: prompt_start + prompt_len
                                    ].unsqueeze(0)
            gen_seq = autoregressive_generate(model, prompt,
                                                n_new=args.seq_len - prompt_len,
                                                vocab_size=vocab_size,
                                                temperature=0.8)
            with torch.no_grad():
                gen_logits = model(gen_seq[:, :args.seq_len])
                h = compute_harmony_grounded(gen_logits, vocab_size,
                                              harmony_kind, sig_char, sig_ms)
            samples_scored.append((gen_seq[0], h.item()))
        samples_scored.sort(key=lambda x: x[1])
        n_keep = max(1, int(samples_per_cycle * keep_top_frac))
        top_samples = samples_scored[:n_keep]
        scores = [s[1] for s in samples_scored]
        kept_scores = [s[1] for s in top_samples]
        print(f"  harmony scores: all={[round(s, 4) for s in scores]}")
        print(f"  kept (top {n_keep}): {[round(s, 4) for s in kept_scores]}")

        # Grow corpus with the top-harmony generations.
        for s in top_samples:
            corpus_tokens = torch.cat([corpus_tokens, s[0]])

    # Final generation sample for inspection.
    prompt = train_seed[:16].unsqueeze(0)
    final_gen = autoregressive_generate(model, prompt, n_new=240,
                                          vocab_size=vocab_size,
                                          temperature=0.8)
    return {"name": name, "mode": "multi_cycle", "n_params": n_params,
             "best_val": best_val, "best_step": best_step,
             "wall": time.time() - t0,
             "final_corpus_size": corpus_tokens.numel(),
             "generated_tokens": final_gen[0].tolist()}


def train_arm(name, mode, train_seed, val_split, vocab_size, args,
               fib_positions, harmony_kind="char",
               phase_a_frac: float = 0.7):
    """mode in {'baseline', 'with_harmony', 'self_recursive', 'two_phase'}.
    harmony_kind in {'none', 'char', 'multiscale', 'combined'}.
    phase_a_frac: for two_phase mode, fraction of steps spent in
    supervised Phase A before switching to self-recursive Phase B."""
    torch.manual_seed(args.seed)
    gen = torch.Generator(); gen.manual_seed(args.seed + 1)
    model = FibRecLMSubsim(
        vocab_size=vocab_size, d_model=args.d_model, n_blocks=args.n_blocks,
        seq_len=args.seq_len, K=args.K_init, mode="cross", K_sig=args.K_sig,
    )
    optimizer = FibonacciAdamW(model.parameters(), lr=args.lr)
    sched = lambda s, T: K_schedule_tier_walk(s, T, K_init=args.K_init,
                                                 K_min=args.K_min)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[train {name}]  mode={mode}  harmony={harmony_kind}  "
          f"tiny_seed_chars={train_seed.numel()}  params={n_params:,}",
          flush=True)
    t0 = time.time()
    best_val = float("inf"); best_step = -1
    cur_K = None
    eval_every = max(args.steps // 20, 100)
    for step in range(args.steps):
        new_K = sched(step, args.steps)
        if new_K != cur_K:
            set_K_active_recursive(model, new_K)
            cur_K = new_K

        phase_a_steps = int(args.steps * phase_a_frac)
        # In two_phase mode, Phase B starts after phase_a_steps.
        in_phase_b = (mode == "two_phase" and step >= phase_a_steps)

        if (mode == "self_recursive" and step > 0 and step % 5 == 0) \
                or in_phase_b:
            # Self-recursion step: generate from prompt, score harmony.
            # In two_phase, this runs EVERY step in Phase B (no supervised
            # signal -- the model now reviews its own output, refines via
            # the substrate harmony prior).
            prompt_len = 16
            prompt = train_seed[:prompt_len].unsqueeze(0).repeat(
                args.batch_size, 1)
            seq = autoregressive_generate(model, prompt,
                                            n_new=args.seq_len - prompt_len,
                                            vocab_size=vocab_size)
            x = seq[:, :-1]; y = seq[:, 1:]
            logits = model(x)
            harmony = compute_harmony(logits, vocab_size, harmony_kind)
            if in_phase_b:
                # Pure harmony refinement -- no CE target on self-output.
                # Model reviews its own work against the substrate prior.
                loss = harmony
            else:
                # Old self_recursive mode: still uses CE on self-output.
                ce = F.cross_entropy(logits.reshape(-1, vocab_size),
                                       y.reshape(-1))
                loss = ce + args.lambda_harmony * harmony
        else:
            # Supervised step on tiny seed.
            x, y = sample_tiny_batch(train_seed, args.batch_size, args.seq_len,
                                       gen)
            logits = model(x)
            loss = substrate_fft_loss(logits, y, vocab_size,
                                        lambda_substrate=args.lambda_sub)
            if mode in ("with_harmony", "self_recursive"):
                harmony = compute_harmony(logits, vocab_size, harmony_kind)
                loss = loss + args.lambda_harmony * harmony

        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % eval_every == 0 or step == args.steps - 1:
            vl = evaluate(model, val_split, args.batch_size, args.seq_len,
                          fib_positions, gen)
            marker = ""
            if vl < best_val:
                best_val = vl; best_step = step
                marker = " ← BEST"
            print(f"  step {step:5d}  val={vl:.4f}  K={cur_K}  "
                  f"({time.time()-t0:.1f}s){marker}", flush=True)
    # Post-training: generate a sample to qualitatively see the output.
    sample_prompt = train_seed[:16].unsqueeze(0)
    gen_seq = autoregressive_generate(model, sample_prompt,
                                        n_new=240, vocab_size=vocab_size,
                                        temperature=0.8)
    return {"name": name, "mode": mode, "n_params": n_params,
             "best_val": best_val, "best_step": best_step,
             "wall": time.time() - t0,
             "generated_tokens": gen_seq[0].tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--K-init", type=int, default=89)
    parser.add_argument("--K-min", type=int, default=13)   # restore K-shrink

    parser.add_argument("--K-sig", type=int, default=16)
    parser.add_argument("--lambda-sub", type=float, default=0.01)
    parser.add_argument("--lambda-harmony", type=float, default=0.05)
    parser.add_argument("--tiny-chars", type=int, default=1024,
                          help="Size of the tiny training seed in chars")
    parser.add_argument("--out", type=str,
                          default="results_self_recursive.json")
    args = parser.parse_args()

    chars, stoi, itos, encoded = make_dataset(seq_len=args.seq_len,
                                                 source="tinyshakespeare")
    vocab_size = len(chars)
    # Tiny train seed; full val for evaluation
    train_seed = take_tiny_seed(encoded, args.tiny_chars, seed=args.seed)
    val_start = encoded.numel() // 10 * 9
    val_split = encoded[val_start:].clone()
    fib_positions = fib_positions_in_window(args.seq_len)

    print(f"Tiny training seed: {train_seed.numel()} chars; "
          f"val on {val_split.numel()} chars")

    # Multi-cycle adaptive substrate: corpus signatures are the truth
    # (anchor), seed corpus grows with model's most-harmonious generations.
    # We use a large held-out slice of the full corpus to compute target
    # signatures -- the model sees only TINY tokens for CE but the
    # substrate target captures FULL corpus structure.
    anchor_start = 0
    anchor_size = min(20000, val_start)   # 20k chars of corpus structure
    corpus_anchor = encoded[anchor_start: anchor_start + anchor_size].clone()

    # Aggressive search: many mutations, larger steps, shorter eval window.
    arms = [
        ("mutable_char",        "char"),
        ("mutable_multiscale",  "multiscale"),
    ]
    results = {}
    for name, harmony_kind in arms:
        results[name] = train_mutable_substrate(name, train_seed, corpus_anchor,
                                                   val_split, vocab_size, args,
                                                   fib_positions,
                                                   harmony_kind=harmony_kind,
                                                   mutation_every=100,
                                                   mutation_alpha=0.9)

    print()
    print("=" * 92)
    print(f"{'arm':<24} {'params':>10} {'best_val':>10} {'wall':>10}")
    print('-' * 92)
    for name, r in results.items():
        print(f"{name:<24} {r['n_params']:>10,} {r['best_val']:>10.4f} "
              f"{r['wall']:>9.1f}s")

    # Compute deltas vs known references.
    REF_BASELINE = 3.5526       # tiny_baseline (Subsim, no harmony)
    REF_CHAR = 3.4501           # char-level harmony, 1/phi^(pi*k)
    REF_CHAR_REFINED = 3.4920   # char-level harmony, F(k)/phi^(pi*k)
    print()
    print(f"refs:  baseline={REF_BASELINE}  char(pure)={REF_CHAR}  "
          f"char(F-decay)={REF_CHAR_REFINED}")
    for name, r in results.items():
        d_base = (r["best_val"] - REF_BASELINE) / REF_BASELINE * 100
        d_char = (r["best_val"] - REF_CHAR) / REF_CHAR * 100
        print(f"  {name:<24} val={r['best_val']:.4f}  "
              f"vs_baseline={d_base:+.2f}%  vs_char={d_char:+.2f}%")

    # Print decoded generation samples per arm.
    itos_map = {i: c for i, c in enumerate(chars)}
    def decode(toks):
        return ''.join(itos_map.get(int(t), '?') for t in toks)
    print()
    print("=" * 92)
    print("Generated samples (prompt = first 16 chars of seed, temp=0.8)")
    print('-' * 92)
    for name, r in results.items():
        text = decode(r["generated_tokens"])
        print(f"\n[{name}]")
        print(repr(text[:240]))

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

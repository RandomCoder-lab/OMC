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
from creativity_score import (creativity_score as compute_creativity_score,
                                  real_word_fraction)


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


def _single_stage_refine(model, draft, vocab_size, scorer, mode: str,
                            n_iters: int, resample_frac: float,
                            prompt_len: int, temperature: float,
                            patience: int = 5):
    """One refinement stage: optimize a single score until plateau.

    mode: 'min' (harmony, quality) or 'max' (creativity).
    patience: stop after this many consecutive iters with no improvement.
    n_iters acts as a safety cap; the stage typically ends earlier on
    natural plateau.

    Returns (best_seq, trajectory).
    """
    model.eval()
    with torch.no_grad():
        cur = draft.clone()
        cur_score = scorer(cur) if scorer is not None else None
        best_seq = cur.clone()
        best_score = cur_score
        trajectory = [cur_score]
        steps_since_improve = 0
        for it in range(n_iters):
            T = cur.shape[1]
            offset = max(0, T - model.seq_len)
            ctx = cur if T <= model.seq_len else cur[:, -model.seq_len:]
            logits = model(ctx)
            probs = F.softmax(logits / temperature, dim=-1)
            tokens_after_prefix = ctx[:, 1:]
            confidences = probs[:, :-1].gather(
                -1, tokens_after_prefix.unsqueeze(-1)).squeeze(-1)
            prompt_in_ctx = max(0, prompt_len - offset)
            confidences[:, :prompt_in_ctx] = 1.0
            n_avail = confidences.shape[1] - prompt_in_ctx
            n_resample = max(1, int(resample_frac * n_avail))
            n_resample = min(n_resample, max(1, n_avail))
            _, low_idx = confidences[0].topk(n_resample, largest=False)

            new = cur.clone()
            for idx in low_idx.tolist():
                t_draft = idx + 1 + offset
                if t_draft < new.shape[1] and t_draft >= prompt_len:
                    new[0, t_draft] = torch.multinomial(
                        probs[0, idx], num_samples=1).item()

            new_score = scorer(new) if scorer is not None else None
            trajectory.append(new_score)
            improved = False
            if new_score is not None:
                if mode == "max":
                    if best_score is None or new_score > best_score:
                        best_score = new_score; best_seq = new.clone()
                        improved = True
                else:                      # 'min'
                    if best_score is None or new_score < best_score:
                        best_score = new_score; best_seq = new.clone()
                        improved = True
            cur = new
            steps_since_improve = 0 if improved else steps_since_improve + 1
            if steps_since_improve >= patience:
                break
    model.train()
    return best_seq, trajectory


def staged_refine(model, prompt, n_new, vocab_size,
                    harmony_scorer, quality_scorer, creativity_scorer,
                    n_iters_per_stage: int = 200,
                    resample_frac: float = 0.35,
                    prompt_len: int = 16,
                    temperature: float = 0.5):
    """Staircase refinement: hit one score, then the next, then the next.

    Stage 1: substrate alignment (minimize harmony) -- match the shape.
    Stage 2: model coherence (minimize self-perplexity) -- output that
             the model itself finds plausible given the substrate shape.
    Stage 3: Shakespeare creativity (maximize creativity score) -- output
             that matches Shakespeare's char patterns and vocabulary.

    Each stage starts from the PREVIOUS stage's best output. Output of
    one objective becomes the input to the next.
    """
    model.eval()
    with torch.no_grad():
        draft = autoregressive_generate(model, prompt, n_new=n_new,
                                          vocab_size=vocab_size,
                                          temperature=temperature)
    stages_out = {}
    stages_out["initial"] = {"seq": draft.clone(),
                                "harmony": harmony_scorer(draft),
                                "quality": quality_scorer(draft),
                                "creativity": creativity_scorer(draft)
                                  if creativity_scorer else None}
    # Stage 1: harmony.
    draft, h_traj = _single_stage_refine(model, draft, vocab_size,
                                            harmony_scorer, mode="min",
                                            n_iters=n_iters_per_stage,
                                            resample_frac=resample_frac,
                                            prompt_len=prompt_len,
                                            temperature=temperature)
    stages_out["after_harmony"] = {"seq": draft.clone(),
                                       "trajectory": h_traj,
                                       "harmony": harmony_scorer(draft),
                                       "quality": quality_scorer(draft),
                                       "creativity": creativity_scorer(draft)
                                         if creativity_scorer else None}
    # Stage 2: quality.
    draft, q_traj = _single_stage_refine(model, draft, vocab_size,
                                            quality_scorer, mode="min",
                                            n_iters=n_iters_per_stage,
                                            resample_frac=resample_frac,
                                            prompt_len=prompt_len,
                                            temperature=temperature)
    stages_out["after_quality"] = {"seq": draft.clone(),
                                       "trajectory": q_traj,
                                       "harmony": harmony_scorer(draft),
                                       "quality": quality_scorer(draft),
                                       "creativity": creativity_scorer(draft)
                                         if creativity_scorer else None}
    # Stage 3: creativity (if scorer provided).
    if creativity_scorer is not None:
        draft, c_traj = _single_stage_refine(model, draft, vocab_size,
                                                creativity_scorer, mode="max",
                                                n_iters=n_iters_per_stage,
                                                resample_frac=resample_frac,
                                                prompt_len=prompt_len,
                                                temperature=temperature)
        stages_out["after_creativity"] = {"seq": draft.clone(),
                                              "trajectory": c_traj,
                                              "harmony": harmony_scorer(draft),
                                              "quality": quality_scorer(draft),
                                              "creativity": creativity_scorer(draft)}
    model.train()
    return draft, stages_out


def iterative_refine(model, prompt, n_new, vocab_size,
                       n_iters: int = 30,
                       resample_frac: float = 0.35,
                       prompt_len: int = 16,
                       harmony_scorer=None,
                       quality_scorer=None,
                       creativity_scorer=None,
                       temperature: float = 0.5,
                       force_run_all: bool = True):
    """Aggressive inference-time substrate-recursive refinement.

    Selection priority for returning the BEST sequence:
      creativity_scorer (HIGHER is better) > quality_scorer (LOWER) >
      harmony_scorer (LOWER).

    The selection target matters: val/harmony/quality reward exact
    replication or substrate alignment; creativity rewards
    Shakespeare-LIKE patterns without requiring exact word match.
    """
    model.eval()
    with torch.no_grad():
        # Step 1: initial draft.
        draft = autoregressive_generate(model, prompt, n_new=n_new,
                                          vocab_size=vocab_size,
                                          temperature=temperature)
        history = []
        h0 = harmony_scorer(draft) if harmony_scorer is not None else None
        q0 = quality_scorer(draft) if quality_scorer is not None else None
        c0 = creativity_scorer(draft) if creativity_scorer is not None else None
        history.append({"iter": 0, "harmony": h0, "quality": q0,
                          "creativity": c0,
                          "seq": draft.clone(), "n_resampled": 0})

        best_seq = draft.clone()
        # Selection priority: creativity (max), quality (min), harmony (min).
        if c0 is not None:
            best_score = c0; best_mode = "creativity_max"
        elif q0 is not None:
            best_score = q0; best_mode = "quality_min"
        else:
            best_score = h0; best_mode = "harmony_min"

        for it in range(1, n_iters + 1):
            T = draft.shape[1]
            offset = max(0, T - model.seq_len)        # draft -> ctx index offset
            ctx = draft if T <= model.seq_len else draft[:, -model.seq_len:]
            logits = model(ctx)
            probs = F.softmax(logits / temperature, dim=-1)
            tokens_after_prefix = ctx[:, 1:]
            confidences = probs[:, :-1].gather(
                -1, tokens_after_prefix.unsqueeze(-1)).squeeze(-1)
            # Don't touch the prompt portion (in draft coords, indices < prompt_len).
            # In ctx coords, that's indices < (prompt_len - offset).
            prompt_in_ctx = max(0, prompt_len - offset)
            confidences[:, :prompt_in_ctx] = 1.0
            n_avail = confidences.shape[1] - prompt_in_ctx
            n_resample = max(1, int(resample_frac * n_avail))
            n_resample = min(n_resample, max(1, n_avail))
            _, low_idx = confidences[0].topk(n_resample, largest=False)

            new_draft = draft.clone()
            for idx in low_idx.tolist():
                t_ctx = idx + 1                       # position in ctx
                t_draft = t_ctx + offset              # position in draft
                if t_draft < new_draft.shape[1] and t_draft >= prompt_len:
                    new_tok = torch.multinomial(probs[0, idx], num_samples=1)
                    new_draft[0, t_draft] = new_tok.item()

            new_h = (harmony_scorer(new_draft) if harmony_scorer is not None
                      else None)
            new_q = (quality_scorer(new_draft) if quality_scorer is not None
                      else None)
            new_c = (creativity_scorer(new_draft) if creativity_scorer is not None
                      else None)
            history.append({"iter": it, "harmony": new_h, "quality": new_q,
                              "creativity": new_c,
                              "seq": new_draft.clone(),
                              "n_resampled": n_resample})

            # Selection: creativity higher is better, quality/harmony lower.
            if best_mode == "creativity_max" and new_c is not None:
                if best_score is None or new_c > best_score:
                    best_seq = new_draft.clone(); best_score = new_c
            elif best_mode == "quality_min" and new_q is not None:
                if best_score is None or new_q < best_score:
                    best_seq = new_draft.clone(); best_score = new_q
            elif new_h is not None:
                if best_score is None or new_h < best_score:
                    best_seq = new_draft.clone(); best_score = new_h

            draft = new_draft
            if not force_run_all:
                # Early stopping on flat harmony (conservative mode).
                if (new_h is not None and h0 is not None and new_h >= h0):
                    break
                h0 = new_h

    model.train()
    return best_seq, history


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
        """Joint blind perturbation: phi, pi_exp, all fib_weights drift.
        No data signal -- pure random move bounded to substrate-physical
        ranges. Kept for ablation against data-guided perturbation.
        """
        new = self.clone()
        K = len(self.fib_weights)
        d_phi = (rng.random() * 2 - 1) * step_size
        new.phi = max(PHI_LOCAL * 0.8,
                        min(PHI_LOCAL * 1.2, self.phi * (1 + d_phi)))
        d_pi = (rng.random() * 2 - 1) * step_size
        new.pi_exp = max(math.pi * 0.8,
                            min(math.pi * 1.2, self.pi_exp * (1 + d_pi)))
        for k in range(K):
            d_fib = (rng.random() * 2 - 1) * fib_step
            new.fib_weights[k] = max(0.1, self.fib_weights[k] * (1 + d_fib))
        return new

    def data_guided_perturb(self, target_sig: torch.Tensor, rng,
                              step_size: float = 0.05,
                              noise_scale: float = 0.5,
                              K_active: int = None,
                              ) -> "ParametricSubstrate":
        """Mutation BIASED by the corpus signature gradient.

        Computes d|parametric_sig - target_sig|/d(phi, pi_exp, fib_weights)
        via autograd, mutates each constant in the descent direction with
        added noise (noise_scale fraction of step_size). The corpus tells
        the mutation where to push every constant; pure random + reward
        is replaced by data-informed proposal + reward.

        K_active: if set, only the first K_active components contribute
        to the gradient (matches K-harmony schedule).
        """
        K_full = len(self.fib_weights)
        K_use = K_full if K_active is None else min(K_active, K_full)
        # Tensors with grad enabled (must be float).
        phi_t = torch.tensor(float(self.phi), requires_grad=True)
        pi_t = torch.tensor(float(self.pi_exp), requires_grad=True)
        fib_t = torch.tensor([float(x) for x in self.fib_weights],
                              dtype=torch.float32, requires_grad=True)
        ks = torch.arange(K_use, dtype=torch.float)
        # parametric signature at K_use
        unnorm = fib_t[:K_use] / (phi_t ** (pi_t * ks))
        sig = unnorm / (unnorm.sum() + 1e-8)
        # gap to target (truncated to K_use)
        target = target_sig[:K_use]
        target = target / (target.sum() + 1e-8)
        loss = (sig - target).abs().sum()
        loss.backward()
        # Read gradients (descent = -grad).
        g_phi = float(phi_t.grad)
        g_pi = float(pi_t.grad)
        g_fib = fib_t.grad.tolist()

        new = self.clone()
        # phi: step in -g_phi direction (multiplicatively scaled) + noise.
        d_phi_rel = -g_phi * step_size + (rng.random() * 2 - 1) * step_size * noise_scale
        new.phi = max(PHI_LOCAL * 0.8,
                        min(PHI_LOCAL * 1.2, self.phi + d_phi_rel * abs(self.phi)))
        d_pi_rel = -g_pi * step_size + (rng.random() * 2 - 1) * step_size * noise_scale
        new.pi_exp = max(math.pi * 0.8,
                            min(math.pi * 1.2,
                                 self.pi_exp + d_pi_rel * abs(self.pi_exp)))
        for k in range(K_full):
            grad_k = g_fib[k] if k < K_use else 0.0
            d_fib_rel = (-grad_k * step_size
                          + (rng.random() * 2 - 1) * step_size * noise_scale)
            new.fib_weights[k] = max(0.1, self.fib_weights[k]
                                            + d_fib_rel * abs(self.fib_weights[k]))
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


def train_with_self_distillation(name, train_seed, corpus_anchor, val_split,
                                    vocab_size, args, fib_positions,
                                    harmony_kind="multiscale",
                                    itos_map=None,
                                    corpus_text=None,
                                    n_cycles: int = 4,
                                    distill_prob: float = 0.3,
                                    samples_per_cycle: int = 8,
                                    keep_top_k: int = 4,
                                    growth_n_new: int = 128):
    """Self-distillation: model's high-creativity refined outputs become
    training targets for the next cycle.

    Each cycle:
      1. Train for steps_per_cycle on (tiny_seed + distill_buffer)
         -- with prob `distill_prob` each batch comes from buffer.
      2. Generate a draft from current model.
      3. Refine via staged loop targeting creativity (not harmony!).
      4. Score the refined output's creativity.
      5. If creativity > best_seen: add to distill_buffer.

    Substrate stays as the scaffolding; creativity is the compass.
    The model's parameters move toward fixed-points that are both
    substrate-aligned AND linguistically creative (Shakespeare-like).
    """
    import random as _rng_mod
    rng = _rng_mod.Random(args.seed + 7)
    torch.manual_seed(args.seed)
    gen = torch.Generator(); gen.manual_seed(args.seed + 1)

    model = FibRecLMSubsim(
        vocab_size=vocab_size, d_model=args.d_model, n_blocks=args.n_blocks,
        seq_len=args.seq_len, K=args.K_init, mode="cross", K_sig=args.K_sig,
        substrate_embed=True,
    )
    optimizer = FibonacciAdamW(model.parameters(), lr=args.lr)
    sched = lambda s, T: K_schedule_tier_walk(s, T, K_init=args.K_init,
                                                 K_min=args.K_min)
    n_params = sum(p.numel() for p in model.parameters())

    sig_char = corpus_char_signature(corpus_anchor, vocab_size)
    sig_ms = corpus_multiscale_signature(corpus_anchor, vocab_size,
                                            seq_len=args.seq_len)

    full_corpus = corpus_text or ""
    def creativity_fn(seq_tokens):
        text = ''.join(itos_map.get(int(t), '?')
                        for t in seq_tokens[0].tolist())
        return compute_creativity_score(text, full_corpus)["creativity_score"]

    def harmony_fn(seq_tokens):
        with torch.no_grad():
            T = seq_tokens.shape[1]
            ctx = seq_tokens if T <= model.seq_len else seq_tokens[:, -model.seq_len:]
            logits = model(ctx)
            K_h = K_to_K_harmony(cur_K or args.K_init,
                                  K_init=args.K_init, K_min=args.K_min)
            return compute_harmony_grounded(logits, vocab_size, harmony_kind,
                                              sig_char, sig_ms,
                                              K_harmony=K_h).item()

    def quality_fn(seq_tokens):
        with torch.no_grad():
            T = seq_tokens.shape[1]
            ctx = seq_tokens if T <= model.seq_len else seq_tokens[:, -model.seq_len:]
            logits = model(ctx)
            return F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size),
                                     ctx[:, 1:].reshape(-1)).item()

    print(f"\n[self_distill {name}]  harmony={harmony_kind}  "
          f"n_cycles={n_cycles}  distill_prob={distill_prob}  "
          f"params={n_params:,}", flush=True)

    n_new = max(args.seq_len - 16, 32)

    # Compute corpus creativity baseline (over random n_new-char windows of
    # the actual corpus) -- the floor that refined output must beat to
    # be admitted to the active_base. Stops the model from feeding
    # itself sub-corpus-quality material.
    corpus_creativity_samples = []
    import random as _rng_seed
    _rng_seed_inst = _rng_seed.Random(args.seed + 11)
    sample_window = max(64, n_new + 16)
    n_corpus_samples = 50
    if corpus_text is not None and len(corpus_text) > sample_window + 1:
        for _ in range(n_corpus_samples):
            start = _rng_seed_inst.randint(0, len(corpus_text) - sample_window)
            chunk = corpus_text[start: start + sample_window]
            corpus_creativity_samples.append(
                compute_creativity_score(chunk, corpus_text)["creativity_score"])
        corpus_creativity_baseline = sorted(corpus_creativity_samples)[
            len(corpus_creativity_samples) // 2]   # median
    else:
        corpus_creativity_baseline = 0.0
    print(f"  corpus creativity baseline (median of {n_corpus_samples} "
          f"{sample_window}-char windows): {corpus_creativity_baseline:.4f}")

    # Anchor weight: original seed must remain at least this fraction of
    # active_base. Stops the model's mediocre output from dominating.
    seed_min_fraction = 0.70
    orig_seed_chars = train_seed.numel()

    # Active training base: starts as tiny_seed, GROWS by appending each
    # cycle's best refined output -- only if (a) creativity > corpus
    # baseline AND (b) anchor weight constraint still satisfied.
    active_base = train_seed.clone()
    best_creativity = 0.0
    best_refined_seq = None
    cycle_summary = []
    n_rejected_below_baseline = 0
    n_rejected_anchor = 0

    steps_per_cycle = args.steps // n_cycles
    t0 = time.time()
    best_val = float("inf"); best_step = -1
    cur_K = None
    eval_every = max(steps_per_cycle // 4, 100)
    global_step = 0
    prompt = train_seed[:16].unsqueeze(0)

    for cycle in range(n_cycles):
        print(f"\n  --- Cycle {cycle+1}/{n_cycles}  "
              f"active_base_size={active_base.numel()} chars "
              f"best_creativity={best_creativity:.4f} ---", flush=True)
        for s in range(steps_per_cycle):
            new_K = sched(global_step, args.steps)
            if new_K != cur_K:
                set_K_active_recursive(model, new_K)
                cur_K = new_K
            # Train entirely on the active_base (which is seed + appended
            # best-refined outputs). No mixing logic -- the active_base IS
            # the model's corpus, growing with every successful distillation.
            x, y = sample_tiny_batch(active_base, args.batch_size,
                                       args.seq_len, gen)
            logits = model(x)
            ce_fft = substrate_fft_loss(logits, y, vocab_size,
                                          lambda_substrate=args.lambda_sub)
            K_h = K_to_K_harmony(cur_K or args.K_init,
                                  K_init=args.K_init, K_min=args.K_min)
            harmony = compute_harmony_grounded(logits, vocab_size, harmony_kind,
                                                 sig_char, sig_ms,
                                                 K_harmony=K_h)
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

        # End of cycle: HEAVY extrapolation. Generate samples_per_cycle
        # drafts from varied prompts, refine each, score creativity, and
        # APPEND the top-K to active_base. The substrate ratchet turns
        # multiple clicks per cycle.
        samples = []   # list of (refined_seq, creativity)
        for s_idx in range(samples_per_cycle):
            # Diverse prompts: random 16-char windows from active_base.
            start = rng.randint(0, max(0, active_base.numel() - 17))
            prompt_s = active_base[start: start + 16].unsqueeze(0)
            with torch.no_grad():
                draft = autoregressive_generate(
                    model, prompt_s, n_new=growth_n_new,
                    vocab_size=vocab_size, temperature=0.8)
            refined_s, _ = staged_refine(
                model, prompt_s, n_new=growth_n_new, vocab_size=vocab_size,
                harmony_scorer=harmony_fn, quality_scorer=quality_fn,
                creativity_scorer=creativity_fn,
                n_iters_per_stage=30, resample_frac=0.35,
                prompt_len=16, temperature=0.5)
            samples.append((refined_s.squeeze(0).clone(),
                              creativity_fn(refined_s)))
        # Sort by creativity desc, keep top K.
        samples.sort(key=lambda x: x[1], reverse=True)
        kept = samples[:keep_top_k]
        kept_scores = [s[1] for s in kept]
        all_scores = [s[1] for s in samples]
        mean_score = sum(all_scores) / len(all_scores)
        print(f"  cycle {cycle+1}: generated {samples_per_cycle} samples, "
              f"mean creativity={mean_score:.4f}, "
              f"top-{keep_top_k}={[round(s, 4) for s in kept_scores]}")
        # Three filters: quality (> corpus baseline), anchor (seed >= 70%),
        # real_words (>= min fraction). Real-words is the strict gate that
        # stops "fan fan fan" gibberish from entering the corpus.
        real_word_min = 0.6
        n_growth = 0
        n_added_this_cycle = 0
        n_rej_rw_this_cycle = 0
        for ref_seq, cr in kept:
            if cr > best_creativity:
                best_creativity = cr
                best_refined_seq = ref_seq.clone()
            # Decode to check real-word fraction.
            ref_text = ''.join(itos_map.get(int(t), '?')
                                for t in ref_seq.tolist())
            rw = real_word_fraction(ref_text, corpus_text, min_word_len=3)
            new_size = active_base.numel() + ref_seq.numel()
            seed_frac_after = orig_seed_chars / new_size
            passes_q = cr > corpus_creativity_baseline
            passes_a = seed_frac_after >= seed_min_fraction
            passes_rw = rw >= real_word_min
            if passes_q and passes_a and passes_rw:
                active_base = torch.cat([active_base, ref_seq])
                n_growth += ref_seq.numel()
                n_added_this_cycle += 1
            else:
                if not passes_q:
                    n_rejected_below_baseline += 1
                if not passes_a:
                    n_rejected_anchor += 1
                if not passes_rw:
                    n_rej_rw_this_cycle += 1
        cycle_summary.append({
            "cycle": cycle + 1,
            "samples_creativity": all_scores,
            "kept_top_k": kept_scores,
            "n_added": n_added_this_cycle,
            "n_rejected_baseline": n_rejected_below_baseline,
            "n_rejected_anchor": n_rejected_anchor,
            "active_base_after": active_base.numel(),
        })
        print(f"  added {n_added_this_cycle}/{len(kept)} samples  "
              f"(rej_baseline={n_rejected_below_baseline}, "
              f"rej_anchor={n_rejected_anchor}, "
              f"rej_realword(this cycle)={n_rej_rw_this_cycle}) "
              f"active_base={active_base.numel()} chars "
              f"(best ever: {best_creativity:.4f})")

    # Final generation for inspection.
    final_gen = autoregressive_generate(model, prompt, n_new=n_new,
                                          vocab_size=vocab_size,
                                          temperature=0.8)
    final_refined, _ = staged_refine(
        model, prompt, n_new=n_new, vocab_size=vocab_size,
        harmony_scorer=harmony_fn, quality_scorer=quality_fn,
        creativity_scorer=creativity_fn,
        n_iters_per_stage=200, resample_frac=0.35,
        prompt_len=16, temperature=0.5)

    return {"name": name, "mode": "self_distillation",
             "n_params": n_params,
             "best_val": best_val, "best_step": best_step,
             "wall": time.time() - t0,
             "best_creativity_seen": best_creativity,
             "active_base_final_size": active_base.numel(),
             "cycle_summary": cycle_summary,
             "generated_tokens": final_gen[0].tolist(),
             "refined_tokens": final_refined[0].tolist()}


def train_mutable_substrate(name, train_seed, corpus_anchor, val_split,
                              vocab_size, args, fib_positions,
                              harmony_kind="char",
                              mutation_every: int = 200,
                              mutation_alpha: float = 0.9,
                              data_guided: bool = True,
                              itos_map: dict = None,
                              corpus_text: str = None):
    """Parametric substrate mutation with best-revert + data guidance.

    Constants (phi, pi_exp, fib_weights) are the ONLY mutable values.
    When data_guided=True, mutations are biased by the corpus
    signature: compute the gradient of |parametric_sig - corpus_sig|
    w.r.t. each constant via autograd, mutate in descent direction
    with added noise. The corpus tells the mutation where to push;
    val tells us whether to keep or revert.
    """
    import random as _rng_mod
    rng = _rng_mod.Random(args.seed + 7)
    torch.manual_seed(args.seed)
    gen = torch.Generator(); gen.manual_seed(args.seed + 1)
    model = FibRecLMSubsim(
        vocab_size=vocab_size, d_model=args.d_model, n_blocks=args.n_blocks,
        seq_len=args.seq_len, K=args.K_init, mode="cross", K_sig=args.K_sig,
        substrate_embed=True,
    )
    optimizer = FibonacciAdamW(model.parameters(), lr=args.lr)
    sched = lambda s, T: K_schedule_tier_walk(s, T, K_init=args.K_init,
                                                 K_min=args.K_min)
    n_params = sum(p.numel() for p in model.parameters())

    # Canonical substrate -- no mutation, no search. Constants are fixed:
    # phi=1.618, pi_exp=pi, F=Fibonacci. The corpus signature is the
    # harmony target (data's voice grounding the substrate).
    substrate = ParametricSubstrate()
    K_sig = len(_FIB_FREQS_LOCAL)
    history = [(substrate.clone(), float("inf"))]

    sig_char = corpus_char_signature(corpus_anchor, vocab_size)
    sig_ms = corpus_multiscale_signature(corpus_anchor, vocab_size,
                                            seq_len=args.seq_len)
    corpus_target = sig_ms if harmony_kind == "multiscale" else sig_char

    print(f"\n[canonical_substrate {name}]  harmony={harmony_kind}  "
          f"params={n_params:,}", flush=True)
    print(f"  canonical constants: {substrate.summary()}")
    print(f"  corpus sig_char: {[round(x, 4) for x in sig_char.tolist()]}")
    print(f"  corpus sig_ms:   {[round(x, 4) for x in sig_ms.tolist()]}")

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

            # Mutation disabled -- canonical substrate is final. The
            # translation work belongs at inference (refinement), not at
            # training (search for constants that the math already gave us).

    # Final generation: BOTH single-pass and iteratively-refined.
    # n_new sized to fit within model.seq_len so refinement covers the
    # whole draft (no out-of-window positions).
    prompt = train_seed[:16].unsqueeze(0)
    n_new = max(args.seq_len - 16, 32)
    final_gen = autoregressive_generate(model, prompt, n_new=n_new,
                                          vocab_size=vocab_size,
                                          temperature=0.8)
    # Iterative refinement: aggressive output->input loop.
    def harmony_scorer(seq_tokens):
        with torch.no_grad():
            T = seq_tokens.shape[1]
            ctx = seq_tokens if T <= model.seq_len else seq_tokens[:, -model.seq_len:]
            logits = model(ctx)
            K_h = K_to_K_harmony(cur_K or args.K_init,
                                  K_init=args.K_init, K_min=args.K_min)
            return compute_harmony_grounded(logits, vocab_size, harmony_kind,
                                              sig_char, sig_ms,
                                              K_harmony=K_h).item()

    def quality_scorer(seq_tokens):
        """Self-perplexity: how surprising is the sequence to the model."""
        with torch.no_grad():
            T = seq_tokens.shape[1]
            ctx = seq_tokens if T <= model.seq_len else seq_tokens[:, -model.seq_len:]
            logits = model(ctx)
            ce = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size),
                                   ctx[:, 1:].reshape(-1))
            return ce.item()

    creativity_fn = None
    if corpus_text is not None and itos_map is not None:
        def creativity_fn(seq_tokens):
            """Shakespeare-creativity: n-gram + vocab + structural match.
            Higher = more Shakespeare-LIKE without exact replication."""
            text = ''.join(itos_map.get(int(t), '?')
                            for t in seq_tokens[0].tolist())
            return compute_creativity_score(text, corpus_text)["creativity_score"]

    refined_gen, stages_out = staged_refine(
        model, prompt, n_new=n_new, vocab_size=vocab_size,
        harmony_scorer=harmony_scorer,
        quality_scorer=quality_scorer,
        creativity_scorer=creativity_fn,
        n_iters_per_stage=200, resample_frac=0.35, prompt_len=16,
        temperature=0.5)
    print(f"  staged refinement (max 200 per stage, patience=5):")
    for k, v in stages_out.items():
        h = v.get("harmony"); q = v.get("quality"); c = v.get("creativity")
        h_str = f"{h:.4f}" if h is not None else "n/a"
        q_str = f"{q:.4f}" if q is not None else "n/a"
        c_str = f"{c:.4f}" if c is not None else "n/a"
        traj = v.get("trajectory")
        iters_str = f"  (ran {len(traj)-1} iters)" if traj else ""
        print(f"    [{k:<18}]  harmony={h_str}  quality={q_str}  "
              f"creativity={c_str}{iters_str}")
    refine_history = stages_out
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
             "generated_tokens": final_gen[0].tolist(),
             "refined_tokens": refined_gen[0].tolist(),
             "refinement_stages": {
                 k: {"harmony": v.get("harmony"),
                      "quality": v.get("quality"),
                      "creativity": v.get("creativity"),
                      "tokens": v["seq"][0].tolist()}
                 for k, v in stages_out.items()
             }}


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
        substrate_embed=True,
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
        substrate_embed=True,
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

    # Build itos and full corpus text for creativity scoring.
    itos_map = {i: c for i, c in enumerate(chars)}
    full_corpus_text = ''.join(itos_map.get(int(t), '?')
                                  for t in encoded.tolist())

    arms = [
        ("self_distill_multiscale",  "multiscale"),
    ]
    results = {}
    for name, harmony_kind in arms:
        results[name] = train_with_self_distillation(
            name, train_seed, corpus_anchor, val_split, vocab_size, args,
            fib_positions, harmony_kind=harmony_kind,
            itos_map=itos_map, corpus_text=full_corpus_text,
            n_cycles=6, distill_prob=0.3,
            samples_per_cycle=8, keep_top_k=4, growth_n_new=128)

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

    # Print decoded generation samples per arm: single-pass vs refined.
    itos_map = {i: c for i, c in enumerate(chars)}
    def decode(toks):
        return ''.join(itos_map.get(int(t), '?') for t in toks)
    print()
    print("=" * 92)
    print("Generated samples (prompt = first 16 chars of seed, temp=0.8)")
    print("Comparing single-pass vs iterative-refinement (refined = output→input loop)")
    print('-' * 92)
    for name, r in results.items():
        sp = decode(r["generated_tokens"])
        rf = decode(r["refined_tokens"])
        sp_cr = compute_creativity_score(sp, full_corpus_text)
        rf_cr = compute_creativity_score(rf, full_corpus_text)
        print(f"\n[{name}]")
        stages = r.get("refinement_stages", {})
        if stages:
            print(f"  Staircase progression (each stage targets next score):")
            for stage_name, stage in stages.items():
                print(f"    {stage_name:<18}  "
                      f"h={stage['harmony']:.4f}  "
                      f"q={stage['quality']:.4f}  "
                      f"c={stage['creativity']:.4f}")
        print(f"  single-pass [c={sp_cr['creativity_score']:.3f}, "
              f"n3={sp_cr['ngram_3']:.3f}, vocab={sp_cr['vocab_overlap']:.3f}]:")
        print(f"    {repr(sp[:160])}")
        print(f"  refined    [c={rf_cr['creativity_score']:.3f}, "
              f"n3={rf_cr['ngram_3']:.3f}, vocab={rf_cr['vocab_overlap']:.3f}]:")
        print(f"    {repr(rf[:160])}")
        # Print each stage's output for inspection.
        for stage_name, stage in stages.items():
            stage_text = decode(stage["tokens"])
            print(f"  [{stage_name}] {repr(stage_text[:160])}")

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

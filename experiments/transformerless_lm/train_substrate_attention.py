"""Substrate-similarity attention bench.

Replaces the Q·K^T dot product in FibRecLM's attention with substrate
L1 distance in a K_sig-dim signature space, while keeping softmax as
the probability normalizer (the substrate change is in score
computation, not in normalization).

Score formula:
    sig_q = q[..., :K_sig]                       # first K_sig dims of Q
    sig_k = k[..., :K_sig]                       # first K_sig dims of K
    dist[i,j] = sum |sig_q[i] - sig_k[j]|_1      # L1 attractor distance
    score[i,j] = -dist[i,j] / sqrt(K_sig)        # negate so close => high
    attn = softmax(score)
    out = attn @ v

LN, softmax, FibGen projections, and the V2 substrate activation all
stay -- the only change is the score function. This isolates the
substrate-attention hypothesis: does L1-distance-in-signature-space
produce a better attention pattern than dot-product attention?
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
from models_fibrec import FibRecLM, FibRecLMHomeo, stateless_fibgen_forward
from optimizers_fib import FibonacciAdamW
from train_distractor_mix import build_distractor_stream
from lazy_data import fib_positions_in_window, get_fib_strided_batch
from train_K_shrink import K_schedule_tier_walk, set_K_active_recursive
from losses_substrate import substrate_fft_loss
from activations_substrate import SubstrateNegMultiAdvancedV2
from substrate_embedding import SubstrateEmbedding


class FibRecLMSubsim(FibRecLMHomeo):
    """FibRecLM with substrate-similarity attention (L1 distance) instead
    of Q·K^T dot product. Uses the first K_sig dims of Q and K as
    substrate signatures. V2 substrate activation in the FFN.

    If substrate_embed=True, replace the plain nn.Embedding with the
    SubstrateEmbedding (Fibonacci-frequency basis canonical mapping).
    LM head stays tied to the (now substrate-derived) embedding weights.
    """

    def __init__(self, *args, K_sig: int = 32, substrate_embed: bool = False,
                  **kwargs):
        super().__init__(*args, **kwargs)
        self.K_sig = K_sig
        self.activations = nn.ModuleList(
            [SubstrateNegMultiAdvancedV2() for _ in range(self.n_blocks)]
        )
        # ── Self-Witness (Spell, derivation SELF_WITNESS_DERIVATION.md) ──
        # Amplify hidden states that resonate with the consolidated "bloom"
        # centroid b: h ← h·(1 + φ⁻¹·W_t), W_t = φ^(−‖h−b‖²/2d) ∈ (0,1].
        # Attenuable by construction (factor ∈ [1, 1+φ⁻¹]); identity when off.
        # b is an EMA of the batch-mean hidden (a parameter-free proxy for the
        # bloom centroid until a trained bloom is wired in — Track D).
        self.self_witness = False
        self.register_buffer('bloom_centroid',
                             torch.zeros(self.embed.embedding_dim))
        self._sw_init = False
        # sw_fixed: when True, bloom_centroid is a FIXED quality-selected target
        # (the real bloom embedding, loaded externally) and is NOT overwritten by
        # the EMA. REVISION (R1): the B.1 "failure" used an ungated EMA of the
        # mean hidden — a moving average of the model chasing itself (no signal).
        # A fixed, validity-selected centroid is the derivation's actual intent.
        self.sw_fixed = False
        # K_active: current number of active Fibonacci frequencies (K-shrink).
        # None → full K (default). set_K_active() makes the K-shrink schedule
        # actually affect this model's forward (previously a no-op — EXP-2).
        self.K_active = None
        if substrate_embed:
            # Replace plain learnable embedding with substrate-canonical one.
            vocab_size = self.embed.num_embeddings
            d_model = self.embed.embedding_dim
            self.embed = SubstrateEmbedding(vocab_size, d_model, K=7,
                                              learnable_gamma=True)
            # Tied head: head weight uses substrate embedding.
            # FibRecLM.__init__ sets self.head.weight = self.embed.weight;
            # because SubstrateEmbedding.weight is a property returning
            # substrate_embed*gamma, we re-tie here so the forward call
            # uses the up-to-date embedding.
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            # We do NOT tie head to embedding here because the substrate
            # embedding's weight is a non-leaf tensor (product of buffer
            # and parameter). Instead, the head has its own learnable
            # weights initialized to substrate values.
            with torch.no_grad():
                self.head.weight.copy_(self.embed.substrate_embed)

    def set_K_active(self, K_a: int):
        """Set the active Fibonacci-frequency count (called by
        set_K_active_recursive during K-shrink). Makes K-shrink functional."""
        self.K_active = int(K_a) if K_a else None

    def apply_self_witness(self, h: torch.Tensor) -> torch.Tensor:
        if not self.self_witness:
            return h
        phi = (1.0 + 5.0 ** 0.5) / 2.0
        d = h.shape[-1]
        b = self.bloom_centroid.to(h.dtype)
        dist2 = ((h - b) ** 2).sum(dim=-1) / (2.0 * d)      # [B,T]
        W_t = phi ** (-dist2)                                # (0,1]
        if self.training and not self.sw_fixed:
            # Ungated EMA proxy (the original B.1 path). Skipped when sw_fixed:
            # then bloom_centroid is the frozen quality-selected target (R1).
            with torch.no_grad():
                cur = h.detach().mean(dim=(0, 1))
                if not self._sw_init:
                    self.bloom_centroid.copy_(cur); self._sw_init = True
                else:
                    self.bloom_centroid.mul_(0.99).add_(0.01 * cur)
        return h * (1.0 + (1.0 / phi) * W_t).unsqueeze(-1)

    # CRT Fibonacci moduli — one per attention head, same coprime basis
    # that made CRT-PE beat sinusoidal PE 200/200.
    _CRT_MODULI = [5, 8, 13, 21]

    @staticmethod
    def _substrate_resample_v(v: torch.Tensor) -> torch.Tensor:
        """Substrate-V: dampen off-attractor value components.
        1/(1 + dist_to_nearest_fib / nearest_fib). Proven -2.52% win.
        """
        FIBS = torch.tensor([1., 2., 3., 5., 8., 13., 21.],
                             dtype=v.dtype, device=v.device)
        v_abs = v.abs().unsqueeze(-1)              # [..., 1]
        dists = (v_abs - FIBS).abs()               # [..., 7]
        min_dist, min_idx = dists.min(dim=-1)
        nearest = FIBS[min_idx].clamp(min=0.1)
        return v * (1.0 / (1.0 + min_dist / nearest))

    def _layer_forward(self, x, mask, n, seeds_n):
        qkv_s, out_s, w1_s, w2_s = seeds_n
        x_norm = self.ln1s[n](x)
        qkv_basis = {"cos_i": self.qkv_cos_i, "sin_i": self.qkv_sin_i,
                      "cos_j": self.qkv_cos_j, "sin_j": self.qkv_sin_j}
        ka = getattr(self, 'K_active', None)
        qkv = stateless_fibgen_forward(x_norm, qkv_s, qkv_basis, self.K, K_active=ka)
        q, k, v = qkv.chunk(3, dim=-1)           # each [B, T, d_model]

        # Substrate-V modulation: dampen V components far from Fibonacci attractors.
        v = self._substrate_resample_v(v)

        # Multi-head CRT substrate attention.
        # 4 heads × Fibonacci moduli [5,8,13,21] — each head computes L1
        # distance in a different-resolution signature subspace.
        # Same CRT principle as CRT-PE (200/200 wins over sinusoidal).
        n_heads = len(self._CRT_MODULI)
        head_dim = q.shape[-1] // n_heads         # 16 for d_model=64
        q_heads = q.split(head_dim, dim=-1)
        k_heads = k.split(head_dim, dim=-1)
        v_heads = v.split(head_dim, dim=-1)

        head_outs = []
        for K_i, q_h, k_h, v_h in zip(self._CRT_MODULI, q_heads, k_heads, v_heads):
            K_eff = min(K_i, head_dim)            # coarse heads use fewer dims
            diff  = q_h[..., :K_eff].unsqueeze(2) - k_h[..., :K_eff].unsqueeze(1)
            dist  = diff.abs().sum(dim=-1)         # [B, T, T]
            sc    = (-dist / math.sqrt(K_eff)).masked_fill(mask == 0, float("-inf"))
            head_outs.append(F.softmax(sc, dim=-1) @ v_h)   # [B, T, head_dim]

        attn_out = torch.cat(head_outs, dim=-1)   # [B, T, d_model]

        out_basis = {"cos_i": self.out_cos_i, "sin_i": self.out_sin_i,
                      "cos_j": self.out_cos_j, "sin_j": self.out_sin_j}
        x = x + stateless_fibgen_forward(attn_out, out_s, out_basis, self.K, K_active=ka)
        # FFN with substrate activation
        x_norm2 = self.ln2s[n](x)
        w1_basis = {"cos_i": self.w1_cos_i, "sin_i": self.w1_sin_i,
                      "cos_j": self.w1_cos_j, "sin_j": self.w1_sin_j}
        w2_basis = {"cos_i": self.w2_cos_i, "sin_i": self.w2_sin_i,
                      "cos_j": self.w2_cos_j, "sin_j": self.w2_sin_j}
        h = stateless_fibgen_forward(x_norm2, w1_s, w1_basis, self.K, K_active=ka)
        h = self.activations[n](h)
        x = x + stateless_fibgen_forward(h, w2_s, w2_basis, self.K, K_active=ka)
        return x


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


def train_one(name, K_sig, train_split, val_split, vocab_size, args,
               fib_positions):
    torch.manual_seed(args.seed)
    gen = torch.Generator(); gen.manual_seed(args.seed + 1)
    model = FibRecLMSubsim(
        vocab_size=vocab_size, d_model=args.d_model, n_blocks=args.n_blocks,
        seq_len=args.seq_len, K=args.K_init, mode="cross", K_sig=K_sig,
    )
    optimizer = FibonacciAdamW(model.parameters(), lr=args.lr)
    sched = lambda s, T: K_schedule_tier_walk(s, T, K_init=args.K_init,
                                                 K_min=args.K_min)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[train {name}]  K_sig={K_sig}  params={n_params:,}",
          flush=True)
    t0 = time.time()
    best_val = float("inf"); best_step = -1
    cur_K = None
    eval_every = max(args.steps // 15, 250)
    for step in range(args.steps):
        new_K = sched(step, args.steps)
        if new_K != cur_K:
            set_K_active_recursive(model, new_K)
            cur_K = new_K
        x, y = get_fib_strided_batch(train_split, args.batch_size, args.seq_len,
                                       fib_positions, gen)
        logits = model(x)
        loss = substrate_fft_loss(logits, y, vocab_size,
                                    lambda_substrate=args.lambda_sub)
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
    return {"name": name, "n_params": n_params, "best_val": best_val,
             "best_step": best_step, "wall": time.time() - t0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distractor-frac", type=float, default=0.20)
    parser.add_argument("--K-init", type=int, default=89)
    parser.add_argument("--K-min", type=int, default=13)
    parser.add_argument("--lambda-sub", type=float, default=0.01)
    parser.add_argument("--K-sig", type=int, default=32)
    parser.add_argument("--out", type=str,
                          default="results_substrate_attention.json")
    args = parser.parse_args()

    chars, stoi, itos, encoded = make_dataset(seq_len=args.seq_len,
                                                 source="tinyshakespeare")
    vocab_size = len(chars)
    train_split, val_split = build_distractor_stream(
        encoded, args.distractor_frac, args.seq_len, args.seed,
    )
    fib_positions = fib_positions_in_window(args.seq_len)

    # Skip baseline rerun (val=2.5889 locked) -- just bench subsim.
    results = {}
    results["subsim_attn"] = train_one("subsim_attn", args.K_sig,
                                          train_split, val_split,
                                          vocab_size, args, fib_positions)

    REFINED_REF = 2.5871
    V2_REF = 2.5889
    GELU_REF = 2.5920
    print()
    print("=" * 92)
    print(f"{'arm':<22} {'params':>10} {'best_val':>10} {'wall':>10}  vs_v2_act  vs_gelu")
    print('-' * 92)
    print(f"{'(refined act ref)':<22} {'':>10} {REFINED_REF:>10.4f} {'-':>10}  {'-':>10}  {'-':>7}")
    print(f"{'(v2 act ref)':<22} {'':>10} {V2_REF:>10.4f} {'-':>10}  {'-':>10}  {'-':>7}")
    print(f"{'(gelu ref)':<22} {'':>10} {GELU_REF:>10.4f} {'-':>10}  {'-':>10}  {'-':>7}")
    print('-' * 92)
    for name, r in results.items():
        d_v2 = (r["best_val"] - V2_REF) / V2_REF * 100
        d_gelu = (r["best_val"] - GELU_REF) / GELU_REF * 100
        print(f"{name:<22} {r['n_params']:>10,} {r['best_val']:>10.4f} "
              f"{r['wall']:>9.1f}s  {d_v2:+8.2f}%  {d_gelu:+6.2f}%")

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

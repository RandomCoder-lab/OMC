"""D-grow + K-shrink combined schedule on TinyShakespeare.

Theory (The Architect's):
  D-grow:   start at d=d_min (small width, fast convergence on coarse structure),
            expand to d=d_max via Fibonacci tier walk.
            → Each tier explores a wider feature space before zooming out.
  K-shrink: start at K=K_init (many substrate frequencies, high expressiveness),
            compress to K=K_min via Fibonacci tier walk.
            → Distills to the most essential substrate components as width grows.

Both schedules walk Fibonacci tiers simultaneously, driven by a single
step counter. The hypothesis: the model first learns coarse structure
(small d, large K) then refines lexical detail (large d, small K),
mirroring how the substrate hierarchy maps to abstraction levels.

Key architectural fact:
  FibRecLM seeds are K²×4 — independent of D.
  Growing D just uses a larger slice of the pre-computed Fibonacci basis.
  No weight transfer, no padding of stored parameters.

Bench arms:
  dense_baseline     : standard transformer at d=128, 4 blocks (fast reference)
  fibgen_static      : FibRecLM at d=d_max, K=K_init throughout
  d_grow_k_shrink    : combined Fibonacci-tier D-grow + K-shrink

All arms on TinyShakespeare, same step budget, lazy Fibonacci-strided data.
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
from corpus import make_dataset, get_batch
from models import make_model
from models_fibgen import FIBONACCI
from models_fibrec import make_fib_basis
from lazy_data import fib_positions_in_window, get_fib_strided_batch

PHI = (1 + math.sqrt(5)) / 2


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

def D_schedule_tier_walk(step: int, max_steps: int,
                          d_min: int, d_max: int) -> int:
    """Grow D from d_min to d_max, spending equal steps at each Fibonacci tier.

    Tiers are all Fibonacci values in [d_min, d_max], plus d_max itself
    if it is not already Fibonacci.
    """
    tiers = sorted({f for f in FIBONACCI if d_min <= f <= d_max})
    if d_max not in tiers:
        tiers.append(d_max)
    steps_per_tier = max_steps // len(tiers)
    tier_idx = min(step // max(steps_per_tier, 1), len(tiers) - 1)
    return tiers[tier_idx]


def K_schedule_tier_walk(step: int, max_steps: int,
                          K_init: int, K_min: int) -> int:
    """Shrink K from K_init to K_min, spending equal steps at each Fibonacci tier."""
    tiers = sorted({f for f in FIBONACCI if K_min <= f <= K_init}, reverse=True)
    steps_per_tier = max_steps // len(tiers)
    tier_idx = min(step // max(steps_per_tier, 1), len(tiers) - 1)
    return tiers[tier_idx]


# ---------------------------------------------------------------------------
# Model: FibRecLM with runtime D-grow + K-shrink
# ---------------------------------------------------------------------------

class FibRecLM_DGrow(nn.Module):
    """FibRecLM with dynamic active width (d_active) and active depth (K_active).

    Seeds are K_init²×4 throughout. Basis tables are pre-computed for
    full (d_max, K_init). At runtime, the forward uses:
      - first d_active cols of embedding / positional encoding
      - first d_active / out_active rows of each basis matrix
      - first K_active cols of each basis matrix
      - first K_active × K_active block of each seed (reshaped)

    Recurrence matrices A, B remain full K_init×K_init at all times
    (the recurrence runs in seed-space, not feature-space).
    """

    def __init__(self, vocab_size: int, d_max: int, K_init: int,
                 n_blocks: int, seq_len: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_max = d_max
        self.K_init = K_init
        self.n_blocks = n_blocks
        self.seq_len = seq_len
        self.d_active = d_max
        self.K_active = K_init

        # Embedding + CRT-Fibonacci PE (full d_max; slice at runtime)
        self.embed = nn.Embedding(vocab_size, d_max)
        self.register_buffer("pe", self._crt_pe(seq_len, d_max))
        self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len)))

        # Pre-compute basis tables for all layer shapes at full (d_max, K_init)
        for lname, in_dim, out_dim in [
            ("qkv", d_max, 3 * d_max),
            ("out", d_max,     d_max),
            ("w1",  d_max, 4 * d_max),
            ("w2", 4 * d_max,  d_max),
        ]:
            b = make_fib_basis(in_dim, out_dim, K_init)
            self.register_buffer(f"{lname}_cos_i", b["cos_i"])  # [out_dim, K_init]
            self.register_buffer(f"{lname}_sin_i", b["sin_i"])
            self.register_buffer(f"{lname}_cos_j", b["cos_j"])  # [in_dim,  K_init]
            self.register_buffer(f"{lname}_sin_j", b["sin_j"])

        # Seeds: K_init² × 4, two base seeds per layer (blocks 0 and 1)
        n_comp = K_init * K_init
        init_scale = 0.1 / math.sqrt(n_comp)
        for nm in ("qkv", "out", "w1", "w2"):
            for n in (0, 1):
                setattr(self, f"{nm}_seed_{n}",
                        nn.Parameter(torch.randn(n_comp, 4) * init_scale))

        # Recurrence matrices A (near identity), B (near zero)
        for nm in ("qkv", "out", "w1", "w2"):
            setattr(self, f"A_{nm}", nn.Parameter(
                torch.eye(K_init) + 0.01 * torch.randn(K_init, K_init)))
            setattr(self, f"B_{nm}", nn.Parameter(
                0.01 * torch.randn(K_init, K_init)))

        # LayerNorm affine params at d_max; sliced to d_active in forward
        for i in range(n_blocks):
            self.register_parameter(f"ln1_w_{i}", nn.Parameter(torch.ones(d_max)))
            self.register_parameter(f"ln1_b_{i}", nn.Parameter(torch.zeros(d_max)))
            self.register_parameter(f"ln2_w_{i}", nn.Parameter(torch.ones(d_max)))
            self.register_parameter(f"ln2_b_{i}", nn.Parameter(torch.zeros(d_max)))
        self.register_parameter("ln_f_w", nn.Parameter(torch.ones(d_max)))
        self.register_parameter("ln_f_b", nn.Parameter(torch.zeros(d_max)))

    # ------------------------------------------------------------------
    # Schedule setters
    # ------------------------------------------------------------------

    def set_d_active(self, d_a: int):
        self.d_active = max(1, min(d_a, self.d_max))

    def set_K_active(self, K_a: int):
        self.K_active = max(1, min(K_a, self.K_init))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _crt_pe(seq_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(seq_len, d_model)
        pos = torch.arange(seq_len, dtype=torch.float)
        moduli = [5, 8, 13, 21, 34, 55, 89, 144]
        n_pairs = d_model // 2
        for i in range(n_pairs):
            m = moduli[i % len(moduli)]
            angle = 2 * math.pi * (pos % m) / m
            pe[:, 2 * i]     = torch.sin(angle)
            pe[:, 2 * i + 1] = torch.cos(angle)
        return pe

    def _ln(self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        return F.layer_norm(x, [d], w[:d], b[:d])

    def _fib_forward(self, x: torch.Tensor, seed: torch.Tensor,
                     lname: str, in_active: int, out_active: int) -> torch.Tensor:
        """Substrate-FibGen forward using only [in_active, out_active, K_active] slice."""
        K = self.K_init
        Ka = self.K_active
        cos_j = getattr(self, f"{lname}_cos_j")[:in_active,  :Ka]  # [in_a,   Ka]
        sin_j = getattr(self, f"{lname}_sin_j")[:in_active,  :Ka]
        cos_i = getattr(self, f"{lname}_cos_i")[:out_active, :Ka]  # [out_a,  Ka]
        sin_i = getattr(self, f"{lname}_sin_i")[:out_active, :Ka]
        sc = seed.view(K, K, 4)[:Ka, :Ka]                           # [Ka, Ka, 4]
        a, b, c, d_ = sc[..., 0], sc[..., 1], sc[..., 2], sc[..., 3]
        x_cos = x @ cos_j                                            # [B, T, Ka]
        x_sin = x @ sin_j
        y_cos = x_cos @ a.t() + x_sin @ c.t()                       # [B, T, Ka]
        y_sin = x_cos @ b.t() + x_sin @ d_.t()
        return y_cos @ cos_i.t() + y_sin @ sin_i.t()                # [B, T, out_a]

    def _rec_step(self, nm: str, s_p2: torch.Tensor,
                  s_p1: torch.Tensor) -> torch.Tensor:
        K = self.K_init
        A = getattr(self, f"A_{nm}")
        B = getattr(self, f"B_{nm}")
        sp1 = s_p1.view(K, K, 4)
        sp2 = s_p2.view(K, K, 4)
        s_n = (torch.einsum("ik,kjc->ijc", A, sp1) +
               torch.einsum("ik,kjc->ijc", B, sp2))
        return s_n.reshape(K * K, 4)

    def _all_seeds(self):
        base = {nm: (getattr(self, f"{nm}_seed_0"), getattr(self, f"{nm}_seed_1"))
                for nm in ("qkv", "out", "w1", "w2")}
        running = dict(base)
        seeds = []
        for n in range(self.n_blocks):
            if n == 0:
                seeds.append(tuple(base[nm][0] for nm in ("qkv", "out", "w1", "w2")))
            elif n == 1:
                seeds.append(tuple(base[nm][1] for nm in ("qkv", "out", "w1", "w2")))
            else:
                new = {}
                for nm in ("qkv", "out", "w1", "w2"):
                    s_p2, s_p1 = running[nm]
                    new[nm] = (s_p1, self._rec_step(nm, s_p2, s_p1))
                running = new
                seeds.append(tuple(running[nm][1] for nm in ("qkv", "out", "w1", "w2")))
        return seeds

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, T = token_ids.shape
        d = self.d_active

        h = self.embed(token_ids)[:, :, :d] + self.pe[:T, :d]
        mask = self.mask[:T, :T]

        for n, (qkv_s, out_s, w1_s, w2_s) in enumerate(self._all_seeds()):
            # Self-attention
            x_n = self._ln(h, getattr(self, f"ln1_w_{n}"), getattr(self, f"ln1_b_{n}"))
            qkv = self._fib_forward(x_n, qkv_s, "qkv", d, 3 * d)
            q, k, v = qkv.chunk(3, dim=-1)
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
            scores = scores.masked_fill(mask == 0, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            h = h + self._fib_forward(attn @ v, out_s, "out", d, d)

            # FFN
            x_n2 = self._ln(h, getattr(self, f"ln2_w_{n}"), getattr(self, f"ln2_b_{n}"))
            h = h + self._fib_forward(
                F.gelu(self._fib_forward(x_n2, w1_s, "w1", d, 4 * d)),
                w2_s, "w2", 4 * d, d)

        h = F.layer_norm(h, [d], self.ln_f_w[:d], self.ln_f_b[:d])
        return h @ self.embed.weight[:, :d].t()  # tied head

    def storage_summary(self) -> dict:
        stored = sum(p.numel() for p in self.parameters())
        d = self.d_max
        dense_eq = self.n_blocks * (3*d*d + d*d + 4*d*d + 4*d*d + 4*d) + self.vocab_size * d
        return {"stored": stored, "dense_equivalent": dense_eq,
                "compression": dense_eq / max(stored, 1),
                "d_active": self.d_active, "K_active": self.K_active}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def evaluate(model, val_data, batch_size, seq_len, fib_pos, gen, n_batches=16):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = get_fib_strided_batch(val_data, batch_size, seq_len, fib_pos, gen)
            logits = model(x)
            losses.append(F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item())
    model.train()
    return sum(losses) / len(losses)


def train(name: str, model: nn.Module, optimizer, train_data, val_data, args,
          fib_pos, D_fn=None, K_fn=None):
    """
    D_fn(step, max_steps) -> d_active  (or None for static)
    K_fn(step, max_steps) -> K_active  (or None for static)
    """
    torch.manual_seed(args.seed)
    gen = torch.Generator(); gen.manual_seed(args.seed + 1)

    stored = sum(p.numel() for p in model.parameters())
    print(f"\n[{name}]  stored_params={stored:,}", flush=True)
    if hasattr(model, "storage_summary"):
        s = model.storage_summary()
        print(f"  compression={s['compression']:.1f}×  "
              f"d_active={s['d_active']}  K_active={s['K_active']}", flush=True)

    t0 = time.time()
    best_val = float("inf"); best_step = -1
    val_hist = []
    D_hist = []; K_hist = []
    eval_every = max(args.steps // 20, 250)
    cur_d = cur_k = None

    for step in range(args.steps):
        # Apply schedules
        if D_fn is not None:
            new_d = D_fn(step, args.steps)
            if new_d != cur_d:
                model.set_d_active(new_d)
                cur_d = new_d
                D_hist.append((step, new_d))
        if K_fn is not None:
            new_k = K_fn(step, args.steps)
            if new_k != cur_k:
                model.set_K_active(new_k)
                cur_k = new_k
                K_hist.append((step, new_k))

        x, y = get_fib_strided_batch(train_data, args.batch_size, args.seq_len,
                                      fib_pos, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        if step % eval_every == 0 or step == args.steps - 1:
            vl = evaluate(model, val_data, args.batch_size, args.seq_len,
                          fib_pos, gen)
            val_hist.append((step, vl))
            marker = ""
            if vl < best_val:
                best_val = vl; best_step = step; marker = " ← BEST"
            d_tag = f" d={cur_d}" if cur_d else ""
            k_tag = f" K={cur_k}" if cur_k else ""
            print(f"  step {step:5d}  val={vl:.4f}{d_tag}{k_tag}"
                  f"  ({time.time()-t0:.1f}s){marker}", flush=True)

    final_val = evaluate(model, val_data, args.batch_size, args.seq_len,
                          fib_pos, gen, n_batches=32)
    return {
        "name": name, "stored_params": stored,
        "best_val": best_val, "best_step": best_step,
        "final_val": final_val, "wall": time.time() - t0,
        "D_history": D_hist, "K_history": K_hist,
        "val_hist": val_hist,
    }


# ---------------------------------------------------------------------------
# Text sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample(model, prompt_ids: torch.Tensor, n_new: int, seq_len: int,
           temperature: float = 0.8, top_k: int = 10) -> str:
    model.eval()
    out = prompt_ids.clone()
    for _ in range(n_new):
        ctx = out[:, -seq_len:]
        logits = model(ctx)[:, -1, :] / max(temperature, 1e-6)
        if top_k:
            v, _ = logits.topk(top_k)
            logits[logits < v[..., -1:]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        out = torch.cat([out, nxt], dim=-1)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",     type=int,   default=30000)
    parser.add_argument("--batch-size",type=int,   default=32)
    parser.add_argument("--seq-len",   type=int,   default=128)
    parser.add_argument("--n-blocks",  type=int,   default=6)
    parser.add_argument("--d-max",     type=int,   default=384)
    parser.add_argument("--d-min",     type=int,   default=34,
                        help="Starting D tier for D-grow schedule")
    parser.add_argument("--K-init",    type=int,   default=144,
                        help="Starting K for K-shrink schedule")
    parser.add_argument("--K-min",     type=int,   default=3,
                        help="Ending K for K-shrink schedule")
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--prompt",    type=str,   default="ROMEO:\nWhat light through")
    parser.add_argument("--n-new",     type=int,   default=500)
    parser.add_argument("--temperature",type=float,default=0.8)
    parser.add_argument("--top-k",     type=int,   default=10)
    parser.add_argument("--no-dense",  action="store_true",
                        help="Skip dense baseline (saves time)")
    parser.add_argument("--out",       type=str,   default="results_d_shrink.json")
    args = parser.parse_args()

    chars, stoi, itos, encoded = make_dataset(seq_len=args.seq_len,
                                               source="tinyshakespeare")
    vocab_size = len(chars)
    n = encoded.numel()
    split = int(n * 0.9)
    train_data = encoded[:split]
    val_data   = encoded[split:]
    fib_pos = fib_positions_in_window(args.seq_len)

    prompt_ids = torch.tensor(
        [[stoi.get(c, stoi.get(" ", 0)) for c in args.prompt]], dtype=torch.long)

    print(f"TinyShakespeare  vocab={vocab_size}  train={len(train_data):,}  "
          f"val={len(val_data):,}", flush=True)
    print(f"D-grow:   {args.d_min} → {args.d_max}  (Fibonacci tier walk)")
    print(f"K-shrink: {args.K_init} → {args.K_min}  (Fibonacci tier walk)")
    print(f"Steps: {args.steps}  batch={args.batch_size}  seq={args.seq_len}\n")

    # Preview schedules
    print("D-grow schedule preview:")
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        s = int(args.steps * frac)
        d = D_schedule_tier_walk(s, args.steps, args.d_min, args.d_max)
        k = K_schedule_tier_walk(s, args.steps, args.K_init, args.K_min)
        print(f"  step {s:>6} ({frac*100:3.0f}%): d={d:<5} K={k}")

    results = {}
    samples = {}

    # ------------------------------------------------------------------
    # Arm 1: Dense baseline at d=128, 4 blocks (fast reference)
    # ------------------------------------------------------------------
    if not args.no_dense:
        print("\n" + "="*60)
        print("DENSE baseline  d=128  4-block")
        print("="*60)
        m = make_model("crt_only", vocab_size=vocab_size, seq_len=args.seq_len,
                        d_model=128, n_blocks=4)
        opt = torch.optim.AdamW(m.parameters(), lr=args.lr)
        results["dense_baseline"] = train(
            "dense_baseline", m, opt, train_data, val_data, args, fib_pos)
        out_ids = sample(m, prompt_ids, args.n_new, args.seq_len,
                          args.temperature, args.top_k)
        samples["dense_baseline"] = "".join(itos[int(i)] for i in out_ids[0].tolist())

    # ------------------------------------------------------------------
    # Arm 2: FibRecLM static at d=d_max, K=K_init throughout
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"FibRecLM static  d={args.d_max}  K={args.K_init}")
    print("="*60)
    m_static = FibRecLM_DGrow(vocab_size=vocab_size, d_max=args.d_max,
                               K_init=args.K_init, n_blocks=args.n_blocks,
                               seq_len=args.seq_len)
    opt_static = torch.optim.AdamW(m_static.parameters(), lr=args.lr)
    results["fibgen_static"] = train(
        "fibgen_static", m_static, opt_static, train_data, val_data, args, fib_pos)
    out_ids = sample(m_static, prompt_ids, args.n_new, args.seq_len,
                      args.temperature, args.top_k)
    samples["fibgen_static"] = "".join(itos[int(i)] for i in out_ids[0].tolist())

    # ------------------------------------------------------------------
    # Arm 3: D-grow + K-shrink combined schedule
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"D-grow + K-shrink  d: {args.d_min}→{args.d_max}  K: {args.K_init}→{args.K_min}")
    print("="*60)
    m_sched = FibRecLM_DGrow(vocab_size=vocab_size, d_max=args.d_max,
                              K_init=args.K_init, n_blocks=args.n_blocks,
                              seq_len=args.seq_len)
    # Cold-start: zero out embedding dims beyond d_min so that new
    # dimensions activate from zero (not random noise) when D grows.
    # Prevents early random dims from destabilizing learned d_min features.
    with torch.no_grad():
        m_sched.embed.weight[:, args.d_min:].zero_()
    # Start at d_min so the schedule kicks in from step 0
    m_sched.set_d_active(args.d_min)
    m_sched.set_K_active(args.K_init)
    opt_sched = torch.optim.AdamW(m_sched.parameters(), lr=args.lr)
    D_fn = lambda s, T: D_schedule_tier_walk(s, T, args.d_min, args.d_max)
    K_fn = lambda s, T: K_schedule_tier_walk(s, T, args.K_init, args.K_min)
    results["d_grow_k_shrink"] = train(
        "d_grow_k_shrink", m_sched, opt_sched, train_data, val_data, args, fib_pos,
        D_fn=D_fn, K_fn=K_fn)
    # Sample at final (d_max, K_min)
    m_sched.set_d_active(args.d_max)
    m_sched.set_K_active(args.K_min)
    out_ids = sample(m_sched, prompt_ids, args.n_new, args.seq_len,
                      args.temperature, args.top_k)
    samples["d_grow_k_shrink"] = "".join(itos[int(i)] for i in out_ids[0].tolist())

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    UNIFORM_FLOOR = math.log(vocab_size)
    print()
    print("="*80)
    print(f"{'name':<22} {'stored':>10} {'best_val':>10} {'vs uniform':>12} {'wall':>8}")
    print("-"*80)
    for nm, r in results.items():
        gap = (r["best_val"] - UNIFORM_FLOOR) / UNIFORM_FLOOR * 100
        print(f"{nm:<22} {r['stored_params']:>10,} {r['best_val']:>10.4f} "
              f"{gap:>+11.1f}%  {r['wall']:>7.1f}s")

    # Text samples
    sample_path = Path(__file__).parent / "samples_d_shrink.txt"
    with open(sample_path, "w") as f:
        f.write(f"# D-grow + K-shrink samples on TinyShakespeare\n")
        f.write(f"# steps={args.steps}  d_min={args.d_min}  d_max={args.d_max}"
                f"  K_init={args.K_init}  K_min={args.K_min}\n")
        f.write(f"# prompt: {args.prompt!r}\n\n")
        for nm, text in samples.items():
            r = results[nm]
            f.write(f"\n{'='*70}\n{nm}  best_val={r['best_val']:.4f}  "
                    f"params={r['stored_params']:,}\n{'='*70}\n{text}\n")

    print(f"\nSamples → {sample_path}")

    for nm, text in samples.items():
        r = results[nm]
        print(f"\n{'='*70}")
        print(f"{nm}  best_val={r['best_val']:.4f}  params={r['stored_params']:,}")
        print('='*70)
        print(text[:300])

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()

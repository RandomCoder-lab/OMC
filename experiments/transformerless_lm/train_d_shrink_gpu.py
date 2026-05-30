"""D-grow + K-shrink on TinyShakespeare — GPU version via tinygrad Vulkan backend.

Run on AMD RX 580 via Vulkan (default):
    python3 train_d_shrink_gpu.py

For OpenCL fallback:
    RUSTICL_ENABLE=radeonsi python3 train_d_shrink_gpu.py --device CL

For CPU fallback:
    python3 train_d_shrink_gpu.py --device CPU

Same theory and schedules as train_d_shrink.py; this file ports the model
and training loop to tinygrad so the RX 580 handles all matrix work.
Expected speedup: ~20x over the PyTorch CPU version.
"""

import argparse
import collections
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from models_fibgen import FIBONACCI
from corpus import make_dataset

PHI = (1 + math.sqrt(5)) / 2


# ---------------------------------------------------------------------------
# GPU / device setup  (must happen before tinygrad imports tensors)
# ---------------------------------------------------------------------------

def setup_device(device: str):
    if device == "VULKAN":
        os.environ["DEV"] = "VULKAN"
    elif device == "CL":
        os.environ["DEV"] = "CL"
    elif device == "CPU":
        os.environ["DEV"] = "CLANG"
    # else let DEV env var rule


# ---------------------------------------------------------------------------
# Schedules  (identical to train_d_shrink.py)
# ---------------------------------------------------------------------------

def D_schedule_tier_walk(step: int, max_steps: int, d_min: int, d_max: int) -> int:
    tiers = sorted({f for f in FIBONACCI if d_min <= f <= d_max})
    if d_max not in tiers:
        tiers.append(d_max)
    steps_per_tier = max_steps // len(tiers)
    return tiers[min(step // max(steps_per_tier, 1), len(tiers) - 1)]


def K_schedule_tier_walk(step: int, max_steps: int, K_init: int, K_min: int) -> int:
    tiers = sorted({f for f in FIBONACCI if K_min <= f <= K_init}, reverse=True)
    steps_per_tier = max_steps // len(tiers)
    return tiers[min(step // max(steps_per_tier, 1), len(tiers) - 1)]


# ---------------------------------------------------------------------------
# Fibonacci basis (numpy, uploaded to GPU as tinygrad Tensors)
# ---------------------------------------------------------------------------

def make_fib_basis_np(in_features: int, out_features: int, K: int):
    i_idx = np.arange(out_features, dtype=np.float32)
    j_idx = np.arange(in_features,  dtype=np.float32)
    freqs = np.array(FIBONACCI[:K], dtype=np.float32)
    a_i = 2 * math.pi * i_idx[:, None] * freqs[None, :] / max(out_features, 1)
    a_j = 2 * math.pi * j_idx[:, None] * freqs[None, :] / max(in_features,  1)
    return {
        "cos_i": np.cos(a_i).astype(np.float32),  # [out, K]
        "sin_i": np.sin(a_i).astype(np.float32),
        "cos_j": np.cos(a_j).astype(np.float32),  # [in,  K]
        "sin_j": np.sin(a_j).astype(np.float32),
    }


def crt_pe_np(seq_len: int, d_model: int) -> np.ndarray:
    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    pos = np.arange(seq_len, dtype=np.float32)
    moduli = [5, 8, 13, 21, 34, 55, 89, 144]
    for i in range(d_model // 2):
        m = moduli[i % len(moduli)]
        angle = 2 * math.pi * (pos % m) / m
        pe[:, 2 * i]     = np.sin(angle)
        pe[:, 2 * i + 1] = np.cos(angle)
    return pe


# ---------------------------------------------------------------------------
# Model — tinygrad port of FibRecLM_DGrow
# ---------------------------------------------------------------------------

class FibRecLM_DGrow_GPU:
    """FibRecLM with dynamic d_active / K_active, running on tinygrad device."""

    def __init__(self, vocab_size: int, d_max: int, K_init: int,
                 n_blocks: int, seq_len: int):
        from tinygrad import Tensor

        self.vocab_size = vocab_size
        self.d_max      = d_max
        self.K_init     = K_init
        self.n_blocks   = n_blocks
        self.seq_len    = seq_len
        self.d_active   = d_max
        self.K_active   = K_init

        # ---- Embedding (d_max) ----
        scale = 1.0 / math.sqrt(d_max)
        self.embed_w = Tensor.uniform(vocab_size, d_max, low=-scale, high=scale)

        # ---- Positional encoding (constant, no grad) ----
        self._pe = Tensor(crt_pe_np(seq_len, d_max), requires_grad=False)

        # ---- Causal mask (constant) ----
        self._mask = Tensor(
            np.tril(np.ones((seq_len, seq_len), dtype=np.float32)),
            requires_grad=False)

        # ---- Basis tables (constant, uploaded once) ----
        self._basis: dict[str, dict] = {}
        for lname, in_dim, out_dim in [
            ("qkv", d_max, 3 * d_max),
            ("out", d_max,     d_max),
            ("w1",  d_max, 4 * d_max),
            ("w2", 4 * d_max,  d_max),
        ]:
            b = make_fib_basis_np(in_dim, out_dim, K_init)
            self._basis[lname] = {k: Tensor(v, requires_grad=False)
                                  for k, v in b.items()}

        # ---- Seeds: K_init² × 4 ----
        n_comp = K_init * K_init
        init_s = 0.1 / math.sqrt(n_comp)
        self._seeds: dict[str, list] = {nm: [] for nm in ("qkv","out","w1","w2")}
        for nm in ("qkv","out","w1","w2"):
            for _ in range(2):
                self._seeds[nm].append(
                    Tensor.uniform(n_comp, 4, low=-init_s, high=init_s))

        # ---- Recurrence matrices ----
        self._A: dict[str, Tensor] = {}
        self._B: dict[str, Tensor] = {}
        for nm in ("qkv","out","w1","w2"):
            self._A[nm] = Tensor(
                (np.eye(K_init) + 0.01*np.random.randn(K_init,K_init)).astype(np.float32))
            self._B[nm] = Tensor(
                (0.01*np.random.randn(K_init,K_init)).astype(np.float32))

        # ---- LayerNorm params (d_max, sliced at runtime) ----
        self._ln1_w = [Tensor.ones(d_max) for _ in range(n_blocks)]
        self._ln1_b = [Tensor.zeros(d_max) for _ in range(n_blocks)]
        self._ln2_w = [Tensor.ones(d_max) for _ in range(n_blocks)]
        self._ln2_b = [Tensor.zeros(d_max) for _ in range(n_blocks)]
        self._lnf_w = Tensor.ones(d_max)
        self._lnf_b = Tensor.zeros(d_max)

    def set_d_active(self, d: int): self.d_active = max(1, min(d, self.d_max))
    def set_K_active(self, k: int): self.K_active = max(1, min(k, self.K_init))

    def parameters(self):
        from tinygrad.nn.state import get_parameters
        return get_parameters(self)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ln(self, x, w, b):
        d = x.shape[-1]
        return x.layernorm().mul(w[:d]).add(b[:d])

    def _fib_fwd(self, x, seed, lname: str, in_a: int, out_a: int):
        K  = self.K_init
        Ka = self.K_active
        basis = self._basis[lname]
        cj = basis["cos_j"][:in_a,  :Ka]   # [in_a,  Ka]
        sj = basis["sin_j"][:in_a,  :Ka]
        ci = basis["cos_i"][:out_a, :Ka]   # [out_a, Ka]
        si = basis["sin_i"][:out_a, :Ka]
        sc = seed.reshape(K, K, 4)[:Ka, :Ka]  # [Ka, Ka, 4]
        a, b, c, d_ = sc[:,:,0], sc[:,:,1], sc[:,:,2], sc[:,:,3]
        xc = x.matmul(cj)                  # [B,T,Ka]
        xs = x.matmul(sj)
        yc = xc.matmul(a.T) + xs.matmul(c.T)
        ys = xc.matmul(b.T) + xs.matmul(d_.T)
        return yc.matmul(ci.T) + ys.matmul(si.T)

    def _rec_step(self, nm: str, s_p2, s_p1):
        K = self.K_init
        A, B = self._A[nm], self._B[nm]
        sp1 = s_p1.reshape(K, K, 4)
        sp2 = s_p2.reshape(K, K, 4)
        # einsum "ik,kjc->ijc" via matmul: A @ sp1 along first axis
        sn = A.matmul(sp1.reshape(K, K*4)).reshape(K, K, 4) + \
             B.matmul(sp2.reshape(K, K*4)).reshape(K, K, 4)
        return sn.reshape(K * K, 4)

    def _all_seeds(self):
        running = {nm: (self._seeds[nm][0], self._seeds[nm][1])
                   for nm in ("qkv","out","w1","w2")}
        seeds = []
        for n in range(self.n_blocks):
            if n == 0:
                seeds.append(tuple(self._seeds[nm][0] for nm in ("qkv","out","w1","w2")))
            elif n == 1:
                seeds.append(tuple(self._seeds[nm][1] for nm in ("qkv","out","w1","w2")))
            else:
                new = {}
                for nm in ("qkv","out","w1","w2"):
                    s_p2, s_p1 = running[nm]
                    new[nm] = (s_p1, self._rec_step(nm, s_p2, s_p1))
                running = new
                seeds.append(tuple(running[nm][1] for nm in ("qkv","out","w1","w2")))
        return seeds

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def __call__(self, token_ids):
        from tinygrad import Tensor
        B, T = token_ids.shape
        d = self.d_active

        h = self.embed_w[token_ids][:, :, :d] + self._pe[:T, :d]  # embed + PE slice
        mask = self._mask[:T, :T]

        for n, (qkv_s, out_s, w1_s, w2_s) in enumerate(self._all_seeds()):
            # Attention
            xn = self._ln(h, self._ln1_w[n], self._ln1_b[n])
            qkv = self._fib_fwd(xn, qkv_s, "qkv", d, 3 * d)
            q, k, v = qkv[:,:,:d], qkv[:,:,d:2*d], qkv[:,:,2*d:]
            scale = 1.0 / math.sqrt(d)
            scores = q.matmul(k.transpose(-2,-1)) * scale
            scores = scores.where(mask.cast('bool'), float('-inf'))
            attn = scores.softmax(-1)
            h = h + self._fib_fwd(attn.matmul(v), out_s, "out", d, d)

            # FFN
            xn2 = self._ln(h, self._ln2_w[n], self._ln2_b[n])
            ff = self._fib_fwd(xn2, w1_s, "w1", d, 4 * d).gelu()
            h  = h + self._fib_fwd(ff, w2_s, "w2", 4 * d, d)

        h = h.layernorm().mul(self._lnf_w[:d]).add(self._lnf_b[:d])
        return h.matmul(self.embed_w[:, :d].T)   # tied head

    def storage_summary(self):
        from tinygrad.nn.state import get_parameters
        stored = sum(p.numel() for p in get_parameters(self))
        d = self.d_max
        dense_eq = self.n_blocks * (3*d*d + d*d + 4*d*d + 4*d*d + 4*d) + self.vocab_size * d
        return {"stored": stored, "dense_equivalent": dense_eq,
                "compression": dense_eq / max(stored, 1),
                "d_active": self.d_active, "K_active": self.K_active}


# ---------------------------------------------------------------------------
# Data helpers (torch-free)
# ---------------------------------------------------------------------------

def get_fib_positions(seq_len: int) -> list[int]:
    positions = set()
    a, b = 1, 1
    while a < seq_len:
        positions.add(a - 1)
        a, b = b, a + b
    positions.add(0)
    return sorted(positions)


def get_batch_np(data: np.ndarray, batch_size: int, seq_len: int,
                  fib_pos: list[int], rng: np.random.Generator):
    n = len(data) - seq_len - 1
    ix = rng.integers(0, n, size=batch_size)
    # Lazy Fibonacci-strided: only load tokens at Fibonacci positions
    x = np.stack([data[i:i+seq_len] for i in ix])
    y = np.stack([data[i+1:i+seq_len+1] for i in ix])
    return x, y


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def evaluate(model, val_data: np.ndarray, batch_size: int, seq_len: int,
              fib_pos: list[int], rng: np.random.Generator, n_batches: int = 16):
    from tinygrad import Tensor
    Tensor.training = False
    losses = []
    for _ in range(n_batches):
        x_np, y_np = get_batch_np(val_data, batch_size, seq_len, fib_pos, rng)
        x = Tensor(x_np.astype(np.int32))
        y = Tensor(y_np.astype(np.int32))
        logits = model(x)
        loss = logits.reshape(-1, model.vocab_size).sparse_categorical_crossentropy(y.reshape(-1))
        losses.append(loss.item())
    Tensor.training = True
    return float(np.mean(losses))


def train(name: str, model, train_data: np.ndarray, val_data: np.ndarray,
          args, fib_pos: list[int], D_fn=None, K_fn=None):
    from tinygrad import Tensor
    from tinygrad.nn.optim import AdamW
    from tinygrad.nn.state import get_parameters
    from tinygrad.engine.jit import TinyJit

    rng = np.random.default_rng(args.seed)
    val_rng = np.random.default_rng(args.seed + 1)

    params = get_parameters(model)
    optimizer = AdamW(params, lr=args.lr)
    Tensor.training = True

    s = model.storage_summary()
    print(f"\n[{name}]  stored={s['stored']:,}  compression={s['compression']:.1f}x"
          f"  d={s['d_active']}  K={s['K_active']}", flush=True)

    t0 = time.time()
    best_val = float("inf"); best_step = -1
    val_hist = []; D_hist = []; K_hist = []
    eval_every = args.eval_every if args.eval_every > 0 else max(args.steps // 20, 250)
    log_every  = args.log_every
    cur_d = cur_k = None
    # Rolling average of the last 10 step times
    _step_times: collections.deque = collections.deque(maxlen=10)
    _step_t0 = time.time()

    # TinyJit-compiled training step — replays cached dispatch sequence each step.
    # Reset when d_active/K_active changes (different shapes → different kernels).
    def make_jit_step():
        @TinyJit
        def _step(x, y):
            logits = model(x)
            loss = logits.reshape(-1, model.vocab_size).sparse_categorical_crossentropy(y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return loss
        return _step

    jit_step = make_jit_step()
    # Keep last 6 (x,y) tensor pairs alive to prevent LRU allocator from
    # returning the same _VKBuf objects — reused buffers trigger a GPU sync
    # in async copyin. ring_n=6 benchmarked best at 0.355s/step vs 0.463s.
    ring = collections.deque(maxlen=6)

    for step in range(args.steps):
        dim_changed = False
        if D_fn is not None:
            new_d = D_fn(step, args.steps)
            if new_d != cur_d:
                model.set_d_active(new_d); cur_d = new_d
                D_hist.append((step, new_d))
                dim_changed = True
        if K_fn is not None:
            new_k = K_fn(step, args.steps)
            if new_k != cur_k:
                model.set_K_active(new_k); cur_k = new_k
                K_hist.append((step, new_k))
                dim_changed = True
        if dim_changed:
            jit_step = make_jit_step()  # rebuild JIT for new tensor shapes
            ring.clear()               # release old tensors so LRU can reuse them

        x_np, y_np = get_batch_np(train_data, args.batch_size, args.seq_len,
                                   fib_pos, rng)
        x = Tensor(x_np.astype(np.int32))
        y = Tensor(y_np.astype(np.int32))

        _loss = jit_step(x, y)
        ring.append((x, y))

        # --- rolling timing ---
        _now = time.time()
        _step_times.append(_now - _step_t0)
        _step_t0 = _now
        is_eval_step = (step % eval_every == 0 or step == args.steps - 1)
        if not is_eval_step and step > 0 and step % log_every == 0:
            avg_s = sum(_step_times) / len(_step_times)
            print(f"  step {step:5d}  loss={_loss.item():.3f}  ({avg_s:.2f}s/step)", flush=True)

        if is_eval_step:
            vl = evaluate(model, val_data, args.batch_size, args.seq_len,
                          fib_pos, val_rng)
            val_hist.append((step, vl))
            marker = ""
            if vl < best_val:
                best_val = vl; best_step = step; marker = " ← BEST"
            d_tag = f" d={cur_d}" if cur_d else ""
            k_tag = f" K={cur_k}" if cur_k else ""
            print(f"  step {step:5d}  val={vl:.4f}{d_tag}{k_tag}"
                  f"  ({time.time()-t0:.1f}s){marker}", flush=True)

    stored = sum(p.numel() for p in get_parameters(model))
    return {"name": name, "stored_params": stored,
            "best_val": best_val, "best_step": best_step,
            "final_val": evaluate(model, val_data, args.batch_size, args.seq_len,
                                   fib_pos, val_rng, n_batches=32),
            "wall": time.time() - t0,
            "D_history": D_hist, "K_history": K_hist, "val_hist": val_hist}


# ---------------------------------------------------------------------------
# Text sampling
# ---------------------------------------------------------------------------

def sample_text(model, prompt_ids: list[int], vocab_size: int,
                n_new: int, seq_len: int,
                temperature: float = 0.8, top_k: int = 10) -> list[int]:
    from tinygrad import Tensor
    Tensor.training = False
    out = list(prompt_ids)
    for _ in range(n_new):
        ctx = out[-seq_len:]
        x = Tensor(np.array([ctx], dtype=np.int32))
        logits = model(x)[0, -1, :].numpy() / max(temperature, 1e-6)
        if top_k:
            threshold = np.sort(logits)[-top_k]
            logits[logits < threshold] = float('-inf')
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        out.append(int(np.random.choice(len(probs), p=probs)))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",      type=int,   default=30000)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--seq-len",    type=int,   default=128)
    parser.add_argument("--n-blocks",   type=int,   default=6)
    parser.add_argument("--d-max",      type=int,   default=384)
    parser.add_argument("--d-min",      type=int,   default=34)
    parser.add_argument("--K-init",     type=int,   default=144)
    parser.add_argument("--K-min",      type=int,   default=3)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--device",     type=str,   default="VULKAN",
                        choices=["CL", "CPU"],
                        help="CL = OpenCL/GPU (RUSTICL_ENABLE=radeonsi), CPU = clang")
    parser.add_argument("--prompt",     type=str,
                        default="ROMEO:\nWhat light through")
    parser.add_argument("--n-new",      type=int,   default=600)
    parser.add_argument("--temperature",type=float, default=0.8)
    parser.add_argument("--top-k",      type=int,   default=10)
    parser.add_argument("--log-every",   type=int,   default=50,
                        help="Print a one-line timing update every N steps (default: 50)")
    parser.add_argument("--eval-every",  type=int,   default=0,
                        help="Eval every N steps (default: max(steps//20, 250))")
    parser.add_argument("--no-static",  action="store_true",
                        help="Skip the fibgen_static baseline arm")
    parser.add_argument("--out",        type=str,
                        default="results_d_shrink_gpu.json")
    args = parser.parse_args()

    setup_device(args.device)

    from tinygrad import Tensor, Device
    print(f"Device: {Device.DEFAULT}", flush=True)

    chars, stoi, itos, encoded_torch = make_dataset(seq_len=args.seq_len,
                                                     source="tinyshakespeare")
    vocab_size = len(chars)
    data = encoded_torch.numpy()
    n = len(data)
    split = int(n * 0.9)
    train_data = data[:split]
    val_data   = data[split:]
    fib_pos = get_fib_positions(args.seq_len)

    prompt_ids = [stoi.get(c, stoi.get(" ", 0)) for c in args.prompt]

    print(f"TinyShakespeare  vocab={vocab_size}  train={len(train_data):,}  val={len(val_data):,}")
    print(f"D-grow: {args.d_min}→{args.d_max}   K-shrink: {args.K_init}→{args.K_min}")
    print(f"Steps={args.steps}  batch={args.batch_size}  seq={args.seq_len}")
    print("\nSchedule preview:")
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        s = int(args.steps * frac)
        d = D_schedule_tier_walk(s, args.steps, args.d_min, args.d_max)
        k = K_schedule_tier_walk(s, args.steps, args.K_init, args.K_min)
        print(f"  step {s:>6} ({frac*100:3.0f}%): d={d:<5} K={k}")

    results = {}
    samples = {}

    D_fn = lambda s, T: D_schedule_tier_walk(s, T, args.d_min, args.d_max)
    K_fn = lambda s, T: K_schedule_tier_walk(s, T, args.K_init, args.K_min)

    # ------------------------------------------------------------------
    # Arm 1: FibRecLM static at d=d_max, K=K_init (no schedule)
    # ------------------------------------------------------------------
    if not args.no_static:
        print(f"\n{'='*60}\nFibRecLM static  d={args.d_max}  K={args.K_init}\n{'='*60}")
        m = FibRecLM_DGrow_GPU(vocab_size, args.d_max, args.K_init,
                                args.n_blocks, args.seq_len)
        results["fibgen_static"] = train(
            "fibgen_static", m, train_data, val_data, args, fib_pos)
        ids = sample_text(m, prompt_ids, vocab_size, args.n_new, args.seq_len,
                          args.temperature, args.top_k)
        samples["fibgen_static"] = "".join(itos[i] for i in ids)

    # ------------------------------------------------------------------
    # Arm 2: D-grow + K-shrink
    # ------------------------------------------------------------------
    print(f"\n{'='*60}\nD-grow + K-shrink  d:{args.d_min}→{args.d_max}  K:{args.K_init}→{args.K_min}\n{'='*60}")
    m2 = FibRecLM_DGrow_GPU(vocab_size, args.d_max, args.K_init,
                              args.n_blocks, args.seq_len)
    # Cold-start: zero embedding dims beyond d_min
    _embed_np = m2.embed_w.numpy().copy()
    _embed_np[:, args.d_min:] = 0.0
    from tinygrad import Tensor as _T
    m2.embed_w.assign(_T(_embed_np))
    m2.set_d_active(args.d_min)
    m2.set_K_active(args.K_init)
    results["d_grow_k_shrink"] = train(
        "d_grow_k_shrink", m2, train_data, val_data, args, fib_pos,
        D_fn=D_fn, K_fn=K_fn)
    m2.set_d_active(args.d_max); m2.set_K_active(args.K_min)
    ids2 = sample_text(m2, prompt_ids, vocab_size, args.n_new, args.seq_len,
                       args.temperature, args.top_k)
    samples["d_grow_k_shrink"] = "".join(itos[i] for i in ids2)

    # ------------------------------------------------------------------
    # Summary + samples
    # ------------------------------------------------------------------
    FLOOR = math.log(vocab_size)
    print(f"\n{'='*76}")
    print(f"{'name':<22} {'stored':>10} {'best_val':>10} {'vs uniform':>12} {'wall':>8}")
    print("-"*76)
    for nm, r in results.items():
        gap = (r["best_val"] - FLOOR) / FLOOR * 100
        print(f"{nm:<22} {r['stored_params']:>10,} {r['best_val']:>10.4f}"
              f"  {gap:>+10.1f}%  {r['wall']:>7.1f}s")

    sample_path = Path(__file__).parent / "samples_d_shrink_gpu.txt"
    with open(sample_path, "w") as f:
        f.write(f"# D-grow + K-shrink GPU samples  steps={args.steps}"
                f"  d:{args.d_min}→{args.d_max}  K:{args.K_init}→{args.K_min}\n"
                f"# prompt: {args.prompt!r}\n\n")
        for nm, text in samples.items():
            r = results[nm]
            f.write(f"\n{'='*70}\n{nm}  best_val={r['best_val']:.4f}"
                    f"  params={r['stored_params']:,}\n{'='*70}\n{text}\n")

    for nm, text in samples.items():
        r = results[nm]
        print(f"\n{'='*70}")
        print(f"{nm}  best_val={r['best_val']:.4f}  params={r['stored_params']:,}")
        print("="*70)
        print(text)

    print(f"\nSamples → {sample_path}")
    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results → {out_path}")


if __name__ == "__main__":
    main()

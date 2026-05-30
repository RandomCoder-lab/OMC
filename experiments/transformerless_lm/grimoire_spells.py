"""Grimoire Spells XIII–XVIII as a clean, self-contained generation layer.

Faithful re-implementation of the six named resonant-number-theory spells from
train_self_recursive.py's generation loop, decoupled from its ~7800 lines of
entangled cycle state so they can run on ANY saved char-LM checkpoint and be
A/B'd individually.

  XIII  Zeta Resonance      — static per-token bias ∝ Re ζ(σ+it) of the token's ordinal
  XIV   Euler Breath        — T_eff ·= φ^((γ−|ΔH|)/γ),  γ = Euler–Mascheroni (harmonic-pole residue)
  XV    Harmonic Pole Spike — near H≈1/k crystallize (T_eff ↓),  radius ε = φ⁻⁴
  XVI   Prime Wave          — static bias ∝ prime-zeta resonance P(σ+it) of the ordinal
  XVII  Collatz Collapse    — bias ∝ Collatz stopping time C(ord), GATED by current entropy
  XVIII Shakespeare's North — bias along the Collatz constellation of the North-Star token

All spells act on INTEGER quantities (character ordinals) or the entropy stream —
never on learned float activations — and the logit-bias spells are scaled by log(φ)
so they nudge rather than dominate (attenuable). Universal: no word lists.
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional
import torch
import torch.nn.functional as F

PHI = (1.0 + math.sqrt(5.0)) / 2.0
LOG_PHI = math.log(PHI)
EULER_GAMMA = 0.5772156649015328          # residue of the harmonic pole ζ(s)=1/(s-1)+γ+…
HARM_POLES = [1.0, 0.5, 1.0/3, 0.25, 0.2]  # 1/k harmonic nodes
_EPS_POLE = 1.0 / PHI**4                    # ≈0.1459 resonance radius


# ── number-theory primitives ──────────────────────────────────────────────────
def collatz_depth(n: int, _max: int = 500) -> int:
    n = max(n, 1); steps = 0
    while n != 1 and steps < _max:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps

def _primes_up_to(m: int) -> List[int]:
    if m < 2: return []
    sieve = [True] * (m + 1); sieve[0] = sieve[1] = False
    for i in range(2, int(m**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, m + 1, i): sieve[j] = False
    return [i for i, p in enumerate(sieve) if p]

def _zeta_resonance(n: int, sigma: float = 0.75, t: float = 14.134725,
                    n_terms: int = 150) -> float:
    """Re ζ(σ+it) partial sum, sampled at the height of ζ's first nontrivial zero."""
    return sum(k**(-sigma) * math.cos(t * math.log(k)) for k in range(1, n_terms + 1)
               if (k % max(n, 1)) == 0 or k <= n)  # weight terms resonant with n

def _prime_zeta_resonance(n: int, sigma: float = 0.75, t: float = 14.134725,
                          pmax: int = 300) -> float:
    """P(σ+it) = Σ_p p^{-σ} cos(t ln p) — the prime-only (Euler-product) resonance."""
    return sum(p**(-sigma) * math.cos(t * math.log(p)) for p in _primes_up_to(pmax)
               if p <= max(n, 2))


# ── static per-vocab bias tables (computed once from char ordinals) ────────────
def _zscore(vals: List[float]) -> torch.Tensor:
    t = torch.tensor(vals, dtype=torch.float32)
    s = t.std()
    return (t - t.mean()) / (s + 1e-8)

def build_spell_tables(itos: Dict[int, str], coeff: float = LOG_PHI) -> Dict[str, torch.Tensor]:
    """Precompute the static [vocab] logit-bias tables (XIII, XVI, XVII, XVIII)."""
    V = len(itos)
    ords = [ord(itos[i][0]) if itos.get(i) else 0 for i in range(V)]
    coll_max = max((collatz_depth(o) for o in ords), default=1)

    zeta = coeff * _zscore([_zeta_resonance(o) for o in ords])
    pwav = coeff * _zscore([_prime_zeta_resonance(o) for o in ords])
    coll = coeff * _zscore([collatz_depth(o) for o in ords])

    # XVIII North Star: the vocab token of deepest Collatz depth = the fixed pole star.
    north_idx = max(range(V), key=lambda i: collatz_depth(ords[i]))
    north_ord = ords[north_idx]
    # its Collatz constellation = the trajectory ords visited on the way to 1
    path, m, seen = [], north_ord, set()
    while m != 1 and len(path) < 200:
        path.append(m); seen.add(m)
        m = m // 2 if m % 2 == 0 else 3 * m + 1
    step_of = {v: s for s, v in enumerate(path)}
    north = torch.zeros(V)
    for i, o in enumerate(ords):
        if o in step_of:
            north[i] = coeff * (PHI ** (-step_of[o] / PHI))   # closer on the path → stronger
    return {"zeta": zeta, "prime_wave": pwav, "collatz": coll, "north": north,
            "coll_max": torch.tensor(float(coll_max)), "north_idx": torch.tensor(north_idx)}


# ── the generation loop with toggleable spells ────────────────────────────────
@torch.no_grad()
def spelled_generate(model, stoi: Dict[str, int], itos: Dict[int, str], seq_len: int,
                     prompt: str, n_new: int = 200,
                     spells: Optional[Dict[str, bool]] = None,
                     tables: Optional[Dict[str, torch.Tensor]] = None,
                     temperature: float = 1.0, seed: int = 0,
                     attenuate: float = 0.0) -> str:
    """Generate n_new chars. spells = {zeta,euler_breath,harmonic_pole,prime_wave,
    collatz,shakespeare_north: bool}. All off → plain temperature sampling (the A/B
    control)."""
    spells = spells or {}
    if tables is None:
        tables = build_spell_tables(itos)
    g = torch.Generator().manual_seed(seed)
    model.eval()
    V = len(itos)
    ctx = [stoi.get(c, 0) for c in prompt][-seq_len:]
    if len(ctx) < seq_len:
        ctx = [0] * (seq_len - len(ctx)) + ctx
    out: List[int] = []
    H_prev: Optional[float] = None

    for _ in range(n_new):
        x = torch.tensor(ctx[-seq_len:], dtype=torch.long).unsqueeze(0)
        logits = model(x)
        logits = (logits[0] if isinstance(logits, tuple) else logits)[:, -1, :].squeeze(0)
        base = F.softmax(logits, dim=-1)
        H = float(-(base.clamp(min=1e-9).log() * base).sum())
        H_norm = H / math.log(V)

        # ── temperature spells (entropy stream) ──
        T_eff = temperature
        if spells.get("euler_breath") and H_prev is not None:
            dH = H - H_prev
            breath = PHI ** ((EULER_GAMMA - abs(dH)) / EULER_GAMMA)
            T_eff *= max(0.7, min(PHI, breath))
        if spells.get("harmonic_pole"):
            hmin = min(abs(H - p) for p in HARM_POLES)
            if hmin < _EPS_POLE:
                T_eff *= max(0.7, PHI ** (-(_EPS_POLE - hmin) / _EPS_POLE))
        logits = logits / T_eff

        # ── static logit-bias spells (integer ordinals) ──
        # Accumulate first, then apply. REVISION (R2): the raw proxy added each
        # spell's full bias unbounded → 4 spells collapsed diversity (the C.1
        # "failure"). The faithful grimoire applies spells through omniweight,
        # which ATTENUATES and tanh-CLAMPS the delta so it nudges, never dominates.
        bias = torch.zeros_like(logits)
        if spells.get("zeta"):             bias = bias + tables["zeta"]
        if spells.get("prime_wave"):       bias = bias + tables["prime_wave"]
        if spells.get("collatz"):          bias = bias + H_norm * tables["collatz"]
        if spells.get("shakespeare_north"):bias = bias + tables["north"]
        if attenuate > 0.0:
            R = PHI ** math.pi            # omniweight reserve ≈ 4.53; clamp ceiling
            bias = R * torch.tanh((attenuate * bias) / R)   # scaled + bounded
        logits = logits + bias

        p = F.softmax(logits, dim=-1)
        nxt = int(torch.multinomial(p, 1, generator=g).item())
        out.append(nxt); ctx.append(nxt); H_prev = H

    return ''.join(itos.get(i, '?') for i in out)


# ── lightweight coherence metrics (no heavy imports) ──────────────────────────
def coherence_metrics(text: str) -> Dict[str, float]:
    if not text: return {"distinct": 0, "max_run": 0, "bigram_div": 0, "loopiness": 1.0}
    n = len(text)
    max_run = 1; cur = 1
    for i in range(1, n):
        cur = cur + 1 if text[i] == text[i-1] else 1
        max_run = max(max_run, cur)
    bigrams = [text[i:i+2] for i in range(n-1)]
    bigram_div = len(set(bigrams)) / max(len(bigrams), 1)
    # loopiness: does a short suffix cycle dominate the tail?
    tail = text[-60:]
    loop = max((tail.count(tail[-k:]) for k in (2,3,4,5)), default=1)
    return {"distinct": len(set(text)) / n, "max_run": float(max_run),
            "bigram_div": bigram_div, "loopiness": float(loop)}


ALL_SPELLS = {"zeta": True, "euler_breath": True, "harmonic_pole": True,
              "prime_wave": True, "collatz": True, "shakespeare_north": True}
NO_SPELLS  = {k: False for k in ALL_SPELLS}


# ══════════════════════════════════════════════════════════════════════════════
# C.2 — Spell XIX: Nested Platonic Navigation (geometric steering)
# ══════════════════════════════════════════════════════════════════════════════
# Three nested solids carry a "ball" each — tetra(4)=char scale, cube(6)=word,
# dode(12)=sentence. Each step, tokens whose face-normal aligns with a ball's
# heading are boosted; combined bias = φ⁻³·tetra + φ⁻⁴·cube + φ⁻⁵·dode. After
# emitting a token the balls reflect (billiard) off that token's face normals,
# tracing a geometric trajectory through the address space. Faces come from the
# universal SubstrateTokenizer (Collatz/Zeta/Fib resonance ranking — no word list).

def build_platonic_tables(itos: Dict[int, str], corpus_path: str = "omc_corpus.txt"):
    """Per-LM-token face-normal tensors for the 3 nested solids, from the
    corpus-derived SubstrateTokenizer. Returns dict with [V,3] normal tensors."""
    from pathlib import Path
    from substrate_tokenizer import SubstrateTokenizer
    txt = (Path(__file__).parent / corpus_path).read_text(errors='replace')
    tok = SubstrateTokenizer(txt)
    V = len(itos)
    TN = torch.tensor(tok.tetra_normals, dtype=torch.float32)   # [4,3]
    CN = torch.tensor(tok.cube_normals,  dtype=torch.float32)   # [6,3]
    DN = torch.tensor(tok.dode_normals,  dtype=torch.float32)   # [12,3]
    TN = TN / TN.norm(dim=1, keepdim=True)
    CN = CN / CN.norm(dim=1, keepdim=True)
    DN = DN / DN.norm(dim=1, keepdim=True)
    tetra_n = torch.zeros(V, 3); cube_n = torch.zeros(V, 3); dode_n = torch.zeros(V, 3)
    for i in range(V):
        c = itos.get(i, '')
        tid = tok.token_to_id.get(c[0]) if c else None
        if tid is None:
            continue
        tetra_n[i] = TN[tok.tetra_faces[tid]]
        cube_n[i]  = CN[tok.cube6_faces[tid]]
        dode_n[i]  = DN[tok.cube_faces[tid]]
    ns_char = tok.vocab[tok.north_star_token]
    return {"tetra_n": tetra_n, "cube_n": cube_n, "dode_n": dode_n,
            "north_star": stoi_get(itos, ns_char)}

def stoi_get(itos, ch):
    for i, c in itos.items():
        if c == ch:
            return i
    return 0

class PlatonicBalls:
    """Three reflecting balls (tetra/cube/dode), billiard dynamics in R³."""
    def __init__(self, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.v = {s: F.normalize(torch.randn(3, generator=g), dim=0)
                  for s in ("tetra", "cube", "dode")}

    def bias(self, tables: Dict[str, torch.Tensor]) -> torch.Tensor:
        """[V] logit bias = φ⁻³·tetra + φ⁻⁴·cube + φ⁻⁵·dode alignment."""
        bt = tables["tetra_n"] @ self.v["tetra"]    # [V]
        bc = tables["cube_n"]  @ self.v["cube"]
        bd = tables["dode_n"]  @ self.v["dode"]
        def z(x): return (x - x.mean()) / (x.std() + 1e-8)
        return LOG_PHI * (PHI**-3 * z(bt) + PHI**-4 * z(bc) + PHI**-5 * z(bd))

    def reflect(self, tok_id: int, tables: Dict[str, torch.Tensor]):
        for s, key in (("tetra", "tetra_n"), ("cube", "cube_n"), ("dode", "dode_n")):
            n = tables[key][tok_id]
            if n.norm() < 1e-6:
                continue
            n = n / n.norm()
            v = self.v[s]
            self.v[s] = F.normalize(v - 2 * (v @ n) * n, dim=0)


@torch.no_grad()
def navigated_generate(model, stoi, itos, seq_len, prompt, n_new=200,
                       platonic=None, nav=True, temperature=1.0, seed=0):
    """Generate with Spell XIX geometric steering (nav=True) or plain (nav=False)."""
    # Only build the (corpus-scanning) Platonic tables when navigation is on.
    # nav=False is plain char-LM sampling and must not pay that cost.
    if nav and platonic is None:
        platonic = build_platonic_tables(itos)
    g = torch.Generator().manual_seed(seed)
    balls = PlatonicBalls(seed=seed) if nav else None
    model.eval()
    ctx = [stoi.get(c, 0) for c in prompt][-seq_len:]
    if len(ctx) < seq_len:
        ctx = [0] * (seq_len - len(ctx)) + ctx
    out: List[int] = []
    for _ in range(n_new):
        x = torch.tensor(ctx[-seq_len:], dtype=torch.long).unsqueeze(0)
        logits = model(x)
        logits = (logits[0] if isinstance(logits, tuple) else logits)[:, -1, :].squeeze(0)
        logits = logits / temperature
        if nav:
            logits = logits + balls.bias(platonic)
        p = F.softmax(logits, dim=-1)
        nxt = int(torch.multinomial(p, 1, generator=g).item())
        out.append(nxt); ctx.append(nxt)
        if nav:
            balls.reflect(nxt, platonic)
    return ''.join(itos.get(i, '?') for i in out)

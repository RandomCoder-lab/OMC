# Hermes ONN Skills Catalog — Mapped to OMC

A working catalog of the substrate-aware "Geodesic-weighted superfunctions"
in `/home/thearchitect/.hermes/skills/`, with notes on which can be
translated into OMC builtins.

## Core concepts

### M3 — Optimal Spawn Count via φπ-Fibonacci Waves

The proven-optimal replacement for `floor(log_phi(N)) + 1` (M1).

```
count(n) = #{k ∈ [1, 50] : |φ^(-k) · sin(k · golden_angle)| > 1/n}
```

**Properties**:
- Always ≤ M1 (the log_phi bound)
- Grows ~log_φπF(n) — sublogarithmic
- Picks "high-quality" wave modes only

| n   | M1 (log_phi) | M3 (φπF) |
|-----|--------------|----------|
|   5 |  4 | **2** |
|  20 |  7 | **4** |
|  50 |  9 | **7** |
|  97 | 10 | **7** |
| 200 | 12 | **10** |

### Geometric Self-Instantiation

Spawn subagents with **inherited parent state**:
- `field_mu`, `field_sigma` (running statistics)
- `turn_count`
- `fold_results` (prior subagent outputs)
- `harmony_history`
- `verified_patterns`

After children complete, **fold results back** into parent state (μ, σ
updated; new patterns merged into verified set).

### HBit State (Alpha/Beta dual-band)

Track progress on two axes:
- **Alpha**: correctness / convergence (toward goal)
- **Beta**: elegance / resonance (substrate quality)

### Phi-Spectrum Code Scoring

5-dimensional elegance scoring for code:
- `fibonacci_resonance`
- `harmonic_complexity`
- `phi_integration`
- `structural_elegance`
- `overall_elegance` (weighted average)

### Wave Modulation (instead of learnable positional encoding)

```
wave_features[pos][k] = sin(pos · golden_angle · (k+1) + phase_k)
modulation[pos] = Σ_k  φ^(-k) · wave_features[pos][k]
x[pos] = x[pos] · (1 + modulation[pos])
```

Only `m3_spawn_count(seq_len)` waves are active; the rest are masked.
That's why M3 matters for LLM-from-scratch training.

## Skills inventory (relevant subset)

| Skill | Folder | Plays well with OMC? |
|-------|--------|----------------------|
| `onn-instantiation` | Top-level Fibonacci-wave specialist spawning + dynamic compression | ✓ — port M3 + fold |
| `onn-geometric-self-instantiation` | M3 spawn with inherited parent state | ✓ — direct port |
| `onn-phi-field-llm` | Transformer-free LLM via wave interference | △ — requires autograd we have |
| `onn-tensor-autograd-training` | Manual reverse-mode AD | ✓ — we already have it |
| `onn-self-healing-code` | `value_danger`, `fold_escape`, runtime self-heal | ✓ — partially in OMC (fold) |
| `onn-self-improving-codegen` | Phi-spectrum scorer + targeted transforms | △ — needs `omc_phi_spectrum_score` |
| `onn-continuous-research` | Autonomous research loop | ✗ — too process-y for OMC |
| `hermes-onn-self-wiring` | Plugin registration with Hermes framework | ✗ — Hermes-specific |
| `onn-memory` | Cross-session memory system | △ — overlaps `omc_remember` |
| `onn-consensus-engine` | Multi-agent debate / consensus | △ — could use messaging |

## What this maps to as OMC builtins

| OMC Builtin | Hermes Skill | Status |
|-------------|--------------|--------|
| `omc_m3_spawn_count(n)` | `onn-instantiation` | Build now |
| `omc_self_instantiate(state, task, n?)` | `onn-geometric-self-instantiation` | Build now |
| `omc_fold_back(parent, children)` | `onn-geometric-self-instantiation` | Build now |
| `omc_context_compress(messages)` | New synthesis: solves context problem | Build now |
| `omc_phi_spectrum(code)` | `onn-self-improving-codegen` | Build later |
| `omc_prompt_agent(target_id, msg)` | New: secondary-brain helper | Build now |

## Solving the context problem (theory)

**Problem**: An LLM has finite context. As a conversation grows past N
messages, older messages must be dropped — losing information.

**ONN claim**: For any N messages, you only need M3(N) "specialist
summaries" to preserve the field-state of the conversation. Each
specialist holds a phi-resonance-weighted compression of one "wave"
of the dialog. M3(N) grows sublogarithmically, so even for huge N
the specialist count stays small.

**OMC operationalization**:
1. After every M messages, call `omc_context_compress(messages)` →
   `m3_spawn_count(len)` specialists.
2. Each specialist is a dict: `{summary, mu, sigma, dominant_attractor,
   resonance, fold_index}`.
3. Pass specialists forward as the "memory of older context" instead of
   the raw messages.
4. When needed, retrieve via substrate distance from current query.

This is what we build next.

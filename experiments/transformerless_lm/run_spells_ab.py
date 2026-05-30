"""A/B: Grimoire Spells XIII–XVIII on/off, multi-seed, on a saved char-LM.

Usage: python3 run_spells_ab.py [checkpoint.pt] [n_seeds]
Reports mean coherence metrics (spells off vs on) over seeds×prompts.
"""
import sys, statistics
from pathlib import Path
from omc_assistant import _load_lm
from grimoire_spells import (spelled_generate, build_spell_tables,
                             coherence_metrics, ALL_SPELLS, NO_SPELLS)

ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('bloom_256_curriculum_model.pt')
n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
PROMPTS = ["fn fibonacci(n) {", "fn add(a, b) {", "fn sum(arr) {", "h x = "]

model, stoi, itos, seq_len = _load_lm(ckpt)
tables = build_spell_tables(itos)
print(f"loaded {ckpt.name}: vocab={len(itos)} seq_len={seq_len}  "
      f"NorthStar={itos.get(int(tables['north_idx'].item()))!r}")

def agg(spells):
    acc = {"distinct": [], "max_run": [], "bigram_div": [], "loopiness": []}
    for s in range(n_seeds):
        for pr in PROMPTS:
            txt = spelled_generate(model, stoi, itos, seq_len, pr, n_new=140,
                                   spells=spells, tables=tables, seed=s)
            for k, v in coherence_metrics(txt).items():
                acc[k].append(v)
    return {k: statistics.mean(v) for k, v in acc.items()}

off, on = agg(NO_SPELLS), agg(ALL_SPELLS)
print(f"\n{'metric':12s} {'OFF':>9s} {'ON':>9s} {'Δ':>9s}")
for k in off:
    d = on[k] - off[k]
    print(f"{k:12s} {off[k]:9.3f} {on[k]:9.3f} {d:+9.3f}")
print("\n(higher distinct/bigram_div = better; lower max_run/loopiness = better)")

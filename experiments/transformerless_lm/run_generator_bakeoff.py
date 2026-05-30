"""C.3 — Generator bake-off with a real generation-quality metric.

Compares generators on CORPUS N-GRAM VALIDITY (fraction of generated k-grams that
actually occur in the training corpus = "does it look like real OMC?") plus the
crude diversity metrics. Includes a corpus k-gram Markov chain as the ceiling
reference (it emits only corpus k-grams by construction → validity ≈ 1.0).

Generators:
  char_lm          — plain temperature sampling from the navigator LM
  char_lm+navXIX   — LM + Spell XIX nested-Platonic geometric steering
  char_lm+spells   — LM + named Spells XIII–XVIII
  corpus_markov4   — pure 4-gram Markov over the corpus (no NN; the ceiling)
"""
import sys, statistics, random
from collections import defaultdict, Counter
from pathlib import Path
import torch

from omc_assistant import _load_lm
from grimoire_spells import (spelled_generate, navigated_generate, build_spell_tables,
                             build_platonic_tables, coherence_metrics,
                             ALL_SPELLS, NO_SPELLS)

HERE = Path(__file__).parent
K = 4
PROMPTS = ["fn fibonacci(n) {", "fn add(a, b) {", "fn sum(arr) {", "h x = "]
SEEDS = [0, 1, 2]
NGEN = 140

print("loading corpus + building 4-gram reference ...", flush=True)
text = (HERE / 'omc_corpus.txt').read_text(errors='replace')
sample = text[:20_000_000]
grams = set()
for i in range(len(sample) - K):
    grams.add(sample[i:i+K])
print(f"  corpus={len(text):,}  distinct {K}-grams (20M sample)={len(grams):,}", flush=True)

# 4-gram Markov table (prev K-1 chars -> next char counts), built on a smaller slice
markov = defaultdict(Counter)
msl = sample[:8_000_000]
for i in range(len(msl) - K):
    markov[msl[i:i+K-1]][msl[i+K-1]] += 1

def ngram_validity(s: str) -> float:
    if len(s) < K: return 0.0
    hit = sum(1 for i in range(len(s)-K) if s[i:i+K] in grams)
    return hit / max(len(s)-K, 1)

def markov_gen(prompt, n, seed):
    rnd = random.Random(seed)
    s = prompt
    for _ in range(n):
        ctx = s[-(K-1):]
        nxt = markov.get(ctx)
        if not nxt:
            ctx = s[-1:] if s else ' '
            cands = [g for g in grams if g.startswith(ctx)]
            s += (rnd.choice(cands)[len(ctx)] if cands else ' '); continue
        chars, wts = zip(*nxt.items())
        s += rnd.choices(chars, weights=wts)[0]
    return s[len(prompt):]

model, stoi, itos, seq_len = _load_lm(HERE / 'bloom_256_curriculum_model.pt')
stab = build_spell_tables(itos)
plat = build_platonic_tables(itos)

GENS = {
    "corpus_markov4":  lambda pr, s: markov_gen(pr, NGEN, s),
    "char_lm":         lambda pr, s: navigated_generate(model, stoi, itos, seq_len, pr, NGEN, platonic=plat, nav=False, seed=s),
    "char_lm+navXIX":  lambda pr, s: navigated_generate(model, stoi, itos, seq_len, pr, NGEN, platonic=plat, nav=True,  seed=s),
    "char_lm+spells":  lambda pr, s: spelled_generate(model, stoi, itos, seq_len, pr, NGEN, spells=ALL_SPELLS, tables=stab, seed=s),
}

print(f"\n{'generator':16s} {'ngram_valid':>11s} {'distinct':>9s} {'bigram_div':>10s} {'loopiness':>9s}")
results = {}
for name, gen in GENS.items():
    vals = {"v": [], "distinct": [], "bigram_div": [], "loopiness": []}
    for s in SEEDS:
        for pr in PROMPTS:
            t = gen(pr, s)
            vals["v"].append(ngram_validity(t))
            m = coherence_metrics(t)
            vals["distinct"].append(m["distinct"]); vals["bigram_div"].append(m["bigram_div"])
            vals["loopiness"].append(m["loopiness"])
    mean = {k: statistics.mean(v) for k, v in vals.items()}
    results[name] = mean
    print(f"{name:16s} {mean['v']:11.3f} {mean['distinct']:9.3f} {mean['bigram_div']:10.3f} {mean['loopiness']:9.3f}")

print("\nngram_validity = fraction of generated 4-grams found in the corpus (higher=more OMC-like)")
best = max((n for n in results if n != 'corpus_markov4'), key=lambda n: results[n]['v'])
print(f"best NN generator by validity: {best}  ({results[best]['v']:.3f})  "
      f"vs markov ceiling {results['corpus_markov4']['v']:.3f}")

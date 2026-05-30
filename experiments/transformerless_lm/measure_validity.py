"""Measure a checkpoint's generation quality = corpus 4-gram validity,
GUARDED against degenerate repetition.

EXP-6 exposed that raw 4-gram validity is gameable: a collapsed model that
emits mostly whitespace/common chars scores ~0.82 validity (every "    " is a
corpus 4-gram) while producing garbage (distinct≈0.05). So we report a guarded
score that zeroes out degenerate output: validity counts only if the text is
sufficiently diverse (distinct-char ratio ≥ 0.15 and max single-char run ≤ 40).
Usage: python3 measure_validity.py <checkpoint.pt>
Prints: gen_validity (guarded), raw_validity, distinct — so collapse is visible.
"""
import sys, statistics
from pathlib import Path
from omc_assistant import _load_lm
from grimoire_spells import navigated_generate

HERE = Path(__file__).parent
K = 4
corpus = (HERE / 'omc_corpus.txt').read_text(errors='replace')
grams = set(corpus[i:i+K] for i in range(0, min(len(corpus), 20_000_000) - K))

def raw_validity(s):
    return 0.0 if len(s) < K else sum(1 for i in range(len(s)-K) if s[i:i+K] in grams)/(len(s)-K)

def distinct_ratio(s):
    return len(set(s)) / max(len(s), 1)

def max_run(s):
    mr = cur = 1
    for i in range(1, len(s)):
        cur = cur + 1 if s[i] == s[i-1] else 1
        mr = max(mr, cur)
    return mr

def guarded_validity(s):
    """Raw validity, but 0 if the text is degenerate (collapsed/repetitive)."""
    if distinct_ratio(s) < 0.15 or max_run(s) > 40:
        return 0.0
    return raw_validity(s)

if __name__ == '__main__':
    ck = Path(sys.argv[1])
    m, stoi, itos, sl = _load_lm(ck)
    prompts = ["fn ", "h x = ", "fn add(a, b) {", "return "]
    texts = [navigated_generate(m, stoi, itos, sl, pr, 140, nav=False, seed=s)
             for s in range(3) for pr in prompts]
    gv  = statistics.mean(guarded_validity(t) for t in texts)
    rv  = statistics.mean(raw_validity(t) for t in texts)
    dr  = statistics.mean(distinct_ratio(t) for t in texts)
    # Keep 'gen_validity=' as the guarded score (what the ledger/parsers read).
    print(f"{ck.name}: gen_validity={gv:.3f}  raw_validity={rv:.3f}  distinct={dr:.2f}  (n={len(texts)})")

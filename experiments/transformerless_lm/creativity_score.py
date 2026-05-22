"""Shakespeare-aware creativity scoring.

Replaces val=CE-on-next-token (which only rewards exact reproduction)
with metrics that measure whether GENERATED text is Shakespeare-LIKE
without being identical:

  - n-gram overlap: fraction of n-char windows in generated text that
    appear ANYWHERE in the corpus. Measures Shakespearean character
    patterns without exact-word requirement.
  - vocab overlap: fraction of generated tokens (whitespace-separated)
    that match corpus vocabulary. Real English/Shakespeare words even
    if not in the same sentence.
  - line structure: avg line length, ratio of letters to total chars.
    Captures stanza/line-break patterns.
  - vowel-consonant transition rate: English alternates v/c; random
    text doesn't. Score the alternation pattern.

Use these to evaluate creative output of substrate-aligned model. A
model that produces statistically-Shakespearean GIBBERISH gets ~0;
a model that produces creative but recognizable English gets high.
"""

import string
from collections import Counter


VOWELS = set("aeiouAEIOU")
LETTERS = set(string.ascii_letters)
WHITESPACE = set(" \n\t")


def char_ngram_overlap(generated: str, corpus_text: str, n: int) -> float:
    """Fraction of n-char windows in generated that appear in corpus.
    Higher = more Shakespearean char-pattern overlap."""
    if len(generated) < n:
        return 0.0
    corpus_ngrams = set(corpus_text[i:i+n] for i in range(len(corpus_text) - n + 1))
    gen_ngrams = [generated[i:i+n] for i in range(len(generated) - n + 1)]
    if not gen_ngrams:
        return 0.0
    matches = sum(1 for g in gen_ngrams if g in corpus_ngrams)
    return matches / len(gen_ngrams)


def vocab_overlap(generated: str, corpus_text: str) -> float:
    """Fraction of generated 'words' (whitespace-split) that appear in
    the corpus vocabulary. Punctuation stripped for comparison."""
    def clean(s):
        return s.lower().strip(string.punctuation)
    corpus_vocab = set(clean(w) for w in corpus_text.split() if clean(w))
    gen_words = [clean(w) for w in generated.split() if clean(w)]
    if not gen_words:
        return 0.0
    matches = sum(1 for w in gen_words if w in corpus_vocab)
    return matches / len(gen_words)


def line_structure_stats(generated: str) -> dict:
    """Line-level statistics: line count, mean line length, std line
    length. Compare to corpus to see if the model matches Shakespeare's
    typical line structure."""
    lines = [ln for ln in generated.split("\n") if ln.strip()]
    if not lines:
        return {"n_lines": 0, "mean_line_len": 0.0, "std_line_len": 0.0}
    lengths = [len(ln) for ln in lines]
    mean = sum(lengths) / len(lengths)
    var = sum((L - mean) ** 2 for L in lengths) / len(lengths)
    return {"n_lines": len(lines),
             "mean_line_len": mean,
             "std_line_len": var ** 0.5}


def vc_alternation_rate(generated: str) -> float:
    """Vowel-consonant alternation rate. English alternates v/c more
    often than random text. Returns the fraction of adjacent letter
    pairs that are (v,c) or (c,v) -- alternating, not same class."""
    letters = [c for c in generated if c in LETTERS]
    if len(letters) < 2:
        return 0.0
    alts = 0
    for i in range(len(letters) - 1):
        a, b = letters[i] in VOWELS, letters[i+1] in VOWELS
        if a != b:
            alts += 1
    return alts / (len(letters) - 1)


def line_length_match(generated: str, corpus_text: str) -> float:
    """How close is the generated line-length distribution to the
    corpus's? L1 distance over normalized histograms (lower = closer
    to Shakespeare's line structure)."""
    def hist(text, max_len=80):
        lines = [ln for ln in text.split("\n") if ln.strip()]
        h = [0] * (max_len + 1)
        for ln in lines:
            L = min(len(ln), max_len)
            h[L] += 1
        total = sum(h) or 1
        return [x / total for x in h]
    gen_h = hist(generated)
    corp_h = hist(corpus_text)
    return sum(abs(g - c) for g, c in zip(gen_h, corp_h))


def creativity_score(generated: str, corpus_text: str) -> dict:
    """Comprehensive Shakespeare-creativity score. Returns a dict of
    individual metrics and a composite creativity_score in [0, 1].
    """
    n2 = char_ngram_overlap(generated, corpus_text, 2)
    n3 = char_ngram_overlap(generated, corpus_text, 3)
    n4 = char_ngram_overlap(generated, corpus_text, 4)
    vocab = vocab_overlap(generated, corpus_text)
    vc = vc_alternation_rate(generated)
    line_dist = line_length_match(generated, corpus_text)
    line_stats = line_structure_stats(generated)
    # Composite: weighted blend, higher = more creative & Shakespeare-aligned.
    # n3 + vocab are the strongest signal; vc shows English-like; line_dist
    # is inverted because it's a distance (lower = better).
    composite = 0.35 * n3 + 0.25 * vocab + 0.15 * n2 + 0.15 * vc \
                + 0.10 * max(0.0, 1.0 - line_dist)
    return {
        "ngram_2": n2,
        "ngram_3": n3,
        "ngram_4": n4,
        "vocab_overlap": vocab,
        "vc_alternation": vc,
        "line_dist": line_dist,
        "line_stats": line_stats,
        "creativity_score": composite,
    }

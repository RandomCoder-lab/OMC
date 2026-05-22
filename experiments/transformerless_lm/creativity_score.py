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


def common_word_presence(generated: str, corpus_text: str,
                            top_k: int = 50) -> float:
    """How many of the corpus's top-K most-common words appear in the
    generated text. This is the strongest anti-gibberish signal:
    Shakespeare uses 'the', 'and', 'of', 'my', 'I' frequently;
    gibberish doesn't.
    """
    def clean(s):
        return s.lower().strip(string.punctuation)
    corpus_words = [clean(w) for w in corpus_text.split() if clean(w)]
    corpus_freq = Counter(corpus_words)
    top_words = set(w for w, _ in corpus_freq.most_common(top_k))
    gen_words = set(clean(w) for w in generated.split() if clean(w))
    if not top_words:
        return 0.0
    overlap = len(gen_words & top_words)
    return overlap / len(top_words)


def avg_word_length_match(generated: str, corpus_text: str) -> float:
    """How close is generated avg word length to corpus avg?
    Returns 1.0 - normalized_distance, clamped to [0, 1]."""
    def clean(s):
        return s.lower().strip(string.punctuation)
    def avg(text):
        words = [clean(w) for w in text.split() if clean(w)]
        return (sum(len(w) for w in words) / len(words)) if words else 0.0
    g = avg(generated); c = avg(corpus_text)
    if c == 0:
        return 0.0
    return max(0.0, 1.0 - abs(g - c) / c)


def ngram_diversity(generated: str, n: int = 3) -> float:
    """Fraction of n-grams in the generated text that are UNIQUE.
    1.0 = every n-gram appears once (max diversity).
    0.0 = all n-grams identical (max repetition).
    Counter-Goodhart against the model gaming overlap by repetition."""
    if len(generated) < n:
        return 0.0
    ngrams = [generated[i:i+n] for i in range(len(generated) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def repetition_penalty(generated: str, n: int = 4,
                         max_freq_threshold: int = 3) -> float:
    """Penalty in [0, 1] for excessive n-gram repetition. 0 = no penalty.

    For each n-gram appearing more than max_freq_threshold times, add a
    penalty proportional to the excess. Strong signal against the
    'fan fan, fan, fan' failure mode.
    """
    if len(generated) < n:
        return 0.0
    ngrams = [generated[i:i+n] for i in range(len(generated) - n + 1)]
    counts = Counter(ngrams)
    excess = sum(max(0, c - max_freq_threshold) for c in counts.values())
    # Normalize by total ngrams; cap penalty at 1.0
    return min(1.0, excess / max(1, len(ngrams)))


def lexical_diversity(generated: str) -> float:
    """Type-token ratio over 'words' (whitespace-split). Higher = more
    varied vocabulary, lower = repetitive word use."""
    import string as _s
    words = [w.lower().strip(_s.punctuation) for w in generated.split()]
    words = [w for w in words if w]
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def creativity_score(generated: str, corpus_text: str) -> dict:
    """Comprehensive Shakespeare-creativity score with anti-gibberish.

    Penalties added in v2 to counter Goodhart's failure (model gaming
    overlap metrics by repetition):
      - ngram_diversity (multiplier; low = repetitive output)
      - lexical_diversity (multiplier; low = same word over and over)
      - repetition_penalty (subtractive; n-gram appears too many times)
    """
    n2 = char_ngram_overlap(generated, corpus_text, 2)
    n3 = char_ngram_overlap(generated, corpus_text, 3)
    n4 = char_ngram_overlap(generated, corpus_text, 4)
    vocab = vocab_overlap(generated, corpus_text)
    vc = vc_alternation_rate(generated)
    line_dist = line_length_match(generated, corpus_text)
    line_stats = line_structure_stats(generated)
    # Strong anti-gibberish: common-word presence and word-length match.
    cw = common_word_presence(generated, corpus_text, top_k=50)
    awl = avg_word_length_match(generated, corpus_text)
    # Repetition penalty: only severe excess counts now (threshold scales
    # with text length so real text's natural repetition doesn't penalize).
    threshold = max(2, len(generated) // 50)
    rep_pen = repetition_penalty(generated, n=4, max_freq_threshold=threshold)

    composite = (
        0.20 * cw +              # common-word presence (anti-gibberish)
        0.25 * vocab +           # any vocab overlap (length-weighted via cw)
        0.15 * awl +             # word-length sanity
        0.20 * n3 +              # 3-gram match (corpus patterns)
        0.10 * n4 +              # 4-gram match (longer patterns)
        0.10 * max(0.0, 1.0 - line_dist)   # line structure
    ) - 0.3 * rep_pen
    composite = max(0.0, min(1.0, composite))
    return {
        "ngram_2": n2,
        "ngram_3": n3,
        "ngram_4": n4,
        "vocab_overlap": vocab,
        "common_word_presence": cw,
        "avg_word_len_match": awl,
        "vc_alternation": vc,
        "line_dist": line_dist,
        "line_stats": line_stats,
        "repetition_penalty": rep_pen,
        "creativity_score": composite,
    }

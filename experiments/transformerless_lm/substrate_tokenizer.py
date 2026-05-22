"""Substrate-aware WORD-level tokenizer (v2).

Char-frequency tokenization (v1) captured sub-word fragments ("ns",
"he") as much as real words. v2 builds a WORD-LEVEL vocabulary:

  1. Split the corpus into words (whitespace + punctuation).
  2. Sort words by frequency.
  3. Top N words become substrate tokens, ranked by Fibonacci tier.
  4. Punctuation kept as single-char tokens.
  5. Unknown words decompose into single characters (fallback).

The vocabulary's STRUCTURE is Fibonacci-tiered: position 0..F(1)-1 =
tier 0 (most common); positions F(1)..F(2)-1 = tier 1 (next); etc.
Within a tier, words are sorted by frequency. This gives the model a
vocab where token ID position carries substrate meaning.

Tokenization: longest-prefix match at word boundaries; otherwise
single-char fallback.
"""

import re
from collections import Counter
from typing import List


def _word_split(text: str) -> List[str]:
    """Split text into (word|punct|whitespace) tokens preserving order."""
    return re.findall(r"[A-Za-z']+|\d+|[^\w\s]|\s", text)


class SubstrateTokenizer:
    """Word-level Fibonacci-tier-ranked tokenizer with char fallback."""

    def __init__(self, corpus_text: str, max_vocab_size: int = 500):
        chars = sorted(set(corpus_text))
        words = [tok for tok in _word_split(corpus_text)
                  if re.fullmatch(r"[A-Za-z']+|\d+", tok)]
        word_counts = Counter(w.lower() for w in words)
        char_budget = len(chars)
        word_budget = max(0, max_vocab_size - char_budget)
        ranked_words = [w for w, _ in word_counts.most_common(word_budget)]
        self.vocab: List[str] = list(chars) + ranked_words
        self.vocab = self.vocab[:max_vocab_size]
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.multichar_tokens = set(t for t in self.vocab if len(t) > 1)
        self.lengths_desc = sorted(
            set(len(t) for t in self.multichar_tokens), reverse=True)

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        i = 0
        n = len(text)
        while i < n:
            best_len = 0
            best_id = None
            for L in self.lengths_desc:
                if i + L > n:
                    continue
                cand = text[i:i+L].lower()
                if cand in self.token_to_id and cand.isalpha():
                    # Word boundary check on both sides.
                    starts_at_boundary = (i == 0 or not text[i-1].isalpha())
                    ends_at_boundary = (i + L == n or not text[i+L].isalpha())
                    if starts_at_boundary and ends_at_boundary:
                        best_len = L
                        best_id = self.token_to_id[cand]
                        break
            if best_id is not None:
                ids.append(best_id)
                i += best_len
            else:
                ch = text[i]
                ids.append(self.token_to_id.get(ch,
                                                  self.token_to_id.get(' ', 0)))
                i += 1
        return ids

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.vocab[int(i)] for i in ids)

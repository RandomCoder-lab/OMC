"""bpe_tokenizer.py — tokenizers for the written-memory generator.

Char-level capped generation coherence (a d=96 char-LM can't write fluent code no matter how much
memory we attach). Subword BPE gives the generator real units (keywords, identifiers, indentation runs)
so each step is a meaningful chunk, not a single character.

Two tokenizers behind one interface (encode / decode / encode_with_offsets / vocab_size / to_dict):
  * CharTokenizer — the proven char-level baseline (for an honest A/B).
  * BPETokenizer  — byte-pair merges learned on the corpus (word-frequency training, the standard
                    efficient scheme). Token 0 is reserved <unk> for out-of-vocab chars at query time.

encode_with_offsets returns, per token, the source CHAR offset of its first character — so the
written-memory datastore keeps provenance (which snippet a retrieval points at) even with subwords.
"""
from __future__ import annotations
import re
from collections import Counter
from typing import List, Tuple

# Pre-tokenization: runs of whitespace (indentation!), identifier runs, or a single punctuation char.
# Keeping whitespace runs whole lets BPE learn "    " (4-space indent) etc. as single code tokens.
WORD_RE = re.compile(r"\s+|[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")


class CharTokenizer:
    kind = "char"

    def __init__(self, chars: List[str]):
        self.id2str = [""] + list(chars)          # id 0 = <unk> ("")
        self.stoi = {c: i + 1 for i, c in enumerate(chars)}

    @classmethod
    def train(cls, text: str, **_) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.id2str)

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(c, 0) for c in text]

    def encode_with_offsets(self, text: str) -> Tuple[List[int], List[int]]:
        ids = self.encode(text)
        return ids, list(range(len(text)))

    def decode(self, ids) -> str:
        return "".join(self.id2str[int(i)] if 0 <= int(i) < len(self.id2str) else "?" for i in ids)

    def to_dict(self) -> dict:
        return {"kind": "char", "chars": self.id2str[1:]}

    @classmethod
    def from_dict(cls, d: dict) -> "CharTokenizer":
        return cls(d["chars"])


class BPETokenizer:
    kind = "bpe"

    def __init__(self, id2str: List[str], merges: List[Tuple[int, int]], base_stoi: dict):
        self.id2str = id2str                       # token id -> its string expansion (id 0 = <unk> "")
        self.merges = merges                       # ordered list of (a_id, b_id)
        self.base_stoi = base_stoi                 # char -> base id
        self.rank = {pair: i for i, pair in enumerate(merges)}
        base = len(base_stoi) + 1                   # ids 1..C are base chars; merges start at C+1
        self.pair_new = {pair: base + i for i, pair in enumerate(merges)}

    @property
    def vocab_size(self) -> int:
        return len(self.id2str)

    # ── training (word-frequency BPE) ──────────────────────────────────────────
    @classmethod
    def train(cls, text: str, vocab_size: int = 2048, max_unique_words: int = 60000,
              log=lambda *_: None) -> "BPETokenizer":
        chars = sorted(set(text))
        base_stoi = {c: i + 1 for i, c in enumerate(chars)}   # 0 reserved for <unk>
        id2str = [""] + chars[:]                              # id 0 = <unk>
        # pre-tokenize -> word frequencies; represent each unique word as a list of base ids
        word_freq = Counter(WORD_RE.findall(text))
        if len(word_freq) > max_unique_words:                 # keep the most common words (cap cost)
            word_freq = Counter(dict(word_freq.most_common(max_unique_words)))
        splits = {w: [base_stoi[c] for c in w] for w in word_freq}
        freqs = {w: f for w, f in word_freq.items()}
        n_merges = max(0, vocab_size - len(id2str))
        merges: List[Tuple[int, int]] = []
        words = list(splits.keys())
        for step in range(n_merges):
            pair_freq: Counter = Counter()
            for w in words:
                syms = splits[w]
                f = freqs[w]
                for i in range(len(syms) - 1):
                    pair_freq[(syms[i], syms[i + 1])] += f
            if not pair_freq:
                break
            best, bf = pair_freq.most_common(1)[0]
            if bf < 2:                                        # nothing repeats — stop early
                break
            new_id = len(id2str)
            id2str.append(id2str[best[0]] + id2str[best[1]])
            merges.append(best)
            a, b = best
            for w in words:                                   # apply the merge to every word
                syms = splits[w]
                if len(syms) < 2:
                    continue
                out = []
                i = 0
                while i < len(syms):
                    if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                        out.append(new_id); i += 2
                    else:
                        out.append(syms[i]); i += 1
                splits[w] = out
            if step % max(1, n_merges // 5) == 0:
                log(f"[bpe]   merge {step}/{n_merges}  '{id2str[new_id]!r}'  (freq {bf})")
        log(f"[bpe] trained: vocab={len(id2str)} ({len(merges)} merges) from {len(words)} unique words")
        return cls(id2str, merges, base_stoi)

    # ── encode / decode ────────────────────────────────────────────────────────
    def _encode_word(self, word: str) -> List[int]:
        syms = [self.base_stoi.get(c, 0) for c in word]
        if len(syms) < 2:
            return syms
        while True:
            # lowest-rank adjacent pair present in this word
            best_rank = None; best_pair = None
            for i in range(len(syms) - 1):
                r = self.rank.get((syms[i], syms[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r; best_pair = (syms[i], syms[i + 1])
            if best_pair is None:
                break
            new_id = self.pair_new[best_pair]
            a, b = best_pair
            out = []; i = 0
            while i < len(syms):
                if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                    out.append(new_id); i += 2
                else:
                    out.append(syms[i]); i += 1
            syms = out
        return syms

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for m in WORD_RE.finditer(text):
            ids.extend(self._encode_word(m.group()))
        return ids

    def encode_with_offsets(self, text: str) -> Tuple[List[int], List[int]]:
        ids: List[int] = []; offs: List[int] = []
        for m in WORD_RE.finditer(text):
            start = m.start()
            within = 0
            for tid in self._encode_word(m.group()):
                ids.append(tid); offs.append(start + within)
                within += len(self.id2str[tid])
        return ids, offs

    def decode(self, ids) -> str:
        return "".join(self.id2str[int(i)] if 0 <= int(i) < len(self.id2str) else "?" for i in ids)

    def to_dict(self) -> dict:
        return {"kind": "bpe", "id2str": self.id2str, "merges": [list(p) for p in self.merges],
                "base_stoi": self.base_stoi}

    @classmethod
    def from_dict(cls, d: dict) -> "BPETokenizer":
        return cls(d["id2str"], [tuple(p) for p in d["merges"]], d["base_stoi"])


def tokenizer_from_dict(d: dict):
    return CharTokenizer.from_dict(d) if d.get("kind") == "char" else BPETokenizer.from_dict(d)


def make_tokenizer(kind: str, text: str, vocab_size: int = 2048, log=lambda *_: None):
    if kind == "char":
        return CharTokenizer.train(text)
    return BPETokenizer.train(text, vocab_size=vocab_size, log=log)


if __name__ == "__main__":
    sample = ("def add(a, b):\n    return a + b\n\ndef mul(a, b):\n    return a * b\n" * 200)
    tok = BPETokenizer.train(sample, vocab_size=300, log=print)
    ids, offs = tok.encode_with_offsets("def add(x, y):")
    print("vocab:", tok.vocab_size, "| n_tokens for 'def add(x, y):':", len(ids))
    print("round-trip OK:", tok.decode(tok.encode(sample)) == sample)
    print("tokens:", [tok.id2str[i] for i in ids])
    print("offsets:", offs)
    # compression vs chars
    full = tok.encode(sample)
    print(f"chars={len(sample)}  bpe_tokens={len(full)}  ratio={len(sample)/len(full):.2f} chars/token")

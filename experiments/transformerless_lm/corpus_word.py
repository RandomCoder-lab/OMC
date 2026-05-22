"""Word-level tokenizer for TinyShakespeare.

The char-level vocab (65 chars) requires the model to learn that
letters form words before it can learn word structure. Word-level
tokenization gives the model atomic semantic units directly — the
model's per-step prediction is a meaningful WORD, not a letter.

Splits on whitespace + punctuation. Keeps punctuation as separate
tokens (so 'ROMEO:' becomes ['ROMEO', ':']). Lowercase'd to keep
vocab small.

For TinyShakespeare (1.1 MB) the word vocab is roughly 25K unique
tokens — much larger than 65 chars but each token carries more
semantic weight per step.
"""

import os
import re

import torch


_TOKEN_PATTERN = re.compile(r"[A-Za-z]+|[0-9]+|[^A-Za-z0-9\s]|\n+|\s+")


def tokenize_text(text: str) -> list[str]:
    """Split text into word-like tokens. Keeps newlines as their own
    tokens so the model can learn line structure."""
    tokens = _TOKEN_PATTERN.findall(text)
    # Lowercase alphabetic tokens to shrink vocab. Keep punctuation as-is.
    return [t.lower() if t.isalpha() else t for t in tokens]


def make_word_dataset(source: str = "tinyshakespeare"):
    """Returns (vocab, stoi, itos, encoded) for word-level tokenization.

    vocab: list of unique tokens, sorted
    stoi: token -> int
    itos: int -> token
    encoded: 1-D int tensor of token ids
    """
    base = os.path.dirname(__file__)
    if source == "tinyshakespeare":
        path = os.path.join(base, "tinyshakespeare.txt")
    elif source == "omc":
        path = os.path.join(base, "omc_codebase.txt")
    else:
        raise ValueError(f"unknown source: {source}")
    with open(path) as f:
        text = f.read()
    tokens = tokenize_text(text)
    vocab = sorted(set(tokens))
    stoi = {t: i for i, t in enumerate(vocab)}
    itos = {i: t for t, i in stoi.items()}
    encoded = torch.tensor([stoi[t] for t in tokens], dtype=torch.long)
    return vocab, stoi, itos, encoded


def detokenize(token_ids, itos) -> str:
    """Inverse of tokenize_text. Reconstructs text by joining tokens —
    keeps newlines/whitespace tokens visible so the line structure
    is preserved in the output."""
    out = []
    prev_alpha = False
    for tid in token_ids:
        t = itos[int(tid)]
        # Add a space between alphanumeric runs; whitespace/newline
        # tokens are emitted directly.
        if t.isalnum():
            if prev_alpha:
                out.append(" ")
            out.append(t)
            prev_alpha = True
        else:
            out.append(t)
            prev_alpha = False
    return "".join(out)


if __name__ == "__main__":
    for src in ("tinyshakespeare", "omc"):
        vocab, stoi, itos, enc = make_word_dataset(src)
        print(f"{src}:")
        print(f"  total tokens: {enc.numel():,}")
        print(f"  unique vocab: {len(vocab):,}")
        sample = detokenize(enc[:30].tolist(), itos)
        print(f"  first 30 detok: {sample!r}")
        print()

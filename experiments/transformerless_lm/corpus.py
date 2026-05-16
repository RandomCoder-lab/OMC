"""Tiny corpus for the transformerless-LM bench. We hand-write a small
text rather than depend on a download — keeps the experiment fully
reproducible and fast on CPU.

The corpus is a few paragraphs of stylistically-consistent English.
The task is just "predict the next character" — a classical mini-LM
benchmark that any architecture should be able to fit. The point of
this experiment is to compare LOSS CURVES across architectures, not
to produce a useful language model.
"""

CORPUS = """\
The substrate is the architecture. Every value carries a shadow,
every shadow carries a tension, every tension is a measurement of
how far the value sits from the nearest harmonic attractor. The
attractors are Fibonacci numbers because Fibonacci is what self-
similar growth looks like when it has memory of its previous step.

The classical band carries the user-visible value. The harmonic band
carries the substrate-aligned shadow. Coherence between the two is
the signal. When coherence is high the computation is on the manifold;
when it drops, something has moved off the manifold and we should
take notice. This is the whole architecture in one paragraph.

Positions in a sequence are not just numbers. They are residues
modulo small Fibonacci attractors. By the Chinese Remainder Theorem
the residue tuple uniquely identifies the position within a window
much larger than any single modulus. This is how we encode position
without losing distinctness past the wrap of any single period.

Attention is not just similarity. It is similarity weighted by how
on-manifold the candidate is. A key that sits at a Fibonacci
attractor passes through the gate with full weight. A key that has
drifted off-manifold gets attenuated. The gate is cheap to compute
and never pays a cost when the key is on the substrate.
"""


def make_dataset(seq_len: int = 64, source: str = "embedded"):
    """Return (vocab, encoded_text) where encoded_text is a 1-D
    int tensor of token indices. Char-level vocab built from the
    corpus's unique characters.

    `source` chooses corpus:
      - "embedded": the small 1.5KB inline CORPUS (default; kept for
                    fast smoke tests and the original tiny-bench)
      - "tinyshakespeare": load tinyshakespeare.txt (1.1 MB) — used
                           by the scale experiment
    """
    import os
    import torch
    if source == "tinyshakespeare":
        path = os.path.join(os.path.dirname(__file__), "tinyshakespeare.txt")
        with open(path, "r") as f:
            text = f.read()
    else:
        text = CORPUS
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    encoded = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return chars, stoi, itos, encoded


def get_batch(encoded, batch_size: int, seq_len: int, generator=None):
    """Return (x, y) where x is [batch, seq_len] and y is the next-token
    target [batch, seq_len]. Sampled uniformly from the encoded text."""
    import torch
    n = encoded.numel()
    if generator is None:
        ix = torch.randint(0, n - seq_len - 1, (batch_size,))
    else:
        ix = torch.randint(0, n - seq_len - 1, (batch_size,), generator=generator)
    x = torch.stack([encoded[i:i + seq_len] for i in ix])
    y = torch.stack([encoded[i + 1:i + seq_len + 1] for i in ix])
    return x, y

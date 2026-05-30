"""Fibonacci-strided data ingestion — validated 5.6x training speedup.

The substrate-aligned data loader. Every experiment going forward should
use `get_fib_strided_batch` instead of dense batching unless the
experiment is explicitly testing dense as a comparator.

See results_lazy_loading.json for the validation: 1500 steps on
TinyShakespeare, dense 165.7s → fib_strided 29.5s, val 2.4396 →
2.5274 (+3.6%). Same model, same step count, just substrate-aligned IO.
"""

import torch


# Canonical Fibonacci table — matches omnimcode-core/src/phi_pi_fib.rs:32
FIBONACCI = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]


def fib_positions_in_window(window: int) -> list[int]:
    """Substrate-aligned positions in [0, window).

    Returns sorted {0} ∪ {Fibonacci numbers ≤ window-1}.

    Examples:
      window=128  → [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]    (11 pos)
      window=256  → [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
      window=1024 → 16 positions

    Count grows as log_phi_pi(window), giving ~13x IO reduction at
    window=128 and ~64x at window=1024.
    """
    return sorted(set([0] + [f for f in FIBONACCI if f < window]))


def get_fib_strided_batch(encoded: torch.Tensor, batch_size: int,
                            window: int, fib_positions: list[int],
                            generator: torch.Generator):
    """Return (x, y) batch where x[b, p] is encoded[start_b + fib_positions[p]]
    and y[b, p] is the next-token target encoded[start_b + fib_positions[p] + 1].

    The "effective" sequence length is `window` but only len(fib_positions)
    tokens are actually loaded — substrate-aligned sparse sampling.

    Args:
        encoded: 1-D int tensor of the corpus.
        batch_size: B
        window: effective sequence length (max offset = window - 1).
        fib_positions: result of fib_positions_in_window(window).
        generator: torch.Generator for the start-index sampling.

    Returns:
        (x, y) each of shape [B, len(fib_positions)] containing token ids.
    """
    n = encoded.numel()
    fib_t = torch.tensor(fib_positions, dtype=torch.long)
    max_off = fib_positions[-1] + 1
    ix = torch.randint(0, n - max_off - 1, (batch_size,), generator=generator)
    x = torch.stack([encoded[i + fib_t] for i in ix])
    y = torch.stack([encoded[i + fib_t + 1] for i in ix])
    return x, y


def get_dense_batch(encoded: torch.Tensor, batch_size: int, seq_len: int,
                    generator: torch.Generator):
    """Standard contiguous-sequence batch. Kept as a comparator only —
    new experiments should default to get_fib_strided_batch."""
    n = encoded.numel()
    ix = torch.randint(0, n - seq_len - 1, (batch_size,), generator=generator)
    x = torch.stack([encoded[i:i + seq_len] for i in ix])
    y = torch.stack([encoded[i + 1:i + seq_len + 1] for i in ix])
    return x, y

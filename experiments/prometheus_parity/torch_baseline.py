"""PyTorch baseline for the Prometheus tinyLM bigram task.

Same architecture, same training loop, same seed → measure how close
Prometheus' pure-OMC training matches PyTorch's hand-optimized loop.

The point isn't to BEAT PyTorch — it's to prove parity: same task,
same model, identical math, similar numbers. That's what makes
Prometheus a real framework instead of a toy.

Setup mirrors examples/prometheus_tinylm.omc exactly:
  vocab = 3 (a/b/c)
  hidden = 8
  architecture: Linear(3,8) → ReLU → Linear(8,3)
  loss: MSE against one-hot target
  optimizer: SGD lr=0.05
  steps: 200
  initialization: rng seed 42, Xavier-uniform bound 0.5
  data: bigram cycle "abcabcabc..." (26 train pairs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_corpus():
    text = "abcabcabcabcabcabcabcabcabc"
    ids = [{"a": 0, "b": 1, "c": 2}[ch] for ch in text]
    return ids


def lcg(state):
    """Same LCG Prometheus uses — same init weights when same seed."""
    return (state * 1103515245 + 12345) % 2147483648


def make_matrix(rows, cols, bound, state):
    """Match _prom_random_matrix from prometheus.omc byte-for-byte."""
    m = torch.empty(rows, cols)
    s = state
    for i in range(rows):
        for j in range(cols):
            s = lcg(s)
            r = s / 2147483648.0
            m[i, j] = (r * 2.0 - 1.0) * bound
    return m, s


class TinyLM(nn.Module):
    def __init__(self, vocab, hidden, seed):
        super().__init__()
        W1, s = make_matrix(vocab, hidden, 0.5, seed)
        W2, _ = make_matrix(hidden, vocab, 0.5, s)
        # Match Prometheus' "b is 1 x out_dim" shape.
        self.W1 = nn.Parameter(W1)
        self.b1 = nn.Parameter(torch.zeros(1, hidden))
        self.W2 = nn.Parameter(W2)
        self.b2 = nn.Parameter(torch.zeros(1, vocab))

    def forward(self, x):
        h = F.relu(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2


def one_hot(idx, vocab):
    v = torch.zeros(1, vocab)
    v[0, idx] = 1.0
    return v


def main():
    ids = make_corpus()
    vocab = 3
    hidden = 8
    n_pairs = len(ids) - 1

    model = TinyLM(vocab, hidden, seed=42)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    tail_losses = []
    for step in range(200):
        k = step % n_pairs
        x = one_hot(ids[k], vocab)
        target = one_hot(ids[k + 1], vocab)
        pred = model(x)
        loss = F.mse_loss(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step >= 180:
            tail_losses.append(loss.item())

    final_loss = sum(tail_losses) / len(tail_losses)

    # Predictions
    chars = ["a", "b", "c"]
    print("=== PyTorch baseline (same architecture, same task) ===")
    print(f"  final tail-mean loss: {final_loss:.6f}")
    model.eval()
    with torch.no_grad():
        for c in range(vocab):
            x = one_hot(c, vocab)
            logits = model(x)
            pred_idx = int(logits.argmax(dim=-1).item())
            expected = (c + 1) % vocab
            ok = "ok" if pred_idx == expected else "x"
            print(f"  {chars[c]} -> {chars[pred_idx]}  (expected {chars[expected]}) {ok}")


if __name__ == "__main__":
    main()

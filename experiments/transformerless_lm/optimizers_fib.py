"""Fibonacci-momentum optimizer — substrate-canonical SGD.

The golden ratio φ ≈ 1.618 is the fixed-point ratio of the Fibonacci
recurrence F(n)/F(n-1). Standard momentum-SGD uses a momentum
coefficient β (usually 0.9). Fibonacci-momentum uses β = 1/φ ≈ 0.618:

  v_{t+1} = (1/φ) · v_t + grad
  W_{t+1} = W_t - lr · v_{t+1}

The momentum decay matches the substrate's canonical contraction
ratio. Whether this gives a meaningful training advantage over
standard β=0.9 is an empirical question.
"""

import math
import torch
from torch.optim import Optimizer


PHI = (1 + math.sqrt(5)) / 2


class FibonacciMomentumSGD(Optimizer):
    """SGD with golden-ratio momentum β = 1/φ ≈ 0.618."""

    def __init__(self, params, lr=3e-4, weight_decay=0.0,
                 beta: float = 1.0 / PHI):
        defaults = dict(lr=lr, weight_decay=weight_decay, beta=beta)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            beta = group["beta"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0:
                    g = g.add(p, alpha=wd)
                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p)
                buf = state["momentum"]
                buf.mul_(beta).add_(g)
                p.add_(buf, alpha=-lr)
        return loss


class FibonacciAdamW(Optimizer):
    """AdamW with golden-ratio first-moment decay and Fibonacci-spaced
    epsilon. β1 = 1/φ ≈ 0.618 instead of standard 0.9. β2 = 1/φ²
    ≈ 0.382 instead of 0.999.

    The substrate intuition: the moment estimates should DECAY at the
    substrate's contraction ratio, matching the geometric structure
    of the gradient signal in a substrate-aligned optimization.
    """

    def __init__(self, params, lr=3e-4, beta1=1.0/PHI, beta2=1.0/(PHI**2),
                 eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1 = group["beta1"]
            b2 = group["beta2"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                m, v = state["m"], state["v"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                # Bias-corrected
                m_hat = m / (1 - b1 ** t)
                v_hat = v / (1 - b2 ** t)
                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
        return loss

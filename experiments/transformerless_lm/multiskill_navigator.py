"""Multi-Skill Address Navigator.

N weight matrices (skills) live at addresses in the same geometric space
as text and tools. A learned router selects a soft mixture of skills based
on the current context fingerprint.

  forward(fp_cur) → fp_pred   (same API as AddressNavigator)

Skill emergence: windows from similar contexts route to the same W_i.
After training, different faces of the dodecahedron specialise to different
skills — the geometry is doing the routing.

Parameters (d=64, n=4):
  W_bank:  4 × 64 × 64  = 16,384
  router:  4 × 64       =    256
  Total:                   16,640
"""

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from corpus_address_index import substrate_fingerprint
from hierarchical_address import HAddr, assign_address, addr_str


class MultiskillNavigator(nn.Module):
    """Soft mixture-of-skills address navigator.

    Each skill i is a d×d weight matrix W_i that lives at a fixed address
    skill_fps[i] in the same geometric space as text and tools.  The router
    outputs a softmax distribution over skills; the effective weight matrix
    for a batch item is the weighted average of all W_i.

    Warm-start: router initialises to zero weights → uniform softmax →
    W_avg = identity.  Identical to AddressNavigator at step 0.
    """

    def __init__(self, d_model: int = 64, n_skills: int = 4):
        super().__init__()
        self.d_model  = d_model
        self.n_skills = n_skills

        # Skill weight matrices — identity + small noise to break symmetry
        W_init = (torch.eye(d_model).unsqueeze(0).expand(n_skills, -1, -1).clone()
                  + 0.05 * torch.randn(n_skills, d_model, d_model))
        self.W_bank = nn.Parameter(W_init)                    # [n, d, d]

        # Router: small random init so skills differentiate from step 0
        self.router = nn.Linear(d_model, n_skills, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)

        # Fixed skill addresses — fingerprints of sentinel strings
        skill_fps = torch.stack([
            substrate_fingerprint(f"__skill_{i}__", d_model)
            for i in range(n_skills)
        ])
        self.register_buffer('skill_fps', skill_fps)          # [n, d]

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, fp: torch.Tensor) -> torch.Tensor:
        """fp: [B, d] → [B, d] predicted next fingerprint."""
        weights = F.softmax(self.router(fp), dim=-1)               # [B, n]
        W_avg   = torch.einsum('bn,nij->bij', weights, self.W_bank)  # [B, d, d]
        return torch.bmm(W_avg, fp.unsqueeze(-1)).squeeze(-1)      # [B, d]

    # ── skill inspection ──────────────────────────────────────────────────────

    @torch.no_grad()
    def skill_weights(self, fp: torch.Tensor) -> torch.Tensor:
        """Soft skill distribution for a single fingerprint. [n_skills]"""
        w = fp if fp.dim() == 2 else fp.unsqueeze(0)
        return F.softmax(self.router(w), dim=-1).squeeze(0)

    @torch.no_grad()
    def dominant_skill(self, fp: torch.Tensor) -> int:
        """Index of the most-activated skill for a single fingerprint."""
        return int(self.skill_weights(fp).argmax().item())

    def skill_addr(self, i: int) -> HAddr:
        """Fixed dodecahedral address of skill i."""
        return assign_address(self.skill_fps[i])

    def skill_summary(self) -> str:
        lines = [f"MultiskillNavigator: d={self.d_model}  n={self.n_skills}"]
        for i in range(self.n_skills):
            a = self.skill_addr(i)
            w_norm = self.W_bank[i].norm().item()
            lines.append(f"  skill {i}: addr={addr_str(a)}  |W|={w_norm:.3f}")
        return '\n'.join(lines)

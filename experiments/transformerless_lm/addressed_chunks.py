"""AddressedChunkRegistry — hierarchical Fibonacci tokenization from bloom runs.

Chunks are discovered purely from the model's own output — no dictionary, no
word list. Mechanism works universally across any corpus.

Hierarchy:
  Level 1 — BLOOM T-tier runs detected in generated samples (8–55 chars)
  Level 2 — Two L1 chunks that reliably follow each other → merged super-chunk
  Level 3 — Two L2 chunks that reliably follow each other → even larger
  ...
  Level N — Fibonacci sum: T13+T8=T21, T21+T13=T34, T34+T21=T55, ... → ∞

Each chunk gets a HAddr from the mean of its constituent char/chunk embeddings.
The model emits a chunk token → entire string outputs in one step.

No hard cap on chunk count or size. Fibonacci scales infinitely.
Natural ceiling: model parameter budget (each chunk = 2×d params).

Merging rule:
  Two promoted chunks A, B form a merge candidate when the sequence (tok_A, tok_B)
  appears in ≥ MIN_PAIR_COUNT distinct samples. The merged chunk AB:
    string    = A.string + B.string
    embedding = length-weighted mean(A.embed, B.embed)
    tier      = A.tier + B.tier  (Fibonacci sum)
    addr      = assign_address(merged_embedding)
"""

from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch

from hierarchical_address import HAddr, assign_address, addr_str

MIN_CHUNK_LEN    = 8     # F[6] — minimum chars for L1 chunk
MIN_CHUNK_COUNT  = 2     # appearances before L1 promotion
MIN_PAIR_COUNT   = 2     # consecutive co-occurrences before merge
PARAM_BUDGET_FRAC = 0.40 # stop promoting when chunk params > 40% of base model params


class _ChunkNode:
    """Internal record for one promoted chunk at any level."""
    __slots__ = ('string', 'tier', 'token_idx', 'embed', 'addr',
                 'level', 'left_idx', 'right_idx')

    def __init__(
        self,
        string:    str,
        tier:      int,
        token_idx: int,
        embed:     torch.Tensor,   # [d] cpu float
        addr:      HAddr,
        level:     int = 1,
        left_idx:  Optional[int] = None,   # token idx of left child
        right_idx: Optional[int] = None,   # token idx of right child
    ):
        self.string    = string
        self.tier      = tier
        self.token_idx = token_idx
        self.embed     = embed
        self.addr      = addr
        self.level     = level
        self.left_idx  = left_idx
        self.right_idx = right_idx


class AddressedChunkRegistry:
    """Hierarchical addressed chunk vocabulary, growing unboundedly via Fibonacci merges.

    Usage::

        reg = AddressedChunkRegistry(base_vocab_size=65, model_base_params=321724, d=64)

        # Per bloom-run (L1 observation):
        reg.observe_run(run_text, embed_weight, vocab)

        # Per generated sample (pair observation for merging):
        reg.observe_sequence(token_seq)    # seq is list[int]

        # Per cycle — promote L1 candidates then merge candidates:
        for chunk_str, embed, addr in reg.l1_candidates(embed_weight, vocab):
            idx, bigram = add_bloom_token(model, embed, vocab, chunk_str, bigram)
            reg.promote_l1(chunk_str, idx, embed, addr)

        for left_idx, right_idx, merged_str, merged_embed, merged_addr in reg.merge_candidates():
            idx, bigram = add_bloom_token(model, merged_embed, vocab, merged_str, bigram)
            reg.promote_merge(left_idx, right_idx, idx, merged_str, merged_embed, merged_addr)

        # Decoding:
        text = reg.decode_sequence(token_seq, base_itos)
    """

    def __init__(
        self,
        base_vocab_size:   int = 65,
        model_base_params: int = 321724,
        d:                 int = 64,
    ) -> None:
        self._base_vocab_size   = base_vocab_size
        self._param_budget      = int(model_base_params * PARAM_BUDGET_FRAC)
        self._d                 = d

        # L1 tracking
        self._run_counts:  Counter          = Counter()   # run_str → count
        self._run_embeds:  Dict[str, torch.Tensor] = {}  # run_str → mean embed [d]
        self._run_addrs:   Dict[str, HAddr] = {}

        # All promoted chunks (any level), keyed by token_idx
        self._nodes:       Dict[int, _ChunkNode] = {}    # token_idx → _ChunkNode
        self._str_to_idx:  Dict[str, int]  = {}          # chunk_str → token_idx

        # Pair tracking for merging
        self._pair_counts: Counter = Counter()   # (left_tok_idx, right_tok_idx) → count

    # ── Parameter budget ──────────────────────────────────────────────────────

    def _over_budget(self) -> bool:
        # Each chunk token = 2 × d params (embed row + head row)
        return len(self._nodes) * 2 * self._d >= self._param_budget

    def n_promoted(self) -> int:
        return len(self._nodes)

    def chunk_params(self) -> int:
        return len(self._nodes) * 2 * self._d

    # ── L1 observation (single bloom run) ────────────────────────────────────

    def observe_run(
        self,
        run_str:      str,
        embed_weight: torch.Tensor,   # [V, d] cpu float
        vocab:        list,
    ) -> None:
        """Record a BLOOM T-tier run as an L1 chunk candidate."""
        if len(run_str) < MIN_CHUNK_LEN:
            return
        if run_str in self._str_to_idx:
            return  # already promoted
        self._run_counts[run_str] += 1
        # Defer embedding until count reaches threshold — safe to scan millions of candidates
        if self._run_counts[run_str] >= MIN_CHUNK_COUNT and run_str not in self._run_embeds:
            emb = self._char_mean_embed(run_str, embed_weight, vocab)
            if emb is not None:
                self._run_embeds[run_str] = emb
                self._run_addrs[run_str]  = assign_address(emb)

    def _char_mean_embed(
        self,
        text:         str,
        embed_weight: torch.Tensor,
        vocab:        list,
    ) -> Optional[torch.Tensor]:
        """Mean of character embeddings for every char in text."""
        with torch.no_grad():
            indices = []
            for ch in text:
                try:
                    idx = vocab.index(ch)
                    if idx < embed_weight.shape[0]:
                        indices.append(idx)
                except ValueError:
                    pass
            if not indices:
                return None
            return embed_weight[indices].float().mean(dim=0)

    # ── Pair observation (adjacent chunk tokens in output) ────────────────────

    def observe_sequence(self, token_seq: List[int]) -> None:
        """Scan a generated token sequence for adjacent promoted-chunk pairs."""
        for i in range(len(token_seq) - 1):
            a, b = token_seq[i], token_seq[i + 1]
            if a in self._nodes and b in self._nodes:
                self._pair_counts[(a, b)] += 1

    # ── Candidate queries ─────────────────────────────────────────────────────

    def l1_candidates(
        self,
        embed_weight: torch.Tensor,
        vocab:        list,
    ) -> List[Tuple[str, torch.Tensor, HAddr]]:
        """L1 runs ready for promotion: count ≥ threshold, not yet promoted, budget ok."""
        if self._over_budget():
            return []
        out = []
        for run_str, cnt in self._run_counts.items():
            if cnt < MIN_CHUNK_COUNT:
                continue
            if run_str in self._str_to_idx:
                continue
            # Refresh embed with current weights
            emb = self._char_mean_embed(run_str, embed_weight, vocab)
            if emb is None:
                continue
            self._run_embeds[run_str] = emb
            self._run_addrs[run_str]  = assign_address(emb)
            out.append((run_str, emb, self._run_addrs[run_str]))
        out.sort(key=lambda x: -self._run_counts[x[0]])
        return out

    def merge_candidates(
        self,
    ) -> List[Tuple[int, int, str, torch.Tensor, HAddr]]:
        """Chunk pairs ready to merge: co-occur ≥ MIN_PAIR_COUNT, budget ok.

        Returns list of (left_tok_idx, right_tok_idx, merged_str, merged_embed, merged_addr).
        """
        if self._over_budget():
            return []
        out = []
        for (a_idx, b_idx), cnt in self._pair_counts.items():
            if cnt < MIN_PAIR_COUNT:
                continue
            if a_idx not in self._nodes or b_idx not in self._nodes:
                continue
            node_a = self._nodes[a_idx]
            node_b = self._nodes[b_idx]
            merged_str = node_a.string + node_b.string
            if merged_str in self._str_to_idx:
                continue  # already promoted
            # Length-weighted mean embedding
            la, lb = len(node_a.string), len(node_b.string)
            merged_embed = (node_a.embed * la + node_b.embed * lb) / (la + lb)
            merged_addr  = assign_address(merged_embed)
            merged_tier  = node_a.tier + node_b.tier  # Fibonacci sum
            out.append((a_idx, b_idx, merged_str, merged_embed, merged_addr, merged_tier))
        # Longest first — maximise coverage per promotion slot
        out.sort(key=lambda x: -len(x[2]))
        return [(a, b, s, e, addr) for a, b, s, e, addr, _ in out]

    # ── Promotion ─────────────────────────────────────────────────────────────

    def promote_l1(
        self,
        run_str:   str,
        token_idx: int,
        embed:     torch.Tensor,
        addr:      HAddr,
        tier:      int = 13,   # default T13
    ) -> None:
        node = _ChunkNode(
            string=run_str, tier=tier, token_idx=token_idx,
            embed=embed.cpu(), addr=addr, level=1,
        )
        self._nodes[token_idx]     = node
        self._str_to_idx[run_str]  = token_idx

    def promote_merge(
        self,
        left_idx:    int,
        right_idx:   int,
        token_idx:   int,
        merged_str:  str,
        merged_embed: torch.Tensor,
        merged_addr: HAddr,
    ) -> None:
        node_a = self._nodes[left_idx]
        node_b = self._nodes[right_idx]
        merged_tier  = node_a.tier + node_b.tier
        merged_level = max(node_a.level, node_b.level) + 1
        node = _ChunkNode(
            string=merged_str, tier=merged_tier, token_idx=token_idx,
            embed=merged_embed.cpu(), addr=merged_addr,
            level=merged_level, left_idx=left_idx, right_idx=right_idx,
        )
        self._nodes[token_idx]       = node
        self._str_to_idx[merged_str] = token_idx

    # ── Decoding ──────────────────────────────────────────────────────────────

    def decode_token(self, token_idx: int) -> Optional[str]:
        node = self._nodes.get(token_idx)
        return node.string if node else None

    def is_chunk_token(self, token_idx: int) -> bool:
        return token_idx in self._nodes

    def decode_sequence(self, token_seq: List[int], base_itos: dict) -> str:
        """Decode a full token sequence, expanding chunk tokens to their strings."""
        parts = []
        for idx in token_seq:
            node = self._nodes.get(idx)
            if node is not None:
                parts.append(node.string)
            else:
                parts.append(base_itos.get(idx, '?'))
        return ''.join(parts)

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        n_l1  = sum(1 for n in self._nodes.values() if n.level == 1)
        n_l2p = sum(1 for n in self._nodes.values() if n.level >= 2)
        n_cands = sum(1 for c in self._run_counts.values() if c >= MIN_CHUNK_COUNT)
        n_pairs = sum(1 for c in self._pair_counts.values() if c >= MIN_PAIR_COUNT)
        max_level = max((n.level for n in self._nodes.values()), default=0)
        max_tier  = max((n.tier  for n in self._nodes.values()), default=0)
        max_len   = max((len(n.string) for n in self._nodes.values()), default=0)
        lines = [
            f"[CHUNKS] promoted={self.n_promoted()}  "
            f"L1={n_l1}  L2+={n_l2p}  max_level={max_level}  "
            f"max_tier=T{max_tier}  max_len={max_len}chars  "
            f"chunk_params={self.chunk_params()}  "
            f"l1_cands={n_cands}  merge_cands={n_pairs}"
        ]
        # Show up to 5 recent promotions, longest first
        recent = sorted(self._nodes.values(), key=lambda n: -len(n.string))[:5]
        for node in recent:
            lines.append(
                f"  L{node.level} T{node.tier}  tok={node.token_idx}"
                f"  @ {addr_str(node.addr)}  {repr(node.string[:60])}"
            )
        return '\n'.join(lines)

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        nodes_out = {}
        for idx, n in self._nodes.items():
            nodes_out[idx] = {
                'string':    n.string,
                'tier':      n.tier,
                'level':     n.level,
                'left_idx':  n.left_idx,
                'right_idx': n.right_idx,
                'addr':      (n.addr.face, n.addr.sub_face, sorted(n.addr.zeck)),
            }
        return {
            'run_counts':       dict(self._run_counts),
            'pair_counts':      {f"{a},{b}": c for (a, b), c in self._pair_counts.items()},
            'nodes':            nodes_out,
            'str_to_idx':       self._str_to_idx,
            'base_vocab_size':  self._base_vocab_size,
            'param_budget':     self._param_budget,
            'd':                self._d,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'AddressedChunkRegistry':
        reg = cls(
            base_vocab_size=d.get('base_vocab_size', 65),
            model_base_params=0,   # budget restored directly
            d=d.get('d', 64),
        )
        reg._param_budget = d.get('param_budget', int(321724 * PARAM_BUDGET_FRAC))
        reg._run_counts   = Counter(d.get('run_counts', {}))
        raw_pairs = d.get('pair_counts', {})
        for key, cnt in raw_pairs.items():
            a, b = map(int, key.split(','))
            reg._pair_counts[(a, b)] = cnt
        reg._str_to_idx   = {k: int(v) for k, v in d.get('str_to_idx', {}).items()}

        from hierarchical_address import HAddr
        for idx_str, nd in d.get('nodes', {}).items():
            idx  = int(idx_str)
            face, sub_face, zeck_list = nd['addr']
            addr = HAddr(face=face, sub_face=sub_face, zeck=frozenset(zeck_list))
            node = _ChunkNode(
                string=nd['string'], tier=nd['tier'], token_idx=idx,
                embed=torch.zeros(reg._d),   # embed not serialized; recomputed on first use
                addr=addr, level=nd['level'],
                left_idx=nd.get('left_idx'), right_idx=nd.get('right_idx'),
            )
            reg._nodes[idx] = node
        return reg

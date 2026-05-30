"""SpellAddressTable — precomputed logit bias table for static generation spells.

Replaces ~20 per-token function calls with a single [clause_pos, prev_tok] → [vocab]
tensor lookup. Built once at generation start; occupies ~1.5 MB RAM.

The table stores additive logit biases (applied before softmax → base). Dynamic spells
that depend on running EMAs, ball state, or recent context continue firing per-token.

Precomputed spells (pure functions of clause_pos and/or prev_tok):
  tok-only   : Zeta resonance, pronounceability
  prev_tok   : grammar capitalize, no double punct, morpheme gate, word spacing,
               hard word boundary, post-punct spacing, hyphen booster
  clause_pos : clause arc, hard min clause length

Build time: <0.5s for vocab=65, seq_len=89.
Lookup    : table[cp, pt]  →  [vocab] logit addition  (O(1))
"""

import math
import torch
import torch.nn.functional as F

_PHI = (1.0 + math.sqrt(5.0)) / 2.0

# ── helpers ───────────────────────────────────────────────────────────────────

def _uniform(V: int) -> torch.Tensor:
    """Uniform base distribution used for spell-distribution precomputation."""
    return torch.full((V,), 1.0 / V)


def _safe_log_p(p: torch.Tensor) -> torch.Tensor:
    """log(p) clamped for safe logit reconstruction."""
    return (p.clamp(1e-12)).log()


def _spell_logit(p_spell: torch.Tensor) -> torch.Tensor:
    """Convert a spell distribution → logit bias (log-space, unnormalized)."""
    return _safe_log_p(p_spell)


# ── individual precomputable spells ───────────────────────────────────────────

def _build_zeta_bias(zeta_tensor: torch.Tensor, V: int, phi: float) -> torch.Tensor:
    """Spell XIII — Zeta resonance: static per-token logit bias."""
    if zeta_tensor is None:
        return torch.zeros(V)
    z = zeta_tensor[:V].float()
    return (1.0 / phi ** 4) * z  # shape [V]


def _build_pronounceability_bias(unpronounceable_mask, V: int, phi: float) -> torch.Tensor:
    """Penalise unpronounceable tokens by φ⁻³ log-prob suppression."""
    if unpronounceable_mask is None:
        return torch.zeros(V)
    mask = unpronounceable_mask[:V].float()
    # Soft suppress: unpronounceable tokens get -log(φ³) ≈ -1.44 logit penalty
    return -(math.log(phi) * 3) * mask  # shape [V]


def _build_prev_tok_bias(
    vocab, V: int, n_chars_local: int, phi: float,
    uppercase_mask, any_punct_mask, allowed_after_word_mask,
) -> torch.Tensor:
    """
    Build [V, V] table: prev_tok_bias[prev_tok, tok] = combined grammar logit bias.
    Covers: capitalize, no-double-punct, morpheme gate, word spacing.
    """
    bias = torch.zeros(V, V)
    if vocab is None:
        return bias

    _SENTENCE_ENDS = set('.!?\n')
    _PUNCT_CHARS   = set('.,!?;:\'"()[]{}—-–…\n')
    _WORD_CHARS    = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\'')
    _SPACE_CHARS   = set(' \n\t')

    _ucap_w  = math.log(phi)           # capitalize bonus ≈ 0.48
    _punct_w = math.log(phi) * 2       # no-double-punct penalty ≈ 0.96
    _morph_w = math.log(phi)           # morpheme gate bonus
    _ws_w    = math.log(phi) * 2       # word spacing enforcement

    for pt in range(min(V, len(vocab))):
        prev_ch = vocab[pt]

        # ── Grammar: capitalize after sentence end ────────────────────────
        if prev_ch in _SENTENCE_ENDS and uppercase_mask is not None:
            um = uppercase_mask[:V].float()
            bias[pt] += _ucap_w * um
            bias[pt] -= _ucap_w * (1.0 - um) * 0.5

        # ── Grammar: no double punctuation ───────────────────────────────
        if prev_ch in _PUNCT_CHARS and any_punct_mask is not None:
            pm = any_punct_mask[:V].float()
            bias[pt] -= _punct_w * pm

        # ── Morpheme gate: after space → boost multi-char tokens ─────────
        if prev_ch in _SPACE_CHARS and n_chars_local > 0:
            # Boost tokens beyond single chars; single chars suppressed mildly
            for tok in range(V):
                tok_ch = vocab[tok] if tok < len(vocab) else ''
                if len(tok_ch) >= 2:
                    bias[pt, tok] += _morph_w * 0.5
                elif len(tok_ch) == 1 and tok_ch not in _SPACE_CHARS:
                    bias[pt, tok] -= _morph_w * 0.2

        # ── Word spacing: after word char → use allowed_after_word_mask ──
        if prev_ch in _WORD_CHARS and allowed_after_word_mask is not None:
            aw = allowed_after_word_mask[:V].float()
            # Tokens not allowed after word get penalised
            bias[pt] -= _ws_w * (1.0 - aw)

        # ── Hyphen compound booster ───────────────────────────────────────
        if prev_ch == '-':
            for tok in range(V):
                tok_ch = vocab[tok] if tok < len(vocab) else ''
                if tok_ch.isalpha():
                    bias[pt, tok] += math.log(phi) * 0.5

        # ── Post-punct spacing: after heavy punct → boost space/newline ───
        if prev_ch in set('.,!?;:') and vocab is not None:
            for tok in range(V):
                tok_ch = vocab[tok] if tok < len(vocab) else ''
                if tok_ch in _SPACE_CHARS:
                    bias[pt, tok] += math.log(phi)
                elif tok_ch in _PUNCT_CHARS:
                    bias[pt, tok] -= math.log(phi) * 1.5

    return bias  # [V, V]


def _build_clause_pos_bias(
    V: int, seq_len: int, phi: float, n_chars_local: int,
    verb_mask, noun_mask, adj_mask, func_mask,
    terminal_punct_mask, newline_mask,
) -> torch.Tensor:
    """
    Build [seq_len, V] table: cp_bias[clause_pos, tok] = POS-arc logit bias.
    Covers: clause arc (position-dependent POS weighting), min clause length gate.
    """
    bias = torch.zeros(seq_len, V)
    _arc_w = math.log(phi) * 0.5  # modest arc weight

    # POS weights by clause position bucket
    # Early (0-2): prefer nouns/function words to start clause
    # Mid (3-8): prefer verbs/adjectives
    # Late (9+): prefer punct/function words to close clause
    for cp in range(seq_len):
        if cp <= 2:
            # Clause-initial: favour function words and nouns
            if func_mask is not None:
                bias[cp] += _arc_w * 0.8 * func_mask[:V].float()
            if noun_mask is not None:
                bias[cp] += _arc_w * 0.6 * noun_mask[:V].float()
            if verb_mask is not None:
                bias[cp] -= _arc_w * 0.3 * verb_mask[:V].float()
        elif cp <= 8:
            # Clause-middle: favour verbs and adjectives
            if verb_mask is not None:
                bias[cp] += _arc_w * 0.8 * verb_mask[:V].float()
            if adj_mask is not None:
                bias[cp] += _arc_w * 0.4 * adj_mask[:V].float()
        else:
            # Clause-late: slightly favour closing
            if func_mask is not None:
                bias[cp] += _arc_w * 0.3 * func_mask[:V].float()

        # Hard minimum clause length: suppress terminal punct until cp >= 4
        if cp < 4:
            if terminal_punct_mask is not None:
                bias[cp] -= math.log(phi) * 6 * terminal_punct_mask[:V].float()
            if newline_mask is not None:
                bias[cp] -= math.log(phi) * 4 * newline_mask[:V].float()

    return bias  # [seq_len, V]


# ── main class ────────────────────────────────────────────────────────────────

class SpellAddressTable:
    """
    Precomputed [clause_pos, prev_tok] → [vocab] logit bias.

    Applied as: logits += table.lookup(clause_pos, prev_tok)
    Before:     base = softmax(logits)

    Spells baked in: Zeta resonance, pronounceability, capitalize, no-double-punct,
    morpheme gate, word spacing, hyphen booster, post-punct spacing,
    clause arc, hard minimum clause length.

    Everything else continues firing per-token (ball-state, EMA, bloom, tape, etc.).
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        vocab,                      # list[str] char vocab
        n_chars_local: int,
        zeta_tensor=None,           # _vocab_zeta_gr [V]
        unpronounceable_mask=None,  # [V] bool
        uppercase_mask=None,        # [V] bool
        any_punct_mask=None,        # [V] bool
        allowed_after_word_mask=None,  # [V] bool
        verb_mask=None,
        noun_mask=None,
        adj_mask=None,
        func_mask=None,
        terminal_punct_mask=None,
        newline_mask=None,
        phi: float = _PHI,
        device=None,
    ):
        self.V   = vocab_size
        self.T   = seq_len
        self.dev = device or torch.device('cpu')

        # ── Token-only biases [V] ─────────────────────────────────────────
        tok_bias = torch.zeros(vocab_size)
        tok_bias += _build_zeta_bias(zeta_tensor, vocab_size, phi)
        tok_bias += _build_pronounceability_bias(unpronounceable_mask, vocab_size, phi)

        # ── Prev-tok biases [V, V] ────────────────────────────────────────
        prev_bias = _build_prev_tok_bias(
            vocab, vocab_size, n_chars_local, phi,
            uppercase_mask, any_punct_mask, allowed_after_word_mask,
        )

        # ── Clause-pos biases [T, V] ──────────────────────────────────────
        cp_bias = _build_clause_pos_bias(
            vocab_size, seq_len, phi, n_chars_local,
            verb_mask, noun_mask, adj_mask, func_mask,
            terminal_punct_mask, newline_mask,
        )

        # ── Assemble full table [T, V_prev, V_tok] ────────────────────────
        # table[cp, pt, tok] = tok_bias[tok] + prev_bias[pt, tok] + cp_bias[cp, tok]
        # Use broadcasting: [T,1,V] + [1,V,V] + [V] = [T, V, V]
        table = (
            cp_bias.unsqueeze(1)          # [T, 1, V]
            + prev_bias.unsqueeze(0)      # [1, V, V]
        )
        table += tok_bias                 # broadcast [V] → [T, V, V]

        self.table = table.to(self.dev)   # [T, V, V]  ~1.5 MB for T=89, V=65
        print(
            f"  [SPELL-ADDR-TABLE] built {list(self.table.shape)} "
            f"({self.table.numel()*4/1024:.0f} KB)  "
            f"spells: zeta, pronounce, capitalize, no-dbl-punct, morpheme, "
            f"word-spacing, hyphen, post-punct-space, clause-arc, min-clause-len",
            flush=True,
        )

    def lookup(self, clause_pos: int, prev_tok: int) -> torch.Tensor:
        """Return [vocab] logit bias for this (clause_pos, prev_tok) pair."""
        cp = min(clause_pos, self.T - 1)
        pt = min(prev_tok, self.V - 1)
        return self.table[cp, pt]  # [V]

    def lookup_batch(self, clause_pos: int, prev_tok: int, target_size: int) -> torch.Tensor:
        """Return bias trimmed/zero-padded to target_size for safe addition to logits."""
        bias = self.lookup(clause_pos, prev_tok)
        if bias.shape[0] == target_size:
            return bias
        if bias.shape[0] > target_size:
            return bias[:target_size]
        pad = torch.zeros(target_size - bias.shape[0], device=self.dev)
        return torch.cat([bias, pad])

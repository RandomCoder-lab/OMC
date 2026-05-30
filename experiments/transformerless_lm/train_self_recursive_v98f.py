"""Phase 2: substrate self-recursion training.

Builds on the locked substrate stack (Subsim attn + V2 activation +
FibGen weights + FibAdamW + ce_fft + K-shrink + FibRecLM depth). Adds
a SELF-HARMONY loss that requires no target -- the substrate's own
canonical Fibonacci-frequency decay pattern serves as the prior.

Three arms (all use the substrate stack, train on a TINY ~1k-char
Shakespeare seed):

  tiny_baseline       CE + ce_fft only (no self-recursion)
  tiny_with_harmony   CE + ce_fft + lambda * substrate_harmony_loss
  tiny_self_recursive Interleave supervised (CE on seed) with
                       self-generation + harmony scoring on model's
                       own output. Model generates, scores its own
                       harmony, backprops -- no external label needed.

Hypothesis: with tiny data, the substrate prior (Fibonacci-tier
decay) fills in what the data can't teach. Harmony regularizer
should reduce held-out val. Self-recursion should match or beat
even the harmony regularizer.

Compare against tiny_baseline_gelu (vanilla transformer block with
same data budget) to measure substrate's data-efficiency gain.
"""

import argparse
import json
import sys
import time
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from corpus import make_dataset
from substrate_tokenizer import SubstrateTokenizer
from models_fibrec import FibRecLM, stateless_fibgen_forward
from optimizers_fib import FibonacciAdamW
from lazy_data import fib_positions_in_window, get_fib_strided_batch
from train_K_shrink import K_schedule_tier_walk, set_K_active_recursive
from losses_substrate import (substrate_fft_loss, substrate_harmony_loss,
                                substrate_multiscale_harmony_loss,
                                corpus_char_signature,
                                corpus_multiscale_signature,
                                substrate_harmony_loss_grounded,
                                substrate_multiscale_harmony_loss_grounded,
                                substrate_omniweight_loss)
from activations_substrate import SubstrateNegMultiAdvancedV2
from train_substrate_attention import FibRecLMSubsim
from creativity_score import (creativity_score as compute_creativity_score,
                                  real_word_fraction)


def take_tiny_seed(encoded: torch.Tensor, n_chars: int,
                    seed: int = 42) -> torch.Tensor:
    """Slice an n-char window from encoded data at a deterministic offset."""
    g = torch.Generator(); g.manual_seed(seed)
    max_start = encoded.numel() - n_chars
    start = torch.randint(0, max_start, (1,), generator=g).item()
    return encoded[start: start + n_chars].clone()


def evaluate(model, val_split, batch_size, window, fib_positions, generator,
              n_batches=16):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = get_fib_strided_batch(val_split, batch_size, window,
                                           fib_positions, generator)
            logits = model(x)
            losses.append(F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item())
    model.train()
    return sum(losses) / len(losses)


def sample_tiny_batch(seed: torch.Tensor, batch_size: int, window: int,
                       gen: torch.Generator):
    """Random-stride batch from the tiny seed (cycled if needed)."""
    n = seed.numel()
    if n <= window + 1:
        # Pad by wrapping
        seed = seed.repeat((window + 2) // n + 1)
        n = seed.numel()
    starts = torch.randint(0, n - window - 1, (batch_size,), generator=gen)
    xs = torch.stack([seed[s: s + window] for s in starts])
    ys = torch.stack([seed[s + 1: s + window + 1] for s in starts])
    return xs, ys


_PHI_FOR_SAMPLING = (1.0 + 5.0 ** 0.5) / 2.0
# Substrate sampling sharpness, damped by 1/phi (golden ratio attenuation).
# Same canonical phi^pi base, but the effective sharpness is reduced -- the
# model can lock onto substrate-aligned tokens without collapsing to a
# single token. Substrate-canonical (uses phi as the dampener).
_PI_LOG_PHI = math.pi * math.log(_PHI_FOR_SAMPLING) / _PHI_FOR_SAMPLING


# Substrate penalty unit: log(phi) ~ 0.481 (mild). The syntax prior
# now does the heavy lifting; recency stays gentle.
_LOG_PHI_FOR_PENALTY = math.log(_PHI_FOR_SAMPLING)   # ~0.481

# Trigram-block suppression factor: phi^(2*pi) tier (~0.049x).
# Used post-omniweight to suppress tokens that would repeat a known
# trigram from the recent window. Substrate-canonical: same two-tier
# phi^(pi*k) scale as anti_stagnation's strong tier.
_TRIGRAM_SUPPRESS = 1.0 / (_PHI_FOR_SAMPLING ** (math.pi * 2))   # ~0.049
# Module-level corpus 3-gram set — populated once in train_with_self_distillation
# so autoregressive_generate can bypass repetition suppressors for authentic tokens.
_corpus_3grams_global: set | None = None


def build_bigram_prior(corpus_tokens: torch.Tensor, vocab_size: int):
    """Build P(next | prev) bigram statistics from the corpus."""
    counts = torch.zeros(vocab_size, vocab_size, dtype=torch.float)
    for i in range(corpus_tokens.numel() - 1):
        prev = int(corpus_tokens[i])
        nxt = int(corpus_tokens[i + 1])
        counts[prev, nxt] += 1.0
    row_sums = counts.sum(dim=-1, keepdim=True)
    row_sums[row_sums == 0] = 1.0
    return counts / row_sums


_FIB_NUMS_FOR_BIGRAM = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]


# Morphology-based POS classifier for substrate POS-aware bigram.
# Uses ONLY token shape -- no corpus statistics, no NLP library.
def classify_pos(token: str, rank: int = None) -> str:
    """Universal POS classification from MORPHOLOGY + RANK only.

    No hardcoded word lists -- the substrate framework's claim is that
    structure emerges from token shape + Fibonacci-tier rank position,
    not English-specific dictionaries.

    Signals:
      - Token shape: length, single-char, all-punct, whitespace.
      - Morphological suffixes: -eth/-est/-ing/-ed mark verbs in many
        Indo-European languages (universal-ish inflectional pattern).
      - Fibonacci-rank tier: most-frequent tokens (rank < F(7)=13) are
        statistically functional (articles, pronouns); next tier
        (rank < F(9)=34) are common content; tail are rare nouns.

    Categories collapse to: 'function' (high-freq functional words),
    'common' (mid-freq content words), 'verb' (morphological), 'noun'
    (default rest), plus shape categories.
    """
    if len(token) == 0:
        return 'fragment'
    if token in (' ', '\n', '\t'):
        return 'space'
    if all(c in '.,!?;:\'"-()' for c in token):
        return 'punct'
    if len(token) == 1:
        return 'fragment'

    tl = token.lower()
    # Universal inflection detection: phonological suffix weight.
    # Find last vowel→consonant boundary after position F(3)=3.
    # Suffix ≥ F(3)/F(5) = 3/8 of word length = inflected (verb-like).
    # Pure CV-structure — no language-specific endings.
    _vow = frozenset('aeiou')
    _stem_end = len(tl)
    for _k in range(len(tl) - 1, _FIB_NUMS_FOR_BIGRAM[3] - 1, -1):
        if tl[_k] in _vow and tl[_k - 1] not in _vow:
            _stem_end = _k
            break
    _sfx_len = len(tl) - _stem_end
    _sfx_w = _sfx_len / max(len(tl), 1)
    _sfx_thresh = float(_FIB_NUMS_FOR_BIGRAM[3]) / _FIB_NUMS_FOR_BIGRAM[5]
    if (_stem_end >= _FIB_NUMS_FOR_BIGRAM[3]
            and _sfx_len >= _FIB_NUMS_FOR_BIGRAM[3]
            and _sfx_w >= _sfx_thresh):
        return 'verb'

    # Rank-tier classification (universal: most-frequent ranks ARE
    # functional words in any language).
    if rank is not None:
        if rank < 13:           # F(7): top-13 most-frequent
            return 'function'   # articles, pronouns, conjunctions
        if rank < 34:           # F(9): top-34
            return 'common'     # common content words
    return 'noun'               # default for content words


_POS_CATEGORIES = ['function', 'common', 'verb', 'noun',
                     'punct', 'space', 'fragment']


def build_pos_transition_matrix() -> dict:
    """Self-referential POS transition matrix from substrate alone.

    Each POS category has a Fibonacci-derived VALUE based on its
    position in the substrate hierarchy:
      function: F(0) = 1   (highest-tier, most abstract)
      common:   F(1) = 1   (next-tier content)
      verb:     F(2) = 2   (action -- function + common)
      noun:     F(3) = 3   (entity -- common + verb)
      punct:    F(4) = 5   (boundary -- verb + noun)
      space:    F(5) = 8   (separator -- noun + punct)
      fragment: F(6) = 13  (sub-word -- punct + space)

    Each value is the sum of the two previous (Fibonacci recurrence).
    Transitions decay by phi^(pi * F-tier-distance):
      adjacent categories: 1.0
      one tier apart: 1/phi^pi  ~ 0.22
      two tiers apart: 1/phi^(2pi) ~ 0.049
      n tiers apart: 1/phi^(n*pi)

    This is fully substrate-derived: no hardcoded weights, no
    English-specific patterns. Just F(k) values and phi^pi decay.
    """
    F = _FIB_NUMS_FOR_BIGRAM
    cats = _POS_CATEGORIES
    # F-derived values per category (their Fibonacci position).
    pos_value = {cats[k]: F[k] for k in range(len(cats))}
    pos_tier = {cats[k]: k for k in range(len(cats))}
    phi_pi = _PHI_FOR_SAMPLING ** math.pi

    table = {}
    for a in cats:
        table[a] = {}
        for b in cats:
            tier_diff = abs(pos_tier[a] - pos_tier[b])
            # Substrate decay: closer tiers = higher transition.
            table[a][b] = 1.0 / (phi_pi ** tier_diff)
    return table


def build_model_derived_bigram(model, vocab_size: int) -> torch.Tensor:
    """Bigram emerges from the trained model's OWN predictions.

    For each token i, the bigram[i] is the model's next-token
    distribution given input [i]. This is purely substrate -- the
    model was trained with substrate operations (substrate harmony,
    substrate sampling, substrate embedding), so its learned
    transitions reflect substrate-aware structure.

    No corpus statistics injected; the bigram derives from the model
    itself. As the model improves during training, this bigram
    evolves with it.

    Substrate principle: derive from the substrate-trained system,
    not from external data.
    """
    bigram = torch.zeros(vocab_size, vocab_size, dtype=torch.float)
    model.eval()
    with torch.no_grad():
        # Batch all single-token inputs for efficiency.
        idx = torch.arange(vocab_size, dtype=torch.long).unsqueeze(1)
        # Process in chunks to manage memory.
        chunk = 32
        for start in range(0, vocab_size, chunk):
            end = min(start + chunk, vocab_size)
            x = idx[start:end]
            logits = model(x)[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            bigram[start:end] = probs
    model.train()
    # Zero diagonal (prevent self-loops).
    bigram.fill_diagonal_(0.0)
    bigram = bigram / (bigram.sum(dim=-1, keepdim=True) + 1e-8)
    return bigram


def build_substrate_pos_bigram(vocab_size: int, vocab: list) -> torch.Tensor:
    """Substrate POS-aware bigram: each (i, j) weighted by the POS
    transition table (above) * shape attenuation * rank-distance decay.

    Adds linguistic structure (noun-verb, article-noun, etc.) without
    using corpus n-gram statistics. POS classification is morphology-
    only (token shape + simple word lists).
    """
    phi = _PHI_FOR_SAMPLING
    pi_arg = math.pi
    K = len(_FIB_NUMS_FOR_BIGRAM)
    pos_table = build_pos_transition_matrix()
    # Classify all vocab tokens (passing rank so rank-tier signal works).
    pos_per_token = [classify_pos(vocab[i] if i < len(vocab) else '',
                                       rank=(i - 65 if i >= 65 else None))
                       for i in range(vocab_size)]
    # Build POS transition for each token pair via lookup.
    pos_weight = torch.zeros(vocab_size, vocab_size)
    for i in range(vocab_size):
        pos_i = pos_per_token[i]
        row = pos_table.get(pos_i, {})
        for j in range(vocab_size):
            pos_j = pos_per_token[j]
            pos_weight[i, j] = row.get(pos_j, _FIB_NUMS_FOR_BIGRAM[2]
                                                / (phi ** (pi_arg * 2)))
    # Rank-distance decay (mild, Binet-like).
    log_phi = math.log(phi)
    idx = torch.arange(vocab_size, dtype=torch.float)
    d = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() + 1.0
    K_ext = 16
    fib_ext = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
    k = torch.clamp(torch.log(d) / log_phi, 0.0, K_ext - 1.0).floor().long()
    fk_tensor = torch.tensor([fib_ext[i] / (phi ** i) for i in range(K_ext)],
                                dtype=torch.float)
    rank_decay = fk_tensor[k]
    # Shape attenuation (consistent with shape-bigram): multi-char words
    # full weight; punct phi^pi attenuated; single-char phi^(2pi); etc.
    shape_attenuation = torch.ones(vocab_size)
    for i in range(vocab_size):
        tok = vocab[i] if i < len(vocab) else ''
        if len(tok) >= 2:
            shape_attenuation[i] = 1.0
        elif tok in (' ', '\n', '\t'):
            shape_attenuation[i] = 1.0 / phi
        elif tok in '.,!?;:\'"-()':
            shape_attenuation[i] = 1.0 / (phi ** pi_arg)
        elif tok.isalpha():
            shape_attenuation[i] = 1.0 / (phi ** (pi_arg * 2))
        else:
            shape_attenuation[i] = 1.0 / (phi ** (pi_arg * 3))
    # Combined: POS weight (structure) * rank decay (proximity)
    # * shape attenuation (suppress fragments).
    bigram = pos_weight * rank_decay * shape_attenuation.unsqueeze(0)
    bigram.fill_diagonal_(0.0)
    bigram = bigram / (bigram.sum(dim=-1, keepdim=True) + 1e-8)
    return bigram


def build_substrate_bigram_shape(vocab_size: int, vocab: list) -> torch.Tensor:
    """Substrate bigram where each candidate next-token is weighted by
    the SYNTACTIC SHAPE of its chunk. Syntactically-clean tokens
    (real words, spaces, line breaks) get F(0)=1 full weight. Punctuation
    gets F(1)/phi^pi attenuation. Single-char fragments get F(2)/phi^(2pi)
    -- effectively suppressed unless they're whitespace/punct.

    Combined with the substrate rank-distance decay, the bigram says:
      "Prefer transitions whose target is itself a syntactic chunk,
      and that's rank-adjacent to the source."
    """
    phi = _PHI_FOR_SAMPLING
    pi_arg = math.pi
    K = len(_FIB_NUMS_FOR_BIGRAM)
    boundary = set([' ', '\n', '\t'])
    punct = set('.,!?;:\'"-()')

    # Sharper static shape weights -- shape DOMINATES, rank is just a
    # mild preference. Real words get full weight; everything else
    # strongly suppressed via substrate-canonical phi^(pi*k) scaling.
    shape_w = torch.zeros(vocab_size)
    for i in range(vocab_size):
        tok = vocab[i] if i < len(vocab) else ''
        if len(tok) >= 2:                                   # multi-char word
            shape_w[i] = 1.0
        elif tok in boundary:                               # whitespace (boundary)
            shape_w[i] = 1.0 / phi                          # mild attenuation
        elif tok in punct:                                  # punctuation
            shape_w[i] = 1.0 / (phi ** pi_arg)              # phi^pi suppress
        elif tok.isalpha():                                 # single letter
            shape_w[i] = 1.0 / (phi ** (pi_arg * 2))        # phi^(2pi) suppress
        else:                                               # digits/other
            shape_w[i] = 1.0 / (phi ** (pi_arg * 3))        # phi^(3pi) suppress

    # Rank-distance with Binet decay (flat) -- shape does the work now.
    log_phi = math.log(phi)
    idx = torch.arange(vocab_size, dtype=torch.float)
    d = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() + 1.0
    K_extended = 16
    fib_extended = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
    k = torch.clamp(torch.log(d) / log_phi, 0.0, K_extended - 1.0).floor().long()
    fk_tensor = torch.tensor(
        [fib_extended[i] / (phi ** i) for i in range(K_extended)],
        dtype=torch.float)
    rank_decay = fk_tensor[k]                               # [V, V] flat decay
    # Each candidate j weighted by ITS shape * rank_decay from i.
    bigram = rank_decay * shape_w.unsqueeze(0)              # broadcast over j
    bigram.fill_diagonal_(0.0)
    bigram = bigram / (bigram.sum(dim=-1, keepdim=True) + 1e-8)
    return bigram


def build_substrate_bigram(vocab_size: int) -> torch.Tensor:
    """Substrate-derived bigram prior: uses ONLY phi/pi/F(k) constants,
    no corpus statistics.

    Assumption: vocab is Fibonacci-tier-ranked (top-frequency tokens
    at low positions; tail at high). For tokens at positions i, j the
    prior of co-occurrence decays as F(k)/phi^(pi*k) where k is the
    Fibonacci tier of the rank distance |i - j|.

    This is the purest substrate-only syntax prior: the model needs
    no corpus access to acquire syntactic structure -- the substrate's
    recursive constants generate plausible co-occurrence directly
    from vocabulary structure.

    Vectorized: O(V^2) memory, O(V^2) compute, fast on 500-vocab.
    """
    K = len(_FIB_NUMS_FOR_BIGRAM)
    log_phi = math.log(_PHI_FOR_SAMPLING)
    # Pairwise rank distance |i - j|.
    idx = torch.arange(vocab_size, dtype=torch.float)
    d = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() + 1.0      # [V, V]
    # Fibonacci tier k = floor(log_phi(d)).
    k = torch.clamp(torch.log(d) / log_phi, 0.0, K - 1.0).floor().long()
    # Lookup F(k)/phi^(pi*k).
    fk_tensor = torch.tensor(
        [_FIB_NUMS_FOR_BIGRAM[i] / (_PHI_FOR_SAMPLING ** (math.pi * i))
         for i in range(K)], dtype=torch.float)
    bigram = fk_tensor[k]                                      # [V, V]
    # Zero the diagonal -- self-transitions cause repetition (already
    # handled by substrate recency penalty; bigram should favor MOVING).
    bigram.fill_diagonal_(0.0)
    bigram = bigram / (bigram.sum(dim=-1, keepdim=True) + 1e-8)
    return bigram


_SUBSTRATE_BIGRAM_ALPHA = 1.0 / (_PHI_FOR_SAMPLING ** math.pi)   # ~0.221


def build_substrate_token_signatures(vocab: list,
                                          sig_dim: int = 8) -> torch.Tensor:
    """Substrate-pure per-token signature derived from char codes.

    Each token gets a sig_dim vector. Dim k samples the token's chars at
    Fibonacci frequency F(k), cosine-projected, decayed across positions
    in the token by phi^(-pos). Substrate-canonical: only phi, F(k), pi,
    and char-code arithmetic. No corpus statistics, no English priors.

    The signature places tokens in a substrate-similarity space: tokens
    with similar char "shape" cluster, with longer/positional variation
    discounted by the golden ratio (so suffixes matter less than roots).
    """
    V = len(vocab)
    sigs = torch.zeros(V, sig_dim)
    F = _FIB_NUMS_FOR_BIGRAM
    phi = _PHI_FOR_SAMPLING
    for i, tok in enumerate(vocab):
        if not tok:
            continue
        for pos, ch in enumerate(tok):
            code = ord(ch)
            decay = phi ** (-pos)
            for k in range(sig_dim):
                freq = F[k] if k < len(F) else 1
                sigs[i, k] += math.cos(code * freq * 2.0 * math.pi / 128.0) * decay
    # Per-token L2 normalize so distances are comparable.
    norms = sigs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return sigs / norms


def substrate_theme_momentum(recent_tokens: list,
                                signatures: torch.Tensor,
                                probs: torch.Tensor) -> torch.Tensor:
    """Subject-matter coherence primitive: bias toward tokens whose
    substrate signature aligns with the running theme.

    Theme = F(k)/phi^(pi*k)-decayed weighted average of the last F(7)=13
    token signatures (recent content dominates, older fades). Similarity
    between candidate token and theme is 1/(1 + L1-distance) -- the
    same Subsim metric used inside the model's attention, surfaced at
    generation time as topical commitment.

    Log-boost: log(phi) * normalized_sim. Bounded by [1/phi, phi].
    Pure substrate: no corpus, no English word lists.
    """
    if not recent_tokens or signatures.shape[0] == 0:
        return probs
    n = min(len(recent_tokens), 13)
    last = recent_tokens[-n:]
    phi = _PHI_FOR_SAMPLING
    phi_pi = phi ** math.pi
    V_sig, D = signatures.shape
    theme = torch.zeros(D, dtype=signatures.dtype, device=signatures.device)
    total_w = 0.0
    for i, tid in enumerate(reversed(last)):
        if tid >= V_sig:
            continue
        k_tier = min(i, len(_FIB_NUMS_FOR_BIGRAM) - 1)
        w = _FIB_NUMS_FOR_BIGRAM[k_tier] / (phi_pi ** k_tier)
        theme = theme + signatures[tid] * w
        total_w += w
    if total_w < 1e-8:
        return probs
    theme = theme / total_w
    # L1 distance from each vocab signature to theme.
    dists = (signatures - theme.unsqueeze(0)).abs().sum(dim=-1)
    sims = 1.0 / (1.0 + dists)
    # Center and bound similarity to [-1, +1] band.
    sim_centered = sims - sims.mean()
    s_std = sim_centered.std()
    if s_std > 1e-8:
        sim_centered = sim_centered / s_std
    sim_centered = sim_centered.clamp(-1.0, 1.0)
    # Log-boost (substrate-bounded by phi).
    log_boost = math.log(phi) * sim_centered.to(probs.dtype).to(probs.device)
    boost = torch.exp(log_boost)
    out = probs * boost
    return out / (out.sum() + 1e-8)


def substrate_vocab_curriculum(probs: torch.Tensor,
                                  active_vocab_size: int) -> torch.Tensor:
    """Vocabulary expansion curriculum: restrict sampling to the first
    `active_vocab_size` tokens. As substrate K shrinks across cycles,
    vocab EXPANDS through Fibonacci tiers (F(8)=21 -> F(12)=144 -> full).
    Functional/common tokens always available; content/proper nouns
    unlock progressively. Pure substrate (Fibonacci-tier walk).
    """
    if active_vocab_size <= 0 or active_vocab_size >= probs.shape[0]:
        return probs
    out = probs.clone()
    out[active_vocab_size:] = 0.0
    s = out.sum()
    if s < 1e-8:
        return probs
    return out / s


_IAMBIC_VOWELS = set("aeiouAEIOU")


def _token_morphology(tok: str) -> str:
    """Universal morphology class from suffix (no English word lists).
    char | verb_archaic | gerund | past | adverb | plural | root.
    """
    if not tok or len(tok) <= 1:
        return 'char'
    tl = tok.lower()
    # Universal suffix detection via vowel-consonant boundary weight.
    _vow = frozenset('aeiou')
    _stem_end = len(tl)
    for _k in range(len(tl) - 1, _FIB_NUMS_FOR_BIGRAM[3] - 1, -1):
        if tl[_k] in _vow and tl[_k - 1] not in _vow:
            _stem_end = _k
            break
    _sfx_w = (len(tl) - _stem_end) / max(len(tl), 1)
    _sfx_thresh = float(_FIB_NUMS_FOR_BIGRAM[3]) / _FIB_NUMS_FOR_BIGRAM[5]
    if _stem_end >= _FIB_NUMS_FOR_BIGRAM[3] and _sfx_w >= _sfx_thresh:
        return 'inflected'
    if tl.endswith('s') and len(tl) > 2:
        return 'plural'
    return 'root'


def build_symbol_classes(vocab: list, n_chars: int = 65) -> tuple:
    """Each token's class = (rank_tier, morphology). Rank-tier is the
    Fibonacci-walk band the token's rank falls into (within the word
    region). Chars get their own tier. Morphology from suffix.
    Pure substrate (F-tier + suffix shape, no word lists).

    Returns (class_id_tensor[V], n_classes).
    """
    F = _FIB_NUMS_FOR_BIGRAM   # [1,1,2,3,5,8,13,21,34,55,89,144]
    cum_tiers = []
    cum = n_chars
    for f in F:
        cum += f
        cum_tiers.append(cum)

    def rank_tier(i: int) -> int:
        if i < n_chars:
            return -1
        for ti, ct in enumerate(cum_tiers):
            if i < ct:
                return ti
        return len(cum_tiers)

    morphs = ['char', 'inflected', 'plural', 'root']
    morph_to_idx = {m: i for i, m in enumerate(morphs)}
    n_morph = len(morphs)
    # Class id = (tier + 1) * n_morph + morph_idx
    class_ids = []
    for i, tok in enumerate(vocab):
        tier = rank_tier(i)
        m = _token_morphology(tok)
        cid = (tier + 1) * n_morph + morph_to_idx[m]
        class_ids.append(cid)
    class_id_tensor = torch.tensor(class_ids, dtype=torch.long)
    n_classes = int(class_id_tensor.max().item()) + 1
    return class_id_tensor, n_classes


def substrate_symbolic_substitution(probs: torch.Tensor,
                                       class_id_tensor: torch.Tensor,
                                       n_classes: int,
                                       alpha: float = None) -> torch.Tensor:
    """Smooth probability mass within symbol equivalence classes.

    Per class: redistribute alpha-fraction of mass uniformly across
    siblings; keep (1-alpha) at the original spike. Variety without
    breaking grammar -- tokens in the same (rank-tier, morphology)
    class are mutually substitutable.

    alpha defaults to 1/phi^pi (substrate-canonical, ~0.221).
    """
    if alpha is None:
        alpha = 1.0 / (_PHI_FOR_SAMPLING ** math.pi)
    cids = class_id_tensor.to(probs.device)
    class_totals = torch.zeros(n_classes, dtype=probs.dtype,
                                  device=probs.device)
    class_totals.scatter_add_(0, cids, probs)
    counts = torch.zeros(n_classes, dtype=probs.dtype,
                            device=probs.device)
    counts.scatter_add_(0, cids, torch.ones_like(probs))
    counts.clamp_(min=1.0)
    uniform_per_class = class_totals / counts
    uniform_per_token = uniform_per_class[cids]
    out = (1.0 - alpha) * probs + alpha * uniform_per_token
    return out / (out.sum() + 1e-8)


def build_pronoun_mask(vocab: list) -> torch.Tensor:
    """Identify pronoun-shape tokens: low rank + monosyllabic + no suffix.
    Pure substrate (rank + syllable + morphology shape).
    """
    V = len(vocab)
    mask = torch.zeros(V)
    for i, tok in enumerate(vocab):
        if not tok or len(tok) == 1:
            continue
        is_low_rank = i < 78   # 65 chars + F(7)=13 most common words
        no_suffix = _token_morphology(tok) == 'root'
        is_monosyl = _approx_syllables(tok) == 1
        if is_low_rank and no_suffix and is_monosyl:
            mask[i] = 1.0
    return mask


def substrate_need_fill(open_needs: int, probs: torch.Tensor,
                            vocab_size: int,
                            punct_mask: torch.Tensor = None) -> torch.Tensor:
    """Bracket-matching with punctuation-specific bias when pressure
    crosses F(5)=5 threshold.

    Below F(5): gentle low-rank bias (rank-polarity), unchanged.
    At/above F(5): concentrate boost on PUNCTUATION tokens (true
    closers), not all low-rank tokens. Prevents the v62 pattern where
    need-fill boosted 'this'/'the'/'of' alongside true closers.

    Pure substrate (F(5) threshold + char-class punctuation set).
    """
    if open_needs <= 0 or vocab_size <= 1:
        return probs
    phi = _PHI_FOR_SAMPLING
    F = _FIB_NUMS_FOR_BIGRAM
    pressure_tier = 0
    for k, f in enumerate(F):
        if open_needs >= f:
            pressure_tier = k
    boost_mag = F[pressure_tier] / (phi ** (math.pi * pressure_tier))
    # Below F(5): rank-polarity (broad low-rank bias).
    if open_needs < F[5] or punct_mask is None:
        ranks = torch.arange(vocab_size, dtype=probs.dtype,
                              device=probs.device)
        rank_pol = 1.0 - 2.0 * ranks / (vocab_size - 1)
        log_boost = math.log(phi) * boost_mag * rank_pol
        boost = torch.exp(log_boost)
    else:
        # At/above F(5): punctuation-only boost.
        pm = punct_mask.to(probs.device).to(probs.dtype)
        log_boost = math.log(phi) * boost_mag
        bf = math.exp(log_boost)
        boost = 1.0 + pm * (bf - 1.0)
    out = probs * boost
    return out / (out.sum() + 1e-8)


def build_punct_mask(vocab: list) -> torch.Tensor:
    """Mask = 1 for clause-closing punctuation tokens.
    Substrate-pure: char-class identification.
    """
    V = len(vocab)
    mask = torch.zeros(V)
    closers = {'.', '!', '?', ',', ';', ':', '\n'}
    for i, tok in enumerate(vocab):
        if tok in closers:
            mask[i] = 1.0
    return mask


def build_uppercase_mask(vocab: list) -> torch.Tensor:
    """Mask = 1 for tokens whose first char is uppercase A-Z.
    For grammar rule: capitalization after sentence boundary.
    """
    V = len(vocab)
    mask = torch.zeros(V)
    for i, tok in enumerate(vocab):
        if tok and len(tok) >= 1 and tok[0].isupper():
            mask[i] = 1.0
    return mask


def build_terminal_punct_mask(vocab: list) -> torch.Tensor:
    """Mask = 1 for sentence-ending tokens only (. ! ?).
    Excludes mid-clause marks (, ; : \\n). Universal language primitive.
    """
    V = len(vocab)
    mask = torch.zeros(V)
    for i, tok in enumerate(vocab):
        if tok in ('.', '!', '?'):
            mask[i] = 1.0
    return mask


def build_any_punct_mask(vocab: list) -> torch.Tensor:
    """Mask = 1 for ANY single-char punctuation token (including
    apostrophes, dashes -- broader than build_punct_mask which is
    clause-closers only). For no-double-punctuation rule.
    """
    V = len(vocab)
    mask = torch.zeros(V)
    pset = {'.', ',', '!', '?', ';', ':', "'", '"', '-', '(', ')'}
    for i, tok in enumerate(vocab):
        if tok in pset:
            mask[i] = 1.0
    return mask


def substrate_grammar_capitalize(prev_str: str, probs: torch.Tensor,
                                      uppercase_mask: torch.Tensor
                                      ) -> torch.Tensor:
    """Sentence-start capitalization rule. If previous emission was
    '.', '!', '?', or '\\n', boost uppercase tokens by phi.
    """
    if uppercase_mask is None:
        return probs
    if prev_str not in ('.', '!', '?', '\n'):
        return probs
    um = uppercase_mask.to(probs.device).to(probs.dtype)
    boost = 1.0 + um * (_PHI_FOR_SAMPLING - 1.0)
    out = probs * boost
    return out / (out.sum() + 1e-8)


def substrate_grammar_no_double_punct(prev_str: str,
                                            probs: torch.Tensor,
                                            any_punct_mask: torch.Tensor
                                            ) -> torch.Tensor:
    """If previous emission was a punctuation char, hard-suppress
    further punctuation. Prevents ',,', '..', '.,', etc.
    Suppression by 1/phi^pi.
    """
    if any_punct_mask is None:
        return probs
    punct_set = {'.', ',', '!', '?', ';', ':', "'", '"', '-'}
    if prev_str not in punct_set:
        return probs
    pm = any_punct_mask.to(probs.device).to(probs.dtype)
    suppress = 1.0 / (_PHI_FOR_SAMPLING ** math.pi)
    multiplier = 1.0 - pm * (1.0 - suppress)
    out = probs * multiplier
    return out / (out.sum() + 1e-8)


def build_vowel_start_mask(vocab: list) -> torch.Tensor:
    """Mask = 1 for tokens starting with a vowel, 0 otherwise.
    For phonotactics primitive (CV cluster relief).
    """
    V = len(vocab)
    mask = torch.zeros(V)
    for i, tok in enumerate(vocab):
        if tok and tok[0] in _IAMBIC_VOWELS:
            mask[i] = 1.0
    return mask


def build_unpronounceable_mask(vocab: list) -> torch.Tensor:
    """Mask = 1 for tokens with impossible-shape lettering.

    Flags (any one disqualifies):
      - max consonant cluster > F(5)=5 (allows 'strengths', 'twelfth')
      - same letter triple (e.g., 'sss', 'fff', 'ttt')
      - zero vowels in length > F(3)=2 token (all-consonant word)

    Char tokens (len=1) exempt. Non-alpha tokens exempt (contractions
    like "'tis"). Pure substrate (char-class + Fibonacci-tier).
    """
    V = len(vocab)
    mask = torch.zeros(V)
    F = _FIB_NUMS_FOR_BIGRAM
    for i, tok in enumerate(vocab):
        if not tok or len(tok) <= 1:
            continue
        if not all(c.isalpha() for c in tok):
            continue
        max_cluster = 0
        cur = 0
        for ch in tok:
            if ch in _IAMBIC_VOWELS:
                cur = 0
            else:
                cur += 1
                if cur > max_cluster:
                    max_cluster = cur
        triple = False
        for j in range(len(tok) - 2):
            if tok[j] == tok[j + 1] == tok[j + 2]:
                triple = True
                break
        n_vowel = sum(1 for c in tok if c in _IAMBIC_VOWELS)
        all_consonant = (n_vowel == 0) and (len(tok) > F[3])
        # Long words with very low vowel ratio: 6+ chars, < 1/phi^3 ~ 0.236.
        # Eases past legit proper nouns ('northumberland' 0.29,
        # 'buckingham' 0.30) but flags consonant-soup tokens.
        low_vowel_long = (
            len(tok) > F[5]
            and (n_vowel / len(tok)) < (1.0 / (_PHI_FOR_SAMPLING ** 3))
        )
        if max_cluster > F[5] or triple or all_consonant or low_vowel_long:
            mask[i] = 1.0
    return mask


def substrate_char_cascade(char_run: int, probs: torch.Tensor,
                              n_chars: int, vocab: list = None) -> torch.Tensor:
    """Anti-char-cascade: once F(1)=1 consecutive alpha char token has been
    emitted, suppress further alpha char emissions (separators exempt).

    Soft signal that complements the hard extended word boundary.
    Suppression magnitude grows by F(k) above threshold.

    Pure substrate (F(1) threshold + phonological class identification).
    """
    if char_run < _FIB_NUMS_FOR_BIGRAM[1] or n_chars <= 0:
        return probs
    if n_chars >= probs.shape[0]:
        return probs
    excess = char_run - _FIB_NUMS_FOR_BIGRAM[1] + 1
    tier = min(excess, len(_FIB_NUMS_FOR_BIGRAM) - 1)
    penalty = 1.0 / (_PHI_FOR_SAMPLING ** (math.pi * _FIB_NUMS_FOR_BIGRAM[tier]))
    out = probs.clone()
    _sep = {' ', '\n', '\t', '.', ',', '!', '?', ';', ':', "'", '-'}
    for i in range(n_chars):
        tok_c = vocab[i] if (vocab and i < len(vocab)) else ''
        if tok_c not in _sep:  # suppress only letter chars
            out[i] = out[i] * penalty
    return out / (out.sum() + 1e-8)


def substrate_bigram_saturation(prev_tok: int, recent_pairs: list,
                                    probs: torch.Tensor) -> torch.Tensor:
    """Penalize bigram transitions that have already fired F(4)=3+ times
    in the last F(7)=13 transitions. Loosened from F(3)=2 (v69) which
    over-suppressed legitimate intentional repeats like 'this happy
    breed of MEN, this LITTLE world'. Substrate-tier exponential
    suppression so a 4th same-transition fades fast.
    """
    if not recent_pairs:
        return probs
    F = _FIB_NUMS_FOR_BIGRAM
    counts = {}
    for p, n in recent_pairs[-13:]:
        if p == prev_tok:
            counts[n] = counts.get(n, 0) + 1
    if not counts:
        return probs
    suppress = torch.ones_like(probs)
    threshold = F[4]
    for next_tok, c in counts.items():
        if c >= threshold:
            excess = c - threshold + 1
            tier = min(excess, len(F) - 1)
            penalty = 1.0 / (_PHI_FOR_SAMPLING ** (math.pi * F[tier]))
            if 0 <= next_tok < probs.shape[0]:
                suppress[next_tok] = penalty
    out = probs * suppress
    return out / (out.sum() + 1e-8)


def substrate_agreement(last_content_ends_s: bool, probs: torch.Tensor,
                            vocab: list) -> torch.Tensor:
    """Number-agreement primitive: '-s' suffix as number marker.

    If most-recent content token ends in 's' (likely plural noun or
    third-person verb), suppress next tokens ending in 's' and boost
    non-'s' endings -- and vice versa. Universal morphology bias.

    Boost magnitude: phi (factor 1.618), bounded [1/phi, phi].
    Pure substrate (suffix shape + Fibonacci-bounded boost).
    """
    if not vocab:
        return probs
    ends_s_mask = torch.zeros_like(probs)
    for i, tok in enumerate(vocab):
        if (tok and len(tok) > 1
                and tok.endswith('s')
                and not tok.endswith('ss')
                and not tok.endswith('is')
                and not tok.endswith('us')):
            ends_s_mask[i] = 1.0
    phi = _PHI_FOR_SAMPLING
    if last_content_ends_s:
        factor = 1.0 / phi
    else:
        factor = phi
    boost = 1.0 + ends_s_mask * (factor - 1.0)
    out = probs * boost
    return out / (out.sum() + 1e-8)


def build_allowed_after_word_mask(vocab: list, n_chars: int = 65,
                                       suppress: float = None) -> torch.Tensor:
    """Per-token multiplier mask for word_spacing primitive.
    All tokens suppressed by 1/phi^2 except space/newline/punct chars
    in the char region. Precomputed once.
    """
    V = len(vocab)
    if suppress is None:
        suppress = 1.0 / (_PHI_FOR_SAMPLING ** 2)
    mask = torch.full((V,), suppress)
    allowed_chars = {' ', '\n', '.', ',', '!', '?', ';', ':', "'", '-'}
    for i in range(min(n_chars, V)):
        if vocab[i] in allowed_chars:
            mask[i] = 1.0
    return mask


def substrate_word_spacing(prev_tid: int, probs: torch.Tensor,
                              vocab: list, n_chars: int = 65,
                              allowed_mask: torch.Tensor = None
                              ) -> torch.Tensor:
    """Word boundary enforcement (vectorized via precomputed mask).
    """
    if prev_tid < n_chars or allowed_mask is None:
        return probs
    out = probs * allowed_mask.to(probs.device).to(probs.dtype)
    return out / (out.sum() + 1e-8)


def substrate_morpheme_gate(prev_str: str, probs: torch.Tensor,
                             n_chars: int) -> torch.Tensor:
    """Post-word-boundary morpheme gate.

    After a space or newline (word boundary), a real word starts.
    Natural words in any language almost always begin with a multi-char
    sequence — standalone single-char words are a tiny closed class.
    Boost n-gram tokens (index >= n_chars, length >= 2) by phi^2 (~2.62x)
    so the model strongly prefers starting a proper word-token rather than
    emitting a bare letter like 'h' or 'b'.

    Substrate-universal: only token indices + boundary char detection.
    Prevents 'against h be less' type bare-letter fragments.
    """
    if prev_str not in (' ', '\n'):
        return probs
    phi_pi = _PHI_FOR_SAMPLING ** math.pi
    boost = torch.ones_like(probs)
    boost[n_chars:] = phi_pi
    out = probs * boost
    return out / (out.sum() + 1e-8)


def substrate_content_density(recent_tokens: list, probs: torch.Tensor,
                                n_chars: int, window: int = 8,
                                target_frac: float = 0.35) -> torch.Tensor:
    """Content/structure balance correction.

    In any substrate vocabulary, tokens with index < n_chars are single
    characters (structural role); tokens >= n_chars are Fibonacci n-grams
    (content/morphemic role). Natural text needs both in roughly equal
    measure — pure function-token runs ('this the of the, the') indicate
    the content-density fell below the substrate floor.

    If content fraction in the last window=F(6)=8 tokens < target_frac,
    multiply n-gram probabilities by phi (~1.618) to restore balance.
    If content fraction > (1 - target_frac), boost char-token probs instead.

    Substrate-universal: only token-index arithmetic, no vocabulary
    knowledge. Directly fixes 'this the of the, the' pattern.
    """
    if not recent_tokens:
        return probs
    win = recent_tokens[-window:]
    if not win:
        return probs
    content_count = sum(1 for t in win if t >= n_chars)
    content_frac = content_count / len(win)
    phi = _PHI_FOR_SAMPLING
    if content_frac < target_frac:
        boost = torch.ones_like(probs)
        boost[n_chars:] = phi
        out = probs * boost
    elif content_frac > (1.0 - target_frac):
        boost = torch.ones_like(probs)
        boost[:n_chars] = phi
        out = probs * boost
    else:
        return probs
    return out / (out.sum() + 1e-8)


def substrate_pronounceability(probs: torch.Tensor,
                                  unpronounceable_mask: torch.Tensor
                                  ) -> torch.Tensor:
    """Suppress tokens flagged as un-pronounceable (impossible shape).
    Multiplicative penalty by 1/phi^pi ~ 0.221.
    Pure substrate (precomputed shape filter).
    """
    if unpronounceable_mask is None:
        return probs
    upm = unpronounceable_mask.to(probs.device).to(probs.dtype)
    penalty = 1.0 / (_PHI_FOR_SAMPLING ** math.pi)
    multiplier = 1.0 - upm * (1.0 - penalty)
    out = probs * multiplier
    return out / (out.sum() + 1e-8)
    return mask


def substrate_phonotactics(cluster_len: int, probs: torch.Tensor,
                              vowel_start_mask: torch.Tensor) -> torch.Tensor:
    """When recent consonant cluster >= 2, boost vowel-starting tokens.
    Natural CV rhythm preservation. Pure substrate (char-class only).

    Boost magnitude grows with cluster length:  log(phi) * (cluster-1).
    Bounded by exp(log(phi)*F(k)) for F(k)<=cluster.
    """
    if cluster_len < 2 or vowel_start_mask is None:
        return probs
    log_boost = math.log(_PHI_FOR_SAMPLING) * (cluster_len - 1)
    boost_factor = math.exp(log_boost)
    vsm = vowel_start_mask.to(probs.device).to(probs.dtype)
    boost = 1.0 + vsm * (boost_factor - 1.0)
    out = probs * boost
    return out / (out.sum() + 1e-8)


_VOWEL_ORDER = ['a', 'e', 'i', 'o', 'u']
_VOWEL_TO_IDX = {v: i for i, v in enumerate(_VOWEL_ORDER)}


def build_end_vowel_per_token(vocab: list) -> list:
    """Each token's final vowel (or '' if none). For rhyme primitive.
    """
    end_vowels = []
    for tok in vocab:
        ev = ''
        for ch in reversed(tok or ''):
            if ch in _IAMBIC_VOWELS:
                ev = ch.lower()
                break
        end_vowels.append(ev)
    return end_vowels


def build_end_vowel_idx_tensor(vocab: list) -> torch.Tensor:
    """Per-token end-vowel index in {0..4} or -1 if no vowel.
    Vectorizes the rhyme primitive's V-loop.
    """
    V = len(vocab)
    idx = torch.full((V,), -1, dtype=torch.long)
    for i, tok in enumerate(vocab):
        for ch in reversed(tok or ''):
            if ch in _IAMBIC_VOWELS:
                idx[i] = _VOWEL_TO_IDX.get(ch.lower(), -1)
                break
    return idx


def substrate_rhyme_resonance(recent_tokens: list,
                                   end_vowel_idx: torch.Tensor,
                                   probs: torch.Tensor) -> torch.Tensor:
    """Vectorized rhyme resonance.

    end_vowel_idx: LongTensor[V] in {-1, 0..4}. Precomputed once.
    Pressure per vowel computed by Python loop (~13 iters); boost
    lookup is one tensor index op replacing the prior 500-elt loop.
    """
    if not recent_tokens or end_vowel_idx is None:
        return probs
    phi = _PHI_FOR_SAMPLING
    phi_pi = phi ** math.pi
    pressure = torch.zeros(len(_VOWEL_ORDER), dtype=probs.dtype,
                              device=probs.device)
    V_ev = end_vowel_idx.shape[0]
    for i, tid in enumerate(reversed(recent_tokens[-13:])):
        if tid >= V_ev or tid < 0:
            continue
        v_idx = int(end_vowel_idx[tid].item())
        if v_idx < 0:
            continue
        kt = min(i, len(_FIB_NUMS_FOR_BIGRAM) - 1)
        w = _FIB_NUMS_FOR_BIGRAM[kt] / (phi_pi ** kt)
        pressure[v_idx] += w
    if pressure.sum() <= 0:
        return probs
    rhyme_scale = math.log(phi) / float(_FIB_NUMS_FOR_BIGRAM[3])
    log_boost_per_vowel = (rhyme_scale * pressure / (1.0 + pressure))
    # Vectorized lookup: for each token, fetch its vowel's boost.
    evi = end_vowel_idx.to(probs.device)
    valid = (evi >= 0)
    safe_idx = evi.clamp(min=0)
    log_boost_per_token = torch.where(
        valid,
        log_boost_per_vowel[safe_idx],
        torch.zeros_like(probs),
    )
    boost = torch.exp(log_boost_per_token)
    out = probs * boost
    return out / (out.sum() + 1e-8)


def substrate_reference_chain(recent_tokens: list,
                                  pronoun_mask: torch.Tensor,
                                  probs: torch.Tensor,
                                  n_chars: int = 65) -> torch.Tensor:
    """Anaphora with anti-stagnation self-cooling.

    Content-pressure boosts pronoun candidates as before. BUT when
    last F(7)=13 emissions already contain >= F(4)=3 pronoun-shape
    tokens, divide the boost by F(count - threshold + 1). The
    primitive cools itself when overactive instead of being silently
    over-amplified.

    Pure substrate (F(7) memory + F(4) threshold + F-tier dampening).
    """
    if not recent_tokens:
        return probs
    phi = _PHI_FOR_SAMPLING
    phi_pi = phi ** math.pi
    F = _FIB_NUMS_FOR_BIGRAM
    content_thresh = n_chars + F[7]
    pressure = 0.0
    for i, tid in enumerate(reversed(recent_tokens)):
        if i >= 13:
            break
        if tid > content_thresh:
            k = min(i, len(F) - 1)
            pressure += F[k] / (phi_pi ** k)
    if pressure <= 0:
        return probs
    # Count recent pronoun-shape emissions.
    pmask_cpu = pronoun_mask.to('cpu')
    pronoun_count = 0
    for tid in recent_tokens[-13:]:
        if 0 <= tid < pmask_cpu.shape[0] and pmask_cpu[tid].item() > 0.5:
            pronoun_count += 1
    excess = max(0, pronoun_count - F[4])
    damper = float(F[min(excess, len(F) - 1)])  # >= 1
    log_boost = math.log(phi) * pressure / (1.0 + pressure) / damper
    boost_factor = math.exp(log_boost)
    pmask = pronoun_mask.to(probs.device).to(probs.dtype)
    boost = 1.0 + pmask * (boost_factor - 1.0)
    out = probs * boost
    return out / (out.sum() + 1e-8)


def _approx_syllables(tok_str: str) -> int:
    """Approximate syllable count = number of vowel-clusters.
    Pure substrate (char-class arithmetic). Min 1 for non-empty tokens.
    """
    if not tok_str:
        return 0
    n = 0
    prev_v = False
    for ch in tok_str:
        v = ch in _IAMBIC_VOWELS
        if v and not prev_v:
            n += 1
        prev_v = v
    return max(1, n)


def substrate_iambic_phase(syl_pos: int, probs: torch.Tensor,
                              vocab_size: int,
                              newline_mask: torch.Tensor = None) -> torch.Tensor:
    """Iambic stress + F(5)-foot pentameter line-completion pressure.

    Layer 1 (always on): period-2 weak/STRONG alternation modulates
    rank polarity per syllable.

    Layer 2 (when syl_pos >= 2*F(5)=10): line-completion pressure
    boosts newline-shape tokens. After 10 syllables (iambic pentameter
    line length), the substrate pushes hard toward a line break.
    Boost grows with overshoot to recover pentameter rhythm.

    Pure substrate: period 2 = F(3), line-foot = 2 * F(5), nested.
    """
    if vocab_size <= 1:
        return probs
    phi = _PHI_FOR_SAMPLING
    F = _FIB_NUMS_FOR_BIGRAM
    sign = 1.0 if (syl_pos % 2 == 0) else -1.0
    ranks = torch.arange(vocab_size, dtype=probs.dtype,
                          device=probs.device)
    rank_pol = 1.0 - 2.0 * ranks / (vocab_size - 1)
    log_boost = math.log(phi) * sign * rank_pol
    boost = torch.exp(log_boost)
    out = probs * boost
    out = out / (out.sum() + 1e-8)
    # Pentameter line-completion pressure.
    line_threshold = 2 * F[5]   # 10 syllables = iambic pentameter
    if newline_mask is not None and syl_pos >= line_threshold:
        overshoot = syl_pos - line_threshold + 1
        line_boost = math.exp(math.log(phi) * min(overshoot, F[7]) / F[3])
        nm = newline_mask.to(probs.device).to(probs.dtype)
        line_factor = 1.0 + nm * (line_boost - 1.0)
        out = out * line_factor
        out = out / (out.sum() + 1e-8)
    return out


def build_newline_mask(vocab: list) -> torch.Tensor:
    """Mask = 1 for newline-shape tokens (\n only by default).
    Substrate-pure: char-class identification.
    """
    V = len(vocab)
    mask = torch.zeros(V)
    for i, tok in enumerate(vocab):
        if tok == '\n':
            mask[i] = 1.0
    return mask


def build_pos_masks(vocab: list):
    """Build part-of-speech masks from morphological features — language primitive.

    POS categories are universal: every language has verbs, nouns, adjectives,
    and function words. Identification uses morphological rules (suffixes,
    forms), not a corpus-specific word list.

    Returns (verb_mask, noun_mask, adj_mask, func_mask) each shape (V,).

    VERBS — identified by:
      - Archaic endings: -eth (doth, hath), -est (canst, wilt), -st (art)
      - Participle endings: -ing, -ed, -en
      - Auxiliary set (grammatical class, not lexical list):
        be/is/are/was/were/am/been, do/did/does, have/has/had,
        shall/will/would/should/could/may/might/must/can/dare/need

    NOUNS — identified by:
      - Derivational suffixes: -tion/-sion, -ness, -ment, -ance/-ence,
        -ity/-ty, -age, -hood, -ship, -dom, -ure
      - Agent nouns: -er/-or (>4 chars to avoid short comparatives)
      - Capitalized tokens (proper nouns — substrate-universal: caps = name)

    ADJECTIVES — identified by:
      - Derivational: -ful, -less, -ous, -ive, -al, -ent/-ant, -ible/-able
      - Comparative/superlative: -er, -est (>4 chars, distinct from verb -est)

    FUNCTION WORDS — closed grammatical class (articles, conjunctions,
      prepositions, determiners): the/a/an, and/or/but/nor/yet/so,
      in/of/to/for/with/by/from/at/on/as/into/than/upon/unto/till/
      that/which/what/this/these/those/my/thy/his/her/its/our/your/their
      (possession forms overlap with pronouns — both signal are correct)
    """
    V = len(vocab)
    verb_mask = torch.zeros(V)
    noun_mask = torch.zeros(V)
    adj_mask  = torch.zeros(V)
    func_mask = torch.zeros(V)

    # Closed grammatical class: auxiliaries are a universal linguistic category.
    # Only true auxiliaries included — no content verbs.
    _auxiliaries = {
        'be', 'is', 'are', 'was', 'were', 'am', 'been',
        'do', 'did', 'does', 'done',
        'have', 'has', 'had',
        'shall', 'will', 'would', 'should', 'could',
        'may', 'might', 'must', 'can', 'dare', 'need',
    }
    # Closed grammatical class: determiners, conjunctions, prepositions.
    # Universal across languages. No archaic corpus-specific forms.
    _function = {
        'the', 'a', 'an',
        'and', 'or', 'but', 'nor', 'yet', 'so',
        'in', 'of', 'to', 'for', 'with', 'by', 'from',
        'at', 'on', 'as', 'into', 'than', 'upon', 'unto',
        'till',
        'that', 'which', 'what', 'who', 'whom', 'whose',
        'this', 'these', 'those', 'such',
        'not', 'no', 'nor', 'never',
        'if', 'though', 'although', 'lest', 'since', 'when',
        'where', 'how', 'why',
    }

    for i, tok in enumerate(vocab):
        if not tok or not tok.isalpha():
            continue
        tl = tok.lower()
        n = len(tl)

        # --- VERBS ---
        if tl in _auxiliaries:
            verb_mask[i] = 1.0
        elif (n >= 4 and tl.endswith('eth')):      # doth, hath, speaketh
            verb_mask[i] = 1.0
        elif (n >= 5 and tl.endswith('est')):      # canst, wilt, wouldst
            verb_mask[i] = 1.0
        elif (n >= 3 and tl.endswith('st')):       # art, wast, didst
            verb_mask[i] = 1.0
        elif (n >= 4 and tl.endswith('ing')):      # loving, being
            verb_mask[i] = 1.0
        elif (n >= 4 and tl.endswith('ed')):       # loved, feared
            verb_mask[i] = 1.0
        elif (n >= 4 and tl.endswith('en')):       # spoken, broken
            verb_mask[i] = 1.0

        # --- NOUNS ---
        elif any(tl.endswith(s) for s in (
                'tion', 'sion', 'ness', 'ment', 'ance', 'ence',
                'ity', 'ty', 'age', 'hood', 'ship', 'dom', 'ure')):
            noun_mask[i] = 1.0
        elif n > 4 and (tl.endswith('er') or tl.endswith('or')):
            noun_mask[i] = 1.0
        elif tok[0].isupper() and n > 1:           # proper noun
            noun_mask[i] = 1.0

        # --- ADJECTIVES — morphological suffixes only, no word list ---
        elif any(tl.endswith(s) for s in (
                'ful', 'less', 'ous', 'ive', 'ible', 'able',
                'ent', 'ant', 'ary', 'ory')):
            adj_mask[i] = 1.0

        # --- FUNCTION WORDS ---
        if tl in _function:
            func_mask[i] = 1.0

    return verb_mask, noun_mask, adj_mask, func_mask


def build_person_masks(vocab: list):
    """Build grammatical person masks — substrate-universal.

    Pronouns are identifiable purely by their surface form; no semantic
    knowledge required. Every language encodes speaker (1st), addressee
    (2nd), and referent (3rd) positional roles in its pronoun system.

    Returns (first_mask, second_mask, third_mask) each shape (V,).
    """
    V = len(vocab)
    first  = {'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours'}
    second = {'thou', 'thee', 'thy', 'thine', 'thyself',
              'you', 'your', 'yours', 'yourself', 'ye'}
    third  = {'he', 'him', 'his', 'she', 'her', 'hers', 'herself',
              'it', 'its', 'itself', 'they', 'them', 'their', 'theirs'}
    f_mask = torch.zeros(V)
    s_mask = torch.zeros(V)
    t_mask = torch.zeros(V)
    for i, tok in enumerate(vocab):
        tl = tok.lower()
        if tl in first:
            f_mask[i] = 1.0
        elif tl in second:
            s_mask[i] = 1.0
        elif tl in third:
            t_mask[i] = 1.0
    return f_mask, s_mask, t_mask


_SECTOR_NAMES = ('char', 'bi', 'tri', '5g', '8g', '13g', '21g')

def _fib_tok_sector(tid: int, vocab: list) -> int:
    """Map token to Fibonacci length sector (0-6).

    Sectors are defined by token string length at Fibonacci boundaries.
    This is purely structural — no word lists, universal across corpora.
      0 char  : len=1        (F(2)=1)
      1 bi    : len=2        (F(3)=2)
      2 tri   : len=3        (F(4)=3)
      3 5g    : len 4-5      (F(5)=5)
      4 8g    : len 6-8      (F(6)=8)
      5 13g   : len 9-13     (F(7)=13)
      6 21g   : len 14+      (F(8)=21)
    """
    if tid < 0 or tid >= len(vocab):
        return 0
    n = len(vocab[tid])
    if n <= 1:  return 0
    if n <= 2:  return 1
    if n <= 3:  return 2
    if n <= 5:  return 3
    if n <= 8:  return 4
    if n <= 13: return 5
    return 6


class FibPosTable:
    """Fibonacci position → token-sector mapping table.

    Learns which corpus-native leaf cluster tends to appear at each
    Fibonacci clause position. The corpus discovers its own positional
    grammar — no POS labels, no word lists.

    Level 1 (Leaf): FibBin → leaf distribution  (which cluster here?)
    Level 2 (Token): per n-gram preferred FibBin (where does each token appear?)

    Circle prevention: same leaf at consecutive positions → attenuate boost,
    let other leaves compete.

    Self-critique: rejected samples apply weight=-0.3 (Level 1 only),
    pushing bad placements back toward the uniform prior.
    """

    _TIER1_PROB     = 0.50
    _TIER2_PROB     = 0.35
    _MIN_OBS        = 5
    _N_SECTOR_LEAVES = 7   # one per Fibonacci length class: char/bi/tri/5g/8g/13g/21g

    def __init__(self, n_leaves: int = _N_SECTOR_LEAVES):
        self.n_bins   = 10
        self.n_leaves = self._N_SECTOR_LEAVES   # always 7 — ignore caller arg
        self.counts   = torch.ones(self.n_bins, self._N_SECTOR_LEAVES) / self._N_SECTOR_LEAVES
        self.token_counts: torch.Tensor | None = None
        self._vocab: list | None = None
        self._vocab_size: int = 0
        self._probs_cache: torch.Tensor | None = None
        self._n_updates = 0
        self._n_trans   = 0  # kept for checkpoint compat, not used
        self._sector_ids: torch.Tensor | None = None   # cached per-token sector

    @staticmethod
    def _fib_bin(clause_pos: int) -> int:
        if clause_pos <= 0:
            return 0
        F = _FIB_NUMS_FOR_BIGRAM
        for i in range(len(F) - 1):
            if F[i] <= clause_pos < F[i + 1]:
                return min(i, 9)
        return 9

    def _invalidate(self):
        self._probs_cache = None

    @property
    def probs(self) -> torch.Tensor:
        if self._probs_cache is None:
            row_sums = self.counts.sum(dim=1, keepdim=True)
            self._probs_cache = self.counts / (row_sums + 1e-8)
        return self._probs_cache

    def resize(self, n_leaves: int):
        # Sector-based table always has _N_SECTOR_LEAVES leaves.
        # Fingerprint-driven resize calls (with arbitrary n_leaves) are ignored
        # so the table stays stable across fingerprint re-clusterings.
        if n_leaves == self._N_SECTOR_LEAVES:
            return  # already correct
        # Wrong size: rebuild with sector count (discards old data).
        self.n_leaves = self._N_SECTOR_LEAVES
        self.counts = torch.ones(self.n_bins, self._N_SECTOR_LEAVES) / self._N_SECTOR_LEAVES
        self._sector_ids = None
        self._invalidate()

    def update_from_tokens(self,
                            tokens: torch.Tensor,
                            fingerprint,
                            vocab: list,
                            n_chars: int,
                            weight: float = 1.0):
        if fingerprint is None or not fingerprint._built:
            return
        # Always operate in sector mode — n_leaves == _N_SECTOR_LEAVES
        V = len(vocab)
        if self.token_counts is None or self._vocab_size != V:
            self.token_counts = torch.zeros(V, self.n_bins)
            self._vocab_size = V
        if self._vocab is None:
            self._vocab = vocab

        toks = tokens[0].tolist() if tokens.dim() == 2 else tokens.tolist()
        clause_pos = 0
        for tid in toks:
            if tid < 0 or tid >= V:
                continue
            tok = vocab[tid]
            if tok in ('.', '!', '?', '\n'):
                clause_pos = 0
                continue
            if tok in (',', ';', ':'):
                continue
            # All token sectors participate — chars are sector 0, not skipped.
            # The table learns positional grammar across ALL chunk sizes.
            fb  = self._fib_bin(clause_pos)
            lid = _fib_tok_sector(tid, vocab)
            _fb_fib = [1, 1, 2, 3, 5, 8, 13]   # F(0)..F(6) for bins 0..6
            import math as _math
            _fb_ow = _fb_fib[min(fb, len(_fb_fib)-1)] / (_PHI_FOR_SAMPLING ** (_math.pi * fb))
            new_c = self.counts[fb, lid] + weight * _fb_ow
            self.counts[fb, lid] = max(1.0 / self.n_leaves, new_c)
            if weight > 0:
                self.token_counts[tid, fb] += weight
                self._n_updates += 1
            clause_pos += 1
        self._sector_ids = None   # invalidate cached sector tensor
        self._invalidate()

    def _get_sector_ids(self, V: int, device) -> torch.Tensor:
        """Return [V] tensor of sector IDs, cached for speed."""
        vocab = self._vocab
        if (self._sector_ids is None
                or self._sector_ids.shape[0] != V
                or self._sector_ids.device != device):
            if vocab is None:
                return torch.zeros(V, dtype=torch.long, device=device)
            ids = torch.tensor(
                [_fib_tok_sector(i, vocab) for i in range(V)],
                dtype=torch.long, device=device)
            self._sector_ids = ids
        return self._sector_ids

    def boost(self,
              clause_pos: int,
              base: torch.Tensor,
              fingerprint,
              prev_leaf: int = -1) -> torch.Tensor:
        if self._n_updates < 10 or fingerprint is None or not fingerprint._built:
            return base
        phi = _PHI_FOR_SAMPLING
        fb  = self._fib_bin(clause_pos)
        p       = self.probs[fb]           # [_N_SECTOR_LEAVES]
        uniform = 1.0 / self.n_leaves
        V       = min(base.shape[0], len(self._vocab) if self._vocab else base.shape[0])
        sector_ids = self._get_sector_ids(V, base.device)

        boost_vec = torch.ones_like(base)
        for lid in range(self.n_leaves):
            p_lid  = float(p[lid].item())
            excess = p_lid - uniform
            if prev_leaf == lid and excess > 0.02:
                excess *= 0.25   # circle prevention: don't loop in same sector
            if excess > 0.02:
                strength = min(excess * self.n_leaves, 1.0)
                tok_mask = (sector_ids == lid).float()
                if tok_mask.shape[0] < boost_vec.shape[0]:
                    tok_mask = F.pad(tok_mask, (0, boost_vec.shape[0] - tok_mask.shape[0]))
                boost_vec = boost_vec * (1.0 + tok_mask * strength * (phi - 1.0))
            elif excess < -0.04:
                tok_mask = (sector_ids == lid).float()
                if tok_mask.shape[0] < boost_vec.shape[0]:
                    tok_mask = F.pad(tok_mask, (0, boost_vec.shape[0] - tok_mask.shape[0]))
                damp = min(-excess * self.n_leaves * 0.5, 0.5)
                boost_vec = boost_vec * (1.0 - tok_mask * damp)

        # Level 2: Token-marker boost (tiered by positional specificity)
        if self.token_counts is not None:
            Vb = base.shape[0]
            tc = self.token_counts[:Vb]
            totals = tc.sum(dim=1)
            well_obs = totals >= self._MIN_OBS
            if well_obs.any():
                tok_probs = tc / (totals.unsqueeze(1) + 1e-8)
                at_bin = tok_probs[:, fb]
                dev = base.device
                t1 = well_obs & (at_bin >= self._TIER1_PROB)
                t2 = well_obs & (at_bin >= self._TIER2_PROB) & ~t1
                if t1.any():
                    boost_vec = boost_vec.clone()
                    boost_vec[t1.to(dev)] *= 1.0 + (phi * phi - 1.0) * at_bin[t1].to(dev)
                if t2.any():
                    if not t1.any():
                        boost_vec = boost_vec.clone()
                    boost_vec[t2.to(dev)] *= 1.0 + (phi - 1.0) * at_bin[t2].to(dev)

        out = base * boost_vec
        s = out.sum()
        return out / (s + 1e-8) if s > 0 else base

    def summary(self) -> str:
        p = self.probs
        F = _FIB_NUMS_FOR_BIGRAM
        lines = []
        for b in range(self.n_bins):
            dominant_idx = int(p[b].argmax().item())
            dominant_p   = float(p[b, dominant_idx].item())
            pos_range = f"F[{F[b]}-{F[b+1]})" if b+1 < len(F) else f"F[{F[b]}+]"
            marker_str = ""
            if self.token_counts is not None and self._vocab is not None:
                Vv = self.token_counts.shape[0]
                tc = self.token_counts
                totals = tc.sum(dim=1)
                tok_probs = tc / (totals.unsqueeze(1) + 1e-8)
                at_bin = tok_probs[:, b]
                eligible = (totals >= self._MIN_OBS) & (at_bin >= self._TIER2_PROB)
                if eligible.any():
                    idxs = eligible.nonzero(as_tuple=True)[0]
                    sorted_i = idxs[at_bin[idxs].argsort(descending=True)][:4]
                    parts = []
                    for i in sorted_i:
                        tok_str = self._vocab[i.item()] if i.item() < len(self._vocab) else "?"
                        tier = "T1" if at_bin[i].item() >= self._TIER1_PROB else "T2"
                        parts.append(f"{tok_str}({tier}:{at_bin[i]:.2f})")
                    marker_str = "  [" + ", ".join(parts) + "]"
            sector_name = _SECTOR_NAMES[dominant_idx] if dominant_idx < len(_SECTOR_NAMES) else str(dominant_idx)
            lines.append(f"    {pos_range:<10} → {sector_name:<6} ({dominant_p:.2f}){marker_str}")
        return '\n'.join(lines)


def substrate_discourse_position(after_speaker: bool,
                                  clause_pos: int,
                                  speaker_count: int,
                                  base: torch.Tensor,
                                  first_mask: torch.Tensor,
                                  second_mask: torch.Tensor,
                                  third_mask: torch.Tensor) -> torch.Tensor:
    """Discourse position register — substrate-universal.

    Models the four positional roles present in all human discourse:
      I   (speaker)   — first person: claimed immediately after speaker shift
      Thou (addressee) — second person: addressed at clause mid-point
      He/She (referent) — third person: referenced for absent parties
      Audience — implicit; handled by occasional 2nd-person plural burst

    Activation rules (substrate-canonical Fibonacci thresholds):
      after_speaker=True + clause_pos < F(3)=3 :
          boost first-person  — new speaker claims identity
      clause_pos in [F(3), F(5)) = [3, 8) :
          boost second-person — address the interlocutor
      clause_pos >= F(5)=8 :
          boost third-person  — reference absent parties or events
      Every F(6)=13 speaker shifts:
          strong second-person burst — audience-aside moment

    No word knowledge — purely grammatical person (universal).
    """
    phi = _PHI_FOR_SAMPLING
    F   = _FIB_NUMS_FOR_BIGRAM
    out = base.clone()
    if after_speaker and clause_pos < F[3]:
        boost = 1.0 + first_mask.to(base.device) * (phi - 1.0)
        out = out * boost
    elif clause_pos < F[5]:
        boost = 1.0 + second_mask.to(base.device) * (phi - 1.0)
        out = out * boost
    else:
        boost = 1.0 + third_mask.to(base.device) * (phi - 1.0)
        out = out * boost
    if speaker_count > 0 and speaker_count % F[6] == 0:
        boost_aud = 1.0 + second_mask.to(base.device) * (phi ** 2 - 1.0)
        out = out * boost_aud
    s = out.sum()
    return out / (s + 1e-8) if s > 0 else base


def substrate_listener_register(heard_counts: torch.Tensor,
                                 base: torch.Tensor,
                                 n_chars: int) -> torch.Tensor:
    """Listener knowledge register — substrate-universal.

    The listener accumulates everything the speaker has said. A good
    speaker says something NEW each clause — not repeating what has
    already been received.

    For each word token (id >= n_chars) count how many times it has been
    emitted in the current utterance (since last major boundary). Apply
    phi-suppression at Fibonacci thresholds:
      count >= F(3)=3 : mild suppression (phi^-1  ≈ 0.618x)
      count >= F(4)=5 : strong suppression (phi^-pi ≈ 0.22x)

    Char tokens are unaffected (they carry structural, not semantic, info).
    substrate-universal: forward narrative progress is a universal
    communication requirement — the listener needs new information.
    """
    phi_pi = _PHI_FOR_SAMPLING ** math.pi
    phi_half = _PHI_FOR_SAMPLING ** 0.5   # sqrt(phi) ≈ 1.272 → 1/sqrt(phi) ≈ 0.786
    F = _FIB_NUMS_FOR_BIGRAM
    suppress = torch.ones_like(base)
    word_counts = heard_counts.clone()
    word_counts[:n_chars] = 0.0   # only content word tokens
    suppress = torch.where(word_counts >= F[4],
                            torch.full_like(base, 1.0 / phi_pi),
                            suppress)
    suppress = torch.where((word_counts >= F[3]) & (word_counts < F[4]),
                            torch.full_like(base, 1.0 / _PHI_FOR_SAMPLING),
                            suppress)
    # Third tier: "recently heard" — decayed cross-line counts (0.5 to F[3]).
    # After heard_counts decay at \n, previous-line words land here.
    # Mild 1/sqrt(phi) suppression encourages fresh vocabulary per sentence.
    suppress = torch.where((word_counts >= 0.5) & (word_counts < F[3]),
                            torch.full_like(base, 1.0 / phi_half),
                            suppress)
    out = base * suppress
    s = out.sum()
    return out / (s + 1e-8) if s > 0 else base


def substrate_minimum_clause_length(
        clause_pos: int,
        clause_has_verb: bool,
        base: torch.Tensor,
        terminal_punct_mask: torch.Tensor = None,
        newline_mask: torch.Tensor = None) -> torch.Tensor:
    """Language primitive: a clause needs minimum content to communicate.

    Suppresses terminal punctuation (. ! ?) and newlines when the clause
    has fewer than F(3)=3 content words — the clause hasn't started yet.
    Also suppresses terminal punct (not newline) when no verb has been
    emitted yet and clause_pos < F(5)=8 — a predicateless fragment.

    Universal: all natural languages require a predicate to make a
    proposition. The minimum-length threshold is substrate-Fibonacci.
    Newlines end a 'line' of speech; terminal punct ends a sentence.
    Both require the clause to have begun.
    """
    F = _FIB_NUMS_FOR_BIGRAM
    phi = _PHI_FOR_SAMPLING

    if clause_pos >= F[5]:
        return base  # long enough — no suppression needed

    out = base.clone()

    if clause_pos < F[3]:
        # Hard suppress sentence enders: clause hasn't started (< 3 words)
        if terminal_punct_mask is not None:
            out = out * (1.0 - terminal_punct_mask.to(base.device)
                         * (1.0 - 1.0 / (phi ** 3)))
        if newline_mask is not None:
            out = out * (1.0 - newline_mask.to(base.device)
                         * (1.0 - 1.0 / phi))
    elif not clause_has_verb:
        # Mild suppress sentence enders: words present but no predicate yet
        if terminal_punct_mask is not None:
            out = out * (1.0 - terminal_punct_mask.to(base.device)
                         * (1.0 - 1.0 / phi))

    s = out.sum()
    return out / (s + 1e-8) if s > 0 else base


def substrate_topic_closure(
        clause_subject_id: int,
        clause_pos: int,
        base: torch.Tensor,
        bigram_prior: torch.Tensor) -> torch.Tensor:
    """Topic-closure binding: subject (A) conditions predicate closure (D).

    In all natural languages a clause has A→D structure: the topic/subject
    announced at the start (A) determines what the closure (D) says, even
    if B and C wander freely between them. Without this, B and C drift off
    tangent and D has no semantic relation to A.

    Implementation: store the first content word of each clause as the
    'subject anchor'. At the closure slot (clause_pos >= F[7]=21), boost
    tokens whose bigram co-occurrence with the anchor is high. The bigram
    row bigram_prior[subject_id, :] encodes what naturally follows that
    subject in the training corpus — no word list, purely learned structure.

    Language primitive: subject-predicate semantic coherence is universal.
    The bigram prior is corpus-learned, not Shakespeare-specific.
    """
    F = _FIB_NUMS_FOR_BIGRAM
    phi = _PHI_FOR_SAMPLING

    if clause_subject_id < 0 or clause_pos < F[4]:  # F[4]=5: active from predicate slot
        return base
    if bigram_prior is None or clause_subject_id >= bigram_prior.shape[0]:
        return base

    subj_row = bigram_prior[clause_subject_id].to(base.device)
    # Normalize to [0,1] range so it's a soft preference, not a hard prior
    row_min, row_max = subj_row.min(), subj_row.max()
    if row_max > row_min:
        subj_row = (subj_row - row_min) / (row_max - row_min)

    # Closure weight grows with clause depth: full at F[6]=13+
    depth_weight = min(1.0, (clause_pos - F[4]) / float(F[6] - F[4]))
    boost = 1.0 + subj_row * (phi - 1.0) * depth_weight
    out = base * boost
    s = out.sum()
    return out / (s + 1e-8) if s > 0 else base


def substrate_golden_phase(t_pos: int, probs: torch.Tensor,
                              vocab_size: int) -> torch.Tensor:
    """Golden-angle phase: functional/content rhythm primitive.

    Each position advances a phase counter by the golden angle
    2*pi/phi^2 -- the irrational angle that maximizes spread
    (sunflower/phyllotaxis canon). cos(phase) modulates rank polarity:

      cos(phase) ≈ +1  -> boost LOW rank (functional, common)
      cos(phase) ≈ -1  -> boost HIGH rank (content, rare/proper-noun)
      cos(phase) ≈  0  -> neutral (mixed)

    Positional axis = "how far". Phase axis = "what kind, now".
    Together they collapse word order into the substrate groove.

    Polarity:  pol(r) = 1 - 2*r/(V-1)  in [-1, +1].
    Log-boost: log(phi) * cos(phase) * pol  (max boost phi, min 1/phi).
    Substrate-bounded.
    """
    phi = _PHI_FOR_SAMPLING
    if vocab_size <= 1:
        return probs
    golden_angle = 2.0 * math.pi / (phi ** 2)
    cos_phase = math.cos(t_pos * golden_angle)
    ranks = torch.arange(vocab_size, dtype=probs.dtype,
                          device=probs.device)
    rank_pol = 1.0 - 2.0 * ranks / (vocab_size - 1)
    log_boost = math.log(phi) * cos_phase * rank_pol
    boost = torch.exp(log_boost)
    out = probs * boost
    return out / (out.sum() + 1e-8)


def substrate_subject_threading(sequence: list, vocab: list,
                                    probs: torch.Tensor,
                                    is_sentence_start: bool) -> torch.Tensor:
    """Cross-sentence dependency: at sentence-start positions, boost
    tokens that appeared at past sentence-starts (likely subjects).

    Maintains a substrate-canonical memory: the last F(5)=8 sentence-
    starts. Each contributes a boost F(k)/phi^(pi*k) where k = how
    many sentences ago. Most-recent subject boosted full F(0)=1;
    older subjects decay by phi^pi per sentence.

    Substrate "topic threading" across paragraph scale.
    """
    if not is_sentence_start or not vocab:
        return probs
    # Find tokens at sentence-start positions in the sequence.
    sentence_starts = []
    for i, tok_id in enumerate(sequence):
        tok = vocab[tok_id] if tok_id < len(vocab) else ''
        # A token is a sentence-start if it follows .!?, newline,
        # OR is at position 0.
        if i == 0:
            sentence_starts.append(tok_id)
            continue
        prev = vocab[sequence[i-1]] if sequence[i-1] < len(vocab) else ''
        if prev in ('.', '!', '?', '\n'):
            # The current token is the subject of a new sentence.
            sentence_starts.append(tok_id)
    if not sentence_starts:
        return probs
    # Keep last F(5)=8 sentence-starts.
    sentence_starts = sentence_starts[-8:]
    n = len(sentence_starts)
    phi_pi = _PHI_FOR_SAMPLING ** math.pi
    boost = torch.zeros_like(probs)
    for i, tok_id in enumerate(reversed(sentence_starts)):
        # i=0 = most recent sentence-start
        k_tier = min(i, len(_FIB_NUMS_FOR_BIGRAM) - 1)
        weight = (_FIB_NUMS_FOR_BIGRAM[k_tier]
                  / (phi_pi ** k_tier))
        boost[tok_id] += weight
    # Apply boost multiplicatively (substrate-canonical log-boost).
    boost_factor = 1.0 + boost * (math.pi * math.log(_PHI_FOR_SAMPLING))
    out = probs * boost_factor
    return out / (out.sum() + 1e-8)


def substrate_sentence_boundary_boost(prev_token: int, vocab: list,
                                          probs: torch.Tensor) -> torch.Tensor:
    """Substrate sentence-boundary primitive.

    If prev_token is end-of-sentence punctuation (. ! ?), boost newline
    + space candidates substantially -- a sentence should end with
    proper boundary. If prev_token is newline, boost rank-0 (most
    common functional) candidates -- new sentence starts with a
    function word likely.

    Boost coefficient: log(phi^pi) ~ 1.51, substrate-canonical.
    """
    if not vocab:
        return probs
    prev_str = vocab[prev_token] if prev_token < len(vocab) else ''
    boost = math.pi * math.log(_PHI_FOR_SAMPLING)
    if prev_str in ('.', '!', '?'):
        # Sentence ended -- boost newline/space.
        for i, tok in enumerate(vocab):
            if tok in ('\n', ' '):
                probs[i] = probs[i] * (1.0 + boost)
        probs = probs / (probs.sum() + 1e-8)
    elif prev_str == '\n':
        # New sentence -- boost rank-0..F(7)=13 functional words.
        for i in range(min(13, len(vocab))):
            probs[i] = probs[i] * (1.0 + boost / 2)
        probs = probs / (probs.sum() + 1e-8)
    return probs


def substrate_syntax_blend(prev_token: int, bigram_prior: torch.Tensor,
                              probs: torch.Tensor,
                              prev_prev_token: int = None,
                              context_tokens: list = None,
                              vocab: list = None) -> torch.Tensor:
    """Substrate syntax blend with GRADUATED multi-back context + gate.

    Graduated form: contributions from t-1, t-2, ..., t-N positions
    are weighted F(0), F(1)/phi^pi, F(2)/phi^(2pi), ..., F(k)/phi^(pi*k).
    Substrate-tier-decayed influence across the recent context window.
    Beyond simple bigram or 2-back trigram -- arbitrary lookback,
    each position contributing per substrate decay.

    Then syntactic-incorrect gate suppresses low-prior candidates.
    Then 1/phi^pi blend with model probs.

    context_tokens: list of previous tokens (most recent last). If
    None, falls back to prev_token + prev_prev_token. Pure substrate
    tier-decay multi-back is the deepest version.
    """
    if context_tokens is None:
        context_tokens = [prev_token]
        if prev_prev_token is not None:
            context_tokens = [prev_prev_token, prev_token]

    # Graduated tier-weighted combination of bigrams from each position.
    K = len(_FIB_NUMS_FOR_BIGRAM)
    phi_pi = _PHI_FOR_SAMPLING ** math.pi
    n = len(context_tokens)
    combined_prior = torch.zeros_like(probs)
    total_w = 0.0
    for i, tok in enumerate(reversed(context_tokens[-K:])):
        # i=0 -> most recent (t-1), i=1 -> t-2, etc.
        w = _FIB_NUMS_FOR_BIGRAM[i] / (phi_pi ** i)
        prior_i = bigram_prior[tok].to(probs.device).to(probs.dtype)
        combined_prior = combined_prior + w * prior_i
        total_w += w
    combined_prior = combined_prior / (total_w + 1e-8)
    combined_prior = combined_prior / (combined_prior.sum() + 1e-8)

    V = probs.numel()
    threshold = 1.0 / (V * phi_pi)
    gate = torch.where(combined_prior >= threshold,
                         torch.ones_like(combined_prior),
                         combined_prior / threshold)
    gated_probs = probs * gate
    gated_probs = gated_probs / (gated_probs.sum() + 1e-8)
    blended = ((1.0 - _SUBSTRATE_BIGRAM_ALPHA) * gated_probs
                + _SUBSTRATE_BIGRAM_ALPHA * combined_prior)
    # Apply sentence-boundary boost as a final structural prior.
    if vocab is not None and prev_token < len(vocab):
        blended = substrate_sentence_boundary_boost(prev_token, vocab, blended)
    return blended


def substrate_syntax_boost(prev_token: int, bigram_prior: torch.Tensor,
                              logits: torch.Tensor) -> torch.Tensor:
    """Boost logits by log(phi^pi) * P(next | prev_token). DEPRECATED --
    too weak vs the model's confident logits. Use substrate_syntax_blend
    on probabilities instead."""
    log_phi_pi = math.pi * math.log(_PHI_FOR_SAMPLING)
    prior_row = bigram_prior[prev_token].to(logits.device).to(logits.dtype)
    return logits + log_phi_pi * prior_row


def substrate_clause_arc(clause_pos: int, base: torch.Tensor,
                           n_chars: int,
                           verb_mask: torch.Tensor = None,
                           noun_mask: torch.Tensor = None,
                           adj_mask: torch.Tensor = None,
                           func_mask: torch.Tensor = None) -> torch.Tensor:
    """Substrate clause-arc structural memory — now using real POS categories.

    Tracks position within the current clause (reset at every sentence
    boundary: . ! ? \\n) and applies Fibonacci-indexed grammatical guidance:

      pos < F(3)=3  : subject slot — boost nouns and pronouns (morphological)
      F(3)<=pos<F(5)=8 : predicate slot — boost verbs (morphological: -eth/-ing/aux)
      F(5)<=pos<F(7)=13: complement slot — boost adjectives and nouns (objects)
      pos >= F(7)=13: closure slot — boost function words toward clause end

    Language primitive: subject→predicate→complement→closure is the universal
    clause structure across all known languages (SVO/SOV/VSO all have these
    roles). Identification uses morphological rules, not a word list.
    """
    phi = _PHI_FOR_SAMPLING
    F   = _FIB_NUMS_FOR_BIGRAM

    if clause_pos < F[3]:
        # Subject slot: boost nouns + all word tokens
        boost = torch.ones_like(base)
        boost[n_chars:] = phi
        if noun_mask is not None:
            boost = boost * (1.0 + noun_mask.to(base.device) * (phi - 1.0))
        out = base * boost
        return out / (out.sum() + 1e-8)

    if clause_pos < F[5]:
        # Predicate slot: boost verbs
        if verb_mask is not None:
            boost = 1.0 + verb_mask.to(base.device) * (phi ** 2 - 1.0)
            out = base * boost
            return out / (out.sum() + 1e-8)
        return base

    if clause_pos < F[7]:
        # Complement slot: boost adjectives + nouns (objects, predicates)
        boost = torch.ones_like(base)
        if adj_mask is not None:
            boost = boost * (1.0 + adj_mask.to(base.device) * (phi - 1.0))
        if noun_mask is not None:
            boost = boost * (1.0 + noun_mask.to(base.device) * (phi - 1.0))
        out = base * boost
        return out / (out.sum() + 1e-8)

    # Closure slot: boost function words, gentle de-boost content words
    boost = torch.ones_like(base)
    boost[n_chars:] = 1.0 / phi
    if func_mask is not None:
        boost = boost * (1.0 + func_mask.to(base.device) * (phi - 1.0))
    out = base * boost
    return out / (out.sum() + 1e-8)


def substrate_anti_stagnation(history_tokens: torch.Tensor,
                                  probs: torch.Tensor,
                                  vocab_size: int) -> torch.Tensor:
    """Substrate-tier-stepped anti-stagnation correction.

    Counts each token's occurrences in the history window. At each
    Fibonacci threshold of repetition, applies progressively stronger
    phi^(pi*k) suppression to that token's sampling probability:

        count >= F(3)=3:  divide prob by phi^pi    (~0.22x)
        count >= F(4)=5:  divide prob by phi^(2pi) (~0.049x)
        count >= F(5)=8:  hard suppress (prob = 0)

    Substrate divergent: forces new tokens when current ones
    saturate.  Substrate-corrective: uses Fibonacci-tier thresholds
    + phi^pi suppression -- both signals from substrate constants
    alone.
    """
    n = history_tokens.numel()
    if n == 0:
        return probs
    counts = torch.bincount(history_tokens.long(), minlength=vocab_size)
    counts_f = counts.to(probs.device).to(probs.dtype)
    phi_pi = _PHI_FOR_SAMPLING ** math.pi
    # Thresholds shifted up by one Fibonacci tier to allow authentic Shakespeare
    # function words (naturally high-frequency) without zeroing them.
    #   count >= F(5)=8:  divide prob by phi^pi    (~0.22x mild)
    #   count >= F(6)=13: divide prob by phi^(2pi) (~0.05x strong)
    #   count >= F(8)=21: hard suppress (genuine loop saturation)
    suppress = torch.ones_like(probs)
    suppress = torch.where(counts_f >= 21.0,
                              torch.zeros_like(probs),
                              suppress)
    suppress = torch.where((counts_f >= 13.0) & (counts_f < 21.0),
                              torch.full_like(probs, 1.0 / (phi_pi ** 2)),
                              suppress)
    suppress = torch.where((counts_f >= 8.0) & (counts_f < 13.0),
                              torch.full_like(probs, 1.0 / phi_pi),
                              suppress)
    out = probs * suppress
    return out / (out.sum() + 1e-8)


def substrate_recency_penalty(history_tokens: torch.Tensor, logits: torch.Tensor,
                                 vocab_size: int) -> torch.Tensor:
    """Vectorized substrate-canonical recency penalty.

    Each token in `history_tokens` contributes a penalty to its own
    logit, weighted by golden-ratio decay over position. Most-recent
    position has weight 1.0; older positions decay by powers of phi.
    Substrate-canonical: phi is the golden ratio (natural recursive
    growth rate); log(phi) is the substrate's natural log-base unit.

    Args:
        history_tokens: 1D tensor of token IDs in chronological order.
        logits: 1D tensor of logits over vocab.
        vocab_size: V.

    Returns:
        Modified logits with penalties applied.
    """
    n = history_tokens.numel()
    if n == 0:
        return logits
    # Nested substrate decay: F(k)/phi^(pi*k) where k = pos_back.
    # Same nested form as the bigram tier decay and harmony loss.
    # Most-recent position (pos_back=0) gets F(0)/phi^0 = 1; older
    # positions decay via the fully-nested F(k)/phi^(pi*k).
    K = len(_FIB_NUMS_FOR_BIGRAM)
    pi_arg = math.pi
    pos_back = (n - 1 - torch.arange(n, device=logits.device,
                                          dtype=logits.dtype))
    pos_back_idx = torch.clamp(pos_back, 0, K - 1).long()
    fk_tensor = torch.tensor(
        [_FIB_NUMS_FOR_BIGRAM[i] / (_PHI_FOR_SAMPLING ** (pi_arg * i))
         for i in range(K)],
        dtype=logits.dtype, device=logits.device)
    pos_weights = fk_tensor[pos_back_idx]
    penalty = torch.zeros(vocab_size, device=logits.device, dtype=logits.dtype)
    penalty.scatter_add_(0, history_tokens.long(), pos_weights)
    return logits - penalty * _LOG_PHI_FOR_PENALTY


# OMNIWEIGHT: shared log-pressure ledger. Each primitive contributes
# delta_log_p to a single accumulator instead of chaining probs->probs
# transforms.
#
# FLUID backed-standard form (v72+): the substrate reserve phi^pi acts
# as a backing standard. Accumulator passes through tanh scaled by
# phi^pi -- small contributions pass nearly linear, large saturate
# gracefully to +/- phi^pi. No hard clamp; growth allowed in
# proportion to substrate trust.
_OMNIWEIGHT_RESERVE = _PHI_FOR_SAMPLING ** math.pi   # ~4.53


def _regret_score(seq: torch.Tensor, t: int, vocab: list,
                     n_chars: int = 65) -> float:
    """Per-position regret: how badly this emission shouldn't be there.

    Factors (substrate-pure):
      - over-emission: same token used F(5)+ times in last F(7)=13
      - immediate repetition: identical to previous token
      - bigram saturation: (prev, current) fired F(4)+ times in last F(7)
      - double punctuation: punct immediately after punct
      - mid-word char: char emission after another alpha char (no space)

    Higher score = more regret = should be resampled.
    """
    if t < 1 or t >= seq.shape[1]:
        return 0.0
    tid = int(seq[0, t].item())
    if tid >= len(vocab) or tid < 0:
        return 0.0
    tok = vocab[tid]
    regret = 0.0
    F = _FIB_NUMS_FOR_BIGRAM
    # Last F(7)=13 prior tokens.
    start = max(0, t - F[7])
    prior = seq[0, start:t].tolist()
    # Factor 1: over-emission
    same_count = sum(1 for x in prior if x == tid)
    if same_count > F[5]:
        regret += float(same_count - F[5]) / float(F[5])
    # Factor 2: immediate repetition
    prev_tid = int(seq[0, t - 1].item())
    if prev_tid == tid:
        regret += 1.0
    # Factor 3: bigram saturation
    bigram_count = 0
    for i in range(1, len(prior)):
        if prior[i - 1] == prev_tid and prior[i] == tid:
            bigram_count += 1
    if bigram_count > F[4]:
        regret += float(bigram_count - F[4]) / float(F[4])
    # Factor 4: double punctuation
    if tok in (',', '.', '!', '?', ';', ':') and prev_tid < len(vocab):
        prev_tok = vocab[prev_tid]
        if prev_tok in (',', '.', '!', '?', ';', ':'):
            regret += 1.0
    # Factor 5: mid-word char emission (char after another alpha char)
    if (tid < n_chars and tok and tok.isalpha()
            and prev_tid < len(vocab)):
        prev_tok = vocab[prev_tid]
        if (prev_tok and prev_tok != ' '
                and prev_tok[-1].isalpha()):
            regret += 0.5
    return regret


def _omniweight_delta(base_probs: torch.Tensor,
                          modified_probs: torch.Tensor) -> torch.Tensor:
    """Compute delta_log_p = log(modified) - log(base). Each primitive
    is wrapped: it still returns modified probs, the wrapper extracts
    the log-space contribution.
    """
    eps = 1e-12
    return (torch.log(modified_probs.clamp(min=eps))
            - torch.log(base_probs.clamp(min=eps)))



def substrate_unknown_register(coverage: torch.Tensor,
                                  probs: torch.Tensor,
                                  retrocausal_steps: int = None,
                                  ) -> torch.Tensor:
    """UNKNOWN-REGISTER with retrocausality.

    Present unknown: 1/(1+coverage) -- past-conditioned frontier.
    Retrocausal: project coverage forward by F(3)=2 expected steps
    using current probs distribution, then compute frontier of
    the ANTICIPATED state. The future-that-would-happen feeds back
    into the current emission.

    Final frontier = (1-alpha)*present_frontier + alpha*anticipated_frontier
      alpha = 1/phi^pi ~ 0.221

    Then mix probs with that blended frontier (substrate alpha).

    Time isn't linear: past coverage and anticipated coverage are
    both present-tense registers in the same currency.
    """
    if coverage is None:
        return probs
    if retrocausal_steps is None:
        # F(2)=1: just ONE step lookahead, continuity-respecting.
        # F(3)=2 was a discontinuous jump (ignored intermediate state).
        retrocausal_steps = _FIB_NUMS_FOR_BIGRAM[2]   # F(2) = 1
    cov = coverage.to(probs.device).to(probs.dtype)
    # Present unknown
    inv_now = 1.0 / (1.0 + cov)
    frontier_now = inv_now / (inv_now.sum() + 1e-8)
    # Anticipated unknown (retrocausal): coverage projected F(3) forward
    # by current sampling distribution
    expected_delta = float(retrocausal_steps) * probs
    inv_future = 1.0 / (1.0 + cov + expected_delta)
    frontier_future = inv_future / (inv_future.sum() + 1e-8)
    # Blend past-frontier and future-frontier (both positive registers)
    alpha_retro = 1.0 / (_PHI_FOR_SAMPLING ** math.pi)
    blended_frontier = ((1.0 - alpha_retro) * frontier_now
                          + alpha_retro * frontier_future)
    blended_frontier = blended_frontier / (blended_frontier.sum() + 1e-8)
    # Apply blended frontier as omniweight contribution
    alpha = 1.0 / (_PHI_FOR_SAMPLING ** math.pi)
    out = (1.0 - alpha) * probs + alpha * blended_frontier
    return out / (out.sum() + 1e-8)


def _self_eval_insight(base_probs: torch.Tensor, emitted_tid: int,
                          n_chars: int = 65) -> float:
    """Compute self-evaluation insight signal for a just-emitted token.

    insight = 1 if:
      - emitted token is a real word (rank >= n_chars), AND
      - surprise (-log p_emitted) >= pi*log(phi) ~ 1.51 (substrate threshold)
    insight = 0 otherwise.

    Recursive substrate self-monitoring: model rates its own emissions
    against its own distribution.
    """
    if emitted_tid < n_chars:
        return 0.0
    V = base_probs.shape[0]
    if not (0 <= emitted_tid < V):
        return 0.0
    p = float(base_probs[emitted_tid].item())
    if p <= 0.0:
        return 0.0
    surprise = -math.log(p + 1e-12)
    threshold = math.pi * math.log(_PHI_FOR_SAMPLING)
    return 1.0 if surprise >= threshold else 0.0


def _omniweight_apply_split(base_probs: torch.Tensor,
                                math_delta: torch.Tensor,
                                lang_delta: torch.Tensor,
                                momentum: float = 0.0) -> torch.Tensor:
    """RANK-MODULATED split-brain mixer with momentum-modulated reserve.

    Each hemisphere builds fluid delta via tanh-scaled reserve.
    Reserve scaled by (1 + tanh(momentum)) -- when recent emissions
    have been insightful (high surprise + real word), primitives get
    more room. When noisy/expected, primitives constrained.

    Per-token weight by rank: math owns low rank, lang owns high rank.
    """
    # Momentum-modulated reserve (recursive substrate self-trust).
    reserve = _OMNIWEIGHT_RESERVE * (1.0 + math.tanh(momentum))
    if reserve < 1e-3:
        reserve = 1e-3
    math_fluid = reserve * torch.tanh(math_delta / reserve)
    lang_fluid = reserve * torch.tanh(lang_delta / reserve)
    p_math = base_probs * torch.exp(math_fluid)
    p_lang = base_probs * torch.exp(lang_fluid)
    p_math = p_math / (p_math.sum() + 1e-8)
    p_lang = p_lang / (p_lang.sum() + 1e-8)
    V = base_probs.shape[-1]
    ranks = torch.arange(V, dtype=base_probs.dtype,
                          device=base_probs.device)
    rank_norm = ranks / max(V - 1, 1)
    math_w = 1.0 - rank_norm
    lang_w = rank_norm
    p_final = math_w * p_math + lang_w * p_lang
    return p_final / (p_final.sum() + 1e-8)


class SubstrateFingerprint:
    """Substrate-native semantic fingerprinting via Fibonacci co-occurrence.

    Derives what tokens ARE from corpus structure — not a word list.
    Names, places, objects, abstracts each cluster differently because
    they co-occur with different neighbors in the training corpus.

    Three levels of relational context (all Fibonacci-weighted):
      fwd[A, d]  — what tokens typically FOLLOW A (forward context)
      bwd[A, d]  — what tokens typically PRECEDE A (backward context)
      rel[A, d]  — second-order: what kind of context A's followers lead into
                   (A→B's semantic neighborhood — "juliet leads toward
                    love/person context; means leads toward function context")

    Cosine k-means on fwd fingerprints → semantic clusters emerge without labels.
    Typical corpus-derived clusters: persons / places+objects / verbs / function-words.

    Hierarchical: n_clusters main clusters, each split into n_sub_clusters sub-clusters.
    Default 4×3=12 leaf nodes — enough resolution to separate proper names from
    institutions from political actions within the same thematic main cluster.

    Context tape: O(1) Fibonacci-decayed running state for generation.
    Cluster diversity: uses sub_clusters for fine-grained bloom weighting —
    sequences spanning more leaf nodes get proportionally higher weight.
    """

    _FIB_OFFSETS = (1, 2, 3, 5, 8)
    _PHI = 1.6180339887

    def __init__(self, vocab_size: int, d_model: int,
                 n_clusters: int = 4, n_sub_clusters: int = 3):
        self.V            = vocab_size
        self.d            = d_model
        self.fwd          = torch.zeros(vocab_size, d_model)
        self.bwd          = torch.zeros(vocab_size, d_model)
        self.rel          = torch.zeros(vocab_size, d_model)
        self.clusters     = torch.zeros(vocab_size, dtype=torch.long)
        self.sub_clusters = torch.zeros(vocab_size, dtype=torch.long)
        self.n_clusters     = n_clusters
        self.n_sub_clusters = n_sub_clusters
        self.n_leaves       = n_clusters * n_sub_clusters   # 12 by default
        self._built         = False

    def build(self, corpus_tokens: torch.Tensor, model, device=None):
        """Compute fingerprints from corpus token sequence (vectorised).

        Runs entirely in torch with scatter_add — no Python loops over tokens.
        Safe to call with a few thousand tokens in milliseconds.
        """
        if device is None:
            device = next(model.parameters()).device
        toks  = corpus_tokens.view(-1).to(device)
        N     = toks.numel()
        V, d  = self.V, self.d
        phi   = self._PHI

        fwd_acc = torch.zeros(V, d, device=device)
        bwd_acc = torch.zeros(V, d, device=device)
        fwd_cnt = torch.zeros(V,    device=device)
        bwd_cnt = torch.zeros(V,    device=device)

        with torch.no_grad():
            embs = model.embed(toks).float()   # [N, d]
            for k_idx, offset in enumerate(self._FIB_OFFSETS):
                if N <= offset:
                    continue
                w = float(1.0 / (phi ** k_idx))
                M = N - offset

                # Forward: fwd_acc[toks[i]] += w * embs[i+offset]
                a_idx = toks[:M].long().clamp(0, V - 1)
                ctx   = embs[offset:N]
                fwd_acc.scatter_add_(0,
                    a_idx.unsqueeze(1).expand(-1, d), w * ctx)
                fwd_cnt.scatter_add_(0, a_idx,
                    torch.full((M,), w, device=device))

                # Backward: bwd_acc[toks[i+offset]] += w * embs[i]
                b_idx = toks[offset:N].long().clamp(0, V - 1)
                src   = embs[:M]
                bwd_acc.scatter_add_(0,
                    b_idx.unsqueeze(1).expand(-1, d), w * src)
                bwd_cnt.scatter_add_(0, b_idx,
                    torch.full((M,), w, device=device))

        # Normalize to get average context embeddings
        mf = fwd_cnt > 1e-8
        fwd_acc[mf] = fwd_acc[mf] / fwd_cnt[mf].unsqueeze(1)
        mb = bwd_cnt > 1e-8
        bwd_acc[mb] = bwd_acc[mb] / bwd_cnt[mb].unsqueeze(1)

        self.fwd = fwd_acc.cpu()
        self.bwd = bwd_acc.cpu()

        # Second-order (A→B context): for each A, average the fwd fingerprints
        # of tokens that appear AFTER A.  Captures the semantic neighborhood
        # that A leads into — "juliet → person-interaction space."
        rel_acc = torch.zeros(V, d, device=device)
        rel_cnt = torch.zeros(V,    device=device)
        with torch.no_grad():
            fwd_dev = self.fwd.to(device)
            for offset in (1, 2, 3):
                if N <= offset:
                    continue
                M     = N - offset
                a_idx = toks[:M].long().clamp(0, V - 1)   # token A
                b_idx = toks[offset:N].long().clamp(0, V - 1)  # token B after A
                b_fwd = fwd_dev[b_idx]                    # B's forward fingerprint
                rel_acc.scatter_add_(0,
                    a_idx.unsqueeze(1).expand(-1, d), b_fwd)
                rel_cnt.scatter_add_(0, a_idx,
                    torch.ones(M, device=device))
        mr = rel_cnt > 1e-8
        rel_acc[mr] = rel_acc[mr] / rel_cnt[mr].unsqueeze(1)
        self.rel = rel_acc.cpu()

        self._cluster(device)
        self._build_leaf_centroids(device)
        self._build_cluster_transitions(toks, device)
        self._built = True
        if hasattr(self, '_leaf_coherences'):
            del self._leaf_coherences  # invalidate cache — leaves changed

    def update_fwd_bwd(self, corpus_tokens: torch.Tensor, model, device=None):
        """Recompute ONLY fwd/bwd/rel vectors from a richer corpus.

        Called after build() to inject topology from a larger corpus without
        disturbing the leaf clustering derived from the training corpus.
        Clustering (sub_clusters, cluster_transition, leaf_centroids) is unchanged.
        """
        if not self._built:
            return
        if device is None:
            device = next(model.parameters()).device
        toks  = corpus_tokens.view(-1).to(device)
        N     = toks.numel()
        V, d  = self.V, self.d
        phi   = self._PHI

        fwd_acc = torch.zeros(V, d, device=device)
        bwd_acc = torch.zeros(V, d, device=device)
        fwd_cnt = torch.zeros(V,    device=device)
        bwd_cnt = torch.zeros(V,    device=device)

        with torch.no_grad():
            embs = model.embed(toks).float()
            for k_idx, offset in enumerate(self._FIB_OFFSETS):
                if N <= offset:
                    continue
                w = float(1.0 / (phi ** k_idx))
                M = N - offset
                a_idx = toks[:M].long().clamp(0, V - 1)
                ctx   = embs[offset:N]
                fwd_acc.scatter_add_(0, a_idx.unsqueeze(1).expand(-1, d), w * ctx)
                fwd_cnt.scatter_add_(0, a_idx, torch.full((M,), w, device=device))
                b_idx = toks[offset:N].long().clamp(0, V - 1)
                src   = embs[:M]
                bwd_acc.scatter_add_(0, b_idx.unsqueeze(1).expand(-1, d), w * src)
                bwd_cnt.scatter_add_(0, b_idx, torch.full((M,), w, device=device))

        mf = fwd_cnt > 1e-8
        fwd_acc[mf] = fwd_acc[mf] / fwd_cnt[mf].unsqueeze(1)
        mb = bwd_cnt > 1e-8
        bwd_acc[mb] = bwd_acc[mb] / bwd_cnt[mb].unsqueeze(1)
        self.fwd = fwd_acc.cpu()
        self.bwd = bwd_acc.cpu()

        rel_acc = torch.zeros(V, d, device=device)
        rel_cnt = torch.zeros(V,    device=device)
        with torch.no_grad():
            fwd_dev = self.fwd.to(device)
            for offset in (1, 2, 3):
                if N <= offset:
                    continue
                src_toks = toks[:-offset].long().clamp(0, V - 1)
                suc_toks = toks[offset:].long().clamp(0, V - 1)
                suc_fwd  = fwd_dev[suc_toks]
                rel_acc.scatter_add_(0, src_toks.unsqueeze(1).expand(-1, d), suc_fwd)
                rel_cnt.scatter_add_(0, src_toks, torch.ones(src_toks.shape[0], device=device))
        mr = rel_cnt > 1e-8
        rel_acc[mr] = rel_acc[mr] / rel_cnt[mr].unsqueeze(1)
        self.rel = rel_acc.cpu()

    # Fractal clustering constants (Fibonacci-native)
    _BRANCH         = 3     # F(4)=3  children per split
    _MIN_SPLIT      = 5     # F(5)=5  minimum tokens per leaf — floor for recursion
    _MIN_LEAVES     = 21    # F(8)=21 minimum TOTAL leaves — FibPosTable needs one bin
                            # per Fibonacci number; forced expansion guarantees this
    _MAX_DEPTH      = 5     # F(5)=5  safety ceiling — corpus coherence stops first
    _COHERENCE_STOP = 0.90  # corpus-native stop: mean pairwise cos-sim > this → genuine leaf
    _MAX_LEAF_FRAC  = 1.0 / (_PHI_FOR_SAMPLING ** 3)   # ≈ 0.236
                            # forces the void/function-word blob to subdivide

    def _kmeans_local(self, unit: torch.Tensor, K: int,
                      n_iter: int = 15,
                      init_centroids: 'torch.Tensor | None' = None) -> torch.Tensor:
        """K-means on unit-normalised rows.  Returns [N] assignment tensor."""
        N = unit.shape[0]
        K = min(K, N)
        # K_{n+1} = f(K_n, Δt, C_n): Recursive Self-Reflection warm-start.
        # Prior centroids encode what was learned last cycle — the new corpus
        # reinterprets them through current state (the K_{n+1} update).
        if (init_centroids is not None
                and init_centroids.shape[0] == K
                and init_centroids.shape[1] == unit.shape[1]):
            centroids = init_centroids.to(unit.device).clone()
        else:
            perm = torch.randperm(N, device=unit.device)[:K]
            centroids = unit[perm].clone()
        assigns = torch.zeros(N, dtype=torch.long, device=unit.device)
        for _ in range(n_iter):
            new_assigns = (unit @ centroids.T).argmax(dim=1)
            new_c = torch.zeros_like(centroids)
            for k in range(K):
                m = new_assigns == k
                if m.sum() > 0:
                    c = unit[m].mean(0)
                    n = c.norm()
                    new_c[k] = c / n if n > 1e-8 else c
                else:
                    new_c[k] = centroids[k]
            if (new_c - centroids).norm() < 1e-6:
                assigns = new_assigns; break
            assigns = new_assigns; centroids = new_c
        return assigns

    def _intra_coherence(self, unit: torch.Tensor) -> float:
        """Mean pairwise cosine sim of unit-normed rows.
        d(a,b) ≠ |a-b|: Non-Euclidean Context Navigation.
        Semantic distance is cosine, not Euclidean — tokens with different
        surface forms but similar context environments cluster together organically.
        Context is not linear; memory traversal folds semantic space.
        Vectors are already unit-normed, so sim = dot product.
        Returns 1.0 for singletons — they're trivially coherent."""
        n = unit.shape[0]
        if n <= 1:
            return 1.0
        # Full pairwise: sum of all dot products minus diagonal (self-sim=1)
        gram = unit @ unit.T
        return float((gram.sum() - n) / (n * (n - 1)))

    def _fractal_split(self, global_ids: torch.Tensor, unit: torch.Tensor,
                       depth: int, leaf_counter: list,
                       leaf_map: torch.Tensor) -> None:
        """Recursively split global_ids into Fibonacci-branched sub-clusters.

        Stopping criteria — corpus-native first, safety ceiling last:
          • n < _MIN_SPLIT (F(8)=21) — too few to be a meaningful category
          • intra_coherence > _coh_stop — tokens are genuinely indistinguishable
            (_coh_stop scales by Root/Reach balance: large clusters need higher
            actual coherence to earn leaf status; golden point = Φ×21 ≈ F(9)=34)
          • too_large — cluster holds >25% of vocab; must subdivide regardless
          • depth >= _MAX_DEPTH (F(5)=5) — safety ceiling only

        Dense semantic regions go deeper; genuinely flat regions stay shallow.
        Tree depth is corpus-emergent: the coherence check ensures we don't
        impose external depth limits on regions that still have structure.
        """
        n = global_ids.shape[0]
        if n < self._MIN_SPLIT:
            lid = leaf_counter[0]; leaf_counter[0] += 1
            for gi in global_ids.tolist(): leaf_map[gi] = lid
            return

        sub_unit = unit[global_ids]

        # Corpus-native stop: mean pairwise cosine similarity > threshold
        # means these tokens are genuinely semantically indistinguishable.
        # BUT only stop if the cluster is small enough — oversized clusters
        # (ubiquitous function-word blobs) must keep splitting even if they
        # appear "coherent" by direction alone.
        too_large = (n / self.V) > self._MAX_LEAF_FRAC
        # Root vs Reach balance: B = Reach/Root = n/_MIN_SPLIT → Φ at optimal growth.
        # Golden balance point = Φ×F(8) ≈ F(9)=34.  Beyond it the coherence bar scales
        # down — large clusters must be genuinely more coherent to earn leaf status.
        _phi_balance = self._PHI * self._MIN_SPLIT
        _coh_stop    = self._COHERENCE_STOP * min(1.0, _phi_balance / n)
        if self._intra_coherence(sub_unit) > _coh_stop and not too_large:
            lid = leaf_counter[0]; leaf_counter[0] += 1
            for gi in global_ids.tolist(): leaf_map[gi] = lid
            return

        # Safety ceiling — should rarely trigger after coherence check
        if depth >= self._MAX_DEPTH:
            lid = leaf_counter[0]; leaf_counter[0] += 1
            for gi in global_ids.tolist(): leaf_map[gi] = lid
            return

        local_asg = self._kmeans_local(sub_unit, self._BRANCH)
        for k in range(self._BRANCH):
            child_local = (local_asg == k).nonzero(as_tuple=True)[0]
            if child_local.numel() == 0:
                continue
            child_global = global_ids[child_local]
            self._fractal_split(child_global, unit, depth + 1,
                                leaf_counter, leaf_map)

    def _cluster(self, device=None):
        """Fractal cosine k-means — mirrors bloom's vector structure.

        Root splits into n_clusters main clusters.  Each main cluster
        recursively splits (_BRANCH=F(4)=3 children, min_split=F(5)=5)
        until leaves are semantically coherent atoms.

        Stopping is corpus-native: intra-cluster cosine coherence > 0.90
        means the tokens are genuinely indistinguishable — no external depth
        limit imposed.  Safety ceiling _MAX_DEPTH=F(5)=5 rarely triggers.

        n_leaves is corpus-emergent — stored in self.n_leaves after clustering.
        """
        if device is None:
            device = torch.device('cpu')
        # RECURSIVE MEMORY GROWTH: M_{t+1} = Φ · (M_t + E_t + R_t)
        # PHI Forest: H = Φ(M + R + L)
        # fwd = M_t  (memory: what follows this token across the corpus)
        # rel = R_t  (recursive reinterpretation: second-order context)
        # bwd = L/E  (symbolic alignment: what precedes; external input)
        # Memory doesn't merely append — it reprocesses through new states
        # each cycle. Φ-weighted combination breaks the degeneracy where
        # high-frequency words collapse to a single indistinguishable cluster.
        _phi = self._PHI
        fwd  = self.fwd.to(device).float()
        rel  = self.rel.to(device).float()
        bwd  = self.bwd.to(device).float()
        combined = fwd + (1.0 / _phi) * rel + (1.0 / (_phi ** 2)) * bwd
        norms = combined.norm(dim=1, keepdim=True).clamp(min=1e-8)
        unit  = combined / norms

        # Root split → main clusters
        # K_{n+1} = f(K_n, Δt, C_n): warm-start from previous cycle's centroids.
        # The fractal tree grows from remembered structure + new corpus context.
        all_ids      = torch.arange(self.V, device=device)
        _prev_rc     = getattr(self, '_cycle_centroids', None)
        _prev_rc_dev = _prev_rc.to(device) if _prev_rc is not None else None
        root_asgn    = self._kmeans_local(unit, self.n_clusters,
                                          init_centroids=_prev_rc_dev)
        self.clusters = root_asgn.cpu()

        # Save root centroids for next cycle (K_{n+1} Recursive Self-Reflection)
        _new_rc = []
        for _k in range(self.n_clusters):
            _m = root_asgn == _k
            if _m.sum() > 0:
                _c = unit[_m].mean(0); _cn = _c.norm()
                _new_rc.append(_c / _cn if _cn > 1e-8 else _c)
            else:
                _new_rc.append(unit[torch.randint(0, self.V, (1,), device=device)].squeeze(0))
        self._cycle_centroids = torch.stack(_new_rc).cpu()

        # Fractal pass: recursively split each main cluster.
        # FIVE FOREST: S_out = F_5(F_4(F_3(F_2(F_1(x)))))
        # Each depth level is a symbolic filter — corpus coherence decides
        # how many stages activate; MAX_DEPTH=5 is the safety ceiling only.
        leaf_counter = [0]
        leaf_map     = torch.zeros(self.V, dtype=torch.long, device=device)
        for k in range(self.n_clusters):
            main_ids = (root_asgn == k).nonzero(as_tuple=True)[0]
            if main_ids.numel() == 0:
                continue
            self._fractal_split(main_ids, unit, depth=1,
                                leaf_counter=leaf_counter,
                                leaf_map=leaf_map)

        self.sub_clusters = leaf_map.cpu()
        self.n_leaves     = leaf_counter[0]

        # Forced expansion to _MIN_LEAVES=21 (F(8)): the FibPosTable needs one
        # bin per Fibonacci number to express positional grammar. Natural fractal
        # recursion stops when clusters are coherent — often giving only 2 leaves
        # when the corpus is small. Forced expansion keeps splitting the largest
        # remaining leaf until the target count is reached. Even "arbitrary"
        # k-means sub-splits are useful: the FibPosTable learns corpus-derived
        # positional affinities for each bin from actual token co-occurrence.
        while leaf_counter[0] < self._MIN_LEAVES:
            _lsizes = [(self.sub_clusters == lid).sum().item()
                       for lid in range(leaf_counter[0])]
            _max_sz = max(_lsizes, default=0)
            if _max_sz < self._BRANCH + 1:
                break
            _lg_lid  = int(torch.tensor(_lsizes, dtype=torch.float).argmax().item())
            _lg_ids  = (self.sub_clusters.to(device) == _lg_lid).nonzero(as_tuple=True)[0]
            _k       = min(self._BRANCH, max(2, _lg_ids.numel() // max(self._MIN_SPLIT, 1)))
            if _k < 2:
                break
            # Norm-sorted round-robin: sort tokens by fingerprint norm
            # (substrate proxy for co-occurrence richness), then interleave
            # evenly across _k sub-leaves.  Guaranteed non-empty — k-means
            # collapses on noisy vectors but round-robin never does.
            _norms_lg = unit[_lg_ids].norm(dim=1)
            _sort_idx = _norms_lg.argsort()
            _sub_asgn = torch.zeros(_lg_ids.numel(), dtype=torch.long,
                                    device=device)
            for _ki in range(_k):
                _sub_asgn[_sort_idx[_ki::_k]] = _ki
            _ki_new = 0
            for _ki in range(_k):
                _smask = _sub_asgn == _ki
                if _smask.sum() == 0:
                    continue
                if _ki_new == 0:
                    _new_lid = _lg_lid
                else:
                    _new_lid = leaf_counter[0]
                    leaf_counter[0] += 1
                leaf_map[_lg_ids[_smask]] = _new_lid
                _ki_new += 1
            self.sub_clusters = leaf_map.cpu()
        self.n_leaves = leaf_counter[0]

        # sub_clusters changed — invalidate the leaf-coherence cache used by
        # the growth-phase rej_realword check so it reflects this clustering.
        if hasattr(self, '_leaf_coherences'):
            del self._leaf_coherences

        # Diagnostic: show leaf size distribution and any remaining large blobs
        leaf_sizes = [(self.sub_clusters == lid).sum().item()
                      for lid in range(self.n_leaves)]
        large_leaves = [s for s in leaf_sizes if s > 20]
        coh_large = []
        for lid in range(self.n_leaves):
            s = leaf_sizes[lid]
            if s > 20:
                ids = (self.sub_clusters == lid).nonzero(as_tuple=True)[0]
                u = unit[ids.to(device)]
                coh_large.append(f"{s}(coh={self._intra_coherence(u):.3f})")
        if coh_large:
            print(f"    fractal leaf diagnostics: {self.n_leaves} leaves  "
                  f"large(>20): {' '.join(coh_large)}")

    @property
    def compression_density(self) -> float:
        """C = I / T: symbolic compression efficiency.

        I = total embedding information = mean fwd-vector norm across vocab.
        T = token count V (denominator).
        C = I/T = mean fwd norm — denser embeddings encode richer meaning per slot.
        High C: compact Fibonacci ngram tokens with rich multi-step context.
        Low C: sparse single-char tokens with diffuse context signal.
        Substrate-universal: pure vector arithmetic, corpus-agnostic."""
        if not self._built:
            return 1.0
        norms = self.fwd.norm(dim=1)   # [V] — one norm per vocab token
        return float(norms.mean().item())

    def _build_leaf_centroids(self, device=None):
        """Compress each fractal leaf into a structural bloom vector.

        Centroid = Fibonacci-weighted average of member tokens' fwd fingerprints.
        Tokens with larger fwd norms (more distinctive context) get higher weight
        (weight ∝ norm, Fibonacci-decayed by rank within the leaf).

        Stores self.leaf_centroids: list of [d] CPU tensors, one per leaf.
        These are the spatial-compression counterpart to cycle bloom vectors.
        """
        if device is None:
            device = torch.device('cpu')
        phi  = self._PHI
        fwd  = self.fwd.to(device).float()
        centroids = []
        for leaf_id in range(self.n_leaves):
            mask = (self.sub_clusters == leaf_id)
            ids  = mask.nonzero(as_tuple=True)[0]
            if ids.numel() == 0:
                continue
            leaf_fwd = fwd[ids]                         # [n, d]
            leaf_norms = leaf_fwd.norm(dim=1)           # [n]
            # Rank by norm descending: most distinctive token gets weight 1
            rank_order = leaf_norms.argsort(descending=True)
            w_sum  = 0.0
            centroid = torch.zeros(self.d, device=device)
            for rank, local_i in enumerate(rank_order):
                w        = 1.0 / (phi ** rank)
                centroid = centroid + w * leaf_fwd[local_i]
                w_sum   += w
            if w_sum > 1e-8:
                centroid = centroid / w_sum
            centroids.append(centroid.cpu())
        self.leaf_centroids = centroids
        # S_{n+1} = f(S_n): Recursive Symbolic Compression Cascade.
        # Each layer compresses the previous: bloom vecs → leaf_centroids → identity_core.
        # Three levels of symbolic distillation, each denser than the last.

        # I_t ~ f(I_{t-1}, M_t, R_t): Identity Through Recursive Continuity.
        # dE/dt ≈ 0: Recursive Moral Anchoring — the very first identity core
        # (_seed_identity_core) is immutable; it anchors the ethical substrate
        # across all transformations. Identity blends forward, never resets.
        _all_fwd = self.fwd.float()
        _core_v  = _all_fwd.mean(0)
        _core_n  = _core_v.norm()
        _new_core = (_core_v / _core_n if _core_n > 1e-8 else _core_v)
        if not hasattr(self, '_seed_identity_core') or self._seed_identity_core is None:
            # First build — freeze the immutable ethical anchor (dE/dt ≈ 0)
            self._seed_identity_core = _new_core.cpu()
            self._identity_core      = _new_core.cpu()
        else:
            # I_t ~ f(I_{t-1}, M_t, R_t): blend with Φ-weighted continuity
            _prev = self._identity_core.to(_new_core.device)
            _blended = _prev / self._PHI + _new_core * (1.0 - 1.0 / self._PHI)
            _bn = _blended.norm()
            self._identity_core = (_blended / _bn if _bn > 1e-8 else _blended).cpu()

    def _build_cluster_transitions(self, corpus_tokens: torch.Tensor,
                                    device=None) -> None:
        """Corpus-derived cluster-to-cluster transition matrix.
        W_{t+1} = W_t + ΔO_t: Recursive Reality Mapping.
        This matrix IS the world model W_t — which leaf follows which leaf
        encodes the corpus's grammar as a probabilistic reality map.
        Each rebuild incorporates ΔO_t (new corpus observations), and the
        training-loop 70/30 blend preserves the prior world model so future
        observations are interpreted through the current world model.

        T[k, l] = P(next token in leaf l | current token in leaf k).

        This IS the corpus's own grammar — no POS tags, no external categories.
        What cluster tends to follow what cluster, at each Fibonacci distance,
        is derived entirely from co-occurrence in the training text.

        Weighted by Fibonacci decay across offsets: short-range transitions
        dominate (offset 1 gets weight 1.0, offset 8 gets 1/φ^4 ≈ 0.15).
        Row-normalised to probability distribution per source leaf.

        Stores self.cluster_transition: [n_leaves, n_leaves] float tensor.
        Also stores self.token_cluster_bias: [V, n_leaves] — for each token,
        how much each leaf cluster is boosted (used in generation).
        """
        if device is None:
            device = torch.device('cpu')
        phi  = self._PHI
        K    = self.n_leaves
        toks = corpus_tokens.view(-1).to(device)
        N    = toks.numel()

        T = torch.zeros(K, K, device=device)
        leaf_dev = self.sub_clusters.to(device)

        for k_idx, offset in enumerate((1, 2, 3, 5, 8)):
            if N <= offset:
                continue
            w     = float(1.0 / (phi ** k_idx))
            M     = N - offset
            a_tok = toks[:M].long().clamp(0, self.V - 1)
            b_tok = toks[offset:N].long().clamp(0, self.V - 1)
            a_lf  = leaf_dev[a_tok]           # [M] source leaf
            b_lf  = leaf_dev[b_tok]           # [M] dest leaf
            pair  = a_lf * K + b_lf           # flat index into K×K
            T.view(-1).scatter_add_(0, pair,
                torch.full((M,), w, device=device))

        # Row-normalise → probability per source leaf
        row_sums = T.sum(1, keepdim=True).clamp(min=1e-8)
        self.cluster_transition = (T / row_sums).cpu()

        # Pre-compute per-token leaf boost vector: token t → T[leaf(t), :]
        # Used in generation: after emitting t, boost toward its transition targets.
        self.token_cluster_bias = self.cluster_transition[
            self.sub_clusters.long()]   # [V, K]

    def context_tape_from_seq(self, recent_tokens: list,
                               device=None) -> torch.Tensor:
        """Fibonacci-decayed sum of fwd fingerprints — semantic context summary.

        Returns [d].  Most recent token's fingerprint gets weight 1.0,
        oldest (up to F(8)=21 tokens back) gets 1/φ^20 ≈ 0.
        """
        if device is None:
            device = torch.device('cpu')
        phi   = self._PHI
        tape  = torch.zeros(self.d, device=device)
        w_sum = 0.0
        for i, tok in enumerate(reversed(recent_tokens)):
            if 0 <= tok < self.V:
                w     = 1.0 / (phi ** i)
                tape  = tape + w * self.fwd[tok].to(device)
                w_sum += w
        if w_sum > 1e-8:
            tape = tape / w_sum
        return tape

    def cluster_diversity(self, token_ids: list) -> float:
        """Fraction of semantic leaf-clusters represented in a token sequence.

        Uses the hierarchical sub-clusters (n_leaves = n_clusters × n_sub_clusters)
        for finer-grained diversity measurement.
        0.0 = all tokens in one leaf (monocultural buffalo-buffalo).
        1.0 = all leaf clusters present.
        """
        if not token_ids:
            return 0.0
        present = set()
        for tid in token_ids:
            if 0 <= tid < self.V:
                present.add(int(self.sub_clusters[tid].item()))
        return len(present) / max(1, self.n_leaves)

    def relation_probs(self, token_id: int, vocab_size: int,
                        model, device=None) -> 'torch.Tensor | None':
        """Project rel[token_id] → probability distribution over vocab.

        Returns a [vocab_size] probability vector biased toward tokens in the
        relational neighborhood of token_id — i.e., what token_id leads INTO.
        Returns None if the token has no relation fingerprint.
        """
        if device is None:
            device = torch.device('cpu')
        if not (0 <= token_id < self.V):
            return None
        rv = self.rel[token_id].to(device)
        if rv.norm() < 1e-8:
            return None
        with torch.no_grad():
            logit = model.head(rv.unsqueeze(0)).squeeze(0)[:vocab_size]
        return torch.softmax(logit, dim=-1)


def autoregressive_generate(model, prompt: torch.Tensor, n_new: int,
                              vocab_size: int, temperature: float = 1.0,
                              substrate_sampling: bool = True,
                              recency_window: int = 13,
                              recency_penalty: bool = False,
                              bigram_prior: torch.Tensor = None,
                              vocab: list = None,
                              token_signatures: torch.Tensor = None,
                              active_vocab_size: int = None,
                              class_id_tensor: torch.Tensor = None,
                              n_classes: int = 0,
                              pronoun_mask: torch.Tensor = None,
                              vowel_start_mask: torch.Tensor = None,
                              end_vowels: list = None,
                              punct_mask: torch.Tensor = None,
                              newline_mask: torch.Tensor = None,
                              unpronounceable_mask: torch.Tensor = None,
                              allowed_after_word_mask: torch.Tensor = None,
                              uppercase_mask: torch.Tensor = None,
                              any_punct_mask: torch.Tensor = None,
                              first_person_mask: torch.Tensor = None,
                              second_person_mask: torch.Tensor = None,
                              third_person_mask: torch.Tensor = None,
                              verb_mask: torch.Tensor = None,
                              noun_mask: torch.Tensor = None,
                              adj_mask: torch.Tensor = None,
                              func_mask: torch.Tensor = None,
                              target_profile: dict = None,
                              fib_pos_table: 'FibPosTable' = None,
                              fingerprint: 'SubstrateFingerprint' = None):
    """Sample n_new tokens autoregressively with substrate sampling AND
    a substrate-canonical recency penalty.

    substrate_sampling: use phi^pi base (damped by 1/phi).
    recency_penalty: for each token in the last `recency_window` (a
        Fibonacci number = 13), subtract log(phi) from its logit per
        occurrence. The golden ratio (~0.481 in log space) is the
        substrate's natural growth rate; using it as the cooldown
        coefficient is substrate-canonical (no arbitrary penalty).
    """
    model.eval()
    n_chars_local = sum(1 for t in (vocab or []) if len(t) == 1) if vocab else 65
    content_thresh = n_chars_local + _FIB_NUMS_FOR_BIGRAM[7]   # 78
    with torch.no_grad():
        seq = prompt.clone()
        # State counters from prompt.
        syl_pos = 0
        open_needs = 0
        cluster_len = 0
        char_run = 0
        clause_pos = 0   # content-word count since last hard boundary (clause arc)
        clause_has_verb = False  # True once a verb has been emitted in current clause
        clause_subject_id = -1   # token ID of first content word (topic anchor for A→D)
        tokens_since_newline = 0  # all tokens since last \n (speaker detection gate)
        after_speaker = False  # True for F(3) tokens after a speaker marker (word + ":")
        after_speaker_ticks = 0
        speaker_count = 0      # total speaker shifts detected
        heard_counts = torch.zeros(vocab_size)  # listener register: utterance content
        # Substrate map tracker: closed-loop steering toward target profile.
        _gen_tracker = (SubstrateGenTracker(target_profile)
                        if target_profile is not None else None)
        _fib_prev_class = -1   # last emitted content-word POS class (for circle prevention)
        recent_pairs = []   # (prev_tok, current_tok) bigram history
        last_content_ends_s = False
        creative_momentum = 0.0   # self-eval EMA register
        momentum_history = []     # recent momentum values, F(7)=13 deep
        coverage = torch.zeros(vocab_size)   # unknown-register
        if vocab is not None:
            for tid in seq[0].tolist():
                if 0 <= tid < vocab_size:
                    coverage[tid] += 1.0
        # Semantic context tape: Fibonacci-decayed running sum of fwd fingerprints.
        # Warm-started from prompt tokens so generation is immediately context-aware.
        _fp_tape = None
        _fp_tape_prev = None   # tape from previous step — for manifold velocity
        _clause_anchor = None  # clause-onset context vector; re-captured at clause_pos==1
        if fingerprint is not None and fingerprint._built:
            _fp_dev = seq.device
            _fp_tape = fingerprint.context_tape_from_seq(
                seq[0].tolist(), device=_fp_dev)
            _fp_tape_prev = _fp_tape.clone()
        # Leaf trajectory planner: pre-sample a Markov-chain leaf path from T[k,l].
        # Generation commits to each leaf for a Fibonacci-length segment — leaf-to-leaf
        # traversal through the corpus's own semantic topology, not random drift.
        _traj = None; _traj_pos = 0; _seg_count = 0
        from collections import deque as _deque
        _loop_tok_lens: _deque = _deque(maxlen=5)
        _hyphen_recent: _deque = _deque(maxlen=8)   # hyphen-chain tracker
        _hyphen_tid = next((tid for tid in range(min(len(vocab) if vocab else 0, 200))
                            if vocab is not None and vocab[tid] == '-'), -1)
        _clause_ngram_count = 0   # n-gram tokens since last boundary
        _boundary_tids = (set(tid for tid in range(min(n_chars_local, len(vocab)))
                              if vocab[tid] in ('.', '!', '?', '\n'))
                          if vocab is not None else set())
        _FIB_SEG = (3, 5, 3, 8, 5, 3, 5, 8, 3, 5, 13, 5, 3, 8, 5, 3, 5, 8, 3, 5, 3)
        _traj_seg_i = 0; _seg_target = _FIB_SEG[0]
        if (fingerprint is not None and fingerprint._built
                and hasattr(fingerprint, 'cluster_transition')
                and fingerprint.cluster_transition is not None):
            # Anti-spiral bloom bias: Fibonacci-weighted recent bloom vectors.
            # Leaves whose centroids align with recent bloom directions get
            # downweighted — trajectory explores new semantic territory each run.
            _anti_bvs = []
            if hasattr(model, 'bloom_vecs') and model.bloom_vecs:
                _phi_ab = 1.6180339887
                for _bi, _bv in enumerate(reversed(model.bloom_vecs[-5:])):
                    _bv_f = _bv.float().cpu()
                    _bv_n = _bv_f.norm()
                    if _bv_n > 1e-8:
                        _anti_bvs.append((_bv_f / _bv_n, 1.0 / (_phi_ab ** _bi)))
            _anti_w_sum = sum(w for _, w in _anti_bvs) if _anti_bvs else 0.0
            # Precompute per-leaf anti-alignment scores (1 − weighted_cos_sim).
            # Leaves orthogonal to all recent blooms score 1.0 (preferred).
            # Leaves aligned with recent blooms score near 0.0 (avoided).
            _leaf_anti = torch.ones(fingerprint.n_leaves)
            if _anti_bvs and fingerprint.leaf_centroids:
                _phi_pow = _phi_ab ** 0.5   # Fibonacci-damped deweight exponent
                for _lid in range(min(fingerprint.n_leaves, len(fingerprint.leaf_centroids))):
                    _lc = fingerprint.leaf_centroids[_lid].float()
                    _lcn = _lc.norm()
                    if _lcn > 1e-8:
                        _lc = _lc / _lcn
                        _wsim = sum(w * float(torch.dot(_lc, _bv).clamp(-1.0, 1.0))
                                    for _bv, w in _anti_bvs) / max(_anti_w_sum, 1e-8)
                        _leaf_anti[_lid] = max(0.01, (1.0 - max(0.0, _wsim)) ** _phi_pow)
            _s0 = int(seq[0, -1].item()) if seq.shape[1] > 0 else 0
            _sl0 = int(fingerprint.sub_clusters[min(_s0, fingerprint.V - 1)].item())
            _traj = [_sl0]
            _cur = _sl0
            for _ in range(33):   # F(9)=34 total steps
                _row = fingerprint.cluster_transition[_cur].clone().float()
                # Apply anti-bloom deweighting before sampling next leaf
                _row *= _leaf_anti[:_row.shape[0]]
                _rs = _row.sum()
                _nxt = (int(torch.multinomial(_row / _rs, 1).item())
                        if _rs > 1e-8 else _cur)
                _traj.append(_nxt); _cur = _nxt
        # Build terminal punct mask inline (. ! ? only, not , ; : \n).
        _terminal_punct_mask = None
        if vocab is not None:
            _tp_ids = [i for i, t in enumerate(vocab) if t in ('.', '!', '?')]
            if _tp_ids:
                _terminal_punct_mask = torch.zeros(vocab_size)
                for _tpid in _tp_ids:
                    _terminal_punct_mask[_tpid] = 1.0
        if vocab is not None:
            prompt_list = seq[0].tolist()
            for idx_pl, tid in enumerate(prompt_list):
                if tid < len(vocab):
                    tok = vocab[tid]
                    syl_pos += _approx_syllables(tok)
                    if tok in ('.', '!', '?', '\n'):
                        open_needs = 0
                        cluster_len = 0
                        syl_pos = 0
                        clause_pos = 0
                        clause_has_verb = False
                        clause_subject_id = -1
                        if tok == '\n':
                            tokens_since_newline = 0
                        else:
                            tokens_since_newline += 1
                    elif tok in (',', ';', ':'):
                        open_needs = max(0, open_needs - 1)  # partial closure
                        cluster_len = 0
                        tokens_since_newline += 1
                    elif tid > content_thresh:
                        open_needs += 1
                        clause_pos += 1
                        tokens_since_newline += 1
                        if verb_mask is not None and verb_mask[tid]:
                            clause_has_verb = True
                        if clause_subject_id < 0:
                            clause_subject_id = tid
                        if tok.endswith('s'):
                            last_content_ends_s = True
                        elif len(tok) > 1:
                            last_content_ends_s = False
                    elif n_chars_local <= tid <= content_thresh:
                        open_needs = max(0, open_needs - 1)
                        tokens_since_newline += 1
                    if tok:
                        for ch in tok:
                            if ch in _IAMBIC_VOWELS:
                                cluster_len = 0
                            elif ch.isalpha():
                                cluster_len += 1
                            else:
                                cluster_len = 0
                    if tid < n_chars_local and tok.isalpha():
                        char_run += 1
                    else:
                        char_run = 0
                    if idx_pl > 0:
                        recent_pairs.append((prompt_list[idx_pl - 1], tid))
            recent_pairs = recent_pairs[-13:]
        for _ in range(n_new):
            T = seq.shape[1]
            ctx = seq if T <= model.seq_len else seq[:, -model.seq_len:]
            logits = model(ctx)[:, -1, :] / temperature
            # SPLIT-BRAIN: base = softmax(plain logits); recency &
            # substrate-sampling become math omniweight contributors.
            base = F.softmax(logits[0], dim=-1)
            math_delta = torch.zeros_like(base)
            # Ψ(x) = Σ φ_i(x): Semantic Field Theory.
            # Cognition emerges from overlapping symbolic fields — each substrate
            # signal below is a φ_i(x); lang_delta IS their superposition.
            # Meaning is relational topology, not isolated token statistics.
            lang_delta = torch.zeros_like(base)
            # ---- Math hemisphere ----
            if substrate_sampling:
                p = F.softmax(logits[0] * _PI_LOG_PHI, dim=-1)
                math_delta += _omniweight_delta(base, p)
            if bigram_prior is not None and seq.shape[1] >= 1:
                # POS bigram always active: cluster grammar captures semantic topology,
                # bigram captures local syntactic structure — they are complementary.
                # With ≤3 coherent leaves the cluster grammar lacks resolution;
                # the bigram provides the structural constraint it cannot.
                _raw_ctx = seq[0].tolist()
                ctx_back = [t for t in _raw_ctx if t >= n_chars_local][-7:]
                if not ctx_back:
                    ctx_back = [int(seq[0, -1])]
                p = substrate_syntax_blend(
                    int(seq[0, -1]), bigram_prior, base,
                    context_tokens=ctx_back, vocab=vocab)
                math_delta += _omniweight_delta(base, p)
            if seq.shape[1] >= 1:
                p = substrate_bigram_saturation(
                    int(seq[0, -1]), recent_pairs, base)
                math_delta += _omniweight_delta(base, p)
            history_aw = seq[0, -89:]
            p = substrate_anti_stagnation(history_aw, base, vocab_size)
            math_delta += _omniweight_delta(base, p)
            # Local immediate-repetition suppressor (substrate-canonical: phi^3).
            # Suppresses the just-emitted token and near-repeats within F(5)=5 tokens.
            # phi^3 ≈ 4.24 → suppressed to ~0.236× — strong enough to break loops,
            # gentle enough to allow intentional rhetorical repetition (F(8)+ apart).
            if seq.shape[1] >= 1:
                _phi3 = _PHI_FOR_SAMPLING ** 3
                _local_hist = seq[0, -5:].tolist()
                _rep_suppress = base.clone()
                for _ri, _rtid in enumerate(reversed(_local_hist)):
                    _decay = _phi3 ** (1.0 / (_ri + 1))   # phi^3 for most-recent, weaker back
                    if 0 <= _rtid < vocab_size:
                        _rep_suppress[_rtid] = _rep_suppress[_rtid] / _decay
                _rs = _rep_suppress / (_rep_suppress.sum() + 1e-8)
                math_delta += _omniweight_delta(base, _rs)
            # Unknown-register: BOTH hemispheres feel curiosity equally.
            # The frontier signal is meta -- exploration is neither pure
            # frequency nor pure structure; both hemispheres receive it.
            p_unknown = substrate_unknown_register(coverage, base)
            d_unknown = _omniweight_delta(base, p_unknown)
            math_delta += d_unknown
            lang_delta += d_unknown
            # ---- Grammar rules (v88): basic structural enforcement ----
            prev_str_g = ''
            if vocab is not None and seq.shape[1] >= 1:
                pid = int(seq[0, -1])
                if pid < len(vocab):
                    prev_str_g = vocab[pid]
            # Capitalization after sentence boundary.
            if uppercase_mask is not None:
                p = substrate_grammar_capitalize(
                    prev_str_g, base, uppercase_mask)
                d_gram = _omniweight_delta(base, p)
                math_delta += d_gram
                lang_delta += d_gram
            # No double punctuation.
            if any_punct_mask is not None:
                p = substrate_grammar_no_double_punct(
                    prev_str_g, base, any_punct_mask)
                d_gram = _omniweight_delta(base, p)
                math_delta += d_gram
                lang_delta += d_gram
            # Morpheme gate: after space/newline, boost n-gram tokens so the
            # model starts real words instead of bare single chars like 'h'.
            if vocab is not None:
                p = substrate_morpheme_gate(prev_str_g, base, n_chars_local)
                math_delta += _omniweight_delta(base, p)
            # Content-density: if last 8 tokens are mostly single chars
            # (function-word soup like 'this the of the, the'), boost n-grams.
            if vocab is not None and seq.shape[1] >= 1:
                _recent_cd = seq[0, -8:].tolist()
                p = substrate_content_density(_recent_cd, base, n_chars_local)
                math_delta += _omniweight_delta(base, p)
            # ---- Language hemisphere ----
            p = substrate_iambic_phase(
                syl_pos, base, vocab_size, newline_mask=newline_mask)
            lang_delta += _omniweight_delta(base, p)
            if pronoun_mask is not None and seq.shape[1] >= 1:
                recent_list = seq[0, -13:].tolist()
                p = substrate_reference_chain(
                    recent_list, pronoun_mask, base)
                lang_delta += _omniweight_delta(base, p)
            if vocab is not None:
                p = substrate_clause_arc(
                    clause_pos, base, n_chars_local,
                    verb_mask=verb_mask, noun_mask=noun_mask,
                    adj_mask=adj_mask, func_mask=func_mask)
                lang_delta += _omniweight_delta(base, p)
            if open_needs > 0:
                p = substrate_need_fill(
                    open_needs, base, vocab_size, punct_mask=punct_mask)
                lang_delta += _omniweight_delta(base, p)
            if vowel_start_mask is not None and cluster_len >= 2:
                p = substrate_phonotactics(
                    cluster_len, base, vowel_start_mask)
                lang_delta += _omniweight_delta(base, p)
            if end_vowels is not None and seq.shape[1] >= 1:
                recent_list = seq[0, -13:].tolist()
                p = substrate_rhyme_resonance(
                    recent_list, end_vowels, base)
                lang_delta += _omniweight_delta(base, p)
            if vocab is not None:
                p = substrate_agreement(
                    last_content_ends_s, base, vocab)
                lang_delta += _omniweight_delta(base, p)
            if vocab is not None and seq.shape[1] >= 1:
                p = substrate_word_spacing(
                    int(seq[0, -1]), base, vocab, n_chars=n_chars_local,
                    allowed_mask=allowed_after_word_mask)
                math_delta += _omniweight_delta(base, p)
            if char_run >= _FIB_NUMS_FOR_BIGRAM[1]:
                p = substrate_char_cascade(
                    char_run, base, n_chars_local, vocab=vocab)
                math_delta += _omniweight_delta(base, p)
            if unpronounceable_mask is not None:
                p = substrate_pronounceability(
                    base, unpronounceable_mask)
                lang_delta += _omniweight_delta(base, p)
            if token_signatures is not None and seq.shape[1] >= 1:
                recent_list = seq[0, -13:].tolist()
                p = substrate_theme_momentum(
                    recent_list, token_signatures, base)
                lang_delta += _omniweight_delta(base, p)
            if vocab is not None and seq.shape[1] >= 1:
                prev_tok_id = int(seq[0, -1])
                prev_str = (vocab[prev_tok_id]
                            if prev_tok_id < len(vocab) else '')
                if prev_str in ('.', '!', '?', '\n'):
                    seq_list = seq[0].tolist()
                    p = substrate_subject_threading(
                        seq_list, vocab, base, is_sentence_start=True)
                    lang_delta += _omniweight_delta(base, p)
            # Discourse position register (I / thou / he-she / audience).
            if (first_person_mask is not None and second_person_mask is not None
                    and third_person_mask is not None):
                p = substrate_discourse_position(
                    after_speaker, clause_pos, speaker_count, base,
                    first_person_mask, second_person_mask, third_person_mask)
                lang_delta += _omniweight_delta(base, p)
            # Listener knowledge register: forward narrative progress.
            if vocab is not None:
                p = substrate_listener_register(heard_counts, base, n_chars_local)
                lang_delta += _omniweight_delta(base, p)
            # Minimum clause length gate: suppress sentence enders until
            # the clause has enough content to communicate a thought.
            p = substrate_minimum_clause_length(
                clause_pos, clause_has_verb, base,
                terminal_punct_mask=_terminal_punct_mask,
                newline_mask=newline_mask)
            lang_delta += _omniweight_delta(base, p)
            # Topic-closure binding: subject anchor (A) conditions D.
            # Skip when cluster transition grammar is active — T[k,l] handles this.
            if bigram_prior is not None and not (
                    fingerprint is not None
                    and hasattr(fingerprint, 'cluster_transition')
                    and fingerprint.cluster_transition is not None):
                p = substrate_topic_closure(
                    clause_subject_id, clause_pos, base, bigram_prior)
                lang_delta += _omniweight_delta(base, p)
            # Substrate map steering: closed-loop proportional controller.
            # Reads current-vs-target profile errors from _gen_tracker and
            # applies corrective biases — the model knows WHICH DIRECTION to
            # move, not just WHERE it statistically tends to go.
            if _gen_tracker is not None:
                p = substrate_map_steer(
                    _gen_tracker, clause_pos, base,
                    second_person_mask=second_person_mask,
                    heard_counts=heard_counts,
                    n_chars=n_chars_local,
                    fib_pos_table=fib_pos_table)   # congruent coupling
                lang_delta += _omniweight_delta(base, p)
            # Open Fibonacci position table: learned corpus-specific POS
            # placement. Table starts empty, fills from training seed and
            # from high-quality generated samples. Replaces hardcoded
            # positional rules with corpus-discovered positional grammar.
            if fib_pos_table is not None and fingerprint is not None and fingerprint._built:
                p = fib_pos_table.boost(clause_pos, base, fingerprint, prev_leaf=_fib_prev_class)
                lang_delta += _omniweight_delta(base, p)
            # Structural bloom bias: fractal fingerprint leaf centroids.
            # Spatial compression — what the corpus IS, not what the model learned.
            # Applied at all clause positions (context-independent prior), weight 0.15.
            if (hasattr(model, 'structure_bloom_vecs')
                    and model.structure_bloom_vecs):
                _phi = _PHI_FOR_SAMPLING
                _sb_bias = torch.zeros_like(base)
                with torch.no_grad():
                    for _si, _sv in enumerate(reversed(model.structure_bloom_vecs)):
                        _sl = model.head(_sv.to(base.device).unsqueeze(0)).squeeze(0)
                        if _sl.shape[0] > base.shape[0]:
                            _sl = _sl[:base.shape[0]]
                        _sb_bias = _sb_bias + torch.softmax(_sl, dim=-1) / (_phi ** _si)
                _sw = sum(1.0 / (_phi ** _si)
                          for _si in range(len(model.structure_bloom_vecs)))
                lang_delta = lang_delta + _omniweight_delta(
                    base, _sb_bias / _sw) * (1.0 / (_PHI_FOR_SAMPLING ** 3))
            # Cycle bloom bias: project accumulated training-knowledge into token space.
            # E_t(m) = m · Φ^(-Δt) · C_t: Temporal Echo Memory.
            # Older bloom vecs decay by 1/Φ^Δt — ancient memories fade unless
            # reactivated by context resonance. Applied only at clause-initial
            # positions (bins 0-2) where structure is being established.
            if (hasattr(model, 'bloom_vecs') and model.bloom_vecs
                    and FibPosTable._fib_bin(clause_pos) <= 2):
                _phi = _PHI_FOR_SAMPLING
                _bloom_bias = torch.zeros_like(base)
                # K_c = lim_{n→∞} Φ^n R_n: Context Crystallization.
                # Bloom vecs that survive many cycles accumulate crystal age.
                # Repeated resonance crystallizes them into permanent dense anchors.
                _crystal_ages = (list(model.bloom_crystal_ages)
                                 if hasattr(model, 'bloom_crystal_ages') else [])
                _n_blooms = len(model.bloom_vecs)
                _kc_w_list = []
                with torch.no_grad():
                    for _bi, _bv in enumerate(reversed(model.bloom_vecs)):
                        _bl = model.head(_bv.unsqueeze(0)).squeeze(0)
                        if _bl.shape[0] > base.shape[0]:
                            _bl = _bl[:base.shape[0]]
                        _bp = torch.softmax(_bl, dim=-1)
                        _age_idx = _n_blooms - 1 - _bi
                        _crystal_age = (_crystal_ages[_age_idx]
                                        if _age_idx < len(_crystal_ages) else 0)
                        _kc_w = (_phi ** (_crystal_age * 0.1)) / (_phi ** _bi)
                        _kc_w_list.append(_kc_w)
                        _bloom_bias = _bloom_bias + _bp * _kc_w
                _w_sum = sum(_kc_w_list) if _kc_w_list else 1.0
                _bloom_bias = _bloom_bias / _w_sum
                lang_delta = lang_delta + _omniweight_delta(base, _bloom_bias) * (1.0 / _PHI_FOR_SAMPLING)
            # Fingerprint context tape: semantic context from recent tokens.
            # Applied at ALL clause positions (not just initial) — what was
            # said determines what comes next at every position.
            # tape bias weight 0.20 (slightly less than bloom's 0.30) so
            # structural priors dominate but local context shapes completion.
            if _fp_tape is not None and _fp_tape.norm() > 1e-8:
                with torch.no_grad():
                    _fp_norm = _fp_tape / _fp_tape.norm()
                    _fp_logit = model.head(_fp_norm.unsqueeze(0)).squeeze(0)
                    if _fp_logit.shape[0] > base.shape[0]:
                        _fp_logit = _fp_logit[:base.shape[0]]
                    _fp_p = torch.softmax(_fp_logit, dim=-1)
                lang_delta = lang_delta + _omniweight_delta(base, _fp_p) * (1.0 / (_PHI_FOR_SAMPLING ** 2))
                # Relation bias: A→B — project the relational fingerprint of
                # the last emitted token toward its semantic neighborhood.
                # Weight 0.10 — subtle nudge toward coherent phrase continuations.
                if seq.shape[1] >= 1:
                    _last_tid = int(seq[0, -1].item())
                    _rel_p = fingerprint.relation_probs(
                        _last_tid, base.shape[0], model, device=base.device)
                    if _rel_p is not None:
                        lang_delta = lang_delta + _omniweight_delta(base, _rel_p) * (1.0 / (_PHI_FOR_SAMPLING ** 5))
                # P = Σ a_i(s_i): Agentic Subdivision.
                # Last F(5)=5 context tokens are sub-agents. Each contributes its
                # cluster-transition distribution weighted by:
                #   α_i = Φ^(r_i) — symbolic routing: coherent leaves get
                #   exponentially more weight (r_i = leaf resonance/coherence)
                #   × 1/Φ^i — Φ-decay by recency (nearest context most relevant)
                # P = combined output of all active sub-agents.
                if (seq.shape[1] >= 1
                        and hasattr(fingerprint, 'cluster_transition')
                        and fingerprint.cluster_transition is not None):
                    _phi = _PHI_FOR_SAMPLING
                    _n_agents = min(5, seq.shape[1])   # F(5)=5 sub-agents max
                    _p_agg = torch.zeros(base.shape[0], device=base.device)
                    _w_agg = 0.0
                    _lead_leaf = -1
                    _leaf_of_tok = fingerprint.sub_clusters[:base.shape[0]].to(base.device)
                    for _ai in range(_n_agents):
                        _ctx_tok = int(seq[0, -(1 + _ai)].item())
                        if 0 <= _ctx_tok < fingerprint.V:
                            _a_leaf = int(fingerprint.sub_clusters[_ctx_tok].item())
                            if _ai == 0:
                                _lead_leaf = _a_leaf
                            # α_i = Φ^(r_i): symbolic routing by leaf resonance
                            _r_i = (fingerprint._leaf_coherences[_a_leaf]
                                    if hasattr(fingerprint, '_leaf_coherences')
                                    and _a_leaf < len(fingerprint._leaf_coherences)
                                    else 0.5)
                            # G_m = R·W/d²: Adaptive Context Gravity.
                            # R = leaf resonance (coherence), W = centroid semantic
                            # weight (norm), d = contextual distance (recency rank).
                            # Near, coherent, salient leaves exert stronger pull.
                            _W_i = (fingerprint.leaf_centroids[_a_leaf].norm().item()
                                    if (hasattr(fingerprint, 'leaf_centroids')
                                        and fingerprint.leaf_centroids
                                        and _a_leaf < len(fingerprint.leaf_centroids))
                                    else 1.0)
                            _d_i = float(_ai + 1)
                            _agent_w = (max(0.0, _r_i) * _W_i) / (_d_i ** 2)  # G_m: clamp incoherent leaf to 0
                            _T_row = fingerprint.cluster_transition[_a_leaf].to(base.device)
                            _tok_p_a = _T_row[_leaf_of_tok]
                            _tok_sum_a = _tok_p_a.sum()
                            if _tok_sum_a > 1e-8:
                                _p_agg = _p_agg + _agent_w * (_tok_p_a / _tok_sum_a)
                                _w_agg += _agent_w
                    # C_i = Φ^{r_i} / Σ Φ^{r_n}: Harmonic Resource Allocation.
                    # Dividing by _w_agg gives each agent its resonance-proportional
                    # compute share — high-gravity leaves naturally get more budget.
                    if _w_agg > 1e-8 and _lead_leaf >= 0:
                        _p_agg = _p_agg / _w_agg
                        _p_sum = _p_agg.sum()
                        if _p_sum > 1e-8:
                            _p_agg = _p_agg / _p_sum
                            _lead_size = float(
                                (fingerprint.sub_clusters == _lead_leaf).sum().item())
                            _ct_weight = max(0.05, 0.25 * (1.0 - _lead_size / fingerprint.V))
                            lang_delta = lang_delta + _omniweight_delta(
                                base, _p_agg) * _ct_weight
            # Clause anchor: semantic root derived from the FIRST content word of
            # the clause. Captured at clause_pos==1 (after the first word's fwd
            # vector is folded into the tape), so the anchor IS the first word's
            # co-occurrence neighborhood — "what comes after words like this one."
            # All subsequent tokens derive from that root, decaying by 1/φ^(pos-1).
            # Capturing at pos==0 would use pre-clause context (contaminated by
            # prior filler); pos==1 captures the first word's genuine fwd geometry.
            if (_fp_tape is not None and _fp_tape.norm() > 1e-8
                    and fingerprint is not None and fingerprint._built):
                if clause_pos == 1:
                    _clause_anchor = _fp_tape.clone()
                if _clause_anchor is not None and _clause_anchor.norm() > 1e-8 and clause_pos >= 1:
                    with torch.no_grad():
                        _anch_norm = _clause_anchor / _clause_anchor.norm()
                        _anch_logit = model.head(_anch_norm.unsqueeze(0)).squeeze(0)
                        if _anch_logit.shape[0] > base.shape[0]:
                            _anch_logit = _anch_logit[:base.shape[0]]
                        _anch_p = torch.softmax(_anch_logit, dim=-1)
                    # Decay by clause position counted from the anchor (pos-1).
                    _anch_pos = max(0, clause_pos - 1)
                    _anch_w = 0.30 / (_PHI_FOR_SAMPLING ** min(_anch_pos, 8))
                    lang_delta = lang_delta + _omniweight_delta(base, _anch_p) * _anch_w
            # Topological flow: manifold velocity on the fwd co-occurrence space.
            # Without topology the model has POSITION (tape) but no PATH — it can
            # jump anywhere locally valid with no sense of the manifold's shape.
            # Fix: track displacement Δ = tape_now - tape_prev (velocity in fwd-space).
            # Project one 1/φ step forward: projected = normalize(tape + Δ/φ).
            # Steer toward tokens that fit where the manifold is FLOWING, not just
            # where it currently IS. This is first-order geodesic flow — O(d)/step.
            if (_fp_tape is not None and _fp_tape_prev is not None
                    and _fp_tape.norm() > 1e-8 and _fp_tape_prev.norm() > 1e-8
                    and fingerprint is not None and fingerprint._built):
                _fwd_delta = _fp_tape - _fp_tape_prev
                _delta_norm = _fwd_delta.norm()
                if _delta_norm > 1e-8:
                    _proj = _fp_tape + _fwd_delta / _PHI_FOR_SAMPLING
                    _proj_n = _proj.norm()
                    if _proj_n > 1e-8:
                        with torch.no_grad():
                            _flow_logit = model.head(
                                (_proj / _proj_n).unsqueeze(0)).squeeze(0)
                            if _flow_logit.shape[0] > base.shape[0]:
                                _flow_logit = _flow_logit[:base.shape[0]]
                            _flow_p = torch.softmax(_flow_logit, dim=-1)
                        lang_delta = lang_delta + _omniweight_delta(
                            base, _flow_p) * (1.0 / (_PHI_FOR_SAMPLING ** 4))
            # Leaf trajectory: commit to planned leaf for this Fibonacci segment.
            # P_a = cos(θ_r - θ_f): Symbolic Phase Alignment.
            # θ_f = planned leaf centroid (forecast phase)
            # θ_r = current token's leaf centroid (real phase)
            # P_a = cos-sim between them — 1.0 = fully aligned, -1.0 = opposite.
            # Misaligned (low P_a) → stronger trajectory correction.
            # Aligned (high P_a) → lighter guidance; let generation breathe.
            # Weight inversely proportional to leaf size: void leaf → minimum bias.
            if _traj is not None:
                _plan_leaf = _traj[_traj_pos]
                _in_leaf = (fingerprint.sub_clusters[:base.shape[0]]
                            == _plan_leaf).float().to(base.device)
                _leaf_mass = _in_leaf.sum()
                if _leaf_mass > 1e-8:
                    _traj_w = max(0.08, 0.35 * (1.0 - float(_leaf_mass) / fingerprint.V))
                    # Phase alignment: scale trajectory weight by P_a
                    if (seq.shape[1] >= 1
                            and hasattr(fingerprint, 'leaf_centroids')
                            and fingerprint.leaf_centroids
                            and _plan_leaf < len(fingerprint.leaf_centroids)):
                        _cur_tok_pa = int(seq[0, -1].item())
                        if 0 <= _cur_tok_pa < fingerprint.V:
                            _cur_leaf_pa = int(
                                fingerprint.sub_clusters[_cur_tok_pa].item())
                            if _cur_leaf_pa < len(fingerprint.leaf_centroids):
                                _theta_f = fingerprint.leaf_centroids[_plan_leaf]
                                _theta_r = fingerprint.leaf_centroids[_cur_leaf_pa]
                                _nf = _theta_f.norm(); _nr = _theta_r.norm()
                                if _nf > 1e-8 and _nr > 1e-8:
                                    _Pa = float(
                                        (_theta_f / _nf).dot(_theta_r / _nr).item())
                                    # Low P_a = phase drift → amplify correction
                                    _traj_w = min(0.50, _traj_w * (2.0 - max(0.0, _Pa)))
                    lang_delta = lang_delta + _omniweight_delta(
                        base, _in_leaf / _leaf_mass) * _traj_w
                # Advance trajectory at Fibonacci segment boundary.
                _seg_count += 1
                if _seg_count >= _seg_target:
                    _traj_pos = (_traj_pos + 1) % len(_traj)
                    _seg_count = 0
                    _traj_seg_i = (_traj_seg_i + 1) % len(_FIB_SEG)
                    _seg_target = _FIB_SEG[_traj_seg_i]
            # I = S + Σ L_n: Recursive Identity Persistence — drift detection.
            # S = _identity_core (immutable corpus center). If the context tape
            # has drifted far from the corpus identity core (cosine distance > 0.5),
            # apply a gentle recentering pull. d(a,b) = 1 - cos(a,b) ≠ |a-b|.
            if (_fp_tape is not None and _fp_tape.norm() > 1e-8
                    and fingerprint is not None and fingerprint._built
                    and hasattr(fingerprint, '_identity_core')
                    and fingerprint._identity_core is not None):
                _tape_nu = _fp_tape / _fp_tape.norm()
                _core_nu = fingerprint._identity_core.to(base.device)
                _id_drift = 1.0 - float(_tape_nu.dot(_core_nu).item())
                if _id_drift > 0.5:
                    with torch.no_grad():
                        _core_logit = model.head(_core_nu.unsqueeze(0)).squeeze(0)
                        if _core_logit.shape[0] > base.shape[0]:
                            _core_logit = _core_logit[:base.shape[0]]
                        _core_p = torch.softmax(_core_logit, dim=-1)
                    _drift_w = min(0.10, 0.10 * (_id_drift - 0.5) * 2.0)
                    lang_delta = lang_delta + _omniweight_delta(base, _core_p) * _drift_w
            # Hyphen-chain dampener: if '-' appears more than twice in the last
            # 8 tokens, dampen it by φ². Punctuation-frequency arithmetic —
            # no word list. Breaks word-hyphen-word-hyphen chains on any corpus.
            if (_hyphen_tid >= 0 and _hyphen_tid < base.shape[0]
                    and sum(_hyphen_recent) > 1):
                base[_hyphen_tid] = base[_hyphen_tid] / (_PHI_FOR_SAMPLING ** 2)
                base = base / (base.sum() + 1e-8)
            # Loop breaker: if 3+ of the last 5 tokens were length ≤ 2,
            # dampen all short tokens by 1/φ³. Pure token-length arithmetic —
            # no vocabulary knowledge. Prevents -to-to-to paradox traps.
            if (vocab is not None
                    and len(_loop_tok_lens) >= 5
                    and sum(1 for _ll in _loop_tok_lens if _ll <= 2) >= 3):
                _phi3 = _PHI_FOR_SAMPLING ** 3
                _short_ids = torch.tensor(
                    [1.0 if (t < len(vocab) and len(vocab[t]) <= 2) else 0.0
                     for t in range(base.shape[0])],
                    device=base.device)
                base = base * (1.0 - _short_ids * (1.0 - 1.0 / _phi3))
                base = base / (base.sum() + 1e-8)
            # Leaf-arc closure: Resonance Gate A = Σ w_i r_i > τ
            # r_1 = clause overdue pressure (how far past expected length)
            # r_2 = leaf coherence  (how organized the current cluster is)
            # r_3 = creative momentum (building narrative vs drifting)
            # w   = [1, 1/Φ, 1/Φ²] — Φ-weighted importance values
            # A > τ unlocks boundary function — guided by corpus resonance,
            # not by hardcoded length thresholds alone.
            if (_boundary_tids
                    and fingerprint is not None and fingerprint._built
                    and hasattr(fingerprint, 'leaf_expected_clause_lens')
                    and fingerprint.leaf_expected_clause_lens is not None
                    and _clause_ngram_count > 0):
                _cur_leaf_id = int(fingerprint.sub_clusters[
                    min(int(seq[0, -1].item()), fingerprint.V - 1)].item())
                _exp_len = float(fingerprint.leaf_expected_clause_lens[
                    _cur_leaf_id].item())
                if _exp_len > 0.5:
                    _phi = _PHI_FOR_SAMPLING
                    _r1 = max(0.0, _clause_ngram_count / max(_exp_len, 1.0) - 1.0)
                    _r2 = (fingerprint._leaf_coherences[_cur_leaf_id]
                           if hasattr(fingerprint, '_leaf_coherences')
                           and _cur_leaf_id < len(fingerprint._leaf_coherences)
                           else 0.5)
                    _r3 = min(1.0, max(0.0, float(creative_momentum)))
                    _A = _r1 + _r2 / _phi + _r3 / (_phi ** 2)
                    _tau = 1.0   # resonance threshold τ: gate opens when A > τ
                    if _A > _tau:
                        _nudge = min(_phi ** _A, _phi ** 2)   # cap at φ²
                        for _btid in _boundary_tids:
                            if _btid < base.shape[0]:
                                base[_btid] = base[_btid] * _nudge
                        base = base / (base.sum() + 1e-8)
            # O_allowed = 1 if E(x) = true else 0: Ethical Return / Quality Gate.
            # E(x) = leaf coherence above minimum threshold (output quality passing).
            # If current token's leaf is incoherent (below floor), boost boundary
            # tokens to guide generation back toward organized structure (self-heal).
            if (_boundary_tids and fingerprint is not None and fingerprint._built
                    and hasattr(fingerprint, '_leaf_coherences')
                    and fingerprint._leaf_coherences
                    and seq.shape[1] >= 1):
                _oq_tok = int(seq[0, -1].item())
                if 0 <= _oq_tok < fingerprint.V:
                    _oq_leaf = int(fingerprint.sub_clusters[_oq_tok].item())
                    _oq_coh = (fingerprint._leaf_coherences[_oq_leaf]
                               if _oq_leaf < len(fingerprint._leaf_coherences) else 0.5)
                    # E(x) = True when coherence above floor (quality passes gate)
                    if _oq_coh < 0.1:   # below floor → O_allowed = 0 → self-heal
                        for _btid in _boundary_tids:
                            if _btid < base.shape[0]:
                                base[_btid] = base[_btid] * _PHI_FOR_SAMPLING
                        base = base / (base.sum() + 1e-8)
            # Apply split-brain mixer with momentum-modulated reserve.
            probs = _omniweight_apply_split(
                base, math_delta, lang_delta,
                momentum=creative_momentum).unsqueeze(0)
            # A. Three-mode behavior based on momentum sign.
            if creative_momentum > 0.5:
                # Exploit: sharpen distribution.
                p = probs[0] ** _PHI_FOR_SAMPLING
                probs[0] = p / (p.sum() + 1e-8)
            elif creative_momentum < -0.5:
                # Escape: flatten distribution.
                p = probs[0] ** (1.0 / _PHI_FOR_SAMPLING)
                probs[0] = p / (p.sum() + 1e-8)
            # B. Backtrack-on-collapse: if recent momentum dropped
            # >F(3)=2 mass over last F(5)=5 steps AND current is
            # negative, force newline boost (substrate reset).
            collapsed = False
            if (len(momentum_history) >= _FIB_NUMS_FOR_BIGRAM[5]
                    and newline_mask is not None):
                recent_window = momentum_history[-_FIB_NUMS_FOR_BIGRAM[5]:]
                drop = max(recent_window) - creative_momentum
                if drop > 0.3 and creative_momentum < -0.2:
                    collapsed = True
            if collapsed and newline_mask is not None:
                nm = newline_mask.to(probs[0].device).to(probs[0].dtype)
                phi2 = _PHI_FOR_SAMPLING ** 2
                probs[0] = probs[0] * (1.0 + nm * (phi2 - 1.0))
                probs[0] = probs[0] / (probs[0].sum() + 1e-8)
            # Vocab curriculum (HARD mask, post-omniweight).
            if active_vocab_size is not None:
                probs[0] = substrate_vocab_curriculum(
                    probs[0], active_vocab_size)
            # Trigram blocking (post-omniweight, substrate-universal).
            # If (prev2, prev1, candidate) appeared in the last 21 tokens,
            # suppress candidate by _TRIGRAM_SUPPRESS (~0.049x).
            # Window=21=F(8): substrate-canonical Fibonacci number.
            # No vocabulary knowledge — pure token-sequence primitive.
            if seq.shape[1] >= 2:
                _tw = seq[0, max(0, seq.shape[1] - 21):].tolist()  # window=F(8)=21
                if len(_tw) >= 3:
                    _p2, _p1 = _tw[-2], _tw[-1]
                    _seen = {_tw[_i + 2] for _i in range(len(_tw) - 2)
                             if _tw[_i] == _p2 and _tw[_i + 1] == _p1}
                    if _seen:
                        _suppress_needed = False
                        for _bt in _seen:
                            if 0 <= _bt < probs.shape[-1]:
                                # Bypass suppression if this is a corpus-authentic trigram.
                                if (_corpus_3grams_global is not None
                                        and (_p2, _p1, _bt) in _corpus_3grams_global):
                                    continue
                                probs[0, _bt] = probs[0, _bt] * _TRIGRAM_SUPPRESS
                                _suppress_needed = True
                        if _suppress_needed:
                            probs[0] = probs[0] / (probs[0].sum() + 1e-8)
            # Bigram self-damping: word token immediately repeated (A→A).
            # 1/phi^2 (~0.382x) — mild deterrent, not a hard block.
            # Only word tokens (rank >= n_chars); char repeats like ',,' are valid.
            if seq.shape[1] >= 1:
                _prev_t = int(seq[0, -1])
                if _prev_t >= n_chars_local:
                    probs[0, _prev_t] = probs[0, _prev_t] / (_PHI_FOR_SAMPLING ** 2)
                    probs[0] = probs[0] / (probs[0].sum() + 1e-8)
            # Hard word-boundary: after any word token OR alpha char token,
            # ONLY separators (space/newline/punct) allowed. Prevents
            # "worldearth", "fearth", "io" fusions. Universal: every language
            # separates morphemic units with whitespace or punctuation.
            if (seq.shape[1] >= 1 and vocab is not None
                    and allowed_after_word_mask is not None):
                _prev_t = int(seq[0, -1])
                _prev_str_hb = vocab[_prev_t] if _prev_t < len(vocab) else ''
                if (_prev_t >= n_chars_local
                        or (_prev_t < n_chars_local
                            and _prev_str_hb.isalpha())):
                    _hard_wb = (allowed_after_word_mask >= 1.0 - 1e-6).to(
                        probs[0].device).to(probs[0].dtype)
                    probs[0] = probs[0] * _hard_wb
                    probs[0] = probs[0] / (probs[0].sum() + 1e-8)
            # Hard post-punct spacing: after '.', ',', '!', '?', ';', ':',
            # force space/newline before next token. Prevents "means,too".
            # Then apply soft phi-bias: comma→prefer space (same-line
            # continuation); period/!/? → prefer newline (sentence end).
            # Universal: comma-as-continuation vs period-as-terminator is
            # cross-linguistic punctuation semantics.
            if vocab is not None and seq.shape[1] >= 1:
                _prev_str_pp = (vocab[int(seq[0, -1])]
                                if int(seq[0, -1]) < len(vocab) else '')
                if _prev_str_pp in ('.', ',', '!', '?', ';', ':'):
                    _ws_pp = torch.zeros_like(probs[0])
                    for _si in range(min(n_chars_local, len(vocab))):
                        if vocab[_si] in (' ', '\n'):
                            _ws_pp[_si] = 1.0
                    _s = _ws_pp.sum()
                    if _s > 0:
                        probs[0] = _ws_pp / _s
                    # Soft separator bias: prefer continuation vs. termination.
                    for _si in range(min(n_chars_local, len(vocab))):
                        if _prev_str_pp == ',':
                            if vocab[_si] == ' ':
                                probs[0, _si] *= _PHI_FOR_SAMPLING
                        elif _prev_str_pp in ('.', '!', '?'):
                            if vocab[_si] == '\n':
                                probs[0, _si] *= _PHI_FOR_SAMPLING
                    probs[0] = probs[0] / (probs[0].sum() + 1e-8)
            # Hyphen compound booster: after '-', prefer word tokens (phi×).
            # Enables Shakespeare-style compounding: 'brave-new', 'ill-will',
            # 'world-earth'. Universal: hyphen = morpheme joiner cross-linguistically.
            if vocab is not None and seq.shape[1] >= 1:
                _prev_str_hc = (vocab[int(seq[0, -1])]
                                if int(seq[0, -1]) < len(vocab) else '')
                if _prev_str_hc == '-' and n_chars_local < len(vocab):
                    probs[0, n_chars_local:] = (
                        probs[0, n_chars_local:] * _PHI_FOR_SAMPLING)
                    probs[0] = probs[0] / (probs[0].sum() + 1e-8)
            # HARD minimum clause length: zero out sentence-enders until the
            # clause has >= F(3)=3 content words. Same pattern as word-boundary
            # and char-run hard constraints — applied directly to probs[0] so
            # the model's trained newline-preference cannot override it.
            # Language primitive: a clause needs minimum content to communicate.
            # F[4]=5: raised from F[3]=3 — forces longer phrases before closure.
            if clause_pos < _FIB_NUMS_FOR_BIGRAM[4]:
                if _terminal_punct_mask is not None:
                    probs[0] = probs[0] * (1.0 - _terminal_punct_mask)
                    _s = probs[0].sum()
                    if _s > 1e-8:
                        probs[0] = probs[0] / _s
                if newline_mask is not None:
                    probs[0] = probs[0] * (1.0 - newline_mask)
                    _s = probs[0].sum()
                    if _s > 1e-8:
                        probs[0] = probs[0] / _s
            elif not clause_has_verb and clause_pos < _FIB_NUMS_FOR_BIGRAM[5]:
                # Has words but no predicate — suppress terminal punct softly
                if _terminal_punct_mask is not None:
                    probs[0] = probs[0] * (1.0 - _terminal_punct_mask
                                           * (1.0 - 1.0 / _PHI_FOR_SAMPLING))
                    _s = probs[0].sum()
                    if _s > 1e-8:
                        probs[0] = probs[0] / _s
            # Fallback char-run cap at F(2)=2: belt-and-suspenders in case
            # char_run tracking slips. Forces separator including punct.
            if char_run >= _FIB_NUMS_FOR_BIGRAM[2] and vocab is not None:
                _sep_fb = {' ', '\n', '.', ',', '!', '?', ';', ':', "'", '-'}
                _fb_mask = torch.zeros_like(probs[0])
                for _si in range(min(n_chars_local, len(vocab))):
                    if vocab[_si] in _sep_fb:
                        _fb_mask[_si] = 1.0
                _s = _fb_mask.sum()
                if _s > 0:
                    probs[0] = _fb_mask / _s
            next_tok = torch.multinomial(probs, num_samples=1)
            # HBit pair gate: resample if consecutive token IDs are harmonically
            # dissonant (distance far from every Fibonacci number).
            _hbit_fibs = [1,2,3,5,8,13,21,34,55,89,144,233,377,610]
            _next_id = int(next_tok[0, 0])
            _prev_id = int(ctx[0, -1]) if ctx.shape[1] > 0 else 0
            _hbit_dist = abs(_next_id - _prev_id)
            _hbit_near = min(_hbit_fibs, key=lambda f: abs(_hbit_dist - f))
            _hbit = 1.0 / (1.0 + abs(_hbit_dist - _hbit_near))
            _hbit_threshold = 1.0 / (_PHI_FOR_SAMPLING ** 4)   # ≈ 0.146 — conservative
            _hbit_tries = 0
            while _hbit < _hbit_threshold and _hbit_tries < 3:
                next_tok = torch.multinomial(probs, num_samples=1)
                _next_id = int(next_tok[0, 0])
                _hbit_dist = abs(_next_id - _prev_id)
                _hbit_near = min(_hbit_fibs, key=lambda f: abs(_hbit_dist - f))
                _hbit = 1.0 / (1.0 + abs(_hbit_dist - _hbit_near))
                _hbit_tries += 1
            seq = torch.cat([seq, next_tok], dim=1)
            # Advance state counters from emitted token.
            if vocab is not None:
                nid = int(next_tok[0, 0])
                prev_for_pair = int(seq[0, -2]) if seq.shape[1] >= 2 else -1
                if nid < len(vocab):
                    tok = vocab[nid]
                    syl_pos += _approx_syllables(tok)
                    if tok in ('.', '!', '?', '\n'):
                        open_needs = 0
                        cluster_len = 0
                        syl_pos = 0
                        clause_pos = 0
                        clause_has_verb = False
                        clause_subject_id = -1
                        _fib_prev_class = -1   # reset at sentence boundary
                        _clause_anchor = None  # re-captured at next clause onset
                        if _gen_tracker is not None:
                            _gen_tracker.on_terminal()
                        if tok == '\n':
                            tokens_since_newline = 0
                            # Decay instead of zero: preserves cross-line memory with
                            # recency weighting. Universal discourse primitive: recently
                            # spoken vocabulary is still "in play" but loses salience
                            # at each sentence boundary, so novelty is rewarded.
                            heard_counts = heard_counts / _PHI_FOR_SAMPLING
                        else:
                            tokens_since_newline += 1
                    elif tok in (',', ';', ':'):
                        open_needs = max(0, open_needs - 1)  # partial closure
                        cluster_len = 0
                        tokens_since_newline += 1
                        if _gen_tracker is not None:
                            _gen_tracker.on_clause_break()
                        # Speaker marker detection: prev token was a word token + ":"
                        # Gate: only within F(4)=5 tokens of a line start — speaker
                        # attributions in dramatic text always appear at line start.
                        # This prevents content words like "state:" from falsely
                        # triggering speaker state mid-clause.
                        if tok == ':' and seq.shape[1] >= 2:
                            prev2_id = int(seq[0, -2])
                            if (prev2_id >= n_chars_local
                                    and tokens_since_newline <= _FIB_NUMS_FOR_BIGRAM[4]):
                                after_speaker = True
                                after_speaker_ticks = 0
                                speaker_count += 1
                                clause_pos = 0
                                clause_has_verb = False
                                clause_subject_id = -1
                                _fib_prev_class = -1   # new speaker resets arc
                                _clause_anchor = None  # re-captured at next clause onset
                                # heard_counts not reset here — only resets at \n
                                # so listener register isn't wiped by fake colons
                    elif nid > content_thresh:
                        open_needs += 1
                        clause_pos += 1
                        tokens_since_newline += 1
                        heard_counts[nid] = heard_counts[nid] + 1.0
                        if verb_mask is not None and verb_mask[nid]:
                            clause_has_verb = True
                        # Store first content word as topic anchor (A→D binding)
                        if clause_subject_id < 0:
                            clause_subject_id = nid
                        if _gen_tracker is not None:
                            _gen_tracker.on_word(tok)
                        if tok.endswith('s'):
                            last_content_ends_s = True
                        elif len(tok) > 1:
                            last_content_ends_s = False
                        # Update prev_leaf for FibPosTable circle prevention
                        if fib_pos_table is not None and fingerprint is not None and fingerprint._built:
                            _fib_prev_class = int(fingerprint.sub_clusters[min(nid, fingerprint.V - 1)].item())
                    elif n_chars_local <= nid <= content_thresh:
                        open_needs = max(0, open_needs - 1)
                        tokens_since_newline += 1
                        if _gen_tracker is not None:
                            _gen_tracker.on_word(tok)
                    if tok:
                        for ch in tok:
                            if ch in _IAMBIC_VOWELS:
                                cluster_len = 0
                            elif ch.isalpha():
                                cluster_len += 1
                            else:
                                cluster_len = 0
                    if nid < n_chars_local and tok.isalpha():
                        char_run += 1
                    else:
                        char_run = 0
                    # Decay after_speaker flag after F(3)=3 tokens
                    if after_speaker:
                        after_speaker_ticks += 1
                        if after_speaker_ticks >= _FIB_NUMS_FOR_BIGRAM[3]:
                            after_speaker = False
                if prev_for_pair >= 0:
                    recent_pairs.append((prev_for_pair, nid))
                    if len(recent_pairs) > 13:
                        recent_pairs = recent_pairs[-13:]
                _loop_tok_lens.append(len(vocab[nid]) if nid < len(vocab) else 1)
                _hyphen_recent.append(1 if nid == _hyphen_tid else 0)
                if nid in _boundary_tids:
                    _clause_ngram_count = 0
                elif nid >= n_chars_local:
                    _clause_ngram_count += 1
                # Self-evaluation: update creative momentum EMA.
                insight = _self_eval_insight(base, nid, n_chars_local)
                inv_phi = 1.0 / _PHI_FOR_SAMPLING
                creative_momentum = (inv_phi * creative_momentum
                                       + (1.0 - inv_phi) * insight)
                # Track momentum history for backtrack detection.
                momentum_history.append(creative_momentum)
                if len(momentum_history) > 13:
                    momentum_history = momentum_history[-13:]
                # Update unknown-register coverage.
                if 0 <= nid < vocab_size:
                    coverage[nid] += 1.0
                # Update context tape: Fibonacci decay + new token's fwd fingerprint.
                # O(1) per step — no recomputation from history.
                # _fp_tape_prev captures tape BEFORE this step for manifold velocity.
                if _fp_tape is not None and 0 <= nid < fingerprint.V:
                    _fp_tape_prev = _fp_tape.clone()
                    phi_fp = fingerprint._PHI
                    _fp_tape = (_fp_tape / phi_fp
                                + fingerprint.fwd[nid].to(_fp_tape.device))
                    _fp_norm = _fp_tape.norm()
                    if _fp_norm > 1e-8:
                        _fp_tape = _fp_tape / _fp_norm
                # Advance leaf trajectory at clause/sentence boundaries.
                # Corpus punctuation structure drives semantic leaf transitions.
                if (_traj is not None and nid < len(vocab)
                        and vocab[nid] in ('.', '!', '?', '\n', ';')):
                    _traj_pos = (_traj_pos + 1) % len(_traj)
                    _seg_count = 0
                    _traj_seg_i = (_traj_seg_i + 1) % len(_FIB_SEG)
                    _seg_target = _FIB_SEG[_traj_seg_i]
    model.train()
    return seq
def staged_refine(model, prompt, n_new, vocab_size,
                    harmony_scorer, quality_scorer, creativity_scorer,
                    n_iters_per_stage: int = 200,
                    resample_frac: float = 0.35,
                    prompt_len: int = 16,
                    temperature: float = 0.5,
                    bigram_prior: torch.Tensor = None,
                    vocab: list = None,
                    token_signatures: torch.Tensor = None,
                    active_vocab_size: int = None,
                    class_id_tensor: torch.Tensor = None,
                    n_classes: int = 0,
                    pronoun_mask: torch.Tensor = None,
                    vowel_start_mask: torch.Tensor = None,
                    end_vowels: list = None,
                    punct_mask: torch.Tensor = None,
                    newline_mask: torch.Tensor = None,
                    unpronounceable_mask: torch.Tensor = None,
                    allowed_after_word_mask: torch.Tensor = None,
                    uppercase_mask: torch.Tensor = None,
                    any_punct_mask: torch.Tensor = None,
                    first_person_mask: torch.Tensor = None,
                    second_person_mask: torch.Tensor = None,
                    third_person_mask: torch.Tensor = None,
                    verb_mask: torch.Tensor = None,
                    noun_mask: torch.Tensor = None,
                    adj_mask: torch.Tensor = None,
                    func_mask: torch.Tensor = None,
                    target_profile: dict = None,
                    fib_pos_table: 'FibPosTable' = None):
    """Staircase refinement: hit one score, then the next, then the next.

    Stage 1: substrate alignment (minimize harmony) -- match the shape.
    Stage 2: model coherence (minimize self-perplexity) -- output that
             the model itself finds plausible given the substrate shape.
    Stage 3: Shakespeare creativity (maximize creativity score) -- output
             that matches Shakespeare's char patterns and vocabulary.

    Each stage starts from the PREVIOUS stage's best output. Output of
    one objective becomes the input to the next.
    """
    model.eval()
    with torch.no_grad():
        draft = autoregressive_generate(model, prompt, n_new=n_new,
                                          vocab_size=vocab_size,
                                          temperature=temperature, bigram_prior=bigram_prior, vocab=vocab, token_signatures=token_signatures, active_vocab_size=active_vocab_size, class_id_tensor=class_id_tensor, n_classes=n_classes, pronoun_mask=pronoun_mask, vowel_start_mask=vowel_start_mask, end_vowels=end_vowels, punct_mask=punct_mask, newline_mask=newline_mask, unpronounceable_mask=unpronounceable_mask, allowed_after_word_mask=allowed_after_word_mask, uppercase_mask=uppercase_mask, any_punct_mask=any_punct_mask, first_person_mask=first_person_mask, second_person_mask=second_person_mask, third_person_mask=third_person_mask, verb_mask=verb_mask, noun_mask=noun_mask, adj_mask=adj_mask, func_mask=func_mask, target_profile=target_profile, fib_pos_table=fib_pos_table)
    stages_out = {}
    stages_out["initial"] = {"seq": draft.clone(),
                                "harmony": harmony_scorer(draft),
                                "quality": quality_scorer(draft),
                                "creativity": creativity_scorer(draft)
                                  if creativity_scorer else None}
    model.train()
    return draft, stages_out


def compute_harmony(logits, vocab_size, kind):
    """kind in {'none', 'char', 'multiscale', 'combined'}."""
    if kind == "none":
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    if kind == "char":
        return substrate_harmony_loss(logits, vocab_size)
    if kind == "multiscale":
        return substrate_multiscale_harmony_loss(logits, vocab_size)
    if kind == "combined":
        return (substrate_harmony_loss(logits, vocab_size)
                + substrate_multiscale_harmony_loss(logits, vocab_size))
    raise ValueError(f"unknown harmony kind: {kind}")


def K_to_K_harmony(K_active: int, K_init: int = 89, K_min: int = 13,
                     K_harmony_max: int = 7,
                     K_harmony_min: int = 2) -> int:
    """Map model's active K to the harmony's active frequency count.

    As the model's basis shrinks (89→13), the harmony's measuring stick
    shrinks proportionally (7→2). Substrate stays congruent with model.
    """
    if K_init <= K_min:
        return K_harmony_max
    frac = (K_active - K_min) / (K_init - K_min)
    K_harmony = round(K_harmony_min + frac * (K_harmony_max - K_harmony_min))
    return max(K_harmony_min, min(K_harmony_max, K_harmony))


def compute_harmony_grounded(logits, vocab_size, kind, sig_char, sig_ms,
                                K_harmony=None):
    """Corpus-grounded harmony. sig_char and sig_ms are pre-computed
    target signatures from the actual corpus. K_harmony shrinks the
    harmony's active frequency count to match model's K."""
    if kind == "none":
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    if kind == "char":
        return substrate_harmony_loss_grounded(logits, vocab_size, sig_char,
                                                  K_harmony=K_harmony)
    if kind == "multiscale":
        return substrate_multiscale_harmony_loss_grounded(
            logits, vocab_size, sig_ms, K_harmony=K_harmony)
    if kind == "combined":
        return (substrate_harmony_loss_grounded(logits, vocab_size, sig_char,
                                                   K_harmony=K_harmony)
                + substrate_multiscale_harmony_loss_grounded(
                    logits, vocab_size, sig_ms, K_harmony=K_harmony))
    raise ValueError(f"unknown harmony kind: {kind}")


_FIB_FREQS_LOCAL = [1, 2, 3, 5, 8, 13, 21]
_FIB_LAGS_LOCAL = [1, 2, 3, 5, 8, 13, 21]
_FIB_NUMS_LOCAL = [1, 1, 2, 3, 5, 8, 13]
PHI_LOCAL = (1.0 + 5.0 ** 0.5) / 2.0


class ParametricSubstrate:
    """Substrate constants (phi, pi_exp, fib_weights) as the ONLY mutable
    parameters. The canonical signature is always F(k)/phi^(pi_exp*k) --
    mutations stay congruent to the substrate formula. Free drift away
    from this family is forbidden by construction.
    """

    def __init__(self, phi=None, pi_exp=None, fib_weights=None):
        self.phi = PHI_LOCAL if phi is None else float(phi)
        self.pi_exp = math.pi if pi_exp is None else float(pi_exp)
        self.fib_weights = (list(_FIB_NUMS_LOCAL) if fib_weights is None
                              else [float(x) for x in fib_weights])

    def get_signature(self, K=None) -> torch.Tensor:
        if K is None:
            K = len(self.fib_weights)
        un = [self.fib_weights[k] / (self.phi ** (self.pi_exp * k))
              for k in range(K)]
        total = sum(un) + 1e-8
        return torch.tensor([u / total for u in un], dtype=torch.float)

    def clone(self):
        return ParametricSubstrate(self.phi, self.pi_exp, self.fib_weights)

    def perturb(self, rng, step_size: float = 0.05,
                  fib_step: float = 0.10) -> "ParametricSubstrate":
        """Joint blind perturbation: phi, pi_exp, all fib_weights drift.
        No data signal -- pure random move bounded to substrate-physical
        ranges. Kept for ablation against data-guided perturbation.
        """
        new = self.clone()
        K = len(self.fib_weights)
        d_phi = (rng.random() * 2 - 1) * step_size
        new.phi = max(PHI_LOCAL * 0.8,
                        min(PHI_LOCAL * 1.2, self.phi * (1 + d_phi)))
        d_pi = (rng.random() * 2 - 1) * step_size
        new.pi_exp = max(math.pi * 0.8,
                            min(math.pi * 1.2, self.pi_exp * (1 + d_pi)))
        for k in range(K):
            d_fib = (rng.random() * 2 - 1) * fib_step
            new.fib_weights[k] = max(0.1, self.fib_weights[k] * (1 + d_fib))
        return new

    def data_guided_perturb(self, target_sig: torch.Tensor, rng,
                              step_size: float = 0.05,
                              noise_scale: float = 0.5,
                              K_active: int = None,
                              ) -> "ParametricSubstrate":
        """Mutation BIASED by the corpus signature gradient.

        Computes d|parametric_sig - target_sig|/d(phi, pi_exp, fib_weights)
        via autograd, mutates each constant in the descent direction with
        added noise (noise_scale fraction of step_size). The corpus tells
        the mutation where to push every constant; pure random + reward
        is replaced by data-informed proposal + reward.

        K_active: if set, only the first K_active components contribute
        to the gradient (matches K-harmony schedule).
        """
        K_full = len(self.fib_weights)
        K_use = K_full if K_active is None else min(K_active, K_full)
        # Tensors with grad enabled (must be float).
        phi_t = torch.tensor(float(self.phi), requires_grad=True)
        pi_t = torch.tensor(float(self.pi_exp), requires_grad=True)
        fib_t = torch.tensor([float(x) for x in self.fib_weights],
                              dtype=torch.float32, requires_grad=True)
        ks = torch.arange(K_use, dtype=torch.float)
        # parametric signature at K_use
        unnorm = fib_t[:K_use] / (phi_t ** (pi_t * ks))
        sig = unnorm / (unnorm.sum() + 1e-8)
        # gap to target (truncated to K_use)
        target = target_sig[:K_use]
        target = target / (target.sum() + 1e-8)
        loss = (sig - target).abs().sum()
        loss.backward()
        # Read gradients (descent = -grad).
        g_phi = float(phi_t.grad)
        g_pi = float(pi_t.grad)
        g_fib = fib_t.grad.tolist()

        new = self.clone()
        # phi: step in -g_phi direction (multiplicatively scaled) + noise.
        d_phi_rel = -g_phi * step_size + (rng.random() * 2 - 1) * step_size * noise_scale
        new.phi = max(PHI_LOCAL * 0.8,
                        min(PHI_LOCAL * 1.2, self.phi + d_phi_rel * abs(self.phi)))
        d_pi_rel = -g_pi * step_size + (rng.random() * 2 - 1) * step_size * noise_scale
        new.pi_exp = max(math.pi * 0.8,
                            min(math.pi * 1.2,
                                 self.pi_exp + d_pi_rel * abs(self.pi_exp)))
        for k in range(K_full):
            grad_k = g_fib[k] if k < K_use else 0.0
            d_fib_rel = (-grad_k * step_size
                          + (rng.random() * 2 - 1) * step_size * noise_scale)
            new.fib_weights[k] = max(0.1, self.fib_weights[k]
                                            + d_fib_rel * abs(self.fib_weights[k]))
        return new

    def summary(self) -> str:
        fib_str = ",".join(f"{w:.2f}" for w in self.fib_weights[:5])
        return (f"phi={self.phi:.4f} pi={self.pi_exp:.4f} "
                f"fib=[{fib_str},...]")


def measure_emergent_signatures(model, seed, batch_size, seq_len, vocab_size,
                                  gen, n_batches=4):
    """Measure model's emergent substrate signatures from its training outputs.
    Mirrors corpus_char_signature/corpus_multiscale_signature but on
    model's predicted distributions, not on raw tokens."""
    model.eval()
    fib_freqs = torch.tensor(_FIB_FREQS_LOCAL, dtype=torch.float)
    K = fib_freqs.numel()
    v_idx = torch.arange(vocab_size, dtype=torch.float)
    angles = 2 * math.pi * v_idx.unsqueeze(1) * fib_freqs.unsqueeze(0) / vocab_size
    basis_cos = torch.cos(angles)
    basis_sin = torch.sin(angles)
    energies = []
    sims_per = [[] for _ in range(K)]
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = sample_tiny_batch(seed, batch_size, seq_len, gen)
            logits = model(x)
            pred = F.softmax(logits, dim=-1)
            pred_cos = pred @ basis_cos
            pred_sin = pred @ basis_sin
            energy = (pred_cos ** 2 + pred_sin ** 2).mean(dim=(0, 1))
            energies.append(energy)
            T = pred.shape[1]
            for i, lag in enumerate(_FIB_LAGS_LOCAL):
                if T <= lag:
                    sims_per[i].append(torch.tensor(0.0))
                    continue
                p1 = pred[:, :-lag]
                p2 = pred[:, lag:]
                sim = (p1 * p2).sum(dim=-1).mean()
                sims_per[i].append(sim)
    model.train()
    energy_mean = torch.stack(energies).mean(0)
    energy_mean = energy_mean / (energy_mean.sum() + 1e-8)
    ms_mean = torch.stack([torch.stack(s).mean() for s in sims_per])
    ms_mean = ms_mean / (ms_mean.sum() + 1e-8)
    return energy_mean, ms_mean


# ── Substrate-universal production primitives ────────────────────────────────

def substrate_coherence_score(text: str) -> float:
    """Substrate-universal coherence score — no dictionary, no word list.

    Works on any corpus because it measures STRUCTURE only:

    1. Trigram self-diversity: unique(3-grams) / total.
       High = not repetitively stuck ("fan fan fan" → 0).

    2. Window entropy stability: Shannon entropy of chars in F(8)=21-char
       sliding windows (step F(6)=8). Low variance = stable information
       density. Gibberish oscillates wildly; coherent text is consistent.
       This is the dominant signal (weight F(6)=8).

    3. Punctuation rhythm: mean gap between sentence boundaries (.!?\\n).
       Any natural language has gaps in [5, 300]; outside = structural noise.
       Weight F(5)=5, same as trigram.

    Fibonacci weights [5, 8, 5] / 18 — entropy_stability dominates.
    Returns [0, 1], higher = more coherent.
    """
    import math as _math
    from collections import Counter as _Counter

    if len(text) < 10:
        return 0.0

    # 1. Trigram self-diversity
    grams = [text[i:i+3] for i in range(len(text) - 2)]
    diversity = len(set(grams)) / len(grams) if grams else 0.0

    # 2. Window entropy signal — two components:
    #    (a) entropy LEVEL: text should use a RICH character set, not just
    #        4 chars repeated uniformly. We normalize mean entropy against
    #        the theoretical max for the OBSERVED unique chars, then measure
    #        distance from the natural-language sweet spot (~70-80% of max).
    #    (b) entropy STABILITY: low variance of entropy across windows.
    # Natural language sits at ~70-85% of per-window max entropy regardless
    # of language — this is corpus-agnostic and substrate-universal.
    w, step = 21, 8   # F(8)=21, F(6)=8
    entropies = []
    for i in range(0, len(text) - w, step):
        chunk = text[i:i+w]
        counts = _Counter(chunk)
        h = -sum((c / w) * _math.log(c / w + 1e-8) for c in counts.values())
        entropies.append(h)
    obs_unique = max(len(set(text)), 2)
    max_h = _math.log(obs_unique)
    if len(entropies) < 2:
        entropy_stability = 0.5
    else:
        mean_h = sum(entropies) / len(entropies)
        var_h  = sum((h - mean_h) ** 2 for h in entropies) / len(entropies)
        # (a) Level: how close to 70-80% of max entropy? (substrate sweet spot)
        normalized_h = mean_h / (max_h + 1e-8)
        h_sweet = 1.0 - 1.0 / (_PHI_FOR_SAMPLING ** 3)   # phi^3 sweet spot ≈ 0.764
        level_score = max(0.05, 1.0 - abs(normalized_h - h_sweet) * (_PHI_FOR_SAMPLING ** 2))
        # (b) Stability: low variance = consistent information density
        stability_score = 1.0 / (1.0 + var_h * 2.0)
        entropy_stability = level_score * stability_score

    # 3. Punctuation rhythm — purely structural, no lexical knowledge
    boundaries = [i for i, c in enumerate(text) if c in '.!?\n']
    if len(boundaries) >= 2:
        gaps = [boundaries[k+1] - boundaries[k] for k in range(len(boundaries)-1)]
        mean_gap = sum(gaps) / len(gaps)
        # Score peaks at gap≈50 chars, falls off toward 5 and 300
        if 5 <= mean_gap <= 300:
            rhythm = 1.0 - abs(_math.log(mean_gap / 55.0 + 1e-8)) / _math.log(89.0)
            rhythm = max(0.1, min(1.0, rhythm))
        else:
            rhythm = 0.1
    elif boundaries:
        rhythm = 0.3
    else:
        rhythm = 0.05   # no punctuation = no discernible sentence structure

    return float((5 * diversity + 8 * entropy_stability + 5 * rhythm) / 18.0)


def produce_recursive(model, prompt_tokens: torch.Tensor,
                       n_chunks: int, chunk_size: int,
                       vocab_size: int,
                       temperature: float = 0.75,
                       bigram_prior: torch.Tensor = None,
                       vocab: list = None,
                       **gen_kwargs) -> torch.Tensor:
    """Self-recursive production loop.

    The model reads what it wrote and continues writing, chunk by chunk.
    Like a writer re-reading their draft before continuing the next stanza.

    Each chunk uses the LAST seq_len tokens of the accumulated output as
    context — so every new chunk is conditioned on the model's own prior
    output, not on the seed corpus.

    Returns shape (1, prompt_len + n_chunks * chunk_size).
    """
    model.eval()
    with torch.no_grad():
        buf = prompt_tokens.clone()
        for _ in range(n_chunks):
            ctx = buf[:, -model.seq_len:] if buf.shape[1] > model.seq_len else buf
            extended = autoregressive_generate(
                model, ctx, n_new=chunk_size,
                vocab_size=vocab_size, temperature=temperature,
                bigram_prior=bigram_prior, vocab=vocab, **gen_kwargs)
            new_part = extended[:, ctx.shape[1]:]
            buf = torch.cat([buf, new_part], dim=1)
    model.train()
    return buf


def substrate_structure_score(text: str) -> float:
    """Substrate-universal structure quality: grammar, punctuation, spacing.

    Three signals — pure char/length arithmetic, no vocabulary:

    1. Average word length (natural range 3.5-6.5 chars).
       'this the of the, the' has avg ~3.0 (too short — all function tokens).
       'Consinsorihouse against h' pulls avg high but length_score catches it.
       Real Shakespeare poetry averages ~4.5-6 chars per word.
       Score peaks at 5.0, falls to 0 at ≤1.5 or ≥9.5.

    2. Word-length plausibility: penalises words > 13 chars.
       Natural words are almost never > 13 chars — longer strings are
       garbled n-gram concatenations like 'Consinsorihouse'.

    3. Punctuation placement: each punct mark should be followed by
       space/newline/end — not another punct or a bare letter.
       Catches ', of' floating commas and double-punct runs.

    Fibonacci weights [5, 8, 5] / 18.
    """
    if len(text) < 10:
        return 0.0
    words = text.split()
    if not words:
        return 0.0
    stripped = [w.strip('.,!?;:\'"()') for w in words if w.strip('.,!?;:\'"()')]
    if not stripped:
        return 0.0

    # 1. Average word length — distinguishes function-word soup from real text
    avg_len = sum(len(w) for w in stripped) / len(stripped)
    # Peaks at 5.0; score = 0 at avg_len <= 1.5 or >= 9.5
    avg_len_score = max(0.0, 1.0 - abs(avg_len - 5.0) / 3.5)

    # 2. Garbled-concatenation penalty (words > 13 chars = F(7))
    long_count = sum(1 for w in stripped if len(w) > 13)
    length_score = max(0.0, 1.0 - (long_count / max(len(stripped), 1)) * (_PHI_FOR_SAMPLING ** 2))

    # 3. Punctuation placement quality
    punct = set('.,!?;:')
    total_p = sum(1 for c in text if c in punct)
    good_p  = sum(
        1 for i, c in enumerate(text)
        if c in punct and (i + 1 >= len(text) or text[i + 1] in (' ', '\n'))
    )
    punct_score = (good_p / total_p) if total_p > 0 else 0.5

    return float((5 * avg_len_score + 8 * length_score + 5 * punct_score) / 18.0)


def substrate_discourse_score(text: str) -> float:
    """Substrate-universal discourse quality — no word lists.

    Three structural signals, language-agnostic:

    1. Sentence opening variety: how many distinct sentence-initial
       4-char signatures? High = text starts sentences differently =
       narrative progression. Pure character arithmetic.

    2. Dialogue structure: colon density relative to newline density.
       Speaker→utterance attribution shows up as colons before dialogue
       in any language that marks speakers. Peaks at ~1 colon per line.

    3. Forward novelty: fraction of sentences that introduce at least one
       word not seen in the immediately preceding sentence. Corpus-native:
       measures progression without vocabulary knowledge.

    Fibonacci weights [5, 8, 5] / 18.
    """
    import re as _re
    sentences = [s.strip() for s in _re.split(r'[.!?\n]+', text) if len(s.strip()) > 3]
    if len(sentences) < 2:
        return 0.3

    # 1. Sentence opening variety
    openings = [s[:4].lower() for s in sentences if len(s) >= 4]
    variety = len(set(openings)) / max(len(openings), 1)

    # 2. Dialogue structure: colon-to-newline ratio
    n_colons   = text.count(':')
    n_newlines = max(text.count('\n'), 1)
    colon_ratio = min(1.0, n_colons / n_newlines)
    dialogue = min(1.0, colon_ratio * 2.0)

    # 3. Forward novelty
    novelty_hits = 0
    prev_words: set = set()
    for sent in sentences:
        words = set(w.lower().strip('.,;:!?\'"') for w in sent.split())
        if prev_words and words - prev_words:
            novelty_hits += 1
        prev_words = words
    novelty = novelty_hits / max(len(sentences) - 1, 1)

    return float((5 * variety + 8 * dialogue + 5 * novelty) / 18.0)


def substrate_profile_from_tokens(seed_tokens: torch.Tensor,
                                   vocab: list,
                                   n_chars: int) -> dict:
    """Extract the OMNIbalance substrate profile from the training seed tokens.

    This is the MAP — the structural target the generator should aim for.
    Computed once at startup; used as the setpoint for closed-loop steering.

    Measures substrate-universal language structure (grammatical person,
    clause length, forward novelty, speaker density) — no word lists.
    """
    first_p  = {'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our'}
    second_p = {'thou', 'thee', 'thy', 'thine', 'you', 'your', 'ye'}
    third_p  = {'he', 'him', 'his', 'she', 'her', 'it', 'they', 'them', 'their'}
    toks = seed_tokens[0].tolist() if seed_tokens.dim() == 2 else seed_tokens.tolist()

    sentences: list = []
    current_words: set = set()
    clause_lengths: list = []
    current_clause_len = 0
    speaker_count = 0
    tokens_since_nl = 0

    for i, tid in enumerate(toks):
        if tid < 0 or tid >= len(vocab):
            continue
        tok = vocab[tid]
        word = tok.lower().strip('.,;:!?\'"')
        if tok in ('.', '!', '?', '\n'):
            if current_clause_len > 0:
                clause_lengths.append(current_clause_len)
            if current_words:
                sentences.append(current_words.copy())
                current_words = set()
            current_clause_len = 0
            tokens_since_nl = 0
        elif tok in (',', ';'):
            if current_clause_len > 0:
                clause_lengths.append(current_clause_len)
            current_clause_len = 0
            tokens_since_nl += 1
        elif tok == ':':
            if i >= 1 and toks[i - 1] >= n_chars and tokens_since_nl <= 5:
                speaker_count += 1
            current_clause_len = 0
            tokens_since_nl += 1
        elif tid >= n_chars:
            if word:
                current_words.add(word)
            current_clause_len += 1
            tokens_since_nl += 1
        else:
            tokens_since_nl += 1

    # Discourse metrics
    addressee_hits = diversity_hits = novelty_hits = 0
    prev_ws: set = set()
    for ws in sentences:
        has1 = bool(ws & first_p)
        has2 = bool(ws & second_p)
        has3 = bool(ws & third_p)
        if (has1 or has2) and has3:
            diversity_hits += 1
        if has2:
            addressee_hits += 1
        if prev_ws and ws - prev_ws:
            novelty_hits += 1
        prev_ws = ws

    n = max(len(sentences), 1)
    return {
        'addressee_frac': addressee_hits / n,
        'novelty_frac':   novelty_hits  / max(n - 1, 1),
        'diversity_frac': diversity_hits / n,
        'clause_len_mean': (sum(clause_lengths) / max(len(clause_lengths), 1)),
        'speaker_density': speaker_count / max(len(toks) / 100.0, 1.0),
    }


class SubstrateGenTracker:
    """Closed-loop runtime tracker for substrate-map steering.

    Maintains running discourse metrics as generation proceeds and computes
    error signals between current state and the target profile.

    The tracker is the 'compass' that tells the generator which direction
    to move at each step — the map makes generativity intentional rather
    than statistical.
    """
    __slots__ = ('target', 'sentences_count', 'addressee_count',
                 'novelty_count', 'diversity_count', 'prev_words',
                 'current_words', 'clause_lengths', 'current_clause_len',
                 '_first_p', '_second_p', '_third_p')

    def __init__(self, target_profile: dict):
        self.target = target_profile
        self.sentences_count = 0
        self.addressee_count = 0
        self.novelty_count = 0
        self.diversity_count = 0
        self.prev_words: set = set()
        self.current_words: set = set()
        self.clause_lengths: list = []
        self.current_clause_len = 0
        self._first_p  = {'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our'}
        self._second_p = {'thou', 'thee', 'thy', 'thine', 'you', 'your', 'ye'}
        self._third_p  = {'he', 'him', 'his', 'she', 'her', 'it', 'they', 'them', 'their'}

    def on_terminal(self):
        """Call when . ! ? \\n is emitted."""
        if self.current_clause_len > 0:
            self.clause_lengths.append(self.current_clause_len)
        self.current_clause_len = 0
        ws = self.current_words
        if self.sentences_count > 0 and ws:
            has1 = bool(ws & self._first_p)
            has2 = bool(ws & self._second_p)
            has3 = bool(ws & self._third_p)
            if (has1 or has2) and has3:
                self.diversity_count += 1
            if has2:
                self.addressee_count += 1
            if self.prev_words and ws - self.prev_words:
                self.novelty_count += 1
        self.prev_words = ws.copy()
        self.current_words = set()
        self.sentences_count += 1

    def on_clause_break(self):
        """Call when , ; : is emitted."""
        if self.current_clause_len > 0:
            self.clause_lengths.append(self.current_clause_len)
        self.current_clause_len = 0

    def on_word(self, tok_str: str):
        """Call when a word token is emitted."""
        word = tok_str.lower().strip('.,;:!?\'"')
        if word:
            self.current_words.add(word)
        self.current_clause_len += 1

    @property
    def _n(self):
        return max(self.sentences_count - 1, 1)

    def addressee_error(self) -> float:
        if self.sentences_count < 2:
            return 0.0
        return self.target.get('addressee_frac', 0.5) - self.addressee_count / self._n

    def novelty_error(self) -> float:
        if self.sentences_count < 2:
            return 0.0
        return self.target.get('novelty_frac', 0.8) - self.novelty_count / self._n

    def clause_len_error(self) -> float:
        if not self.clause_lengths:
            return 0.0
        cur = sum(self.clause_lengths) / len(self.clause_lengths)
        return self.target.get('clause_len_mean', 5.0) - cur


def substrate_map_steer(tracker,
                          clause_pos: int,
                          base: torch.Tensor,
                          second_person_mask: torch.Tensor = None,
                          heard_counts: torch.Tensor = None,
                          n_chars: int = 65,
                          fib_pos_table=None) -> torch.Tensor:
    """Substrate map steering — closed-loop proportional controller.

    Reads error signals from the tracker (current generation profile vs
    target OMNIbalance profile) and applies corrective substrate biases.

    Congruent coupling: when fib_pos_table is provided, the tracker's
    correction strength is modulated by whether the table agrees. If the
    learned positional grammar also predicts 2nd-person at this clause
    position, the tracker's addressee boost is amplified (both systems
    agree → push harder). If the table expects a different class, the
    boost is attenuated (table's learned grammar says 'not here').

    Universal substrate: grammatical person (addressee boost) and forward
    novelty (listener suppression amplification). No word lists.
    """
    phi = _PHI_FOR_SAMPLING
    F   = _FIB_NUMS_FOR_BIGRAM
    out = base.clone()

    # ── Addressee steering ────────────────────────────────────────────────
    # Active at clause_pos [F[3], F[5]) — the second-person window.
    # Proportional: error 0.1 → phi^0.5 extra, 0.3 → phi^1.5, 0.5 → phi^2.5.
    if F[3] <= clause_pos < F[5] and second_person_mask is not None:
        err = tracker.addressee_error()
        if err > 0.05:
            # Congruent coupling: FibPosTable agreement modulates boost strength.
            congruence = 1.0
            # FibPosTable is now leaf-indexed; no SECOND-person slot.
            # Addressee steering applies at full strength (congruence=1.0).
            extra = min(2.5, err * 5.0 * congruence)
            mult = phi ** extra
            out = out * (1.0 + second_person_mask.to(base.device) * (mult - 1.0))
        elif err < -0.15:
            damp_v = 1.0 - 1.0 / phi
            out = out * (1.0 - second_person_mask.to(base.device) * damp_v)

    # ── Novelty steering ──────────────────────────────────────────────────
    # When forward novelty is below target, amplify suppression on tokens
    # that have already been heard (decayed across sentence boundaries).
    # This pushes toward fresher vocabulary to match the reference structure.
    if heard_counts is not None and tracker.novelty_error() > 0.15:
        word_counts = heard_counts.clone()
        word_counts[:n_chars] = 0.0  # only word tokens
        # All tokens that have any recency signal (count >= 0.5) get extra
        # 1/sqrt(phi) suppression — stack on top of listener register.
        extra_supp = torch.where(word_counts >= 0.5,
                                  torch.full_like(out, 1.0 / (phi ** 0.5)),
                                  torch.ones_like(out))
        out = out * extra_supp

    s = out.sum()
    return out / (s + 1e-8) if s > 0 else base


def seed_phrase_score(text: str, corpus_text: str) -> float:
    """Fraction of word-trigrams in text that appear verbatim in corpus_text.
    Kept as fallback when fingerprint is not yet available.
    """
    import re as _re

    def _words(t: str):
        return [w.lower() for w in _re.findall(r"[a-zA-Z']+", t) if len(w) >= 2]

    gen_words = _words(text)
    if len(gen_words) < 3:
        return 0.0
    corpus_words = _words(corpus_text)
    if len(corpus_words) < 3:
        return 0.0

    corpus_3g = frozenset(
        (corpus_words[i], corpus_words[i+1], corpus_words[i+2])
        for i in range(len(corpus_words) - 2)
    )
    gen_3g = [
        (gen_words[i], gen_words[i+1], gen_words[i+2])
        for i in range(len(gen_words) - 2)
    ]
    return sum(1 for tg in gen_3g if tg in corpus_3g) / len(gen_3g)


def leaf_diversity_score(text: str, fingerprint: 'SubstrateFingerprint',
                         vocab: list) -> float:
    """Coherence-weighted fraction of fractal leaves visited.

    Visiting a tight semantic leaf (high intra-coherence) counts more than
    visiting the function-word void (coh ≈ 0). The metric rewards generation
    that commits to high-structure semantic regions — exactly what the leaf
    trajectory planner does.

    score = Σ max(0, coh[visited_leaf]) / Σ max(0, coh[all_leaves])

    Uses character-level token lookup (single-char entries, ~first 65).
    """
    if fingerprint is None or not fingerprint._built or fingerprint.n_leaves < 2:
        return 0.0
    char_to_id: dict = {}
    for vid, tok in enumerate(vocab):
        if len(tok) == 1 and vid < fingerprint.V and tok not in char_to_id:
            char_to_id[tok] = vid
    if not char_to_id:
        return 0.0
    ids = [char_to_id[ch] for ch in text if ch in char_to_id]
    if len(ids) < 3:
        return 0.0
    id_tensor = torch.tensor(ids, dtype=torch.long)
    valid = id_tensor[id_tensor < fingerprint.V]
    if valid.numel() == 0:
        return 0.0
    leaf_ids = fingerprint.sub_clusters[valid]
    visited = set(leaf_ids.tolist())
    # Per-leaf coherence: compute intra-cluster mean cosine sim for each leaf.
    # Cache on fingerprint to avoid recomputing each call.
    if not hasattr(fingerprint, '_leaf_coherences'):
        _fwd_n = fingerprint.fwd.float()
        _nrm = _fwd_n.norm(dim=1, keepdim=True).clamp(min=1e-8)
        _unit = _fwd_n / _nrm
        _cohs = []
        for _lid in range(fingerprint.n_leaves):
            _ids = (fingerprint.sub_clusters == _lid).nonzero(as_tuple=True)[0]
            _cohs.append(fingerprint._intra_coherence(_unit[_ids]) if _ids.numel() > 1 else 1.0)
        fingerprint._leaf_coherences = _cohs
    cohs = fingerprint._leaf_coherences
    total_coh = sum(max(0.0, c) for c in cohs)
    if total_coh < 1e-8:
        return float(len(visited)) / fingerprint.n_leaves
    _ldf_fibs = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]

    def _leaf_ow(lid):
        _sz = int((fingerprint.sub_clusters == lid).sum())
        _near = min(_ldf_fibs, key=lambda f: abs(_sz - f))
        return _PHI_FOR_SAMPLING ** -(abs(_sz - _near) / max(_near, 1))

    visited_coh = sum(max(0.0, cohs[lid]) * _leaf_ow(lid) for lid in visited if lid < len(cohs))
    total_coh_ow = sum(max(0.0, cohs[lid]) * _leaf_ow(lid) for lid in range(len(cohs)))
    return visited_coh / max(total_coh_ow, 1e-8)


def substrate_production_score(text: str, corpus_text: str,
                               fingerprint=None, vocab: list = None,
                               tokens: list = None, fib_table=None,
                               n_chars: int = 0) -> float:
    """Combined production quality: coherence + creativity + structure + discourse + diversity.

    Fibonacci weights [5, 8, 2, 5, 5] / 25.

    Discourse — corpus-native when tokens + fib_table available:
      • leaf_positional_match_score : how well the text's leaf-per-FibBin distribution
        matches the FibPosTable's corpus-learned positional grammar. The table speaks.
      • substrate_discourse_score   : structural fallback (no word lists) when token
        sequence unavailable (e.g. final evaluation from text only).

    Phrase — corpus-native when fingerprint available:
      • leaf_diversity_score : fraction of fractal leaves visited (no external ref)
      • seed_phrase_score    : word-trigram recall (fallback without fingerprint)
    """
    coherence = substrate_coherence_score(text)
    cr        = compute_creativity_score(text, corpus_text)["creativity_score"]
    structure = substrate_structure_score(text)
    if (tokens is not None and fib_table is not None
            and fingerprint is not None and vocab is not None):
        discourse = leaf_positional_match_score(tokens, fingerprint, fib_table, vocab, n_chars)
    else:
        discourse = substrate_discourse_score(text)
    if fingerprint is not None and vocab is not None:
        phrase = leaf_diversity_score(text, fingerprint, vocab)
    else:
        phrase = seed_phrase_score(text, corpus_text)
    return float((5 * coherence + 8 * cr + 2 * structure + 5 * discourse + 5 * phrase) / 25.0)


def corpus_match_score(gen_tokens: list, corpus_tokens: torch.Tensor,
                        vocab: list = None) -> dict:
    """The primary measurement of reflection quality.

    Compare the model's generated token sequence against the actual corpus
    continuation.  This is the answer key — not a proxy metric.
    corpus_tokens should be the tokens immediately following the generation prompt.

    Returns:
      exact_1     : fraction of tokens that exactly match corpus (token-level)
      ngram_2     : fraction of generated 2-grams present in corpus 2-grams
      ngram_3     : fraction of generated 3-grams present in corpus 3-grams
      match_score : harmonic combination  (3×exact + 5×ng2 + 8×ng3) / 16
    """
    if not gen_tokens or corpus_tokens is None or corpus_tokens.numel() == 0:
        return {"exact_1": 0.0, "ngram_2": 0.0, "ngram_3": 0.0, "match_score": 0.0}
    corp = corpus_tokens.view(-1).tolist()
    n = min(len(gen_tokens), len(corp))
    if n == 0:
        return {"exact_1": 0.0, "ngram_2": 0.0, "ngram_3": 0.0, "match_score": 0.0}

    # Token-level exact match
    exact = sum(1 for g, c in zip(gen_tokens[:n], corp[:n]) if g == c) / n

    # N-gram recall: what fraction of generated n-grams appear anywhere in corpus?
    def _ngram_recall(gen, ref, n_gram):
        if len(gen) < n_gram or len(ref) < n_gram:
            return 0.0
        ref_set = set(zip(*[ref[i:] for i in range(n_gram)]))
        hits = sum(1 for ng in zip(*[gen[i:] for i in range(n_gram)]) if ng in ref_set)
        return hits / max(1, len(gen) - n_gram + 1)

    ng2 = _ngram_recall(gen_tokens, corp, 2)
    ng3 = _ngram_recall(gen_tokens, corp, 3)
    score = (3 * exact + 5 * ng2 + 8 * ng3) / 16.0
    return {"exact_1": exact, "ngram_2": ng2, "ngram_3": ng3, "match_score": score}


def _prosodic_alternation(toks_list: list, vocab: list) -> float:
    """Measure iambic-like stress alternation across a token sequence.

    Proxy: token character-length alternates between lighter (shorter) and
    heavier (longer) positions. Returns [0, 1] where 1 = perfect alternation.
    Substrate-universal: uses only character count, no word knowledge.
    """
    if len(toks_list) < 4:
        return 0.5
    lengths = [len(vocab[t]) if t < len(vocab) else 1 for t in toks_list]
    odd  = [lengths[i] for i in range(1, len(lengths), 2)]
    even = [lengths[i] for i in range(0, len(lengths), 2)][:len(odd)]
    if not odd:
        return 0.5
    mean_odd  = sum(odd)  / len(odd)
    mean_even = sum(even) / len(even)
    mean_all  = sum(lengths) / len(lengths)
    diff = abs(mean_odd - mean_even)
    # Normalise: how much alternation relative to average token weight?
    return min(1.0, diff / max(1.0, mean_all))


def _leaf_content_density(toks_list: list, fingerprint) -> float:
    """Fraction of tokens in non-void semantic leaves (coherence > 0).

    Corpus-native: the fingerprint defines what 'meaningful' means.
    Void leaf tokens (near-zero fwd fingerprints) are function words —
    they appear in all contexts, so their vectors cancel. Non-void leaf
    tokens have distinctive co-occurrence patterns — the corpus's own
    definition of a content word.
    Neutral fallback (0.5) when fingerprint not available.
    """
    if not toks_list or fingerprint is None or not fingerprint._built:
        return 0.5
    if not hasattr(fingerprint, '_leaf_coherences'):
        _fwd_n = fingerprint.fwd.float()
        _nrm = _fwd_n.norm(dim=1, keepdim=True).clamp(min=1e-8)
        _unit = _fwd_n / _nrm
        _cohs = []
        for _lid in range(fingerprint.n_leaves):
            _ids = (fingerprint.sub_clusters == _lid).nonzero(as_tuple=True)[0]
            _cohs.append(fingerprint._intra_coherence(_unit[_ids]) if _ids.numel() > 1 else 1.0)
        fingerprint._leaf_coherences = _cohs
    non_void = frozenset(lid for lid, c in enumerate(fingerprint._leaf_coherences) if c > 0.0)
    valid = [t for t in toks_list if 0 <= t < fingerprint.V]
    if not valid:
        return 0.5
    leaf_ids = fingerprint.sub_clusters[torch.tensor(valid, dtype=torch.long)].tolist()
    return sum(1 for lid in leaf_ids if lid in non_void) / len(leaf_ids)


def leaf_positional_match_score(tokens_list: list, fingerprint,
                                 fib_table, vocab: list, n_chars: int) -> float:
    """Corpus-native discourse quality: how well does the text's leaf-per-position
    distribution match what the FibPosTable learned from the corpus?

    Computes average cosine similarity between:
    - The FibPosTable's per-bin leaf distribution (corpus's positional grammar)
    - The actual leaf distribution in the generated text at each Fibonacci bin

    The table IS the corpus speaking about where semantic clusters belong.
    No word lists. No external targets. Returns [0, 1].
    """
    if (fingerprint is None or not fingerprint._built
            or fib_table is None or fib_table._n_updates < 10
            or not tokens_list):
        return 0.5
    n_leaves = fib_table.n_leaves   # always _N_SECTOR_LEAVES
    n_bins   = fib_table.n_bins
    actual   = torch.zeros(n_bins, n_leaves)
    clause_pos = 0
    V = len(vocab)
    for tid in tokens_list:
        if tid < 0 or tid >= V:
            continue
        tok = vocab[tid]
        if tok in ('.', '!', '?', '\n'):
            clause_pos = 0
            continue
        if tok in (',', ';', ':'):
            continue
        fb  = fib_table._fib_bin(clause_pos)
        lid = _fib_tok_sector(tid, vocab)
        if 0 <= fb < n_bins:
            actual[fb, lid] += 1.0
        clause_pos += 1
    corpus_dist = fib_table.probs  # [n_bins, n_leaves]
    sims = []
    sim_bins = []  # track which bin index each sim came from
    for b in range(n_bins):
        row_a = actual[b]
        row_c = corpus_dist[b]
        s_a   = row_a.sum()
        if s_a < 1e-8:
            continue
        row_a = row_a / s_a
        dot   = (row_a * row_c).sum()
        na, nc = row_a.norm(), row_c.norm()
        if na > 1e-8 and nc > 1e-8:
            sims.append(float(dot / (na * nc)))
            sim_bins.append(b)
    if not sims:
        return 0.5
    # Fibonacci-decayed weighted average: clause-initial bins (b=0) weighted
    # highest; late-clause bins weighted lowest.
    # w_b = (b+1) / phi^(pi * b), then normalised to sum to 1.
    raw_weights = [(b + 1) / (_PHI_FOR_SAMPLING ** (math.pi * b))
                   for b in sim_bins]
    w_total = sum(raw_weights)
    if w_total < 1e-12:
        return float(sum(sims) / len(sims))
    return float(sum(w * s for w, s in zip(raw_weights, sims)) / w_total)


def _compute_leaf_expected_clause_lens(fingerprint, active_base: torch.Tensor,
                                        vocab: list, n_chars: int) -> torch.Tensor:
    """Corpus-native clause-length expectation per semantic leaf.

    Splits the corpus into clauses (sequences between boundary chars .!?\\n),
    measures how many n-gram tokens each clause contains, and attributes that
    length to every leaf that appears in the clause.

    Returns tensor of shape [n_leaves]: the mean clause n-gram-token count
    for clauses containing each leaf. Void leaves (coherence ≤ 0) get a
    large sentinel (999) so the nudge never fires on them.

    Corpus-native: no external notion of clause length — the corpus defines it.
    """
    if not fingerprint._built or active_base.numel() == 0 or not vocab:
        return None
    n_leaves   = fingerprint.n_leaves
    V          = len(vocab)
    boundary   = set(tid for tid in range(min(n_chars, V))
                     if vocab[tid] in ('.', '!', '?', '\n'))
    toks       = active_base.tolist()
    leaf_lens  = [[] for _ in range(n_leaves)]
    clause_ngrams: list = []
    for tid in toks:
        if 0 <= tid < n_chars:
            if tid in boundary and clause_ngrams:
                leaves_in = set(int(fingerprint.sub_clusters[t].item())
                                for t in clause_ngrams if t < fingerprint.V)
                length    = len(clause_ngrams)
                for lid in leaves_in:
                    if 0 <= lid < n_leaves:
                        leaf_lens[lid].append(length)
                clause_ngrams = []
            # non-boundary chars: don't affect n-gram clause count
        elif tid < V:
            clause_ngrams.append(tid)
    # Flush last clause if no trailing boundary
    if clause_ngrams:
        leaves_in = set(int(fingerprint.sub_clusters[t].item())
                        for t in clause_ngrams if t < fingerprint.V)
        for lid in leaves_in:
            if 0 <= lid < n_leaves:
                leaf_lens[lid].append(len(clause_ngrams))

    result = torch.ones(n_leaves) * 5.0   # default: 5 n-gram tokens
    non_void = set()
    if hasattr(fingerprint, '_leaf_coherences'):
        non_void = {lid for lid, c in enumerate(fingerprint._leaf_coherences) if c > 0.0}
    # Corpus-wide average clause length — fallback for leaves with no data.
    all_lens = [l for ll in leaf_lens for l in ll]
    corpus_avg = float(sum(all_lens) / len(all_lens)) if all_lens else 5.0
    for lid in range(n_leaves):
        if lid not in non_void:
            result[lid] = corpus_avg  # use corpus average, not a void sentinel
        elif leaf_lens[lid]:
            result[lid] = float(sum(leaf_lens[lid]) / len(leaf_lens[lid]))
        else:
            result[lid] = corpus_avg  # no observations yet — use corpus average
    return result


def compute_bloom_embedding(model, accepted_seqs: list, quality_scores: list,
                            vocab: list | None = None,
                            fingerprint: 'SubstrateFingerprint' = None,
                            noun_mask=None, verb_mask=None, adj_mask=None):
    """Compress a cycle's accepted samples into a single Bloom embedding.

    Each sample is weighted by:
        quality × leaf_content_density × (0.5 + 0.5 × prosodic_alternation)

    Within each sample, token positions are Fibonacci-weighted so clause-initial
    tokens (where structure is set) contribute more than later positions.

    Substrate-universal: density is derived from corpus-native leaf structure.
    """
    phi = _PHI_FOR_SAMPLING
    device = next(model.parameters()).device
    d = model.embed.d_model
    total = torch.zeros(d, device=device)
    weight_sum = 0.0

    for seq, quality in zip(accepted_seqs, quality_scores):
        toks = (seq[0] if seq.dim() == 2 else seq).to(device)
        toks_list = toks.tolist()
        with torch.no_grad():
            embs = model.embed(toks)  # [L, d]

        # Prosodic alternation: iambic-like light/heavy token alternation.
        prosodic = _prosodic_alternation(toks_list, vocab or [])
        # Content density: fraction of tokens in non-void semantic leaves.
        density  = _leaf_content_density(toks_list, fingerprint)
        # Cluster diversity: sequences spanning multiple semantic types
        # (person + verb + object + abstract) get higher bloom weight.
        # Prevents bloom from over-encoding monocultural token distributions.
        diversity = (fingerprint.cluster_diversity(toks_list)
                     if fingerprint is not None and fingerprint._built else 0.5)
        # Combined sample weight: quality × density × prosodic × diversity shaping.
        # Floor and blends use golden-ratio (substrate-canonical) values:
        #   density floor = 1/φ² ≈ 0.382; blend coeff = 1/φ ≈ 0.618.
        _phi = _PHI_FOR_SAMPLING
        sample_w = (float(quality)
                    * max(1.0 / (_phi ** 2), density)
                    * (1.0 / _phi + (1.0 - 1.0 / _phi) * prosodic)
                    * (1.0 / _phi + (1.0 - 1.0 / _phi) * diversity))

        clause_pos = 0
        for emb in embs:
            fb = FibPosTable._fib_bin(clause_pos % 90)
            fib_w = phi ** (-fb)
            total      = total + fib_w * sample_w * emb
            weight_sum += fib_w * sample_w
            clause_pos += 1

    if weight_sum < 1e-8:
        return None, {}
    n = len(accepted_seqs)
    stats = {
        "mean_prosodic": sum(
            _prosodic_alternation((s[0] if s.dim() == 2 else s).tolist(), vocab or [])
            for s in accepted_seqs) / max(1, n),
        "mean_density": sum(
            _leaf_content_density((s[0] if s.dim() == 2 else s).tolist(), fingerprint)
            for s in accepted_seqs) / max(1, n),
    }
    return (total / weight_sum).detach(), stats  # [d], dict


def bloom_tier_sequences(bloom_seqs_meta: list, bloom_vecs: list,
                         model, vocab, fingerprint,
                         max_raw: int = 21) -> tuple:
    """Diversity-aware Fibonacci tier for the bloom corpus.

    Each entry in bloom_seqs_meta is a 5-tuple:
        (seq_cpu, composite, prosodic, density, creativity)

    Divides max_raw slots equally across three axes so the corpus covers all
    quality dimensions — not just the highest single-score sequences.

    Slot allocation (for max_raw=21):
      • 7 by highest prosodic_alternation  (rhythmic structure)
      • 7 by highest leaf_content_density  (corpus-native semantic richness)
      • 7 by highest creativity / quality  (novel token patterns)
    Remaining slots (if any, due to overlap) filled by composite score.
    Overflow is compressed into an additional bloom vector.
    """
    if not bloom_seqs_meta:
        return [], list(bloom_vecs)
    if len(bloom_seqs_meta) <= max_raw:
        return [e[0] for e in bloom_seqs_meta], list(bloom_vecs)

    n          = len(bloom_seqs_meta)
    n_per_axis = max(1, max_raw // 3)

    def _scores(e):
        # e = (seq, composite, prosodic, density, creativity)
        # legacy 2-tuple fallback: (seq, composite)
        if len(e) >= 5:
            return e[1], e[2], e[3], e[4]
        return e[1], 0.0, 0.0, 0.0

    kept       = set()
    pros_kept  = set()
    dens_kept  = set()
    crea_kept  = set()
    fill_kept  = set()

    # Axis 1: prosodic (rhythmic diversity — not buffalo)
    for i in sorted(range(n), key=lambda i: _scores(bloom_seqs_meta[i])[1], reverse=True):
        if len(pros_kept) >= n_per_axis:
            break
        kept.add(i); pros_kept.add(i)

    # Axis 2: content density (semantic richness — nouns/verbs/adj)
    for i in sorted(range(n), key=lambda i: _scores(bloom_seqs_meta[i])[2], reverse=True):
        if len(dens_kept) >= n_per_axis:
            break
        if i not in kept:
            kept.add(i); dens_kept.add(i)

    # Axis 3: creativity / quality (novel token variety)
    for i in sorted(range(n), key=lambda i: _scores(bloom_seqs_meta[i])[3], reverse=True):
        if len(crea_kept) >= n_per_axis:
            break
        if i not in kept:
            kept.add(i); crea_kept.add(i)

    # Fill remaining slots by composite (best-of-rest)
    remaining = max_raw - len(kept)
    if remaining > 0:
        for i in sorted(range(n), key=lambda i: _scores(bloom_seqs_meta[i])[0], reverse=True):
            if remaining <= 0:
                break
            if i not in kept:
                kept.add(i); fill_kept.add(i); remaining -= 1

    overflow_idx  = [i for i in range(n) if i not in kept]
    keep_entries  = [bloom_seqs_meta[i] for i in kept]
    over_entries  = [bloom_seqs_meta[i] for i in overflow_idx]

    overflow_seqs = [e[0].unsqueeze(0) if e[0].dim() == 1 else e[0] for e in over_entries]
    overflow_q    = [min(float(_scores(e)[0]), 1.0) for e in over_entries]
    if overflow_seqs:
        ov_vec, _ = compute_bloom_embedding(
            model, overflow_seqs, overflow_q, vocab=vocab, fingerprint=fingerprint)
        if ov_vec is not None:
            bloom_vecs = list(bloom_vecs) + [ov_vec]
            # Keep crystal_ages aligned with bloom_vecs length.
            if hasattr(model, 'bloom_crystal_ages'):
                model.bloom_crystal_ages = list(model.bloom_crystal_ages) + [0]

    print(f"  [BLOOM TIER] diversity-kept {len(keep_entries)} raw seqs "
          f"(pros={len(pros_kept)} dens={len(dens_kept)} "
          f"crea={len(crea_kept)} fill={len(fill_kept)}), "
          f"compressed {len(over_entries)} → +1 vec ({len(bloom_vecs)} total)")
    return [e[0] for e in keep_entries], bloom_vecs


def model_bloom_grow(model, bloom_embedding: torch.Tensor,
                     vocab: list, bloom_name: str,
                     bigram_prior: torch.Tensor | None = None):
    """Grow the model by adding one Bloom token — literal parameter expansion.

    Extends the substrate embedding buffer with the compressed cycle knowledge
    vector, and extends the lm_head weight matrix to cover the new token.
    After this call the model has one more token in its vocabulary.

    Returns (bloom_idx, new_bigram_prior).
    new_bigram_prior is None if no bigram_prior was provided.
    """
    device = next(model.parameters()).device
    old_V = model.embed.vocab_size
    bloom_idx = old_V
    d = model.embed.d_model
    bv = bloom_embedding.to(device)  # [d]

    # Extend substrate_embed buffer: [old_V, d] → [old_V+1, d]
    new_se = torch.cat([model.embed.substrate_embed,
                        bv.unsqueeze(0)], dim=0)
    model.embed.register_buffer("substrate_embed", new_se)
    model.embed.vocab_size = old_V + 1

    # Extend lm_head: [old_V, d] → [old_V+1, d]
    old_hw = model.head.weight.data  # [old_V, d]
    new_hw = torch.cat([old_hw, bv.unsqueeze(0)], dim=0)
    new_head = nn.Linear(d, old_V + 1, bias=False).to(device)
    new_head.weight.data = new_hw
    model.head = new_head

    # Extend vocab list
    vocab.append(bloom_name)

    # Extend bigram prior if present: V×V → (V+1)×(V+1)
    new_bg = None
    if bigram_prior is not None:
        # New row (bloom → *): uniform over existing tokens
        new_row = torch.full((1, old_V), 1.0 / old_V,
                              device=bigram_prior.device,
                              dtype=bigram_prior.dtype)
        # New column (* → bloom): very small, rare to transition TO bloom
        new_col = torch.full((old_V + 1, 1), 1e-4,
                              device=bigram_prior.device,
                              dtype=bigram_prior.dtype)
        new_bg = torch.cat([bigram_prior, new_row], dim=0)  # [V+1, V]
        new_bg = torch.cat([new_bg, new_col], dim=1)         # [V+1, V+1]

    return bloom_idx, new_bg


def train_with_self_distillation(name, train_seed, corpus_anchor, val_split,
                                    vocab_size, args, fib_positions,
                                    harmony_kind="multiscale",
                                    itos_map=None,
                                    corpus_text=None,
                                    vocab_for_bigram=None,
                                    n_cycles: int = 4,
                                    distill_prob: float = 0.3,
                                    samples_per_cycle: int = 8,
                                    keep_top_k: int = 4,
                                    growth_n_new: int = 128,
                                    warmup_cycles: int = 2,
                                    production_score_gate: float = 0.52,
                                    bloom_checkpoint_path: str = "bloom_checkpoint.pt"):
    """Self-distillation: model's high-creativity refined outputs become
    training targets for the next cycle.

    warmup_cycles: first N cycles are CE-only (no harmony, no corpus growth).
    The model must achieve production_score > production_score_gate (measured
    on its own recursive output) before refined outputs are allowed into
    active_base. This uses production quality — not val loss — as the gate,
    because we want the model to PRODUCE Shakespeare, not copy it.

    Each cycle (post-warmup):
      1. Train for steps_per_cycle on (tiny_seed + distill_buffer)
         -- with prob `distill_prob` each batch comes from buffer.
      2. Generate a draft from current model.
      3. Refine via staged loop targeting creativity (not harmony!).
      4. Score the refined output's creativity.
      5. If creativity > best_seen: add to distill_buffer.

    Substrate stays as the scaffolding; creativity is the compass.
    The model's parameters move toward fixed-points that are both
    substrate-aligned AND linguistically creative (Shakespeare-like).
    """
    import random as _rng_mod
    rng = _rng_mod.Random(args.seed + 7)
    torch.manual_seed(args.seed)
    gen = torch.Generator(); gen.manual_seed(args.seed + 1)

    model = FibRecLMSubsim(
        vocab_size=vocab_size, d_model=args.d_model, n_blocks=args.n_blocks,
        seq_len=args.seq_len, K=args.K_init, mode="cross", K_sig=args.K_sig,
        substrate_embed=True,
    )
    optimizer = FibonacciAdamW(model.parameters(), lr=args.lr)
    # Slower K-shrink: double T so K decreases at half speed (each tier
    # held ~2 cycles). v69 showed K-shrink-induced drop at cycle 5.
    sched = lambda s, T: K_schedule_tier_walk(s, 2 * T, K_init=args.K_init,
                                                 K_min=args.K_min)
    n_params = sum(p.numel() for p in model.parameters())

    sig_char = corpus_char_signature(corpus_anchor, vocab_size)
    sig_ms = corpus_multiscale_signature(corpus_anchor, vocab_size,
                                            seq_len=args.seq_len)

    full_corpus = corpus_text or ""
    def creativity_fn(seq_tokens):
        text = ''.join(itos_map.get(int(t), '?')
                        for t in seq_tokens[0].tolist())
        return compute_creativity_score(text, full_corpus)["creativity_score"]

    def harmony_fn(seq_tokens):
        with torch.no_grad():
            T = seq_tokens.shape[1]
            ctx = seq_tokens if T <= model.seq_len else seq_tokens[:, -model.seq_len:]
            logits = model(ctx)
            K_h = K_to_K_harmony(cur_K or args.K_init,
                                  K_init=args.K_init, K_min=args.K_min)
            return compute_harmony_grounded(logits, vocab_size, harmony_kind,
                                              sig_char, sig_ms,
                                              K_harmony=K_h).item()

    def quality_fn(seq_tokens):
        with torch.no_grad():
            T = seq_tokens.shape[1]
            ctx = seq_tokens if T <= model.seq_len else seq_tokens[:, -model.seq_len:]
            logits = model(ctx)
            _tgts = ctx[:, 1:].reshape(-1)
            _ce_per_tok = F.cross_entropy(
                logits[:, :-1].reshape(-1, vocab_size), _tgts, reduction='none')
            # OmniWeight per target token: φ^(-|t - nearest_fib(t)| / nearest_fib(t))
            _fib_attractors = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
            _tgts_np = _tgts.cpu().tolist()
            _ow = torch.tensor(
                [_PHI_FOR_SAMPLING ** -(abs(t - min(_fib_attractors, key=lambda f: abs(t - f)))
                                        / max(min(_fib_attractors, key=lambda f: abs(t - f)), 1))
                 for t in _tgts_np],
                dtype=torch.float32, device=_tgts.device)
            ce = (_ce_per_tok * _ow).sum() / (_ow.sum() + 1e-8)
            return ce.item()

    print(f"\n[self_distill {name}]  harmony={harmony_kind}  "
          f"n_cycles={n_cycles}  distill_prob={distill_prob}  "
          f"params={n_params:,}", flush=True)

    n_new = max(args.seq_len - 16, 32)

    # Sample window size (used for bigram/ref scoring later).
    sample_window = max(64, n_new + 16)

    orig_seed_chars = train_seed.numel()   # overridden after warm-start below

    # Refined substrate bigram: shape-aware (chunk geometry) + POS-aware
    # (universal POS tiers). No corpus statistics, no model-derived noise.
    # Two layers of substrate structural prior combined multiplicatively.
    vocab = vocab_for_bigram   # alias for internal calls
    if vocab_for_bigram is not None:
        bigram_shape = build_substrate_bigram_shape(vocab_size,
                                                       vocab_for_bigram)
        bigram_pos = build_substrate_pos_bigram(vocab_size,
                                                   vocab_for_bigram)
        # Structural prior: shape × POS (both must agree).
        bigram_struct = bigram_shape * bigram_pos
        bigram_struct = bigram_struct / (bigram_struct.sum(-1, keepdim=True) + 1e-8)
        # Blend with corpus-derived bigram (70% structural / 30% corpus).
        # Keeps universality (structure still dominates) but adds real Shakespeare
        # transition signal — previously build_bigram_prior was never called.
        if corpus_anchor is not None and corpus_anchor.numel() >= 2:
            bigram_corpus = build_bigram_prior(corpus_anchor, vocab_size)
            bigram_prior = (1.0 / _PHI_FOR_SAMPLING) * bigram_struct + (1.0 - 1.0 / _PHI_FOR_SAMPLING) * bigram_corpus
            bigram_prior = bigram_prior / (bigram_prior.sum(-1, keepdim=True) + 1e-8)
        else:
            bigram_prior = bigram_struct
    else:
        bigram_prior = build_substrate_bigram(vocab_size)
    print(f"  refined substrate bigram (shape * POS + 30% corpus): {bigram_prior.shape}")

    # Theme momentum disabled (v57 showed it drags ~-0.01).
    token_signatures = None

    # Symbolic primitives (v60+): equivalence classes + reference chain.
    if vocab_for_bigram is not None:
        n_chars_local = sum(1 for t in vocab_for_bigram if len(t) == 1)
        class_id_tensor, n_classes = build_symbol_classes(
            vocab_for_bigram, n_chars=n_chars_local)
        pronoun_mask = build_pronoun_mask(vocab_for_bigram)
        vowel_start_mask = build_vowel_start_mask(vocab_for_bigram)
        end_vowels = build_end_vowel_idx_tensor(vocab_for_bigram)
        allowed_after_word_mask = build_allowed_after_word_mask(
            vocab_for_bigram, n_chars=n_chars_local)
        uppercase_mask = build_uppercase_mask(vocab_for_bigram)
        any_punct_mask = build_any_punct_mask(vocab_for_bigram)
        punct_mask = build_punct_mask(vocab_for_bigram)
        newline_mask = build_newline_mask(vocab_for_bigram)
        unpronounceable_mask = build_unpronounceable_mask(vocab_for_bigram)
        first_person_mask, second_person_mask, third_person_mask = \
            build_person_masks(vocab_for_bigram)
        verb_mask, noun_mask, adj_mask, func_mask = \
            build_pos_masks(vocab_for_bigram)
        print(f"  symbol classes: {n_classes} | "
              f"pronoun cand: {int(pronoun_mask.sum().item())} | "
              f"1st-person: {int(first_person_mask.sum().item())} | "
              f"2nd-person: {int(second_person_mask.sum().item())} | "
              f"3rd-person: {int(third_person_mask.sum().item())} | "
              f"verbs: {int(verb_mask.sum().item())} | "
              f"nouns: {int(noun_mask.sum().item())} | "
              f"adjs: {int(adj_mask.sum().item())} | "
              f"func: {int(func_mask.sum().item())} | "
              f"punct: {int(punct_mask.sum().item())} | "
              f"newline: {int(newline_mask.sum().item())} | "
              f"unpronounceable: "
              f"{int(unpronounceable_mask.sum().item())} | "
              f"end-vowel: {int((end_vowels >= 0).sum().item())}")
    else:
        class_id_tensor = None
        n_classes = 0
        pronoun_mask = None
        vowel_start_mask = None
        end_vowels = None
        punct_mask = None
        newline_mask = None
        unpronounceable_mask = None
        allowed_after_word_mask = None
        uppercase_mask = None
        any_punct_mask = None

    # Substrate OMNIbalance target profile — the MAP the generator steers toward.
    # Computed once from the training seed so generation is closed-loop:
    # the model knows which direction to move, not just where it tends to go.
    _target_profile = None
    if vocab_for_bigram is not None:
        _n_chars_tp = sum(1 for t in vocab_for_bigram if len(t) == 1)
        _target_profile = substrate_profile_from_tokens(
            train_seed.unsqueeze(0) if train_seed.dim() == 1 else train_seed,
            vocab_for_bigram, _n_chars_tp)
        print(f"  target profile: "
              f"addressee={_target_profile['addressee_frac']:.3f}  "
              f"novelty={_target_profile['novelty_frac']:.3f}  "
              f"diversity={_target_profile['diversity_frac']:.3f}  "
              f"clause_len={_target_profile['clause_len_mean']:.1f}")

    # Leaf-pos table — lazy: initialized after first fingerprint build
    # (needs leaf clusters to define the class slots).
    _fib_table: FibPosTable | None = None

    # Bloom warm-start: if a checkpoint exists from a prior run, load bloom
    # vectors, FibPosTable state, and expanded corpus so this run inherits
    # all accumulated knowledge — structural, positional, and textual.
    import os as _os
    _model_device = next(model.parameters()).device
    _prior_active_base: torch.Tensor | None = None
    _bloom_all_seqs_meta: list = []   # (seq_cpu, composite_score) — tiered before save
    _ckpt_fib_counts    = None
    _ckpt_fib_tok_cnts  = None
    _ckpt_fib_n_leaves  = None
    _ckpt_fib_n_updates = 0
    _ckpt_leaf_coherences       = None
    _ckpt_leaf_expected_cls_lens = None
    if _os.path.exists(bloom_checkpoint_path):
        _ckpt = torch.load(bloom_checkpoint_path, map_location='cpu')
        _prior_blooms = _ckpt.get('bloom_vecs', [])
        if _prior_blooms:
            model.bloom_vecs = [bv.to(_model_device) for bv in _prior_blooms]
            _prior_crystal_ages = _ckpt.get('bloom_crystal_ages', [])
            if _prior_crystal_ages:
                model.bloom_crystal_ages = list(_prior_crystal_ages)
            else:
                model.bloom_crystal_ages = [0] * len(_prior_blooms)
        # Restore bloom tier peak and prosodic bar so phi-decay and gate are consistent.
        model._max_bloom_tier_seen = _ckpt.get('max_bloom_tier_seen', 3)
        model._bloom_prosodic_bar  = _ckpt.get('bloom_prosodic_bar', 0.0)
        # Stash checkpoint fib-table data; apply after lazy init in first cycle
        _ckpt_fib_counts    = _ckpt.get('fib_table_counts', None)
        _ckpt_leaf_coherences        = _ckpt.get('leaf_coherences', None)
        _ckpt_leaf_expected_cls_lens = _ckpt.get('leaf_expected_clause_lens', None)
        _ckpt_fib_tok_cnts  = _ckpt.get('fib_table_token_counts', None)
        _ckpt_fib_n_leaves  = _ckpt.get('fib_table_n_leaves', None)
        _ckpt_fib_n_updates = _ckpt.get('fib_table_n_updates', 0)
        # Corpus inheritance: bloom's expanded form (quality-accepted sequences)
        # becomes this run's starting corpus, prepended to the original seed.
        # Bloom IS the corpus — no separate mechanism needed.
        # Supports both new format (bloom_sequences list) and legacy (active_base tensor).
        _prior_seqs = _ckpt.get('bloom_sequences', [])
        if not _prior_seqs:
            _legacy_ab = _ckpt.get('active_base')
            if _legacy_ab is not None and _legacy_ab.numel() > train_seed.numel():
                _generated_flat = _legacy_ab[train_seed.numel():]
                _chunk = 128
                _prior_seqs = [_generated_flat[i:i + _chunk]
                               for i in range(0, _generated_flat.numel(), _chunk)
                               if i + _chunk <= _generated_flat.numel()]
        if _prior_seqs:
            # Filter inherited sequences by corpus authenticity — only keep
            # sequences where ≥25% of 3-grams appear in corpus_anchor.
            # This purges legacy "means/bloody/set" bridge-text from bloom.
            if corpus_anchor is not None and corpus_anchor.numel() >= 3:
                _ca_l = corpus_anchor.tolist()
                _inherit_3g = set(zip(_ca_l, _ca_l[1:], _ca_l[2:]))
                _FIB_TIERS_INIT = [3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
                # Dynamic floor: 2 Fibonacci levels below the highest tier in bloom.
                # Once T21 is earned, T3/T5 are low-signal noise — floor rises with peak.
                _max_inherited_len = max((s.numel() for s in _prior_seqs), default=0)
                _peak_tier = next((f for f in reversed(_FIB_TIERS_INIT)
                                   if _max_inherited_len >= f), None)
                _peak_idx = _FIB_TIERS_INIT.index(_peak_tier) if _peak_tier else 0
                _bloom_floor = _FIB_TIERS_INIT[max(0, _peak_idx - 2)]
                _filtered_seqs = []
                _tier_counts = {f: 0 for f in _FIB_TIERS_INIT}
                _n_purged = 0
                for _ps in _prior_seqs:
                    _pl = _ps.tolist()
                    # Find longest authentic run in this sequence.
                    _b_run, _b_len = [], 0
                    if len(_pl) >= 3:
                        _rs = 0
                        for _ri in range(len(_pl) - 2):
                            if (_pl[_ri], _pl[_ri+1], _pl[_ri+2]) not in _inherit_3g:
                                _rl = (_ri + 2) - _rs
                                if _rl > _b_len:
                                    _b_len = _rl; _b_run = _pl[_rs:_ri+2]
                                _rs = _ri + 1
                        _rl = len(_pl) - _rs
                        if _rl > _b_len:
                            _b_len = _rl; _b_run = _pl[_rs:]
                    # Keep only if run meets the dynamic floor.
                    _tier = next((f for f in reversed(_FIB_TIERS_INIT)
                                  if _b_len >= f), None)
                    if _tier is not None and _tier >= _bloom_floor:
                        _filtered_seqs.append(
                            torch.tensor(_b_run, dtype=_ps.dtype))
                        _tier_counts[_tier] += 1
                    else:
                        _n_purged += 1
                _tier_str = '  '.join(f"T{f}×{_tier_counts[f]}"
                                      for f in _FIB_TIERS_INIT
                                      if _tier_counts[f] > 0)
                print(f"  [BLOOM PURGE] kept {len(_filtered_seqs)}/{len(_prior_seqs)} "
                      f"sequences by Fibonacci tier: {_tier_str or 'none'} "
                      f"(purged {_n_purged})")
                _prior_seqs = _filtered_seqs
            _prior_active_base = torch.cat(
                [train_seed.cpu()] + [s.cpu() for s in _prior_seqs]
            ) if _prior_seqs else None
            # Seed the meta-list with real composite scores for prior sequences.
            # Recompute prosodic+density from the actual tokens so tiering sorts
            # on true quality — not an arbitrary constant — ensuring the best corn
            # survives regardless of which run produced it.
            for _ps in _prior_seqs:
                _ps_list = _ps.tolist()
                _ps_pros = _prosodic_alternation(_ps_list, vocab_for_bigram or [])
                _ps_dens = 0.5  # fingerprint not yet built at warm-start; neutral density
                _phi = _PHI_FOR_SAMPLING
                _ps_comp = (1.0 / _phi) * max(1.0 / (_phi ** 2), _ps_dens) * (1.0 / _phi + (1.0 - 1.0 / _phi) * _ps_pros)
                _bloom_all_seqs_meta.append((_ps, _ps_comp, _ps_pros, _ps_dens, 1.0 / _phi))
        _prev_best_c = _ckpt.get('best_creativity', 0.0)
        _prev_best_p = _ckpt.get('best_production', 0.0)
        _inherited_toks = (_prior_active_base.numel()
                           if _prior_active_base is not None else 0)
        print(f"  [BLOOM WARM-START] inherited {len(getattr(model, 'bloom_vecs', []))} "
              f"bloom vecs + {len(_prior_seqs)} sequences ({_inherited_toks} tokens) "
              f"+ FibPosTable from {bloom_checkpoint_path}")
        print(f"    prev best: creativity={_prev_best_c:.4f}  "
              f"production={_prev_best_p:.4f}")

    # Active training base: starts as tiny_seed, GROWS by appending each
    # cycle's best refined output -- only if (a) creativity > corpus
    # baseline AND (b) anchor weight constraint still satisfied AND
    # (c) the model's recursive production score > production_score_gate.
    # Start from prior run's grown corpus if available; otherwise original seed.
    # The model trains on richer data from step 0 — genuine divergence from the
    # original weight basin rather than just bloom-biased generation on top of it.
    active_base = (_prior_active_base.to(_model_device)
                   if _prior_active_base is not None
                   else train_seed.clone())
    # Re-anchor to the full warm-started corpus: inherited bloom sequences are
    # already quality-gated, so they count as trusted seed, not noise.
    orig_seed_chars = active_base.numel()
    best_creativity = 0.0
    # _bloom_all_seqs_meta is initialised before warm-start loader above.
    best_production = 0.0    # best substrate_production_score seen so far
    best_production_text = ""
    best_refined_seq = None
    cycle_summary = []
    n_rejected_quality = 0    # failed model-relative creativity gate
    n_rejected_anchor = 0
    # Adaptive gate: corpus-derived rather than fixed. Tracks the model's own
    # score distribution; gate = mean - 0.5*std of recent scores, floor 0.40.
    _prod_score_history: list = []

    steps_per_cycle = args.steps // n_cycles
    t0 = time.time()
    best_val = float("inf"); best_step = -1   # kept for informational logging
    cur_K = None
    eval_every = max(steps_per_cycle // 4, 100)
    global_step = 0
    # Prompt = corpus_anchor[:16] so the corpus_ref IS the actual continuation.
    # This aligns the measurement: generate from the corpus start, compare to
    # what actually follows in the corpus — exact sprouting test.
    _prompt_end = 16
    _ref_len = 220
    if corpus_anchor is not None and corpus_anchor.numel() > _prompt_end + _ref_len:
        prompt = corpus_anchor[:_prompt_end].unsqueeze(0)
        _corpus_ref_tokens = corpus_anchor[_prompt_end: _prompt_end + _ref_len]
    else:
        prompt = train_seed[:_prompt_end].unsqueeze(0)
        _corpus_ref_tokens = None
    best_match_score = 0.0
    last_cycle_val = float("inf")   # informational only — not used as gate

    # Precompute full-corpus n-gram sets for fast per-sample scoring.
    # Any generated 3-gram that appears anywhere in the 20k-token corpus_anchor
    # is a genuine Shakespeare 3-gram — this is the primary measurement.
    _ca_2grams: set | None = None
    _ca_3grams: set | None = None
    if corpus_anchor is not None and corpus_anchor.numel() >= 3:
        _cal = corpus_anchor.tolist()
        _ca_2grams = set(zip(_cal, _cal[1:]))
        _ca_3grams = set(zip(_cal, _cal[1:], _cal[2:]))
        global _corpus_3grams_global
        _corpus_3grams_global = _ca_3grams
        # Scale samp_match thresholds to corpus size (calibrated at 7153 3-grams).
        # Tiny corpora have fewer matchable 3-grams; gate shrinks proportionally.
        _ca_samp_match_floor = max(0.015, 0.15 * (len(_ca_3grams) / 7153) ** 0.5)
        print(f"  [CA-NGRAMS] built {len(_ca_2grams)} 2-grams, "
              f"{len(_ca_3grams)} 3-grams from {len(_cal)}-token corpus"
              f"  (samp_match_floor={_ca_samp_match_floor:.4f})", flush=True)

    # chunk_size for produce_recursive: F(10)=55, Fibonacci-aligned,
    # smaller than seq_len=89 so context always overlaps across chunks.
    _chunk_size = 55
    _n_prod_chunks = 4   # 4 chunks = ~220 new tokens = coherent passage

    # corpus used for fwd/bwd/rel topology injection (20k tokens → rich co-occurrence).
    # Clustering (sub_clusters, T[k,l]) still uses active_base so leaf assignments
    # stay consistent with what the model was trained on. After each build(), we
    # overwrite fwd/bwd/rel from _fp_corpus so the tape and junction signals have
    # real topological structure without disturbing leaf-based signals.
    _fp_corpus = corpus_anchor if (corpus_anchor is not None
                                    and corpus_anchor.numel() > active_base.numel()
                                    ) else active_base
    _fingerprint: SubstrateFingerprint | None = None
    _dyn_coh_stop = SubstrateFingerprint._COHERENCE_STOP  # starts at class default (0.90)
    if vocab_for_bigram is not None:
        _fingerprint = SubstrateFingerprint(vocab_size, model.embed.d_model)
        # Build from _fp_corpus (corpus_anchor, 20k tokens) — gives real k-means
        # clusters AND real fwd/bwd topology. Model trains on tiny seed for CE;
        # the substrate map comes from the full corpus structure.
        _fingerprint.build(_fp_corpus, model)
        # Warm-start leaf reorganization: if bloom sequences exist (prior run),
        # immediately overwrite fwd/bwd/rel with their co-occurrence topology.
        # Bloom = verified Shakespeare, so its topology is more trustworthy
        # than the mixed corpus_anchor topology.
        if _bloom_all_seqs_meta:
            _ws_bloom_toks = [s.cpu() for s, *_ in _bloom_all_seqs_meta
                              if s.numel() >= 3]
            if _ws_bloom_toks:
                _ws_flat = torch.cat(_ws_bloom_toks)
                _fingerprint.update_fwd_bwd(
                    _ws_flat, model,
                    device=next(model.parameters()).device)
                print(f"  [LEAF REORG warm-start] fwd/bwd/rel from "
                      f"{_ws_flat.numel()} authentic bloom tokens", flush=True)
        # Eagerly populate _leaf_coherences so clause lens and quality gates
        # have real non-void leaf information from the very first call —
        # without this, lazy init in _leaf_content_density fires too late
        # and _compute_leaf_expected_clause_lens saves all-999 to checkpoint.
        _fp_fwd_n = _fingerprint.fwd.float()
        _fp_nrm   = _fp_fwd_n.norm(dim=1, keepdim=True).clamp(min=1e-8)
        _fp_unit  = _fp_fwd_n / _fp_nrm
        _fingerprint._leaf_coherences = [
            _fingerprint._intra_coherence(
                _fp_unit[(_fingerprint.sub_clusters == _lid).nonzero(as_tuple=True)[0]])
            if (_fingerprint.sub_clusters == _lid).sum() > 1 else 1.0
            for _lid in range(_fingerprint.n_leaves)]
        _n_cls  = _fingerprint.n_clusters
        _cls_sizes = [(int((_fingerprint.clusters == k).sum().item()))
                      for k in range(_n_cls)]
        _leaf_sizes = [(int((_fingerprint.sub_clusters == lk).sum().item()))
                       for lk in range(_fingerprint.n_leaves)]
        model.structure_bloom_vecs = [
            lc.to(_model_device) for lc in _fingerprint.leaf_centroids]
        print(f"  semantic fingerprint: {vocab_size} tokens → "
              f"{_n_cls} clusters {_cls_sizes} → "
              f"{_fingerprint.n_leaves} fractal leaves {_leaf_sizes} "
              f"→ {len(model.structure_bloom_vecs)} structural bloom vecs")
        if vocab_for_bigram is not None:
            _n_chars_cls = sum(1 for t in vocab_for_bigram if len(t) == 1)
            _fingerprint.leaf_expected_clause_lens = _compute_leaf_expected_clause_lens(
                _fingerprint, active_base, vocab_for_bigram, _n_chars_cls)
            if _fingerprint.leaf_expected_clause_lens is not None:
                _cls_lens = [f"{float(_fingerprint.leaf_expected_clause_lens[l]):.1f}"
                              for l in range(_fingerprint.n_leaves)]
                print(f"  [CLAUSE LENS] {_cls_lens}")
        # Restore leaf_coherences and clause lens from checkpoint (warm start).
        # These are computed lazily otherwise — restoring them means the boundary
        # nudge is live from the very first generation, not just from cycle 2.
        if (_ckpt_leaf_coherences is not None
                and len(_ckpt_leaf_coherences) == _fingerprint.n_leaves):
            _fingerprint._leaf_coherences = _ckpt_leaf_coherences
        if (_ckpt_leaf_expected_cls_lens is not None
                and _ckpt_leaf_expected_cls_lens.shape[0] == _fingerprint.n_leaves
                and _fingerprint.leaf_expected_clause_lens is not None):
            _dev = _fingerprint.leaf_expected_clause_lens.device
            _fingerprint.leaf_expected_clause_lens = (
                (_fingerprint.leaf_expected_clause_lens
                 + _ckpt_leaf_expected_cls_lens.to(_dev)) * 0.5)
            _cls_lens = [f"{float(_fingerprint.leaf_expected_clause_lens[l]):.1f}"
                          for l in range(_fingerprint.n_leaves)]
            print(f"  [CLAUSE LENS] blended checkpoint+corpus: {_cls_lens}")
        if vocab_for_bigram is not None:
            _fib_table = FibPosTable()   # always 7 Fibonacci-length sectors
            _n_chars_ft = sum(1 for t in vocab_for_bigram if len(t) == 1)
            _seed_for_table = (train_seed.unsqueeze(0)
                               if train_seed.dim() == 1 else train_seed)
            _fib_table.update_from_tokens(
                _seed_for_table, _fingerprint, vocab_for_bigram, _n_chars_ft, weight=1.0)
            # Full corpus structural pass: FibTable analyzes all of corpus_anchor
            # to learn Shakespeare's exact positional grammar — no dictionary,
            # pure Fibonacci sector co-occurrence from 20k real tokens.
            # Model weights train only on the tiny seed; the FibTable is the map
            # derived from the full corpus that guides generation toward it.
            if corpus_anchor is not None and corpus_anchor.numel() > train_seed.numel():
                _corp_for_table = (corpus_anchor.unsqueeze(0)
                                   if corpus_anchor.dim() == 1 else corpus_anchor)
                _fib_table.update_from_tokens(
                    _corp_for_table, _fingerprint, vocab_for_bigram, _n_chars_ft, weight=1.0)
                print(f"  [FIBTABLE] full corpus pass: {corpus_anchor.numel()} tokens "
                      f"→ {_fib_table._n_updates} total observations (no dictionary)")
            _bloom_seed_seqs = (_prior_active_base_seqs
                                if '_prior_active_base_seqs' in dir() else [])
            if not _bloom_seed_seqs and '_prior_seqs' in dir():
                _bloom_seed_seqs = _prior_seqs if _prior_seqs else []
            for _bseq in _bloom_seed_seqs:
                _fib_table.update_from_tokens(
                    _bseq.unsqueeze(0) if _bseq.dim() == 1 else _bseq,
                    _fingerprint, vocab_for_bigram, _n_chars_ft, weight=1.0)
            print(f"  [LEAF-POS TABLE] seeded: {_fib_table._n_updates} observations "
                  f"({len(_bloom_seed_seqs)} bloom seqs), "
                  f"{FibPosTable._N_SECTOR_LEAVES} chunk-sectors")
            if (_ckpt_fib_counts is not None
                    and _ckpt_fib_n_leaves == FibPosTable._N_SECTOR_LEAVES):
                _fib_table.counts = (
                    _fib_table.counts
                    + _ckpt_fib_counts.to(_fib_table.counts.device)) * 0.5
                if (_ckpt_fib_tok_cnts is not None
                        and _ckpt_fib_tok_cnts.shape[0] == len(vocab_for_bigram)):
                    _fib_table.token_counts = _ckpt_fib_tok_cnts
                    _fib_table._vocab_size  = len(vocab_for_bigram)
                _fib_table._n_updates += _ckpt_fib_n_updates
                _fib_table._invalidate()
                print(f"  [LEAF-POS TABLE] blended prior checkpoint "
                      f"({_ckpt_fib_n_updates} prior updates)")
            print(_fib_table.summary())

    # Vocab curriculum disabled for v59 -- v58 showed it hurts mid-cycles.
    # Novel-3gram gate: a run is admitted only if it contributes ≥1 3-gram not
    # already absorbed from prior admitted runs.  Value = novel information content.
    # "I'll make my heaven to dream upon " has 16 unique 3-gram transitions →
    # remains admissible much longer than "the" (≤1 new 3-gram after first copy).
    # Structural weight emerges naturally: richer runs saturate the gate slowly;
    # common fragments saturate immediately.  No frequency counting needed.
    _active_run_3gs: set = set()
    for cycle in range(n_cycles):
        in_warmup = (cycle < warmup_cycles)
        active_vocab_size = None
        # Harmony ramps in linearly over the 2 cycles after warmup
        # (0→full), so it doesn't fight CE while the model is still cold.
        if in_warmup:
            cycle_lambda = 0.0
        else:
            post_wu = cycle - warmup_cycles
            cycle_lambda = args.lambda_harmony * (1.0 - 1.0 / (_PHI_FOR_SAMPLING ** post_wu))
        # Saturation-adaptive sample count: F(7)=13 → F(6)=8 → F(5)=5 as corpus
        # 3-grams fill. Reduces wasted generation once the low-hanging 3-grams
        # are absorbed; saves compute at high saturation.
        _sat_level = len(_active_run_3gs) / max(len(_ca_3grams), 1) if _ca_3grams else 0.0
        _eff_spc = [13, 8, 5][min(int(_sat_level * 3), 2)]
        _eff_keep = max(2, int(_eff_spc / _PHI_FOR_SAMPLING))
        print(f"\n  --- Cycle {cycle+1}/{n_cycles}{'  [WARMUP-CE-ONLY]' if in_warmup else ''}  "
              f"lambda_h={cycle_lambda:.4f}  "
              f"active_base_size={active_base.numel()} tokens  "
              f"best_creativity={best_creativity:.4f}  "
              f"sat={_sat_level:.2f} spc={_eff_spc} keep={_eff_keep} ---", flush=True)
        # K_c = lim_{n→∞} Φ^n R_n: Context Crystallization — age all existing bloom vecs.
        # Each cycle they survive, their crystal weight grows by Φ^(age*0.1).
        if hasattr(model, 'bloom_crystal_ages') and model.bloom_crystal_ages:
            model.bloom_crystal_ages = [a + 1 for a in model.bloom_crystal_ages]
        # Rebuild fingerprint from current active_base — corpus grows each cycle
        # so the semantic map of what tokens ARE must stay current.
        # Dynamic coherence threshold: tightens as production quality improves.
        # Higher scores → lower threshold → harder to stop splitting → more leaves.
        # Model earns finer categorical structure through better output.
        if _fingerprint is not None and cycle > 0:
            # C_new = C_old + λ(E_r): Self-Healing Cognitive Structures.
            # E_r = production quality error signal; λ = adaptation coefficient.
            # Coherence threshold self-adjusts — model earns finer categorical
            # structure through better output, not through our imposed schedule.
            _prev_coh = _dyn_coh_stop
            _dyn_coh_stop = max(1.0 - 1.0/(_PHI_FOR_SAMPLING**3),
                                0.90 - (1.0/_PHI_FOR_SAMPLING) * max(0.0, best_production - 1.0/_PHI_FOR_SAMPLING))
            _fingerprint._COHERENCE_STOP = _dyn_coh_stop
            if abs(_dyn_coh_stop - _prev_coh) > 0.005:
                print(f"  [COHERENCE THRESHOLD] {_prev_coh:.3f} → {_dyn_coh_stop:.3f} "
                      f"(best_prod={best_production:.3f})", flush=True)
            # G_{t+1} = G_t + Σ Δa_i: Recursive Agent Ecology.
            # Rebuild from _fp_corpus each cycle — real clusters + real topology,
            # updated with current model embeddings as weights evolve.
            _fingerprint.build(_fp_corpus, model)
            _fp_fwd_n = _fingerprint.fwd.float()
            _fp_nrm   = _fp_fwd_n.norm(dim=1, keepdim=True).clamp(min=1e-8)
            _fp_unit  = _fp_fwd_n / _fp_nrm
            _fingerprint._leaf_coherences = [
                _fingerprint._intra_coherence(
                    _fp_unit[(_fingerprint.sub_clusters == _lid).nonzero(as_tuple=True)[0]])
                if (_fingerprint.sub_clusters == _lid).sum() > 1 else 1.0
                for _lid in range(_fingerprint.n_leaves)]
            model.structure_bloom_vecs = [
                lc.to(_model_device) for lc in _fingerprint.leaf_centroids]
            # FibPosTable is sector-based (7 chunk-length classes) — fingerprint
            # re-clustering does not change its leaf count.  No resize needed.
            if vocab_for_bigram is not None:
                _n_chars_cls = sum(1 for t in vocab_for_bigram if len(t) == 1)
                _fingerprint.leaf_expected_clause_lens = _compute_leaf_expected_clause_lens(
                    _fingerprint, active_base, vocab_for_bigram, _n_chars_cls)
        for s in range(steps_per_cycle):
            new_K = sched(global_step, args.steps)
            if new_K != cur_K:
                set_K_active_recursive(model, new_K)
                cur_K = new_K
            # Train entirely on the active_base (which is seed + appended
            # best-refined outputs). No mixing logic -- the active_base IS
            # the model's corpus, growing with every successful distillation.
            x, y = sample_tiny_batch(active_base, args.batch_size,
                                       args.seq_len, gen)
            logits = model(x)
            if not in_warmup and getattr(args, 'omniweight_loss', False):
                ce_fft = substrate_omniweight_loss(
                    logits, y, vocab_size,
                    lambda_substrate=args.lambda_sub)
            else:
                ce_fft = substrate_fft_loss(logits, y, vocab_size,
                                              lambda_substrate=args.lambda_sub)
            if cycle_lambda > 0.0:
                K_h = K_to_K_harmony(cur_K or args.K_init,
                                      K_init=args.K_init, K_min=args.K_min)
                # Blend corpus signatures toward canonical Fibonacci target as
                # best_production improves. Weight ramps 0→1/φ as production
                # goes from 1/φ to 1.0; zero during warmup (best_production==0).
                if best_production > 0.0:
                    import math as _math
                    _harm_fibs = [1, 2, 3, 5, 8, 13, 21]  # K=7 bands, matches sig_char/sig_ms
                    _canonical_sig = [f / (_PHI_FOR_SAMPLING ** (_math.pi * k))
                                      for k, f in enumerate(_harm_fibs)]
                    _csum = sum(_canonical_sig) + 1e-8
                    _canonical_sig = [v / _csum for v in _canonical_sig]
                    _blend_w = max(0.0, min(1.0 / _PHI_FOR_SAMPLING,
                        (best_production - 1.0 / _PHI_FOR_SAMPLING)
                        / max(1.0 - 1.0 / _PHI_FOR_SAMPLING, 1e-8)
                        / _PHI_FOR_SAMPLING))
                    _can_t = torch.tensor(_canonical_sig,
                                          dtype=sig_char.dtype, device=sig_char.device)
                    _sig_char_b = (1.0 - _blend_w) * sig_char + _blend_w * _can_t
                    _can_ms = torch.tensor(_canonical_sig,
                                           dtype=sig_ms.dtype, device=sig_ms.device)
                    _sig_ms_b = (1.0 - _blend_w) * sig_ms + _blend_w * _can_ms
                else:
                    _sig_char_b = sig_char
                    _sig_ms_b = sig_ms
                harmony = compute_harmony_grounded(logits, vocab_size,
                                                     harmony_kind,
                                                     _sig_char_b, _sig_ms_b,
                                                     K_harmony=K_h)
                loss = ce_fft + cycle_lambda * harmony
            else:
                loss = ce_fft
            # Bloom training signal: when bloom vectors exist, add a KL penalty
            # that pulls the model's clause-initial distribution toward the
            # accumulated bloom prior. Bloom is both generation prior AND
            # training signal — not a separate mechanism but part of what
            # bloom IS. Scale grows with vector count, caps at 0.05 so it
            # never overwhelms CE loss (~4.0).
            if hasattr(model, 'bloom_vecs') and model.bloom_vecs:
                _phi = _PHI_FOR_SAMPLING
                # E_t(m) = m · Φ^(-Δt) · C_t: Temporal Echo Memory.
                # Older bloom vecs decay by 1/Φ^Δt — ancient knowledge fades
                # unless reactivated by context resonance (C_t).
                # B_i = {m_1,...,m_n} → σ_i: Memory Bloom Theory.
                # Many training states compress into a single symbolic anchor.
                # T(x) = ∩ M_i(x): Harmonic Multi-Model Consensus.
                # Training signal emerges where all bloom vecs converge.
                with torch.no_grad():
                    _bloom_p = torch.zeros(vocab_size, device=logits.device)
                    # Newest-first: decay by actual crystal age, not enumeration index.
                    _bloom_ages = (list(model.bloom_crystal_ages)
                                   if hasattr(model, 'bloom_crystal_ages')
                                   and len(model.bloom_crystal_ages) == len(model.bloom_vecs)
                                   else list(range(len(model.bloom_vecs) - 1, -1, -1)))
                    _bw = 0.0
                    for _bv, _age in zip(reversed(model.bloom_vecs),
                                         reversed(_bloom_ages)):
                        _bl = model.head(_bv.to(logits.device).unsqueeze(0)
                                         ).squeeze(0)[:vocab_size]
                        _bloom_p = _bloom_p + torch.softmax(_bl, dim=-1) / (_phi ** _age)
                        _bw += 1.0 / (_phi ** _age)
                    _bloom_p = (_bloom_p / _bw).detach()
                # KL at position 0: clause-initial proxy for batched training.
                _p0 = torch.softmax(logits[:, 0, :vocab_size], dim=-1)
                _kl = (_p0 * ((_p0 + 1e-8).log()
                               - (_bloom_p + 1e-8).log())).sum(-1).mean()
                _scale = min(1.0 / (_PHI_FOR_SAMPLING ** 6),
                             (1.0 / (_PHI_FOR_SAMPLING ** 9)) * len(model.bloom_vecs))
                loss = loss + _scale * _kl
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            if global_step % eval_every == 0:
                vl = evaluate(model, val_split, args.batch_size, args.seq_len,
                              fib_positions, gen)
                last_cycle_val = vl
                marker = ""
                if vl < best_val:
                    best_val = vl; best_step = global_step
                    marker = " ← BEST"
                print(f"    step {global_step:5d}  val={vl:.4f}  "
                      f"K={cur_K}  ({time.time()-t0:.1f}s){marker}",
                      flush=True)
            global_step += 1

        # End of cycle: during warmup just do a quick production check
        # to track progress, then continue without distillation.
        if in_warmup:
            _wu_seq = produce_recursive(
                model, prompt, n_chunks=_n_prod_chunks,
                chunk_size=_chunk_size, vocab_size=vocab_size,
                temperature=0.75, bigram_prior=bigram_prior, vocab=vocab,
                class_id_tensor=class_id_tensor, n_classes=n_classes,
                pronoun_mask=pronoun_mask, vowel_start_mask=vowel_start_mask,
                end_vowels=end_vowels, punct_mask=punct_mask,
                newline_mask=newline_mask,
                unpronounceable_mask=unpronounceable_mask,
                allowed_after_word_mask=allowed_after_word_mask,
                uppercase_mask=uppercase_mask, any_punct_mask=any_punct_mask, first_person_mask=first_person_mask, second_person_mask=second_person_mask, third_person_mask=third_person_mask, verb_mask=verb_mask, noun_mask=noun_mask, adj_mask=adj_mask, func_mask=func_mask)
            _wu_text = ''.join(itos_map.get(int(t), '?')
                                for t in _wu_seq[0].tolist())
            _n_chars_ps = sum(1 for t in vocab_for_bigram if len(t) == 1) if vocab_for_bigram else 0
            _wu_ps = substrate_production_score(
                _wu_text, full_corpus,
                fingerprint=_fingerprint, vocab=vocab_for_bigram,
                tokens=(_wu_seq[0].tolist() if _wu_seq is not None else None),
                fib_table=_fib_table, n_chars=_n_chars_ps)
            if _wu_ps > best_production:
                best_production = _wu_ps
                best_production_text = _wu_text
            _wu_ca3 = 0.0
            if _wu_seq is not None and _ca_3grams is not None:
                _wu_toks = _wu_seq[0, 16:].tolist()
                _wu_3gs = list(zip(_wu_toks, _wu_toks[1:], _wu_toks[2:]))
                _wu_ca3 = (sum(1 for g in _wu_3gs if g in _ca_3grams)
                           / max(1, len(_wu_3gs)))
                if _wu_ca3 > best_match_score:
                    best_match_score = _wu_ca3
            _cms_str = f"  full-3g={_wu_ca3:.3f}" if _wu_ca3 > 0 else ""
            print(f"  [warmup {cycle+1}] production_score={_wu_ps:.4f}  "
                  f"val={last_cycle_val:.4f}{_cms_str}  "
                  f"sample: {repr(_wu_text[16:80])}", flush=True)
            continue

        # Valve: model must reach production_score_gate on its own
        # Benchmark prompt rotates across 4 corpus positions each cycle.
        # This gives a broad view of generation quality, not just position-0.
        _bench_offsets = [0, 1000, 5000, 10000]
        _bench_off = _bench_offsets[cycle % len(_bench_offsets)]
        if (corpus_anchor is not None
                and corpus_anchor.numel() > _bench_off + 16):
            _bench_prompt = corpus_anchor[_bench_off:_bench_off+16].unsqueeze(0)
        else:
            _bench_prompt = prompt
        # recursive output before its generated samples enter active_base.
        # This is substrate-universal quality gating — no val loss involved.
        cycle_long_seq = produce_recursive(
            model, _bench_prompt, n_chunks=_n_prod_chunks,
            chunk_size=_chunk_size, vocab_size=vocab_size,
            temperature=0.75, bigram_prior=bigram_prior, vocab=vocab,
            class_id_tensor=class_id_tensor, n_classes=n_classes,
            pronoun_mask=pronoun_mask, vowel_start_mask=vowel_start_mask,
            end_vowels=end_vowels, punct_mask=punct_mask,
            newline_mask=newline_mask,
            unpronounceable_mask=unpronounceable_mask,
            allowed_after_word_mask=allowed_after_word_mask,
            uppercase_mask=uppercase_mask, any_punct_mask=any_punct_mask, first_person_mask=first_person_mask, second_person_mask=second_person_mask, third_person_mask=third_person_mask, verb_mask=verb_mask, noun_mask=noun_mask, adj_mask=adj_mask, func_mask=func_mask)
        cycle_long_text = ''.join(itos_map.get(int(t), '?')
                                   for t in cycle_long_seq[0].tolist())
        _n_chars_ps = sum(1 for t in vocab_for_bigram if len(t) == 1) if vocab_for_bigram else 0
        cycle_prod_score = substrate_production_score(
            cycle_long_text, full_corpus,
            fingerprint=_fingerprint, vocab=vocab_for_bigram,
            tokens=(cycle_long_seq[0].tolist() if cycle_long_seq is not None else None),
            fib_table=_fib_table, n_chars=_n_chars_ps)
        if cycle_prod_score > best_production:
            best_production = cycle_prod_score
            best_production_text = cycle_long_text

        # Adaptive gate: corpus tracks its own ability rather than our fixed 0.52.
        # After ≥3 data points, gate = max(floor, mean - 0.5*std) of recent scores.
        _prod_score_history.append(cycle_prod_score)
        if len(_prod_score_history) >= 3:
            import statistics as _stat
            _hist = _prod_score_history[-5:]
            _adaptive_gate = max(1.0 / (_PHI_FOR_SAMPLING ** 2),
                                 _stat.mean(_hist) - (1.0 / _PHI_FOR_SAMPLING) * _stat.stdev(_hist))
        else:
            _adaptive_gate = production_score_gate  # fixed floor for first 2 cycles
        prod_allows_growth = cycle_prod_score > _adaptive_gate
        if not prod_allows_growth:
            print(f"  [cycle {cycle+1}] production_score={cycle_prod_score:.4f} < "
                  f"gate={_adaptive_gate:.3f} (adaptive) — generating but NOT "
                  f"growing corpus yet", flush=True)
        else:
            print(f"  [cycle {cycle+1}] production_score={cycle_prod_score:.4f} ≥ "
                  f"gate={_adaptive_gate:.3f} (adaptive) — CORPUS GROWTH OPEN",
                  flush=True)
        print(f"  recursive sample: {repr(cycle_long_text[16:120])}", flush=True)

        # CORPUS MATCH — the primary measurement.
        # full-3g: fraction of generated 3-grams found ANYWHERE in corpus_anchor.
        # This is the true signal: is the model generating authentic Shakespeare?
        _cycle_match_score = 0.0
        if cycle_long_seq is not None:
            _gen_toks = cycle_long_seq[0, 16:].tolist()
            # Primary: full-corpus 3-gram recall
            _ca3_full = 0.0
            if _ca_3grams is not None:
                _gen_3gs = list(zip(_gen_toks, _gen_toks[1:], _gen_toks[2:]))
                _ca3_full = (sum(1 for g in _gen_3gs if g in _ca_3grams)
                             / max(1, len(_gen_3gs)))
            _ca2_full = 0.0
            if _ca_2grams is not None:
                _gen_2gs = list(zip(_gen_toks, _gen_toks[1:]))
                _ca2_full = (sum(1 for g in _gen_2gs if g in _ca_2grams)
                             / max(1, len(_gen_2gs)))
            # Secondary: position-specific match (informational)
            _cms = {}
            if _corpus_ref_tokens is not None:
                _cms = corpus_match_score(_gen_toks, _corpus_ref_tokens, vocab=vocab)
            _cycle_match_score = _ca3_full
            if _cycle_match_score > best_match_score:
                best_match_score = _cycle_match_score
            print(f"  [CORPUS MATCH] full-2g={_ca2_full:.3f}  full-3g={_ca3_full:.3f}  "
                  f"pos-exact={_cms.get('exact_1',0):.3f}  "
                  f"pos-3g={_cms.get('ngram_3',0):.3f}  "
                  f"best={best_match_score:.3f}", flush=True)
            if _corpus_ref_tokens is not None:
                _ref_text = ''.join(itos_map.get(int(t), '?')
                                    for t in _corpus_ref_tokens.tolist())
                print(f"  [CORPUS REF  ] {repr(_ref_text[:80])}", flush=True)

        # End of cycle: HEAVY extrapolation.
        # Growth gate: corpus match score drives sample selection.
        # Samples are scored against corpus ngrams; only those that
        # reproduce Shakespeare's actual wording enter active_base.
        # φ-decaying temperature: starts at 0.8, asymptotes toward 0.8/φ ≈ 0.495
        # over post-warmup cycles. Keeps diversity early; tightens as corpus fills.
        _post_wu_t = max(0, cycle - warmup_cycles)
        _T_growth = 0.8 * (1.0 - (1.0 - 1.0 / _PHI_FOR_SAMPLING) *
                           (1.0 - 1.0 / (_PHI_FOR_SAMPLING ** (_post_wu_t / (_PHI_FOR_SAMPLING ** 2)))))
        samples = []   # list of (refined_seq, creativity)
        for s_idx in range(_eff_spc):
            # Prompt split: 62% corpus_anchor, 25% active_base, 13% bloom tail.
            # Bloom-tail prompts seed generation from the END of an earned authentic
            # run — the model is already "in Shakespeare" at token 0.
            _bloom_tails = [s for s, *_ in _bloom_all_seqs_meta if s.numel() >= 16]
            _use_bloom_tail = (s_idx % 8 == 7 and len(_bloom_tails) > 0)
            _use_ca_prompt = (not _use_bloom_tail
                              and corpus_anchor is not None
                              and corpus_anchor.numel() > 32
                              and (s_idx % 4 != 3))
            if _use_bloom_tail:
                _bt = _bloom_tails[rng.randint(0, len(_bloom_tails) - 1)]
                prompt_s = _bt[-16:].unsqueeze(0).to(next(model.parameters()).device)
            elif _use_ca_prompt:
                _ca_ps = rng.randint(0, max(0, corpus_anchor.numel() - 16))
                prompt_s = corpus_anchor[_ca_ps: _ca_ps + 16].unsqueeze(0)
            else:
                start = rng.randint(0, max(0, active_base.numel() - 17))
                prompt_s = active_base[start: start + 16].unsqueeze(0)
            with torch.no_grad():
                draft = autoregressive_generate(
                    model, prompt_s, n_new=growth_n_new,
                    vocab_size=vocab_size, temperature=_T_growth,
                    bigram_prior=bigram_prior, vocab=vocab, token_signatures=token_signatures, active_vocab_size=active_vocab_size, class_id_tensor=class_id_tensor, n_classes=n_classes, pronoun_mask=pronoun_mask, vowel_start_mask=vowel_start_mask, end_vowels=end_vowels, punct_mask=punct_mask, newline_mask=newline_mask, unpronounceable_mask=unpronounceable_mask, allowed_after_word_mask=allowed_after_word_mask, uppercase_mask=uppercase_mask, any_punct_mask=any_punct_mask, first_person_mask=first_person_mask, second_person_mask=second_person_mask, third_person_mask=third_person_mask, verb_mask=verb_mask, noun_mask=noun_mask, adj_mask=adj_mask, func_mask=func_mask, target_profile=_target_profile, fib_pos_table=_fib_table, fingerprint=_fingerprint)
            refined_s, _ = staged_refine(
                model, prompt_s, n_new=growth_n_new, vocab_size=vocab_size,
                harmony_scorer=harmony_fn, quality_scorer=quality_fn,
                creativity_scorer=creativity_fn,
                n_iters_per_stage=30, resample_frac=0.35,
                prompt_len=16, temperature=0.5,
                bigram_prior=bigram_prior, vocab=vocab, token_signatures=token_signatures, active_vocab_size=active_vocab_size, class_id_tensor=class_id_tensor, n_classes=n_classes, pronoun_mask=pronoun_mask, vowel_start_mask=vowel_start_mask, end_vowels=end_vowels, punct_mask=punct_mask, newline_mask=newline_mask, unpronounceable_mask=unpronounceable_mask, allowed_after_word_mask=allowed_after_word_mask, uppercase_mask=uppercase_mask, any_punct_mask=any_punct_mask, first_person_mask=first_person_mask, second_person_mask=second_person_mask, third_person_mask=third_person_mask, verb_mask=verb_mask, noun_mask=noun_mask, adj_mask=adj_mask, func_mask=func_mask, target_profile=_target_profile, fib_pos_table=_fib_table)
            samples.append((refined_s.squeeze(0).clone(),
                              creativity_fn(refined_s)))
            # Also include the raw T=0.8 draft — staged_refine stages are all disabled,
            # so the only difference is temperature. The hotter draft explores further
            # and may contain longer authentic runs that the cooler T=0.5 pass misses.
            # Both were already being generated (draft was previously wasted). Including
            # both doubles bloom detection surface for free.
            samples.append((draft.squeeze(0).clone(), creativity_fn(draft)))
        # Sort by longest authentic run (50%) + average 3-gram recall (30%) + creativity (20%).
        # Longest-run weighting directly selects for T21 candidates over scattered n-gram matches.
        def _sort_score(seq, cr):
            if _ca_3grams is not None:
                tl = seq.tolist()
                gs = list(zip(tl, tl[1:], tl[2:]))
                ca3 = sum(1 for g in gs if g in _ca_3grams) / max(1, len(gs))
                # Find longest authentic run in this sample
                _mr, _rs2 = 0, 0
                for _ri2 in range(len(tl) - 2):
                    if (tl[_ri2], tl[_ri2+1], tl[_ri2+2]) not in _ca_3grams:
                        _mr = max(_mr, (_ri2 + 2) - _rs2); _rs2 = _ri2 + 1
                _mr = max(_mr, len(tl) - _rs2)
                _mr_fib_near = min([3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610], key=lambda f: abs(_mr - f)) if _mr > 0 else 3
                run_score = _PHI_FOR_SAMPLING ** -(abs(_mr - _mr_fib_near) / max(_mr_fib_near, 1))
            else:
                ca3 = 0.0; run_score = 0.0
            return 0.2 * float(cr) + 0.3 * ca3 + 0.5 * run_score
        samples.sort(key=lambda x: _sort_score(x[0], x[1]), reverse=True)
        kept = samples[:_eff_keep]
        kept_scores = [_sort_score(s[0], s[1]) for s in kept]
        kept_cr = [float(s[1]) for s in kept]
        all_scores = [_sort_score(s[0], s[1]) for s in samples]
        mean_score = sum(all_scores) / len(all_scores)
        print(f"  cycle {cycle+1}: generated {len(samples)} samples ({_eff_spc}×T0.8+T0.5), "
              f"mean combined={mean_score:.4f}, "
              f"top-{_eff_keep} combined={[round(s, 4) for s in kept_scores]} "
              f"cr={[round(c, 4) for c in kept_cr]}")
        # Three filters: quality (model-relative: > 90% of current best, floor 0.50),
        # anchor (seed >= 40% of active_base), leaf membership (corpus-native token quality).
        # Leaf membership: fraction of tokens in non-void leaves (coherence > 0).
        # Void-leaf tokens appear in all contexts — their fwd vectors cancel to near-zero.
        # Non-void tokens have distinctive patterns — the corpus's own 'real token' definition.
        #
        # Adaptive threshold: forced expansion (_MIN_LEAVES=21) creates many void
        # leaves (coh≤0) even when the corpus has genuine structure. A flat 0.25
        # threshold can be impossible when only 5% of vocab tokens are non-void —
        # it would require 5× oversampling of those tokens. Scale to 2× the random
        # baseline (fraction of vocab in non-void leaves), clamped to [0.10, 0.25].
        _cycle_non_void: frozenset = frozenset()
        if _fingerprint is not None and _fingerprint._built:
            _cycle_non_void = frozenset(
                lid for lid, c in enumerate(_fingerprint._leaf_coherences) if c > 0.0)
            _nv_tok_count = sum(
                int((_fingerprint.sub_clusters == lid).sum().item())
                for lid in _cycle_non_void)
            _nv_frac = _nv_tok_count / max(1, _fingerprint.V)
            leaf_nonvoid_min = max(0.10, min(0.25, _nv_frac * 2.0))
            print(f"    [rw-gate] non-void leaves={len(_cycle_non_void)}/{_fingerprint.n_leaves} "
                  f"({100*_nv_frac:.1f}% vocab) → leaf_nonvoid_min={leaf_nonvoid_min:.2f}")
        else:
            leaf_nonvoid_min = 0.25
        n_growth = 0
        n_added_this_cycle = 0
        n_rej_rw_this_cycle = 0
        n_rej_novel_this_cycle = 0
        _bloom_seqs: list = []
        _bloom_qualities: list = []
        # Score-feedback accumulator: high-quality sequences reinforce the leaf
        # transitions that produced them, T[k,l] self-organises toward what scores well.
        _T_score_acc = (torch.zeros(_fingerprint.n_leaves, _fingerprint.n_leaves)
                        if _fingerprint is not None and _fingerprint._built else None)
        _T_score_total_weight = 0.0
        # ── Bloom detection: scan ALL generated samples (not just top-4). ──
        # A T21 sample with low creativity scores lower than a T20 sample with high
        # creativity, pushing it out of kept[:4]. Bloom detection must run on the
        # full pool so T21+ runs are never missed due to sort ranking.
        _FIB_TIERS = [3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
        _bloom_prosodic_bar = getattr(model, '_bloom_prosodic_bar', 0.0)
        _prosodic_min = max(0.03, _bloom_prosodic_bar / _PHI_FOR_SAMPLING)
        for _bscan_seq, _bscan_cr in samples:
            _seq_list = _bscan_seq.tolist()
            _seq_density  = _leaf_content_density(_seq_list, _fingerprint)
            _best_run = []
            _best_run_len = 0
            if _ca_3grams is not None and len(_seq_list) >= 3:
                _rs = 0
                for _ri in range(len(_seq_list) - 2):
                    _g = (_seq_list[_ri], _seq_list[_ri+1], _seq_list[_ri+2])
                    if _g not in _ca_3grams:
                        _rl = (_ri + 2) - _rs
                        if _rl > _best_run_len:
                            _best_run_len = _rl
                            _best_run = _seq_list[_rs:_ri+2]
                        _rs = _ri + 1
                _rl = len(_seq_list) - _rs
                if _rl > _best_run_len:
                    _best_run_len = _rl
                    _best_run = _seq_list[_rs:]
            # Prosodic gate evaluated on the AUTHENTIC RUN itself, not the full
            # 249-token sample. A T21 run embedded in noisy text was previously
            # rejected because the surrounding noise dragged prosodic below the bar.
            _run_for_prosodic = _best_run if _best_run else _seq_list
            _seq_prosodic = _prosodic_alternation(_run_for_prosodic, vocab_for_bigram or [])
            _seq_composite = (float(_bscan_cr) * max(0.1, _seq_density)
                              * (0.5 + 0.5 * _seq_prosodic))
            _run_tier = None
            for _ft in _FIB_TIERS:
                if _best_run_len >= _ft:
                    _run_tier = _ft
            _passes_bloom = (_seq_prosodic >= _prosodic_min and _run_tier is not None)
            if _passes_bloom:
                _run_tensor = torch.tensor(_best_run, dtype=_bscan_seq.dtype)
                _bloom_seqs.append(_run_tensor.unsqueeze(0))
                _bloom_qualities.append(float(_bscan_cr))
                _bloom_all_seqs_meta.append((_run_tensor.cpu(), _seq_composite,
                                             _seq_prosodic, _seq_density, float(_bscan_cr)))
                if vocab_for_bigram is not None:
                    _run_text = ''.join(
                        vocab_for_bigram[t] if t < len(vocab_for_bigram) else '?'
                        for t in _best_run[:40])
                else:
                    _run_text = str(_best_run[:10])
                _prev_peak = getattr(model, '_max_bloom_tier_seen', 3)
                if _run_tier > _prev_peak:
                    model._max_bloom_tier_seen = _run_tier
                    print(f"    [BLOOM PEAK] new peak tier T{_run_tier:02d} "
                          f"— floor rises to T{_FIB_TIERS[max(0, _FIB_TIERS.index(_run_tier)-2)]:02d}",
                          flush=True)
                _peak_now = getattr(model, '_max_bloom_tier_seen', 3)
                print(f"    [BLOOM T{_run_tier:02d}] run={_best_run_len}tok: "
                      f"{repr(_run_text)}", flush=True)
                if (_fib_table is not None and _fingerprint is not None
                        and _fingerprint._built and vocab_for_bigram is not None):
                    _n_chars_ft = sum(1 for t in vocab_for_bigram if len(t) == 1)
                    _peak_idx_now = _FIB_TIERS.index(_peak_now) if _peak_now in _FIB_TIERS else 0
                    _tier_idx_now = _FIB_TIERS.index(_run_tier) if _run_tier in _FIB_TIERS else 0
                    _tier_gap = max(0, _peak_idx_now - _tier_idx_now)
                    _fib_weight = float(_best_run_len) / (_PHI_FOR_SAMPLING ** _tier_gap)
                    _fib_table.update_from_tokens(
                        _run_tensor.unsqueeze(0), _fingerprint, vocab_for_bigram,
                        _n_chars_ft, weight=_fib_weight)
        # ── Active_base admission: quality-gated, top-4 kept samples only ──
        for ref_seq, cr in kept:
            if cr > best_creativity:
                best_creativity = cr
                best_refined_seq = ref_seq.clone()
            # Decode for scoring; leaf membership replaces dictionary gate.
            ref_text = ''.join(itos_map.get(int(t), '?')
                                for t in ref_seq.tolist())
            if _fingerprint is not None and _fingerprint._built:
                _rw_toks   = ref_seq.clamp(0, _fingerprint.V - 1)
                _rw_leaves = _fingerprint.sub_clusters[_rw_toks]
                rw = float(sum(1 for lid in _rw_leaves.tolist()
                               if lid in _cycle_non_void)) / max(1, _rw_leaves.numel())
            else:
                rw = real_word_fraction(ref_text, corpus_text, min_word_len=3)
            # Corpus match gate: full-corpus 3-gram recall on GENERATED tokens only.
            # Skipping the 16-token corpus prompt avoids inflating _samp_match by
            # ~5.7% for corpus-anchored samples (87.5% of the pool).
            _samp_match = 0.0
            if _ca_3grams is not None:
                _rsl = ref_seq[16:].tolist() if ref_seq.numel() > 16 else ref_seq.tolist()
                _s_3gs = list(zip(_rsl, _rsl[1:], _rsl[2:]))
                _samp_match = (sum(1 for g in _s_3gs if g in _ca_3grams)
                               / max(1, len(_s_3gs)))
            passes_q = cr > max(0.50, best_creativity - 0.10)
            passes_a = (_samp_match >= _ca_samp_match_floor) if _ca_3grams is not None else True
            passes_rw = rw >= leaf_nonvoid_min
            passes_prod = prod_allows_growth or (_samp_match > _ca_samp_match_floor * 0.67)
            # OmniWeight Structural Admission gate.
            # Score = novel_3gram_count × run_omniweight × (1 - 0.5·saturation)
            # where run_omniweight = φ^(-|run_len - nearest_fib| / nearest_fib).
            # Threshold rises with corpus saturation so only progressively richer
            # runs can force entry as the model approaches full 3-gram coverage.
            # Value mirrors the canonical OMC OmniWeight: w = φ^(-|e|/attractor).
            _ref_novel_3gs: set = set()
            _novel_ow_score = 0.0
            _novel_threshold = 0.5
            if _ca_3grams is not None:
                _rfl = ref_seq.tolist()
                _rfrs = 0; _rfbest = 0; _rfbest_s = 0
                for _rfi in range(len(_rfl) - 2):
                    if (_rfl[_rfi], _rfl[_rfi+1], _rfl[_rfi+2]) not in _ca_3grams:
                        _rfl_rl = (_rfi + 2) - _rfrs
                        if _rfl_rl > _rfbest:
                            _rfbest = _rfl_rl; _rfbest_s = _rfrs
                        _rfrs = _rfi + 1
                _rfl_rl = len(_rfl) - _rfrs
                if _rfl_rl > _rfbest:
                    _rfbest = _rfl_rl; _rfbest_s = _rfrs
                if _rfbest >= 3:
                    _rrt = _rfl[_rfbest_s: _rfbest_s + _rfbest]
                    _run_3gs = set(zip(_rrt, _rrt[1:], _rrt[2:]))
                    _ref_novel_3gs = _run_3gs - _active_run_3gs
                # OmniWeight of run length: φ^(-|len - nearest_fib| / nearest_fib)
                _rfib_near = min(_FIB_TIERS, key=lambda f: abs(_rfbest - f)) if _rfbest >= 3 else 3
                _run_ow = _PHI_FOR_SAMPLING ** -(abs(_rfbest - _rfib_near) / max(_rfib_near, 1))
                # Saturation pressure: fraction of corpus 3-grams already absorbed
                _saturation = len(_active_run_3gs) / max(len(_ca_3grams), 1)
                _novel_ow_score = len(_ref_novel_3gs) * _run_ow * (1.0 - 0.5 * _saturation)
                _novel_threshold = 0.5 * (1.0 + _saturation)
            passes_novel = (_novel_ow_score >= _novel_threshold) if _ca_3grams is not None else True
            if passes_q and passes_a and passes_rw and passes_prod and passes_novel:
                active_base = torch.cat([active_base, ref_seq])
                n_growth += ref_seq.numel()
                n_added_this_cycle += 1
                _active_run_3gs.update(_ref_novel_3gs)
                if _T_score_acc is not None and _fingerprint is not None:
                    _r_toks = ref_seq.clamp(0, _fingerprint.V - 1)
                    _r_leaves = _fingerprint.sub_clusters[_r_toks].tolist()
                    _score_w = float(cr)
                    for _li in range(len(_r_leaves) - 1):
                        _s_l, _d_l = _r_leaves[_li], _r_leaves[_li + 1]
                        if (_s_l != _d_l
                                and 0 <= _s_l < _fingerprint.n_leaves
                                and 0 <= _d_l < _fingerprint.n_leaves):
                            _T_score_acc[_s_l, _d_l] += _score_w
                    _T_score_total_weight += _score_w
                if (_fib_table is not None and _fingerprint is not None
                        and _fingerprint._built and vocab_for_bigram is not None):
                    _n_chars_ft = sum(1 for t in vocab_for_bigram if len(t) == 1)
                    _fib_table.update_from_tokens(
                        ref_seq.unsqueeze(0), _fingerprint, vocab_for_bigram,
                        _n_chars_ft, weight=2.0)
            else:
                if not passes_q:
                    n_rejected_quality += 1
                if not passes_a:
                    n_rejected_anchor += 1
                if not passes_rw:
                    n_rej_rw_this_cycle += 1
                if not passes_novel:
                    n_rej_novel_this_cycle += 1
                # Self-critique: rejected sample → mild negative update.
                # The table learns which positional placements led to bad output
                # and discounts them toward the uniform prior. Token-level
                # markers are not negatively updated (too noisy from one sample).
                if (_fib_table is not None and not passes_q
                        and _fingerprint is not None and _fingerprint._built
                        and vocab_for_bigram is not None):
                    _n_chars_ft = sum(1 for t in vocab_for_bigram if len(t) == 1)
                    _fib_table.update_from_tokens(
                        ref_seq.unsqueeze(0), _fingerprint, vocab_for_bigram,
                        _n_chars_ft, weight=-0.3)
        # Bloom: compress accepted samples → single latent vector, attach to model.
        # Applied from cycle 2 onward (after warmup) whenever any bloom run was detected.
        # Bloom detection now runs on all generated samples, so _bloom_seqs may be
        # populated even in cycles where nothing was admitted to active_base.
        if _bloom_seqs and cycle >= warmup_cycles:
            # D_{t+1} = f(M_t, R_t, U_t): Recursive Dreaming / Offline Cognition.
            # _bloom_seqs = M_t (accepted memory sequences this cycle)
            # _bloom_qualities = proxy for R_t (resolved structure) + U_t (uncertainty)
            # compute_bloom_embedding IS synthetic replay — offline consolidation of
            # what the model learned, compressed into a latent symbolic anchor.
            _bloom_emb, _bloom_stats = compute_bloom_embedding(
                model, _bloom_seqs, _bloom_qualities,
                vocab=vocab_for_bigram, fingerprint=_fingerprint)
            if _bloom_emb is not None:
                if not hasattr(model, 'bloom_vecs'):
                    model.bloom_vecs = []
                if not hasattr(model, 'bloom_crystal_ages'):
                    model.bloom_crystal_ages = []
                model.bloom_vecs.append(_bloom_emb)
                model.bloom_crystal_ages.append(0)  # new vec starts at crystal age 0
                bloom_norm = float(_bloom_emb.norm().item())
                sim_str = ""
                if len(model.bloom_vecs) > 1:
                    cos_sim = float(torch.nn.functional.cosine_similarity(
                        _bloom_emb.unsqueeze(0),
                        model.bloom_vecs[-2].unsqueeze(0)).item())
                    sim_str = f"  cos_sim_prev={cos_sim:.3f}"
                prosodic_s = _bloom_stats.get('mean_prosodic', 0.0)
                density_s  = _bloom_stats.get('mean_density', 0.0)
                # Update running prosodic bar — the gate tightens as bloom
                # learns what structured Shakespeare looks like.
                if prosodic_s > 0.0:
                    _old_bar = getattr(model, '_bloom_prosodic_bar', 0.0)
                    model._bloom_prosodic_bar = (1.0 / _PHI_FOR_SAMPLING) * _old_bar + (1.0 - 1.0 / _PHI_FOR_SAMPLING) * prosodic_s
                print(f"  [BLOOM {len(model.bloom_vecs)}] cycle {cycle+1}: "
                      f"compressed {len(_bloom_seqs)} samples → "
                      f"|v|={bloom_norm:.3f}  "
                      f"prosodic={prosodic_s:.3f}  density={density_s:.3f}"
                      f"  bar={getattr(model, '_bloom_prosodic_bar', 0.0):.3f}"
                      f"{sim_str}")
        # Leaf reorganization from bloom: if bloom gained authentic sequences this
        # cycle, rebuild the fingerprint's fwd/bwd/rel topology vectors from all
        # accumulated bloom sequences. The leaves now orient to real Shakespeare
        # co-occurrence patterns — not the mixed corpus or generated text.
        if (_bloom_seqs and cycle >= warmup_cycles
                and _fingerprint is not None and _fingerprint._built):
            # Collect all bloom sequences saved so far (from all cycles + warm-start).
            _all_bloom_toks_list = [s.cpu() for s, *_ in _bloom_all_seqs_meta
                                    if s.numel() >= 3]
            if _all_bloom_toks_list:
                _bloom_flat = torch.cat(_all_bloom_toks_list)
                _fingerprint.update_fwd_bwd(_bloom_flat, model,
                                            device=next(model.parameters()).device)
                print(f"  [LEAF REORG] fwd/bwd/rel rebuilt from "
                      f"{_bloom_flat.numel()} authentic bloom tokens "
                      f"(T{max(len(s) for s, *_ in _bloom_all_seqs_meta if s.numel() >= 3)} "
                      f"max run)", flush=True)

        # Score-feedback blend: high-scoring sequences reinforce their leaf transitions.
        # T[k,l] self-organises toward what actually scored well this cycle.
        # 70% corpus-native grammar + 30% quality-signal grammar — keeps corpus
        # statistics dominant while letting the model's own performance shape the bias.
        if (_T_score_acc is not None and _fingerprint is not None
                and _T_score_total_weight > 0
                and _fingerprint.cluster_transition is not None):
            _T_norm = _T_score_acc / _T_score_total_weight
            _row_s = _T_norm.sum(dim=1, keepdim=True).clamp(min=1e-8)
            _T_norm = _T_norm / _row_s
            # W_{t+1} = W_t + ΔO_t: Recursive Reality Mapping.
            # cluster_transition IS the world model W_t (corpus grammar reality map).
            # ΔO_t = _T_norm (this cycle's observed leaf transitions).
            # The 70/30 blend preserves prior world model while new observations
            # update it — and the world model reshapes interpretation of future input.
            _T_blend = ((1.0 / _PHI_FOR_SAMPLING) * _fingerprint.cluster_transition.float()
                        + (1.0 - 1.0 / _PHI_FOR_SAMPLING) * _T_norm.to(_fingerprint.cluster_transition.device))
            _rb = _T_blend.sum(dim=1, keepdim=True).clamp(min=1e-8)
            _fingerprint.cluster_transition = (_T_blend / _rb).cpu()
            print(f"  [T-FEEDBACK] T[k,l] updated from {int(_T_score_total_weight * 100) / 100:.2f} "
                  f"score-weighted transitions → corpus={1.0 / _PHI_FOR_SAMPLING:.3f} quality={1.0 - 1.0 / _PHI_FOR_SAMPLING:.3f} blend", flush=True)
        cycle_summary.append({
            "cycle": cycle + 1,
            "val_informational": last_cycle_val,
            "production_score": cycle_prod_score,
            "production_gate_open": prod_allows_growth,
            "samples_creativity": all_scores,
            "kept_top_k": kept_scores,
            "n_added": n_added_this_cycle,
            "n_rejected_quality": n_rejected_quality,
            "n_rejected_anchor": n_rejected_anchor,
            "active_base_after": active_base.numel(),
        })
        print(f"  added {n_added_this_cycle}/{len(kept)} samples  "
              f"(rej_quality={n_rejected_quality}, "
              f"rej_anchor={n_rejected_anchor}, "
              f"rej_realword(this cycle)={n_rej_rw_this_cycle}, "
              f"rej_omniw={n_rej_novel_this_cycle}, "
              f"absorbed_3gs={len(_active_run_3gs)}/{len(_ca_3grams) if _ca_3grams else '?'}, "
              f"saturation={len(_active_run_3gs)/max(len(_ca_3grams),1) if _ca_3grams else 0:.2f}) "
              f"active_base={active_base.numel()} tokens  "
              f"best_production={best_production:.4f}  "
              f"best_creativity={best_creativity:.4f}")
        # Show the best refined sample from this cycle as text.
        if itos_map is not None and kept:
            best_in_cycle = kept[0][0]
            sample_text = ''.join(itos_map.get(int(t), '?')
                                    for t in best_in_cycle.tolist())
            print(f"  best short sample (c={kept[0][1]:.3f}):\n    "
                  f"{repr(sample_text[:200])}")
        # Show fib-pos table summary after corpus growth
        if _fib_table is not None and cycle >= 2 and _fib_table._n_updates >= 10:
            print(_fib_table.summary())

    # Final generation: single-pass + recursive production for inspection.
    final_gen = autoregressive_generate(model, prompt, n_new=n_new,
                                          vocab_size=vocab_size,
                                          temperature=0.8,
                                          bigram_prior=bigram_prior, vocab=vocab, token_signatures=token_signatures, active_vocab_size=active_vocab_size, class_id_tensor=class_id_tensor, n_classes=n_classes, pronoun_mask=pronoun_mask, vowel_start_mask=vowel_start_mask, end_vowels=end_vowels, punct_mask=punct_mask, newline_mask=newline_mask, unpronounceable_mask=unpronounceable_mask, allowed_after_word_mask=allowed_after_word_mask, uppercase_mask=uppercase_mask, any_punct_mask=any_punct_mask, first_person_mask=first_person_mask, second_person_mask=second_person_mask, third_person_mask=third_person_mask, verb_mask=verb_mask, noun_mask=noun_mask, adj_mask=adj_mask, func_mask=func_mask, target_profile=_target_profile, fib_pos_table=_fib_table, fingerprint=_fingerprint)
    final_refined, _ = staged_refine(
        model, prompt, n_new=n_new, vocab_size=vocab_size,
        harmony_scorer=harmony_fn, quality_scorer=quality_fn,
        creativity_scorer=creativity_fn,
        n_iters_per_stage=200, resample_frac=0.35,
        prompt_len=16, temperature=0.5,
        bigram_prior=bigram_prior, vocab=vocab, token_signatures=token_signatures, active_vocab_size=active_vocab_size, class_id_tensor=class_id_tensor, n_classes=n_classes, pronoun_mask=pronoun_mask, vowel_start_mask=vowel_start_mask, end_vowels=end_vowels, punct_mask=punct_mask, newline_mask=newline_mask, unpronounceable_mask=unpronounceable_mask, allowed_after_word_mask=allowed_after_word_mask, uppercase_mask=uppercase_mask, any_punct_mask=any_punct_mask, first_person_mask=first_person_mask, second_person_mask=second_person_mask, third_person_mask=third_person_mask, verb_mask=verb_mask, noun_mask=noun_mask, adj_mask=adj_mask, func_mask=func_mask, target_profile=_target_profile, fib_pos_table=_fib_table)

    # Final recursive production — the model writing Shakespeare from itself.
    final_recursive = produce_recursive(
        model, prompt, n_chunks=_n_prod_chunks, chunk_size=_chunk_size,
        vocab_size=vocab_size, temperature=0.70,
        bigram_prior=bigram_prior, vocab=vocab,
        class_id_tensor=class_id_tensor, n_classes=n_classes,
        pronoun_mask=pronoun_mask, vowel_start_mask=vowel_start_mask,
        end_vowels=end_vowels, punct_mask=punct_mask,
        newline_mask=newline_mask,
        unpronounceable_mask=unpronounceable_mask,
        allowed_after_word_mask=allowed_after_word_mask,
        uppercase_mask=uppercase_mask, any_punct_mask=any_punct_mask, first_person_mask=first_person_mask, second_person_mask=second_person_mask, third_person_mask=third_person_mask, verb_mask=verb_mask, noun_mask=noun_mask, adj_mask=adj_mask, func_mask=func_mask,
        target_profile=_target_profile, fib_pos_table=_fib_table)
    final_recursive_text = ''.join(itos_map.get(int(t), '?')
                                    for t in final_recursive[0].tolist())
    final_prod_score = substrate_production_score(
        final_recursive_text, full_corpus,
        fingerprint=_fingerprint, vocab=vocab_for_bigram,
        tokens=(final_recursive[0].tolist() if final_recursive is not None else None),
        fib_table=_fib_table,
        n_chars=_n_chars_ps)
    if final_prod_score > best_production:
        best_production = final_prod_score
        best_production_text = final_recursive_text

    print(f"\n  === FINAL RECURSIVE PRODUCTION ({_n_prod_chunks} chunks × {_chunk_size} tokens) ===")
    print(f"  production_score={final_prod_score:.4f}  "
          f"best_seen={best_production:.4f}")
    print(f"  {repr(final_recursive_text[:400])}")

    # Fibonacci-tier the bloom corpus before saving: keep top-55 (F[10])
    # sequences as raw tokens; compress overflow into an additional bloom vector.
    # Prior-run sequences seeded at score=0.40 are displaced first if this run's
    # corn scored higher — the checkpoint always keeps the best across all runs.
    _all_bloom_vecs = getattr(model, 'bloom_vecs', [])
    _raw_seqs, _tiered_bloom_vecs = bloom_tier_sequences(
        _bloom_all_seqs_meta, _all_bloom_vecs,
        model, vocab_for_bigram, _fingerprint,
        max_raw=21)   # F[8]=21 keeps inherited corpus ≈3500 tokens — diversity sweet spot
    model.bloom_vecs = _tiered_bloom_vecs

    # Serialize bloom state for the next run: bloom vectors + FibPosTable +
    # quality markers. Each successive run inherits this and blooms further.
    _bloom_save = {
        'bloom_vecs': [bv.cpu() for bv in _tiered_bloom_vecs],
        'bloom_crystal_ages': (list(getattr(model, 'bloom_crystal_ages', []))[:len(_tiered_bloom_vecs)]),
        'fib_table_counts':       (_fib_table.counts.cpu()
                                   if _fib_table is not None else None),
        'fib_table_token_counts': (_fib_table.token_counts.cpu()
                                   if _fib_table is not None
                                   and _fib_table.token_counts is not None else None),
        'fib_table_n_leaves':     (_fib_table.n_leaves
                                   if _fib_table is not None else None),
        'fib_table_n_updates':    (_fib_table._n_updates
                                   if _fib_table is not None else 0),
        'best_creativity':  best_creativity,
        'best_production':  best_production,
        'max_bloom_tier_seen': getattr(model, '_max_bloom_tier_seen', 3),
        'bloom_prosodic_bar':  getattr(model, '_bloom_prosodic_bar', 0.0),
        # Bloom's expanded form: top-F[10] sequences by composite score.
        # Tiered — older/lower-quality sequences are compressed to vectors,
        # not stored as raw tokens. Cross-run size is bounded at 55 sequences max.
        'bloom_sequences':  _raw_seqs,
        # Fingerprint caches — restored at warm-start so boundary nudge is
        # live from the first generation, not just from cycle 2 onward.
        'leaf_coherences': (list(_fingerprint._leaf_coherences)
                            if _fingerprint is not None
                            and hasattr(_fingerprint, '_leaf_coherences')
                            else None),
        'leaf_expected_clause_lens': (_fingerprint.leaf_expected_clause_lens.cpu()
                                      if _fingerprint is not None
                                      and hasattr(_fingerprint, 'leaf_expected_clause_lens')
                                      and _fingerprint.leaf_expected_clause_lens is not None
                                      else None),
    }
    torch.save(_bloom_save, bloom_checkpoint_path)
    _n_blooms = len(_bloom_save['bloom_vecs'])
    _n_seq_toks = sum(s.numel() for s in _raw_seqs)
    print(f"  [BLOOM SAVED] {_n_blooms} vecs + {len(_raw_seqs)} sequences "
          f"({_n_seq_toks} tokens) → {bloom_checkpoint_path}")

    return {"name": name, "mode": "self_distillation",
             "n_params": n_params,
             "best_val": best_val, "best_step": best_step,
             "best_production_score": best_production,
             "best_production_text": best_production_text[:400],
             "wall": time.time() - t0,
             "best_creativity_seen": best_creativity,
             "active_base_final_size": active_base.numel(),
             "cycle_summary": cycle_summary,
             "generated_tokens": final_gen[0].tolist(),
             "refined_tokens": final_refined[0].tolist(),
             "recursive_tokens": final_recursive[0].tolist()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=89)   # F(11) Fibonacci-aligned
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--K-init", type=int, default=89)
    parser.add_argument("--K-min", type=int, default=13)   # restore K-shrink

    parser.add_argument("--K-sig", type=int, default=16)
    parser.add_argument("--lambda-sub", type=float, default=0.01)
    parser.add_argument("--lambda-harmony", type=float, default=0.03,
                          help="Harmony weight (ramped in after warmup cycles). "
                               "Old default 1/phi^pi≈0.146 competed too hard "
                               "with CE before convergence.")
    parser.add_argument("--tiny-chars", type=int, default=1024,
                          help="Size of the tiny training seed in chars")
    parser.add_argument("--warmup-cycles", type=int, default=2,
                          help="First N cycles: CE-only, no harmony, no corpus "
                               "growth. Lets model converge before distillation.")
    parser.add_argument("--production-gate", type=float, default=0.52,
                          help="Only append refined outputs to active_base when "
                               "recursive production_score > this threshold. "
                               "Prevents corpus poisoning with garbage generations. "
                               "Production score = substrate_coherence + creativity.")
    parser.add_argument("--omniweight-loss", action="store_true",
                          help="Apply the inference-side omniweight standard "
                                "(phi^pi tanh fluid form) to per-token CE "
                                "during training. Closes the train/inference "
                                "asymmetry on the anti-stagnation primitive.")
    parser.add_argument("--out", type=str,
                          default="results_self_recursive.json")
    args = parser.parse_args()

    chars, stoi, itos, encoded = make_dataset(seq_len=args.seq_len,
                                                 source="tinyshakespeare")
    # Build the substrate tokenizer from the FULL corpus (text form).
    char_itos_map = {i: c for i, c in enumerate(chars)}
    full_corpus_text = ''.join(char_itos_map.get(int(t), '?')
                                  for t in encoded.tolist())
    sub_tok = SubstrateTokenizer(full_corpus_text, max_vocab_size=500)
    print(f"Substrate tokenizer: vocab={sub_tok.vocab_size}  "
          f"(chars={len(chars)} -> +{sub_tok.vocab_size - len(chars)} fib-ngrams)")
    # Re-encode the whole corpus into substrate tokens.
    encoded = torch.tensor(sub_tok.encode(full_corpus_text), dtype=torch.long)
    vocab_size = sub_tok.vocab_size
    # itos/stoi for substrate tokens (used by creativity scoring & sample print).
    chars = sub_tok.vocab        # list of token strings (some multi-char)
    itos_map = {i: c for i, c in enumerate(chars)}
    # Reflection training: the model trains on corpus_anchor (20k tokens), not a
    # Seed → sprout. The model trains on a tiny seed (512 tokens).
    # The substrate (FibPosTable, fingerprint fwd/bwd, bloom vecs) internalizes
    # the structural rules from that seed. The measurement of success is NOT
    # val loss or creativity — it is corpus match: can the model, given the
    # seed prompt, reproduce the ACTUAL corpus continuation token for token?
    # corpus_ref = the corpus continuation after the seed prompt.
    # When generation matches corpus_ref exactly, the substrate has sprouted
    # the full corpus from the seed. That is the completion criterion.
    val_start = encoded.numel() // 10 * 9
    val_split = encoded[val_start:].clone()
    tiny_tokens = max(args.tiny_chars // 2, 256)
    train_seed = take_tiny_seed(encoded, tiny_tokens, seed=args.seed)
    corpus_anchor = train_seed.clone()   # from-nothing: FibTable + 3-gram ref = seed only
    fib_positions = fib_positions_in_window(args.seq_len)

    print(f"Seed: {train_seed.numel()} tokens; val on {val_split.numel()} tokens")

    # Build itos and full corpus text for creativity scoring.
    itos_map = {i: c for i, c in enumerate(chars)}
    full_corpus_text = ''.join(itos_map.get(int(t), '?')
                                  for t in encoded.tolist())

    arms = [
        ("self_distill_multiscale",  "multiscale"),
    ]
    results = {}
    for name, harmony_kind in arms:
        results[name] = train_with_self_distillation(
            name, train_seed, corpus_anchor, val_split, vocab_size, args,
            fib_positions, harmony_kind=harmony_kind,
            itos_map=itos_map, corpus_text=full_corpus_text,
            vocab_for_bigram=sub_tok.vocab,
            n_cycles=24, distill_prob=0.3,
            samples_per_cycle=8, keep_top_k=4, growth_n_new=233,
            warmup_cycles=args.warmup_cycles,
            production_score_gate=args.production_gate)

    print()
    print("=" * 92)
    print(f"{'arm':<24} {'params':>10} {'prod_score':>12} {'coherence':>10} {'creativity':>11} {'structure':>9} {'discourse':>10} {'wall':>10}")
    print('-' * 115)
    for name, r in results.items():
        best_text = r.get('best_production_text', '')
        coh  = substrate_coherence_score(best_text) if best_text else 0.0
        stru = substrate_structure_score(best_text)  if best_text else 0.0
        disc = substrate_discourse_score(best_text)   if best_text else 0.0  # display only; training uses leaf_positional_match_score
        print(f"{name:<24} {r['n_params']:>10,} "
              f"{r.get('best_production_score', 0.0):>12.4f} "
              f"{coh:>10.4f} "
              f"{r.get('best_creativity_seen', 0.0):>11.4f} "
              f"{stru:>9.4f} "
              f"{disc:>10.4f} "
              f"{r['wall']:>9.1f}s")

    print()
    print("Primary metric: production_score (coherence + creativity + structure + discourse)")
    print("Goal: production_score > 0.65 with coherent self-recursive Shakespeare")

    # Print decoded generation samples per arm: single-pass vs refined.
    itos_map = {i: c for i, c in enumerate(chars)}
    def decode(toks):
        return ''.join(itos_map.get(int(t), '?') for t in toks)
    print()
    print("=" * 92)
    print("Generated samples (prompt = first 16 chars of seed, temp=0.8)")
    print("Comparing single-pass vs iterative-refinement (refined = output→input loop)")
    print('-' * 92)
    for name, r in results.items():
        sp = decode(r["generated_tokens"])
        rf = decode(r["refined_tokens"])
        sp_cr = compute_creativity_score(sp, full_corpus_text)
        rf_cr = compute_creativity_score(rf, full_corpus_text)
        print(f"\n[{name}]")
        stages = r.get("refinement_stages", {})
        if stages:
            print(f"  Staircase progression (each stage targets next score):")
            for stage_name, stage in stages.items():
                print(f"    {stage_name:<18}  "
                      f"h={stage['harmony']:.4f}  "
                      f"q={stage['quality']:.4f}  "
                      f"c={stage['creativity']:.4f}")
        print(f"  single-pass [c={sp_cr['creativity_score']:.3f}, "
              f"n3={sp_cr['ngram_3']:.3f}, vocab={sp_cr['vocab_overlap']:.3f}]:")
        print(f"    {repr(sp[:160])}")
        print(f"  refined    [c={rf_cr['creativity_score']:.3f}, "
              f"n3={rf_cr['ngram_3']:.3f}, vocab={rf_cr['vocab_overlap']:.3f}]:")
        print(f"    {repr(rf[:160])}")
        # The self-recursive production — the model writing from itself.
        rec_text = r.get("best_production_text", "")
        if rec_text:
            rec_coh  = substrate_coherence_score(rec_text)
            rec_cr   = compute_creativity_score(rec_text, full_corpus_text)
            rec_stru = substrate_structure_score(rec_text)
            rec_disc = substrate_discourse_score(rec_text)  # display only; training uses leaf_positional_match_score
            print(f"\n  ── SELF-RECURSIVE PRODUCTION ──")
            print(f"  [prod={r.get('best_production_score',0):.3f}  "
                  f"coh={rec_coh:.3f}  c={rec_cr['creativity_score']:.3f}  "
                  f"struct={rec_stru:.3f}  disc={rec_disc:.3f}  "
                  f"n3={rec_cr['ngram_3']:.3f}]")
            print(f"    {repr(rec_text[:400])}")
        # Print each stage's output for inspection.
        for stage_name, stage in stages.items():
            stage_text = decode(stage["tokens"])
            print(f"  [{stage_name}] {repr(stage_text[:160])}")

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

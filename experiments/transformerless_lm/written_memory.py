"""written_memory.py — write-don't-train addressed memory, productionized.

The proven recipe (GENERATOR_PLAN.md, 159+ pre-registered runs, 2026-05-30):
  * WRITE knowledge into a datastore instead of training it into weights — a single no-backprop forward
    pass collects (hidden_state -> next_token) pairs. O(1) per item vs gradient descent. (kNN-LM, AM-2/3/5)
  * INDEX the datastore with IVF (k-means Voronoi cells) — the frontier winner: ~99-100% of brute-force
    quality at 400x+ fewer comparisons (AM-6 "Index Done Right"). IVF beat fixed substrate geometry
    (whitened-dodeca ~32x) on these learned-float keys, consistent with the integer-substrate law.
  * The gain is largest on CODE: +48% next-char on a held-out, disjoint code slice vs +14% on prose.

TOKENIZER (BUILD-5): char-level capped generation coherence, so the generator now runs on a pluggable
tokenizer (see bpe_tokenizer.py) — `char` (the proven baseline) or `bpe` (subword units). score()
reports bits-per-char so the two are comparable across different vocabularies.

NOTE: distinct from addressed_memory.py (the 12-face dodecahedral *episodic* memory, 384-entry scale).
This module bundles a small LM (the generator) + an IVF-indexed written datastore, and exposes:
  build / save / load, add_text (incremental write), retrieve, score (LM vs LM+memory, bits/char),
  complete (generate with memory injection), tune_lambda, nearest_contexts (code-RAG provenance).
CPU-friendly, no_grad on every inference path. Shared core for the codemem CLI, the OMC assistant
integration, and the reference for the native Rust primitive (amem_* builtins).
"""
from __future__ import annotations
import sys, json, time, math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from am1_addressed_memory import LM, DenseFFN          # proven char-LM arch (CRT positional encoding)
from corpus import get_batch
from bpe_tokenizer import make_tokenizer, tokenizer_from_dict


@dataclass
class MemConfig:
    d: int = 96
    blocks: int = 3
    seq: int = 128
    ds_keys: int = 400_000
    nlist: int = 1024
    nprobe: int = 4
    kmeans_iters: int = 8
    topk: int = 256
    lam: float = 0.5
    gen_temp: float = 0.8
    tokenizer: str = "bpe"      # "bpe" (subword — better generation, default) | "char" (baseline)
    bpe_vocab: int = 2048
    device: str = "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# IVF index (k-means Voronoi) — vectorized, memory-safe (chunked cdist)
# ─────────────────────────────────────────────────────────────────────────────
def _cdist_argmin(X: torch.Tensor, C: torch.Tensor, ch: int = 20_000) -> torch.Tensor:
    out = torch.empty(X.shape[0], dtype=torch.long)
    for s in range(0, X.shape[0], ch):
        out[s:s + ch] = torch.cdist(X[s:s + ch], C).argmin(1)
    return out


def _kmeans(K: torch.Tensor, nlist: int, iters: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    nlist = min(nlist, K.shape[0])
    C = K[torch.randperm(K.shape[0], generator=g)[:nlist]].clone()
    for _ in range(iters):
        a = _cdist_argmin(K, C)
        C2 = torch.zeros_like(C)
        cnt = torch.zeros(nlist)
        C2.index_add_(0, a, K)
        cnt.index_add_(0, a, torch.ones(K.shape[0]))
        m = cnt > 0
        C2[m] /= cnt[m].unsqueeze(1)
        C2[~m] = C[~m]
        C = C2
    return C


class IVFIndex:
    """Inverted-file index over datastore keys. buckets[cell] -> LongTensor of key rows."""
    def __init__(self, centroids: torch.Tensor, cell_of_key: torch.Tensor):
        self.C = centroids
        self.cell_of_key = cell_of_key
        self._buckets: dict[int, torch.Tensor] = {}
        order = cell_of_key.argsort()
        sc = cell_of_key[order]
        uniq, cnts = torch.unique_consecutive(sc, return_counts=True)
        off = 0
        for c, n in zip(uniq.tolist(), cnts.tolist()):
            self._buckets[c] = order[off:off + n]
            off += n

    @classmethod
    def build(cls, K: torch.Tensor, nlist: int, iters: int, seed: int) -> "IVFIndex":
        C = _kmeans(K, nlist, iters, seed)
        return cls(C, _cdist_argmin(K, C))

    @torch.no_grad()
    def search(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
               nprobe: int, topk: int, vocab: int, qch: int = 128):
        """Each query row -> softmax-weighted next-token distribution from topk neighbours within the
        nprobe nearest cells. Returns (p_knn[Nq,vocab], matched[Nq] bool)."""
        Nq = Q.shape[0]
        nprobe = min(nprobe, self.C.shape[0])
        qcells = torch.cdist(Q, self.C).topk(nprobe, dim=1, largest=False).indices
        run_d = torch.full((Nq, topk), 1e30)
        run_t = torch.zeros((Nq, topk), dtype=torch.long)
        for col in range(nprobe):
            qc = qcells[:, col]
            for cell in torch.unique(qc).tolist():
                keys_c = self._buckets.get(cell)
                if keys_c is None:
                    continue
                qids = (qc == cell).nonzero().flatten()
                if qids.numel() == 0:
                    continue
                Kc, Vc = K[keys_c], V[keys_c]
                kk = min(topk, keys_c.numel())
                for s in range(0, qids.numel(), qch):
                    qi = qids[s:s + qch]
                    dist = torch.cdist(Q[qi], Kc)
                    cd, ci = dist.topk(kk, dim=1, largest=False)
                    allk = torch.cat([run_d[qi], cd], 1)
                    allt = torch.cat([run_t[qi], Vc[ci]], 1)
                    md, mi = allk.topk(topk, dim=1, largest=False)
                    run_d[qi] = md
                    run_t[qi] = torch.gather(allt, 1, mi)
                    del dist, cd, ci, allk, allt
        matched = run_d[:, 0] < 1e29
        w = F.softmax(-run_d, dim=1)
        p_knn = torch.zeros(Nq, vocab)
        p_knn.scatter_add_(1, run_t, w)
        return p_knn, matched


# ─────────────────────────────────────────────────────────────────────────────
# WrittenMemory — the engine
# ─────────────────────────────────────────────────────────────────────────────
class WrittenMemory:
    def __init__(self, cfg: MemConfig, tok, model: LM):
        self.cfg = cfg
        self.tok = tok
        self.vocab = tok.vocab_size
        self.model = model.eval()
        self.K: Optional[torch.Tensor] = None      # [N, d] written hidden states
        self.V: Optional[torch.Tensor] = None      # [N]    next tokens
        self.P: Optional[torch.Tensor] = None      # [N]    source CHAR offset of each entry (provenance)
        self.Ptok: Optional[torch.Tensor] = None   # [N]    source TOKEN index of each entry (span-copy)
        self.source: str = ""
        self.index: Optional[IVFIndex] = None
        self._enc_cache: Optional[torch.Tensor] = None   # corpus token stream (lazy; for span-copy)
        self.copy_threshold: Optional[float] = None      # nearest-distance below which we copy a span

    # ---- tokenization (delegated) ----
    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(self.tok.encode(text), dtype=torch.long)

    def decode(self, ids) -> str:
        return self.tok.decode(ids)

    @torch.no_grad()
    def _hidden_and_logits(self, x: torch.Tensor):
        T = x.shape[1]
        h = self.model.embed(x) + self.model.pe[:T]
        m = self.model.mask[:T, :T]
        for b in self.model.blocks:
            h = b(h, m)
        h = self.model.ln_f(h)
        return self.model.head(h), h

    # ─────────────────────────────────────────────────────────────────────────
    # BUILD: train the generator, then WRITE memory (no backprop), then INDEX
    # ─────────────────────────────────────────────────────────────────────────
    @classmethod
    def build(cls, text: str, cfg: Optional[MemConfig] = None, *,
              train_steps: int = 1500, lr: float = 3e-4, bs: int = 32, seed: int = 0,
              log=print) -> "WrittenMemory":
        cfg = cfg or MemConfig()
        tok = make_tokenizer(cfg.tokenizer, text, vocab_size=cfg.bpe_vocab, log=log)
        ids, char_offs = tok.encode_with_offsets(text)
        enc = torch.tensor(ids, dtype=torch.long)
        offs = torch.tensor(char_offs, dtype=torch.long)
        vocab = tok.vocab_size
        cpt = len(text) / max(1, len(ids))
        log(f"[mem] corpus={len(text):,} chars -> {len(ids):,} {cfg.tokenizer} tokens "
            f"({cpt:.2f} chars/tok)  vocab={vocab}  d={cfg.d} blocks={cfg.blocks} seq={cfg.seq}")

        torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
        model = LM(vocab, cfg.d, cfg.blocks, cfg.seq, lambda: DenseFFN(cfg.d, 4 * cfg.d))
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        model.train()
        t0 = time.time()
        for step in range(train_steps):
            x, y = get_batch(enc, bs, cfg.seq, g)
            loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            if step % max(1, train_steps // 5) == 0:
                log(f"[mem]   train step {step}/{train_steps}  loss={loss.item():.3f}")
        log(f"[mem] generator trained ({time.time()-t0:.0f}s)")

        self = cls(cfg, tok, model)
        self.source = text
        self._write_and_index(enc, offs, seed, log)
        return self

    @torch.no_grad()
    def _write_pass(self, enc: torch.Tensor, char_offs: torch.Tensor, tok_base: int = 0):
        """Sequential sweep over the token stream (full coverage, reproducible). For each window we write
        (hidden_t -> token_{t+1}); provenance = char offset + global token index of the target token."""
        cfg = self.cfg
        seq = cfg.seq
        n = enc.numel()
        max_starts = max(1, cfg.ds_keys // seq)
        stride = max(1, (n - seq - 1) // max_starts)
        starts = list(range(0, n - seq - 1, stride))
        keys, vals, poss, toks = [], [], [], []
        for s in range(0, len(starts), 64):
            bstarts = starts[s:s + 64]
            x = torch.stack([enc[i:i + seq] for i in bstarts])
            y = torch.stack([enc[i + 1:i + seq + 1] for i in bstarts])
            _, h = self._hidden_and_logits(x)
            keys.append(h.reshape(-1, h.shape[-1]))
            vals.append(y.reshape(-1))
            poss.append(torch.stack([char_offs[i + 1:i + seq + 1] for i in bstarts]).reshape(-1))
            # source TOKEN index of each target (for verbatim span-copy at generation)
            toks.append(torch.stack([torch.arange(i + 1, i + seq + 1) + tok_base for i in bstarts]).reshape(-1))
            if sum(k.shape[0] for k in keys) >= cfg.ds_keys:
                break
        K = torch.cat(keys)[:cfg.ds_keys].contiguous()
        V = torch.cat(vals)[:cfg.ds_keys].contiguous()
        P = torch.cat(poss)[:cfg.ds_keys].contiguous()
        Pt = torch.cat(toks)[:cfg.ds_keys].contiguous()
        return K, V, P, Pt

    @torch.no_grad()
    def _write_and_index(self, enc: torch.Tensor, char_offs: torch.Tensor, seed: int, log=print):
        cfg = self.cfg
        self.model.eval()
        t0 = time.time()
        self.K, self.V, self.P, self.Ptok = self._write_pass(enc, char_offs)
        log(f"[mem] wrote {self.K.shape[0]:,} (hidden->token) entries, no backprop ({time.time()-t0:.0f}s)")
        t0 = time.time()
        self.index = IVFIndex.build(self.K, cfg.nlist, cfg.kmeans_iters, seed)
        log(f"[mem] IVF index: {self.index.C.shape[0]} cells over {self.K.shape[0]:,} keys "
            f"({time.time()-t0:.0f}s)")

    @torch.no_grad()
    def add_text(self, text: str, seed: int = 0, log=print):
        """Incrementally WRITE more knowledge and re-index. Uses the existing (fixed) tokenizer."""
        self.model.eval()
        base = len(self.source)
        tok_base = int(self.Ptok.max().item()) + 1 if self.Ptok is not None and self.Ptok.numel() else 0
        self.source = self.source + "\n" + text
        self._enc_cache = None                                  # corpus changed; recompute lazily
        ids, char_offs = self.tok.encode_with_offsets(text)
        enc = torch.tensor(ids, dtype=torch.long)
        offs = torch.tensor(char_offs, dtype=torch.long) + (base + 1)
        addK, addV, addP, addPt = self._write_pass(enc, offs, tok_base=tok_base)
        self.K = torch.cat([self.K, addK]).contiguous()
        self.V = torch.cat([self.V, addV]).contiguous()
        self.P = torch.cat([self.P, addP]).contiguous()
        self.Ptok = torch.cat([self.Ptok, addPt]).contiguous()
        self.index = IVFIndex.build(self.K, self.cfg.nlist, self.cfg.kmeans_iters, seed)
        log(f"[mem] +{addK.shape[0]:,} entries (now {self.K.shape[0]:,}); re-indexed")

    # ─────────────────────────────────────────────────────────────────────────
    # RETRIEVE / provenance
    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def retrieve(self, hidden: torch.Tensor):
        return self.index.search(hidden, self.K, self.V, self.cfg.nprobe, self.cfg.topk, self.vocab)

    @torch.no_grad()
    def nearest_contexts(self, query_text: str, k: int = 5, window: int = 80):
        """code-RAG: nearest written entries to query_text's final hidden state, with source snippets."""
        ctx = self.encode(query_text)[-self.cfg.seq:].unsqueeze(0)
        if ctx.numel() == 0:
            return []
        _, h = self._hidden_and_logits(ctx)
        q = h[0, -1:].reshape(1, -1)
        nprobe = min(self.cfg.nprobe, self.index.C.shape[0])
        qcells = torch.cdist(q, self.index.C).topk(nprobe, dim=1, largest=False).indices[0].tolist()
        present = [c for c in qcells if c in self.index._buckets]
        cand = torch.cat([self.index._buckets[c] for c in present]) if present \
            else torch.arange(self.K.shape[0])
        d = torch.cdist(q, self.K[cand])[0]
        kk = min(k, cand.numel())
        nd, ni = d.topk(kk, largest=False)
        out = []
        for dist, idx in zip(nd.tolist(), ni.tolist()):
            row = int(cand[idx]); pos = int(self.P[row]) if self.P is not None else -1
            snip = self.source[max(0, pos - window):pos + window] if self.source else ""
            out.append(dict(distance=round(dist, 3), position=pos,
                            predicted=self.tok.id2str[int(self.V[row])] if hasattr(self.tok, "id2str") else "?",
                            snippet=snip))
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # SCORE — LM vs LM+memory, reported as token-CE AND bits-per-char (cross-vocab comparable)
    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _windows(self, text: str, max_windows: int):
        enc = self.encode(text)
        seq = self.cfg.seq
        n = min(max_windows, max(1, (enc.numel() - 1) // seq))
        xs = torch.stack([enc[i * seq:(i + 1) * seq] for i in range(n)])
        ys = torch.stack([enc[i * seq + 1:(i + 1) * seq + 1] for i in range(n)])
        return xs, ys

    def _chars_in(self, tgt: torch.Tensor) -> int:
        if hasattr(self.tok, "id2str"):
            return sum(len(self.tok.id2str[int(t)]) for t in tgt) or tgt.numel()
        return tgt.numel()

    @torch.no_grad()
    def score(self, text: str, lam: Optional[float] = None, max_windows: int = 64):
        lam = self.cfg.lam if lam is None else lam
        xs, ys = self._windows(text, max_windows)
        logits, h = self._hidden_and_logits(xs)
        p_lm = F.softmax(logits.reshape(-1, self.vocab), dim=-1)
        tgt = ys.reshape(-1)
        p_knn, _ = self.retrieve(h.reshape(-1, h.shape[-1]))
        nchars = self._chars_in(tgt)
        ln2 = math.log(2)
        lm_nll = F.nll_loss(torch.log(p_lm + 1e-9), tgt, reduction="sum").item()
        mix_nll = F.nll_loss(torch.log((1 - lam) * p_lm + lam * p_knn + 1e-9), tgt, reduction="sum").item()
        return dict(
            lm_loss=lm_nll / tgt.numel(), mem_loss=mix_nll / tgt.numel(),
            lm_bpc=lm_nll / nchars / ln2, mem_bpc=mix_nll / nchars / ln2,
            improve_pct=100.0 * (lm_nll - mix_nll) / lm_nll, lam=lam,
            chars_per_tok=nchars / tgt.numel())

    @torch.no_grad()
    def tune_lambda(self, val_text: str, grid=None, max_windows: int = 64) -> float:
        grid = grid or [i / 20 for i in range(0, 19)]
        xs, ys = self._windows(val_text, max_windows)
        logits, h = self._hidden_and_logits(xs)
        p_lm = F.softmax(logits.reshape(-1, self.vocab), dim=-1)
        tgt = ys.reshape(-1)
        p_knn, _ = self.retrieve(h.reshape(-1, h.shape[-1]))
        best = (1e9, self.cfg.lam)
        for lam in grid:
            L = F.nll_loss(torch.log((1 - lam) * p_lm + lam * p_knn + 1e-9), tgt).item()
            if L < best[0]:
                best = (L, lam)
        self.cfg.lam = best[1]
        return best[1]

    # ─────────────────────────────────────────────────────────────────────────
    # COMPLETE — generate with memory injection
    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def complete(self, prefix: str, n_tokens: int = 200, lam: Optional[float] = None,
                 temp: Optional[float] = None, use_memory: bool = True, greedy: bool = False,
                 seed: int = 0) -> str:
        lam = self.cfg.lam if lam is None else lam
        temp = self.cfg.gen_temp if temp is None else temp
        seq = self.cfg.seq
        g = torch.Generator().manual_seed(seed)
        ids = self.tok.encode(prefix) or [0]
        out = []
        for _ in range(n_tokens):
            ctx = torch.tensor([ids[-seq:]], dtype=torch.long)
            logits, h = self._hidden_and_logits(ctx)
            p_lm = F.softmax(logits[0, -1] / max(temp, 1e-3), dim=-1)
            if use_memory and self.index is not None:
                p_knn, matched = self.retrieve(h[0, -1:].reshape(1, -1))
                p = (1 - lam) * p_lm + lam * p_knn[0] if bool(matched[0]) else p_lm
            else:
                p = p_lm
            nxt = int(p.argmax()) if greedy else int(torch.multinomial(p, 1, generator=g))
            ids.append(nxt); out.append(nxt)
        return prefix + self.decode(out)

    # ─────────────────────────────────────────────────────────────────────────
    # CHUNKED SPAN-COPY — coherence from the corpus, not the tiny generator.
    # When the retrieved context matches confidently, COPY the next `span` tokens verbatim from the
    # corpus at that position; otherwise emit one LM(+memory) token. The LM only has to stitch seams.
    # ─────────────────────────────────────────────────────────────────────────
    def _corpus_tokens(self) -> torch.Tensor:
        if self._enc_cache is None:
            self._enc_cache = torch.tensor(self.tok.encode(self.source), dtype=torch.long)
        return self._enc_cache

    @torch.no_grad()
    def _nearest_row(self, q: torch.Tensor):
        """q [1,d] -> (distance, datastore_row) of the single nearest written key (via IVF cells)."""
        nprobe = min(self.cfg.nprobe, self.index.C.shape[0])
        qcells = torch.cdist(q, self.index.C).topk(nprobe, dim=1, largest=False).indices[0].tolist()
        present = [c for c in qcells if c in self.index._buckets]
        cand = torch.cat([self.index._buckets[c] for c in present]) if present \
            else torch.arange(self.K.shape[0])
        d = torch.cdist(q, self.K[cand])[0]
        i = int(d.argmin())
        return float(d[i]), int(cand[i])

    @torch.no_grad()
    def calibrate_copy(self, holdout_text: str, target_rate: float = 0.6, n_points: int = 200) -> float:
        """Set copy_threshold so ~target_rate of positions on held-out text trigger a span-copy. Higher
        threshold => copy more (more verbatim corpus, more coherent, less novel). Returns the threshold."""
        xs, _ = self._windows(holdout_text, max_windows=8)
        _, h = self._hidden_and_logits(xs)
        hq = h.reshape(-1, h.shape[-1])
        idx = torch.linspace(0, hq.shape[0] - 1, min(n_points, hq.shape[0])).long().tolist()
        dists = sorted(self._nearest_row(hq[i:i + 1])[0] for i in idx)
        self.copy_threshold = dists[min(len(dists) - 1, int(target_rate * len(dists)))]
        return self.copy_threshold

    @torch.no_grad()
    def complete_chunked(self, prefix: str, max_tokens: int = 200, span: int = 8,
                         copy_threshold: Optional[float] = None, lam: Optional[float] = None,
                         temp: Optional[float] = None, seed: int = 0,
                         max_run: int = 24, seam: int = 2, rep_penalty: float = 1.3,
                         mark_seams: bool = False):
        """Generate by COPYING verbatim corpus spans when memory matches, else LM(+memory) token-by-token.

        NOVELTY PRESSURE (so it recombines pieces rather than paste one big chunk):
          * max_run — cap on CONTIGUOUS tokens copied from one source region; on reaching it, force a
            seam (LM tokens), which changes the context so the next match lands on a DIFFERENT region.
          * seam — number of LM tokens to generate at a forced seam.
          * rep_penalty — divide logits of recently-emitted tokens so seams don't loop.
        Returns (text, stats): copy_rate, longest_verbatim_run (max contiguous same-source copy — the
        honest plagiarism signal), distinct_regions (how many corpus places were stitched together)."""
        lam = self.cfg.lam if lam is None else lam
        temp = self.cfg.gen_temp if temp is None else temp
        thr = self.copy_threshold if copy_threshold is None else copy_threshold
        if thr is None:
            thr = float("inf")
        seq = self.cfg.seq
        g = torch.Generator().manual_seed(seed)
        corpus = self._corpus_tokens()
        ids = self.tok.encode(prefix) or [0]
        out, copied, generated = [], 0, 0
        last_end = -1          # source token index just past the last copied span (contiguity tracker)
        run = 0                # current contiguous same-source copy length
        runs, regions, force_lm = [], set(), 0
        seam_ids = []          # token ids that were LM-generated (for seam marking)

        def lm_token(logits, h):
            lv = logits[0, -1].clone()
            for t in set(out[-64:]):                            # repetition penalty (anti-loop at seams)
                lv[t] = lv[t] / rep_penalty if lv[t] > 0 else lv[t] * rep_penalty
            p_lm = F.softmax(lv / max(temp, 1e-3), dim=-1)
            p_knn, matched = self.retrieve(h[0, -1:].reshape(1, -1))
            p = (1 - lam) * p_lm + lam * p_knn[0] if bool(matched[0]) else p_lm
            return int(torch.multinomial(p, 1, generator=g))

        def end_run():
            nonlocal run
            if run > 0:
                runs.append(run); run = 0

        while len(out) < max_tokens:
            logits, h = self._hidden_and_logits(torch.tensor([ids[-seq:]], dtype=torch.long))
            q = h[0, -1:].reshape(1, -1)
            dist, row = self._nearest_row(q)
            j = int(self.Ptok[row]) if self.Ptok is not None else -1
            aligned = 0 <= j < corpus.numel() and int(corpus[j]) == int(self.V[row])
            copy_ok = force_lm == 0 and dist <= thr and aligned
            if copy_ok and run >= max_run:                      # run cap hit -> open a seam, diverge
                force_lm = seam; copy_ok = False
            if copy_ok:
                if j != last_end:                               # new source region -> previous run ended
                    end_run()
                regions.add(j // max(1, span))
                n = 0
                for t in corpus[j:min(j + span, corpus.numel())].tolist():
                    ids.append(t); out.append(t); copied += 1; n += 1
                    if len(out) >= max_tokens:
                        break
                run += n; last_end = j + n
            else:                                               # seam / low-confidence: LM token
                end_run(); last_end = -1
                ids.append(lm_token(logits, h)); out.append(ids[-1]); generated += 1
                seam_ids.append(len(out) - 1)                    # mark this output position as a seam
                force_lm = max(0, force_lm - 1)
        end_run()
        total = copied + generated
        stats = dict(copied=copied, generated=generated, copy_rate=round(copied / max(1, total), 3),
                     longest_verbatim_run=max(runs) if runs else 0, distinct_regions=len(regions))
        if mark_seams:                                          # wrap LM-generated tokens in 〚 ... 〛
            seam_set = set(seam_ids)
            parts = []
            for i, t in enumerate(out):
                s = self.tok.id2str[t] if hasattr(self.tok, "id2str") else self.decode([t])
                parts.append(f"〚{s}〛" if i in seam_set else s)
            return prefix + "".join(parts), stats
        return prefix + self.decode(out), stats

    # ─────────────────────────────────────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────────────────────────────────────
    def save(self, path: str | Path):
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        torch.save({
            "cfg": asdict(self.cfg),
            "tokenizer": self.tok.to_dict(),
            "model": self.model.state_dict(),
            "K": self.K, "V": self.V, "P": self.P, "Ptok": self.Ptok,
            "centroids": self.index.C, "cell_of_key": self.index.cell_of_key,
        }, path / "memory.pt")
        (path / "source.txt").write_text(self.source)
        (path / "meta.json").write_text(json.dumps({
            "keys": int(self.K.shape[0]), "cells": int(self.index.C.shape[0]),
            "vocab": self.vocab, "tokenizer": self.cfg.tokenizer, "cfg": asdict(self.cfg),
        }, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "WrittenMemory":
        path = Path(path)
        blob = torch.load(path / "memory.pt", map_location="cpu", weights_only=False)
        cfg = MemConfig(**blob["cfg"])
        tok = tokenizer_from_dict(blob["tokenizer"])
        model = LM(tok.vocab_size, cfg.d, cfg.blocks, cfg.seq, lambda: DenseFFN(cfg.d, 4 * cfg.d))
        model.load_state_dict(blob["model"]); model.eval()
        self = cls(cfg, tok, model)
        self.K = blob["K"]; self.V = blob["V"]; self.P = blob.get("P"); self.Ptok = blob.get("Ptok")
        src = path / "source.txt"
        self.source = src.read_text() if src.exists() else ""
        self.index = IVFIndex(blob["centroids"], blob["cell_of_key"])
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Self-test / honest A/B: char vs bpe on this module's own source.
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    src = Path(__file__).read_text() * 5
    cut = int(len(src) * 0.85)
    train_text, test_text = src[:cut], src[cut:]
    for kind in ("char", "bpe"):
        cfg = MemConfig(d=64, blocks=2, seq=96, ds_keys=40_000, nlist=128, nprobe=4, topk=64,
                        tokenizer=kind, bpe_vocab=512)
        mem = WrittenMemory.build(train_text, cfg, train_steps=400, seed=0, log=lambda *_: None)
        lam = mem.tune_lambda(train_text[-8000:])
        sc = mem.score(test_text, lam=lam)
        print(f"\n[{kind}]  λ={lam:.2f}  LM bpc={sc['lm_bpc']:.3f} -> +mem bpc={sc['mem_bpc']:.3f}  "
              f"({sc['improve_pct']:+.1f}% token-CE)  chars/tok={sc['chars_per_tok']:.2f}")
        print(f"[{kind}] per-token:  " +
              repr(mem.complete("def ", n_tokens=60, lam=lam, temp=0.6)[:120]))
        if kind == "bpe":
            mem.calibrate_copy(train_text[-8000:], target_rate=0.6)
            txt, st = mem.complete_chunked("def ", max_tokens=60, span=8, lam=lam)
            print(f"[{kind}] span-copy (copy_rate={st['copy_rate']}, thr={mem.copy_threshold:.2f}): "
                  + repr(txt[:160]))

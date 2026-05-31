"""OMC Assistant — addressable computation as a conversational agent.

Everything in one geometric space:
  text   → 648K OMC code windows, retrieved by address proximity
  tool   → omc_eval, omc_search, omc_predict (callable, fingerprinted by docstring)
  memory → conversation turns, each addressed, retrieved by relevance
  weight → skill matrices, addressable, navigable

The model doesn't route to tools explicitly — it navigates to whatever
address is closest to the predicted next fingerprint. If that's a tool, it
calls it. If it's code, it shows it. If it's memory, it references it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from universal_store import UniversalAddressStore, Entry, fingerprint_any
from multiskill_navigator import MultiskillNavigator
from corpus_address_index import substrate_fingerprint
from hierarchical_address import HAddr, assign_address, addr_str


# ─────────────────────────────────────────────────────────────────────────────
# LM generation (optional — needs bloom_checkpoint_model.pt trained on OMC)
# ─────────────────────────────────────────────────────────────────────────────

def _load_lm(ckpt_path: Path):
    """Load a FibRecLMNavigator checkpoint. Returns (model, stoi, itos) or None."""
    try:
        from train_substrate_attention import FibRecLMSubsim
        from train_address_navigator import FibRecLMNavigator
    except ImportError as exc:
        print(f"[LM] import error: {exc}")
        return None

    if not ckpt_path.exists():
        return None

    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    chars    = ckpt['vocab']
    d_model  = ckpt.get('d_model', 64)
    n_blocks = ckpt.get('n_blocks', 4)
    seq_len  = ckpt.get('seq_len', 89)
    K_init   = ckpt.get('K_init', 89)

    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}

    base = FibRecLMSubsim(
        vocab_size=len(chars),
        d_model=d_model,
        n_blocks=n_blocks,
        seq_len=seq_len,
        K=K_init,
        mode="cross",
        K_sig=32,
    )
    model = FibRecLMNavigator(base, d_model)

    # Separate base model state from intent_head state (intent head is optional)
    base_state = {k: v for k, v in ckpt['model_state'].items() if not k.startswith('intent_head.')}
    # strict=False: tolerate newer buffers absent from older checkpoints
    # (e.g. base.bloom_centroid added for Self-Witness — defaults to zeros, inert).
    model.load_state_dict(base_state, strict=False)

    # Load intent_head if checkpoint contains it
    if ckpt.get('has_intent_head'):
        try:
            from train_intent_head import IntentHead
            head_state = {k[len('intent_head.'):]: v
                         for k, v in ckpt['model_state'].items()
                         if k.startswith('intent_head.')}
            vocab_size = len(chars)
            head = IntentHead(d_model, vocab_size)
            head.load_state_dict(head_state)
            head.eval()
            model.intent_head = head
            print(f"  IntentHead loaded ({sum(p.numel() for p in head.parameters())} params)")
        except Exception as exc:
            print(f"  [warn] could not load intent_head: {exc}")

    model.eval()
    return model, stoi, itos, seq_len


@torch.no_grad()
def _generate_from_lm(
    lm_bundle,
    seed: str,
    n_chars: int = 120,
    temperature: float = 0.85,
    store: Optional['UniversalAddressStore'] = None,
    n_hops: int = 2,
    nav: Optional['MultiskillNavigator'] = None,
    dag_seed: str = "",
    intent_fp: Optional[torch.Tensor] = None,
) -> str:
    """Generate OMC code via DAG-seeded, address-navigated generation.

    dag_seed (if provided) is a multi-chunk context assembled by _retrieve_dag().
    It positions the LM *inside real relevant code* before generation starts,
    so the model completes and extends rather than hallucinating from scratch.

    intent_fp (if provided) is a φ-hash of the user's query. If the model has
    an intent_head attribute (after fine-tuning), it biases logits at every step
    toward chars statistically associated with the requested function.

    Each hop: generate chars → navigate to next corpus section via
    MultiskillNavigator → prime next hop from retrieved text.
    """
    model, stoi, itos, seq_len = lm_bundle
    chars_per_hop = max(20, n_chars // max(n_hops, 1))

    # DAG seed positions the LM inside real relevant code (corpus→corpus hops
    # share the same φ-hash length scale, unlike query→corpus which has mismatch)
    effective_seed = dag_seed if dag_seed else seed
    ctx = [stoi.get(c, 0) for c in effective_seed[-seq_len:]]
    if len(ctx) < seq_len:
        ctx = [0] * (seq_len - len(ctx)) + ctx

    output_parts: List[str] = []
    visited: set = set()

    hops = n_hops if (store is not None and store._type_sets.get('text')) else 1
    chars_total = n_chars

    for hop in range(hops):
        this_hop_chars = chars_per_hop if hop < hops - 1 else (chars_total - len(''.join(output_parts)))
        this_hop_chars = max(1, this_hop_chars)

        gen_ids: List[int] = []
        run_ctx = list(ctx)
        for _ in range(this_hop_chars):
            ids    = torch.tensor(run_ctx[-seq_len:], dtype=torch.long).unsqueeze(0)
            logits, addr_emb = model(ids)
            # Intent conditioning: bias logits toward chars matching the user's intent
            if intent_fp is not None and hasattr(model, 'intent_head'):
                bias = model.intent_head(intent_fp)   # [vocab]
                logits = logits + bias.unsqueeze(0).unsqueeze(0)
            probs  = F.softmax(logits[0, -1] / temperature, dim=-1)
            tok    = torch.multinomial(probs, 1).item()
            gen_ids.append(tok)
            run_ctx.append(tok)

        gen_text = ''.join(itos.get(i, '?') for i in gen_ids)
        output_parts.append(gen_text)

        if store is not None and store._type_sets.get('text') and hop < hops - 1:
            # Compute current context fingerprint from the generated text
            ctx_str = ''.join(itos.get(i, '?') for i in run_ctx[-seq_len:])
            ctx_fp  = substrate_fingerprint(ctx_str, model.addr_head.weight.shape[0])

            if nav is not None:
                # MultiskillNavigator: trained for navigation, much better than addr_head
                fp_pred = nav(ctx_fp.unsqueeze(0)).squeeze(0)
            else:
                # LM addr_head fallback (converges poorly, kept for completeness)
                ids_full = torch.tensor(run_ctx[-seq_len:], dtype=torch.long).unsqueeze(0)
                _, addr_emb = model(ids_full)
                fp_pred = addr_emb[0].cpu()

            results = store.retrieve(fp_pred, top_k=8, type_filter='text',
                                     exclude_eids=visited)
            if results:
                best_sim, best_entry = results[0]
                visited.add(best_entry.eid)
                ret_text = best_entry.content if isinstance(best_entry.content, str) else str(best_entry.content)
                nav_tag = 'nav' if nav is not None else 'addr'
                output_parts.append(f'\n/* → {addr_str(best_entry.addr)} sim={best_sim:.3f} [{nav_tag}] */\n')
                ctx = [stoi.get(c, 0) for c in ret_text[-seq_len:]]
                if len(ctx) < seq_len:
                    ctx = [0] * (seq_len - len(ctx)) + ctx
            else:
                ctx = run_ctx

    return ''.join(output_parts)


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Keyword search (φ-hash is length-dependent; short queries need keyword fallback)
# ─────────────────────────────────────────────────────────────────────────────

_CORPUS_PATH = _HERE / 'omc_corpus.txt'

_FNDEFS_PATH = _HERE / 'omc_fndefs.txt'

def _keyword_search(query: str, top_k: int = 5,
                    prefer_fndefs: bool = False) -> List[Tuple[float, str]]:
    """Grep OMC corpus for query terms. Returns [(score, window_text)].

    prefer_fndefs=True (used for synthesis queries): searches omc_fndefs.txt first.
    That file contains clean, complete function definitions (no string literal noise).
    Bare `fn` definitions score +0.5 over quoted occurrences.
    """
    import re, subprocess as sp
    terms = [t for t in re.split(r'\W+', query) if len(t) > 2
             and t.lower() not in {'the','and','for','how','does','work','what',
                                   'show','me','this','that','with','from','into',
                                   'using','can','are','is','in','it','get','let',
                                   'write','generate','create','implement','make',
                                   'function','code','omc'}]
    if not terms:
        return []

    terms_sorted = sorted(terms, key=len, reverse=True)
    results = []
    seen: set = set()

    # Pre-pass: exact function name lookup in omc_fndefs.txt.
    # grep -m 8 on the full file stops at 8 matches; exact `fn term(` may appear
    # later.  Run a targeted grep per term with larger context, then trim to one fn body.
    if prefer_fndefs and _FNDEFS_PATH.exists():
        import subprocess as sp, re as _re2
        def _trim_to_one_fn(lines_in: List[str]) -> str:
            """Keep only the first complete function body (brace-balanced)."""
            depth = 0
            result_lines: List[str] = []
            started = False
            for ln in lines_in:
                if ln == '--':
                    if started and depth == 0:
                        break
                    continue
                result_lines.append(ln.strip())
                for ch in ln:
                    if ch == '{':
                        depth += 1
                        started = True
                    elif ch == '}':
                        depth -= 1
                if started and depth == 0:
                    break
            return '\n'.join(result_lines).strip()

        # Build candidate function names: individual terms + adjacent underscore compounds.
        # "merge sort" → ["merge_sort", "merge", "sort"] so merge_sort(arr) is found first.
        candidate_names: List[str] = []
        for i in range(len(terms_sorted) - 1):
            candidate_names.append(terms_sorted[i] + '_' + terms_sorted[i + 1])
            candidate_names.append(terms_sorted[i + 1] + '_' + terms_sorted[i])
        candidate_names.extend(terms_sorted[:3])

        seen_cands: set = set()
        for cname in candidate_names:
            if cname in seen_cands:
                continue
            seen_cands.add(cname)
            try:
                proc = sp.run(
                    ['grep', '-F', '-m', '2', '-A', '20', f'fn {cname}(', str(_FNDEFS_PATH)],
                    capture_output=True, text=True, timeout=5,
                )
                raw_lines = proc.stdout.splitlines()
                if raw_lines:
                    text = _trim_to_one_fn(raw_lines)
                    if text and text not in seen:
                        seen.add(text)
                        # Compound matches score higher than single-term matches
                        score = 3.0 if '_' in cname else 2.5
                        results.append((score, text))
            except Exception:
                pass

    # Search omc_fndefs.txt first when synthesis is requested — clean fn bodies
    search_paths = []
    if prefer_fndefs and _FNDEFS_PATH.exists():
        search_paths.append((_FNDEFS_PATH, 0.5))   # bonus score for bare defs
    if _CORPUS_PATH.exists():
        search_paths.append((_CORPUS_PATH, 0.0))

    for corpus_file, bonus in search_paths:
        for term in terms_sorted[:3]:
            try:
                proc = sp.run(
                    ['grep', '-F', '-m', '8', '-A', '3', term, str(corpus_file)],
                    capture_output=True, text=True, timeout=5,
                )
                lines = proc.stdout.splitlines()
            except Exception:
                continue
            chunk: List[str] = []
            for line in lines:
                if line == '--':
                    if chunk:
                        text = ' '.join(chunk).strip()
                        if text and text not in seen:
                            seen.add(text)
                            # Penalise lines that look like string literals
                            is_quoted = text.startswith('"') or '\"fn ' in text
                            # Bonus for exact fn name match: fn term( beats fn has_term(
                            import re as _re
                            _fn_name_m = _re.match(r'^fn\s+(\w+)\s*\(', text.lstrip())
                            _exact_name = 1.0 if (_fn_name_m and _fn_name_m.group(1) in terms) else 0.0
                            score = (sum(1 for t in terms if t in text)
                                     / max(len(terms), 1)
                                     + bonus
                                     + _exact_name
                                     - (0.3 if is_quoted else 0.0))
                            results.append((score, text))
                    chunk = []
                else:
                    chunk.append(line.strip())
            if chunk:
                text = ' '.join(chunk).strip()
                if text and text not in seen:
                    seen.add(text)
                    is_quoted = text.startswith('"') or '\"fn ' in text
                    import re as _re
                    _fn_name_m = _re.match(r'^fn\s+(\w+)\s*\(', text.lstrip())
                    _exact_name = 1.0 if (_fn_name_m and _fn_name_m.group(1) in terms) else 0.0
                    score = (sum(1 for t in terms if t in text)
                             / max(len(terms), 1)
                             + bonus
                             + _exact_name
                             - (0.3 if is_quoted else 0.0))
                    results.append((score, text))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# DAG retrieval — builds a connected subgraph of related corpus windows
# ─────────────────────────────────────────────────────────────────────────────

_SYNTHESIS_WORDS = {
    'write', 'generate', 'create', 'implement', 'make', 'build', 'code',
    'give', 'show', 'produce', 'output', 'construct',
}

def _is_synthesis_query(text: str) -> bool:
    """Return True if the user wants code generated (not just retrieved)."""
    words = set(text.lower().split())
    return bool(words & _SYNTHESIS_WORDS)


@torch.no_grad()
def _retrieve_dag(
    query: str,
    store: 'UniversalAddressStore',
    nav: 'MultiskillNavigator',
    d_model: int,
    n_hops: int = 4,
    top_k_seed: int = 2,
) -> str:
    """DAG-navigated retrieval: keyword anchor → navigator hops → assembled context.

    The key advantage over raw φ-hash retrieval: we start from exact keyword
    matches (89-char corpus windows), then navigate window→window at the same
    φ-hash length scale.  Corpus→corpus hops share the same fingerprint space,
    so navigator sims are high (~0.99) rather than the ~0.013 we get when
    fingerprinting a short user query against 89-char indexed windows.

    The assembled result is passed as dag_seed to _generate_from_lm, positioning
    the LM inside real relevant code before it generates a single character.
    """
    kw_hits = _keyword_search(query, top_k=top_k_seed, prefer_fndefs=True)
    if not kw_hits:
        return ""

    visited: set = set()
    anchors: List[str] = []    # keyword hits — most relevant, placed last
    hops:    List[str] = []    # navigator hops — context, placed before anchors

    for _, anchor_text in kw_hits[:top_k_seed]:
        if anchor_text in visited:
            continue
        visited.add(anchor_text)
        anchors.append(anchor_text)

        # Navigate outward from the anchor — corpus→corpus, same length scale
        ctx_fp = substrate_fingerprint(anchor_text, d_model)
        for _ in range(n_hops):
            fp_next = nav(ctx_fp.unsqueeze(0)).squeeze(0)
            hits = store.retrieve(fp_next, top_k=4, type_filter='text')
            if not hits:
                break
            advanced = False
            for sim, entry in hits:
                text = entry.content if isinstance(entry.content, str) else str(entry.content)
                if text not in visited:
                    visited.add(text)
                    hops.append(text)
                    ctx_fp = substrate_fingerprint(text, d_model)
                    advanced = True
                    break
            if not advanced:
                break

    # Anchors go LAST: LM context = effective_seed[-seq_len:] = tail of assembly.
    # This positions the LM inside the anchor (most relevant code), not inside
    # a random hop.  Hops provide surrounding context that the LM has "read".
    return '\n'.join(hops + anchors)


# ─────────────────────────────────────────────────────────────────────────────
# Tool stubs
# ─────────────────────────────────────────────────────────────────────────────

def _tool_search(query: str, store: Optional[UniversalAddressStore] = None) -> str:
    """Search OMC codebase by keyword + address proximity. Returns top code snippets."""
    lines = [f"omc_search({repr(query[:40])})"]

    # Keyword results first (reliable for function names, symbols)
    kw = _keyword_search(query, top_k=4)
    if kw:
        lines.append(f"  keyword hits ({len(kw)}):")
        for score, snippet in kw:
            lines.append(f"    [{score:.2f}] {repr(snippet[:90])}")

    # φ-hash address results (structural similarity)
    if store:
        fp = fingerprint_any(query)
        addr_results = store.retrieve(fp, top_k=3, type_filter='text')
        good = [(s, e) for s, e in addr_results if s > 0.3]
        if good:
            lines.append(f"  address hits ({len(good)}):")
            for sim, entry in good:
                snippet = entry.content.replace('\n', ' ')[:80]
                lines.append(f"    [{addr_str(entry.addr)}] sim={sim:.3f} {repr(snippet)}")

    if len(lines) == 1:
        lines.append("  no results found")
    return '\n'.join(lines)


def _tool_explain(code: str, store: Optional[UniversalAddressStore] = None) -> str:
    """Explain what a piece of OMC code does by retrieving similar patterns from store."""
    if store is None:
        return "[omc_explain] store not available"
    fp = fingerprint_any(code)
    results = store.retrieve(fp, top_k=4, type_filter='text')
    if not results:
        return f"[omc_explain] no similar patterns found for: {repr(code[:60])}"
    lines = [f"omc_explain — {len(results)} similar patterns:"]
    for sim, entry in results:
        snippet = entry.content.replace('\n', ' ')[:100]
        lines.append(f"  [{addr_str(entry.addr)}] sim={sim:.3f}  {repr(snippet)}")
    return '\n'.join(lines)


def _tool_eval(code: str) -> str:
    """Execute OMC code using the omc binary. Returns stdout/stderr output."""
    # Try importing from omc_tools if available
    try:
        from omc_tools import omc_eval  # type: ignore
        return omc_eval(code)
    except ImportError:
        pass

    # Try finding the omc binary
    omc_bin = _HERE.parent.parent / "target" / "release" / "omc"
    if not omc_bin.exists():
        omc_bin_debug = _HERE.parent.parent / "target" / "debug" / "omc"
        if omc_bin_debug.exists():
            omc_bin = omc_bin_debug
        else:
            return "[omc_eval] omc binary not found — build with `cargo build --release`"

    try:
        result = subprocess.run(
            [str(omc_bin)],
            input=code,
            capture_output=True,
            text=True,
            timeout=10,
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        if result.returncode != 0:
            return f"[omc_eval] error (rc={result.returncode}):\n{err}"
        return out if out else "[omc_eval] (no output)"
    except subprocess.TimeoutExpired:
        return "[omc_eval] timeout (10s)"
    except Exception as exc:
        return f"[omc_eval] exception: {exc}"


def _tool_predict(text: str, nav: Optional[MultiskillNavigator] = None) -> str:
    """Predict the next address for a text fragment using the navigator's skill routing."""
    if nav is None:
        return "[omc_predict] navigator not available"
    with torch.no_grad():
        fp = fingerprint_any(text)
        fp_pred = nav(fp.unsqueeze(0)).squeeze(0)
        addr = assign_address(fp_pred)
        sw = nav.skill_weights(fp).tolist()
        dom = nav.dominant_skill(fp)
        sw_str = "  ".join(f"s{i}={w:.2f}" for i, w in enumerate(sw))
        return (
            f"[omc_predict] predicted addr={addr_str(addr)}\n"
            f"  skill weights: {sw_str}  (dominant: skill {dom})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Written-memory recall (the proven write-don't-train + IVF datastore; see written_memory.py)
# ─────────────────────────────────────────────────────────────────────────────
import os as _os

_WM_CACHE: dict = {}          # store_path -> WrittenMemory (load once)


def _find_wm_store() -> Optional[Path]:
    """Locate a codemem store: $CODEMEM_STORE, then repo .codemem, then /tmp/omc_core.codemem."""
    cands = []
    if _os.environ.get("CODEMEM_STORE"):
        cands.append(Path(_os.environ["CODEMEM_STORE"]))
    cands += [_HERE / ".codemem", _HERE.parent.parent / ".codemem",
              Path("/tmp/omc_core_bpe.codemem"), Path("/tmp/omc_core.codemem")]
    return next((p for p in cands if (p / "memory.pt").exists()), None)


def _get_wm(store_path: Optional[Path] = None):
    """Lazily load (and cache) the WrittenMemory store. Returns None if unavailable."""
    store_path = store_path or _find_wm_store()
    if store_path is None:
        return None
    key = str(store_path)
    if key not in _WM_CACHE:
        try:
            from written_memory import WrittenMemory
            _WM_CACHE[key] = WrittenMemory.load(store_path)
        except Exception as e:
            print(f"[Init] WrittenMemory load failed ({store_path}): {e}", flush=True)
            _WM_CACHE[key] = None
    return _WM_CACHE[key]


def _tool_recall(query: str) -> str:
    """Recall real code from the written-memory datastore by hidden-state proximity, WITH provenance
    (source position + the char the context predicts). The strong, proven retrieval path."""
    wm = _get_wm()
    if wm is None:
        return ("[omc_recall] no written-memory store found. Build one:\n"
                "  python codemem.py build <repo> -o .codemem")
    hits = wm.nearest_contexts(query, k=5)
    if not hits:
        return f"[omc_recall] no contexts found for {query!r}"
    lines = [f"omc_recall({query[:50]!r}) — {len(hits)} written contexts ({wm.K.shape[0]:,} entries):"]
    for i, r in enumerate(hits, 1):
        snip = r["snippet"].strip().replace("\n", " ")[:110]
        lines.append(f"  [{i}] dist={r['distance']:.3f} pos={r['position']} →{r['predicted']!r}  {snip}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# OMCAssistant
# ─────────────────────────────────────────────────────────────────────────────

class OMCAssistant:
    """Interactive OMC assistant. Everything lives in one geometric address space.

    On each turn:
      1. Fingerprint user input → store as memory turn
      2. Navigator predicts the next address (fp_pred)
      3. Retrieve: top-3 text entries, top-2 memory entries, top-1 tool
      4. If best tool similarity > 0.3: call it
      5. Compose formatted response
      6. Store assistant response as memory
    """

    def __init__(
        self,
        store: UniversalAddressStore,
        navigator: MultiskillNavigator,
        d_model: int = 64,
        lm_bundle=None,       # optional (model, stoi, itos, seq_len) for char generation
        gen_chars: int = 120, # how many chars to generate per response
        name_reg=None,        # optional name→address registry (O(1) lookup)
    ):
        self.store = store
        self.nav = navigator
        self.d_model = d_model
        self.turn = 0
        self._tool_eids: List[str] = []   # eids of registered tool entries
        self.lm_bundle = lm_bundle
        self.gen_chars = gen_chars
        self.name_reg = name_reg
        # locality-fp retrieval index (ADDR-1/3): built lazily from store text entries
        self._loc_mat = None        # [N, D] normalized char-bigram histograms
        self._loc_eids: List[str] = []
        self._loc_dim = 1024        # hashed bigram dim
        self._loc_built = False

    def _loc_fp(self, text: str) -> 'torch.Tensor':
        """Char-bigram histogram (hashed to _loc_dim), normalized. Locality-bearing:
        similar content → similar fp (unlike φ-hash). Universal, no word list."""
        D = self._loc_dim
        h = torch.zeros(D)
        b = text.encode('utf-8', 'ignore')
        if len(b) > 1:
            a = torch.tensor(list(b[:-1])); c = torch.tensor(list(b[1:]))
            idx = (a * 257 + c) % D
            h.scatter_add_(0, idx, torch.ones(len(idx)))
        n = h.norm()
        return h / n if n > 0 else h

    def _ensure_locality_index(self):
        if self._loc_built:
            return
        self._loc_built = True
        eids = list(self.store._type_sets.get('text', []))
        if not eids:
            return
        import time as _t; t0 = _t.time()
        print(f"[Init] building locality-fp index over {len(eids):,} windows ...", flush=True)
        rows = []
        for e in eids:
            ent = self.store._entries.get(e)
            txt = ent.content if (ent and isinstance(ent.content, str)) else ''
            rows.append(self._loc_fp(txt))
        self._loc_mat = torch.stack(rows)          # already normalized
        self._loc_eids = eids
        print(f"[Init] locality index built in {_t.time()-t0:.1f}s "
              f"({self._loc_mat.shape[0]}×{self._loc_dim})", flush=True)

    def _locality_retrieve(self, query: str, top_k: int = 3):
        """Content-relevant retrieval via locality fp. Returns [(sim, Entry)] or None."""
        self._ensure_locality_index()
        if self._loc_mat is None:
            return None
        import torch.nn.functional as _F
        q = _F.normalize(self._loc_fp(query).unsqueeze(0), dim=1)
        sims = (self._loc_mat @ q.T).squeeze(1)
        out = []
        for i in sims.argsort(descending=True)[:top_k].tolist():
            out.append((float(sims[i]), self.store._entries[self._loc_eids[i]]))
        return out

    # ── Tool registration ─────────────────────────────────────────────────────

    def register_tool(self, fn, **meta) -> Entry:
        """Fingerprint a tool callable by its docstring+signature and store it."""
        entry = self.store.store(fn, type='tool', **meta)
        self._tool_eids.append(entry.eid)
        return entry

    def register_weight(self, W: torch.Tensor, name: str) -> Entry:
        """Store a weight matrix as a navigable address entry."""
        return self.store.store(W, type='weight', name=name)

    # ── Core response ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def respond(self, user_input: str) -> str:
        self.turn += 1

        # ── 1. Fingerprint + store user turn ──────────────────────────────────
        fp_user = fingerprint_any(user_input, self.d_model)
        self.store.store(
            user_input,
            type='memory',
            fp=fp_user,
            turn=self.turn,
            role='user',
        )

        # ── 2. Predict next address via navigator ─────────────────────────────
        fp_pred = self.nav(fp_user.unsqueeze(0)).squeeze(0)
        pred_addr = assign_address(fp_pred)

        # ── 3. Retrieve relevant context ──────────────────────────────────────

        # ── Name registry: O(1) lookup, threshold-free, corpus-agnostic ─────────
        # Try name registry first — no similarity thresholds, no heuristics.
        # φ(name) → dodecahedral address directly; pure substrate, no table.
        _reg_hit = None
        if self.name_reg is not None:
            from name_registry import lookup_terms as _reg_lookup
            _reg_hit = _reg_lookup(self.name_reg, user_input)

        # Code retrieval: LOCALITY-fp (ADDR-3 fix). φ-cosine similarity retrieval was
        # measured ≈ random (recall@5 0.016 vs 0.008); a char-histogram locality fp gives
        # 15× better content relevance. Use locality retrieval; φ stays for keys/buckets.
        if self.store._type_sets.get('text'):
            text_hits = self._locality_retrieve(user_input, top_k=3)
            if text_hits is None:   # locality index unavailable → fall back to φ
                text_hits = self.store.retrieve(fp_user, top_k=3, type_filter='text')
        else:
            text_hits = []
        # Keyword fallback — only when no registry hit and φ-hash weak
        _synthesis_early = _is_synthesis_query(user_input) or len(user_input.split()) <= 5
        kw_hits = (
            _keyword_search(user_input, top_k=3, prefer_fndefs=_synthesis_early)
            if _reg_hit is None and not any(s > 0.7 for s, _ in text_hits) else []
        )

        # Memory retrieval
        _recall_kws = {'first', 'earlier', 'before', 'previous', 'last', 'said', 'asked', 'mentioned'}
        _might_recall = any(w in user_input.lower() for w in _recall_kws)
        latest_mem_eid = f"memory_{self.store._counters.get('memory', 1) - 1:06d}"

        if _might_recall:
            # For recall queries: use ALL stored memory entries (sorted later in compose)
            all_eids = self.store._type_sets.get('memory', set()) - {latest_mem_eid}
            mem_hits = [
                (0.0, self.store._entries[eid])
                for eid in all_eids
                if eid in self.store._entries
                and self.store._entries[eid].meta.get('turn', self.turn) < self.turn
            ]
        else:
            mem_hits = self.store.retrieve(
                fp_user, top_k=4, type_filter='memory',
                exclude_eids={latest_mem_eid},
            )
            mem_hits = [
                (sim, e) for sim, e in mem_hits
                if e.meta.get('turn', self.turn) < self.turn
            ][:2]

        # Top-1 tool match
        tool_hits = self.store.retrieve(fp_pred, top_k=1, type_filter='tool')

        # ── 4. Tool dispatch ──────────────────────────────────────────────────
        # Primary: address geometry routes to tool (sim > 0.3).
        # Fallback: explicit intent prefixes ("run:", "eval:", "search:").
        _lower = user_input.lower().lstrip()
        _forced_tool: Optional[str] = None
        if _lower.startswith(('run:', 'run this:', 'eval:', 'exec:')):
            _forced_tool = 'tool_eval'
        elif _lower.startswith(('search:', 'find:', 'look up:')):
            _forced_tool = 'tool_search'
        elif _lower.startswith(('complete:', 'predict:', 'finish:')):
            _forced_tool = 'tool_complete'

        tool_result: Optional[str] = None
        tool_name:   Optional[str] = None

        # Try forced tool first
        if _forced_tool:
            for _eid in self._tool_eids:
                _e = self.store._entries[_eid]
                if callable(_e.content) and _e.content.__name__ == _forced_tool:
                    code_arg = user_input.split(':', 1)[1].strip()
                    try:
                        tool_result = _e.content(code_arg)
                        tool_name = _forced_tool
                    except Exception as exc:
                        tool_result = f"[tool error: {exc}]"
                    break

        if not tool_result and tool_hits:
            tool_sim, tool_entry = tool_hits[0]
            if tool_sim > 0.3 and callable(tool_entry.content):
                tool_name = tool_entry.content.__name__
                fn = tool_entry.content
                # Pass user_input as argument; tool functions accept (query, store=...)
                try:
                    import inspect as _inspect
                    sig = _inspect.signature(fn)
                    params = list(sig.parameters)
                    if 'store' in params:
                        tool_result = fn(user_input, store=self.store)
                    elif 'nav' in params:
                        tool_result = fn(user_input, nav=self.nav)
                    else:
                        tool_result = fn(user_input)
                except Exception as exc:
                    tool_result = f"[tool error: {exc}]"

        # ── 4b. φ-template synthesis (novel functions, corpus-independent) ──────
        phi_gen: Optional[str] = None
        phi_category: Optional[str] = None
        if (_is_synthesis_query(user_input) or len(user_input.split()) <= 5) and not tool_result:
            try:
                from phi_synthesis import synthesize as _phi_synthesize
                _phi_result = _phi_synthesize(user_input, self.d_model)
                if _phi_result:
                    phi_gen, phi_category, _phi_score = _phi_result
            except Exception:
                pass

        # ── 4c. LM generation (if char model loaded) ─────────────────────────
        lm_gen: Optional[str] = None
        if self.lm_bundle is not None and not tool_result:
            _synthesis = _is_synthesis_query(user_input)

            # Build DAG seed (corpus→corpus navigator hops from keyword anchor)
            dag_seed = ""
            if _synthesis and self.store and self.nav:
                dag_seed = _retrieve_dag(
                    user_input, self.store, self.nav, self.d_model, n_hops=4
                )

            # For synthesis: use the best kw_hit (bare fn def) directly as LM seed.
            # This positions the LM at the START of a real function rather than the
            # middle of the DAG assembly.  The LM then extends from that point.
            if _synthesis and kw_hits:
                best_kw = kw_hits[0][1]
                # Strip leading noise (quoted fragment) — take from first `fn ` if present
                fn_idx = best_kw.find('fn ')
                seed = best_kw[fn_idx:fn_idx + 89] if fn_idx >= 0 else best_kw[:89]
            elif kw_hits:
                seed = kw_hits[0][1][:89]
            elif text_hits and text_hits[0][0] > 0.0:
                best_entry = text_hits[0][1]
                seed = (best_entry.content if isinstance(best_entry.content, str) else str(best_entry.content))[:89]
            else:
                seed = user_input[:89]

            try:
                lm_gen = _generate_from_lm(
                    self.lm_bundle, seed,
                    n_chars=self.gen_chars,
                    store=self.store,
                    n_hops=2,
                    nav=self.nav,
                    dag_seed=dag_seed,
                    intent_fp=fp_user,
                )
            except Exception as exc:
                lm_gen = f"[lm error: {exc}]"

        # ── 5. Compose response ───────────────────────────────────────────────
        # Lead with the answer; diagnostics go below a separator.
        response_body = _compose_response(
            user_input, text_hits, mem_hits, tool_result, kw_hits, lm_gen,
            phi_gen=phi_gen, phi_category=phi_category,
            reg_hit=_reg_hit,
        )

        # Diagnostic footer
        sw = self.nav.skill_weights(fp_user).tolist()
        dom = self.nav.dominant_skill(fp_user)
        sw_str = "  ".join(f"s{i}={w:.2f}" for i, w in enumerate(sw))
        diag: List[str] = ["─" * 60]
        diag.append(
            f"turn={self.turn}  addr={addr_str(pred_addr)}"
            f"  sk={dom}  [{sw_str}]"
        )
        if kw_hits:
            diag.append(f"ctx: {len(kw_hits)} keyword hits")
        elif text_hits:
            diag.append(f"ctx: {len(text_hits)} addr hits  best sim={text_hits[0][0]:.3f}")
        if mem_hits:
            diag.append(f"mem: {len(mem_hits)} related turns")
        if tool_result:
            diag.append(f"tool: {tool_name}")
        if lm_gen:
            dag_tag = f"  dag={len(dag_seed)}chars" if dag_seed else ""
            diag.append(f"gen: {len(lm_gen)} chars{dag_tag}")
        if phi_gen:
            diag.append(f"phi: template={phi_category}")

        response_text = response_body + "\n" + "\n".join(diag)

        # ── 6. Store assistant response as memory ─────────────────────────────
        self.store.store(
            response_body,
            type='memory',
            turn=self.turn,
            role='assistant',
        )

        return response_text


def _compose_response(
    user_input: str,
    text_hits: List[Tuple[float, Entry]],
    mem_hits: List[Tuple[float, Entry]],
    tool_result: Optional[str],
    kw_hits: Optional[List[Tuple[float, str]]] = None,
    lm_gen: Optional[str] = None,
    phi_gen: Optional[str] = None,
    phi_category: Optional[str] = None,
    reg_hit: Optional[dict] = None,
) -> str:
    """Assemble a plain-text response from retrieved pieces."""
    parts: List[str] = []

    # ── Name registry: O(1), threshold-free, corpus-agnostic ─────────────────
    # φ(name) → address → function body; no similarity scoring needed.
    if reg_hit and not tool_result:
        addr = reg_hit['addr']
        parts.append(
            f"Retrieved from name registry  "
            f"[addr=({addr.face},{addr.sub_face},φ({reg_hit['name']}))]:"
        )
        parts.append("")
        parts.append(reg_hit['code'].strip())
        parts.append("")
        return '\n'.join(parts)

    # If there's a direct tool result, lead with it
    if tool_result and '[error' not in tool_result.lower():
        parts.append(tool_result)
        parts.append("")

    # Recall-questions: check conversation memory first, suppress code context
    _recall_keywords = {'first', 'earlier', 'before', 'previous', 'last', 'said', 'asked', 'mentioned'}
    _ui_lower = user_input.lower()
    _is_recall_query = any(w in _ui_lower for w in _recall_keywords)

    if _is_recall_query and mem_hits:
        # Reconstruct all memory entries and sort by turn for temporal queries.
        # Retrieval by similarity alone can miss the chronologically first/last turn.
        _all_mem = list(mem_hits)  # sim-sorted candidates
        _want_role = 'user' if any(w in _ui_lower for w in {'asked', 'ask', 'said'}) else None
        _filtered = [(s, e) for s, e in _all_mem
                     if _want_role is None or e.meta.get('role') == _want_role]
        _candidates = _filtered or _all_mem
        if any(w in _ui_lower for w in {'first', 'earliest', 'beginning', 'start'}):
            _candidates = sorted(_candidates, key=lambda x: x[1].meta.get('turn', 9999))
        elif any(w in _ui_lower for w in {'last', 'latest', 'recent', 'previous', 'before'}):
            _candidates = sorted(_candidates, key=lambda x: x[1].meta.get('turn', 0), reverse=True)
        best_mem_sim, best_mem_entry = _candidates[0]
        role = best_mem_entry.meta.get('role', '?')
        turn_n = best_mem_entry.meta.get('turn', '?')
        raw_mem = best_mem_entry.content if isinstance(best_mem_entry.content, str) else str(best_mem_entry.content)
        parts.append(f"From turn {turn_n} ({role}):")
        parts.append(f"  {raw_mem[:200]}")
        return '\n'.join(parts)

    # Synthesis queries: lead with retrieved function definitions (bare fn defs
    # from omc_fndefs.txt score higher and contain runnable code).  Show them
    # first, then the LM extension below — this is retrieval-augmented synthesis.
    _synthesis_response = _is_synthesis_query(user_input) or (
        len(user_input.split()) <= 5 and not _is_recall_query
    )
    if _synthesis_response and kw_hits and not _is_recall_query:
        # Find the best bare function definition hit — prefer non-test/non-bench functions
        import re as _re_sel
        def _is_test_snippet(s: str) -> bool:
            m = _re_sel.match(r'^fn\s+(\w+)', s.lstrip())
            if not m: return False
            return any(pfx in m.group(1)
                       for pfx in ('test_', 'bench_', 'assert_', 'is_', 'check_'))
        best_fn_hit: Optional[str] = None
        for _, snippet in kw_hits:
            if snippet.lstrip().startswith('fn ') and not _is_test_snippet(snippet):
                best_fn_hit = snippet
                break
        if best_fn_hit is None:
            for _, snippet in kw_hits:
                if snippet.lstrip().startswith('fn '):
                    best_fn_hit = snippet
                    break
        if best_fn_hit is None:
            for _, snippet in kw_hits:
                fi = snippet.find('fn ')
                if fi >= 0:
                    best_fn_hit = snippet[fi:]
                    break
        if best_fn_hit:
            # Resolve helper dependencies: find function calls inside the body
            # that aren't OMC builtins and retrieve their definitions.
            import re as _re3, subprocess as _sp3
            _BUILTINS = {
                'arr_get','arr_set','arr_len','arr_new','arr_push','arr_pop',
                'arr_range','arr_dot','arr_matmul','arr_map','arr_sum',
                'print','str_len','str_slice','str_get','to_float','to_int',
                'exp','log','sqrt','sin','cos','abs','floor','ceil',
                'assert_eq','assert','type_of','file_open','file_read','file_close',
                'py_call','yield','return','if','while','else',
            }
            # Extract this function's own name so we don't retrieve it again
            _own_name_m = _re3.match(r'^fn\s+(\w+)', best_fn_hit.lstrip())
            _own_name = _own_name_m.group(1) if _own_name_m else ''
            _calls = set(_re3.findall(r'\b([a-zA-Z_]\w*)\s*\(', best_fn_hit))
            _helpers_needed = [c for c in _calls
                               if c not in _BUILTINS and c != _own_name]
            _helper_blocks: List[str] = []
            if (_helpers_needed or True) and _FNDEFS_PATH.exists():
                def _trim_one(raw_lines: List[str]) -> str:
                    depth = 0; buf: List[str] = []; started = False
                    for ln in raw_lines:
                        if ln == '--':
                            if started and depth == 0: break
                            continue
                        buf.append(ln.strip())
                        for ch in ln:
                            if ch == '{': depth += 1; started = True
                            elif ch == '}': depth -= 1
                        if started and depth == 0: break
                    return '\n'.join(buf).strip()

                def _fetch_fn(name: str) -> str:
                    try:
                        proc = _sp3.run(
                            ['grep', '-F', '-m', '1', '-A', '40', f'fn {name}(', str(_FNDEFS_PATH)],
                            capture_output=True, text=True, timeout=3,
                        )
                        return _trim_one(proc.stdout.splitlines()) if proc.stdout else ''
                    except Exception:
                        return ''

                # BFS: resolve helpers transitively up to depth 2
                resolved: set = {_own_name}
                queue = list(_helpers_needed)
                for _ in range(2):   # max 2 levels of transitive helpers
                    next_queue: List[str] = []
                    for helper in queue:
                        if helper in resolved:
                            continue
                        resolved.add(helper)
                        blk = _fetch_fn(helper)
                        if blk and blk not in best_fn_hit:
                            _helper_blocks.append(blk)
                            # Find this helper's own dependencies
                            sub_calls = set(_re3.findall(r'\b([a-zA-Z_]\w*)\s*\(', blk))
                            next_queue.extend(
                                c for c in sub_calls
                                if c not in _BUILTINS and c not in resolved
                            )
                    queue = next_queue

            # Check that the retrieved function name actually relates to query terms
            _retrieved_name_m = _re3.match(r'^fn\s+(\w+)', best_fn_hit.lstrip())
            _retrieved_name = _retrieved_name_m.group(1).lower() if _retrieved_name_m else ''
            _query_content_words = {w.lower() for w in _re3.split(r'\W+', user_input)
                                    if len(w) > 2 and w.lower() not in
                                    {'write','generate','create','implement','make',
                                     'build','code','give','show','the','for','and','a','an'}}
            # A name is relevant if any query content word appears in the fn name
            # AND the fn name isn't clearly a test/bench/helper variant
            _is_test_fn = any(pfx in _retrieved_name
                              for pfx in ('test_', 'bench_', 'assert_', 'is_', 'check_'))
            _name_relevant = (
                not _is_test_fn and
                any(w in _retrieved_name or _retrieved_name in w
                    for w in _query_content_words)
            )

            if _name_relevant:
                parts.append("Retrieved function:")
                parts.append("")
                if _helper_blocks:
                    parts.append("# Helper functions:")
                    for hb in _helper_blocks:
                        parts.append(hb)
                        parts.append("")
                    parts.append("# Main function:")
                parts.append(best_fn_hit.strip())
                parts.append("")
                _fn_complete = best_fn_hit.rstrip().endswith('}')
                if lm_gen and not tool_result and not _fn_complete:
                    parts.append("LM extension:")
                    parts.append("")
                    parts.append(lm_gen)
            elif phi_gen and not tool_result:
                # Corpus hit unrelated to query — prefer φ-template synthesis
                parts.append(f"φ-Synthesized function (template: {phi_category}):")
                parts.append("")
                parts.append(phi_gen.replace('{{', '{').replace('}}', '}'))
                parts.append("")
            else:
                # Show corpus hit anyway (best we have)
                parts.append("Retrieved function:")
                parts.append("")
                parts.append(best_fn_hit.strip())
                parts.append("")
        elif phi_gen and not tool_result:
            # No corpus fn found — use φ-template synthesis (derivable, not retrieved)
            parts.append(f"φ-Synthesized function (template: {phi_category}):")
            parts.append("")
            parts.append(phi_gen.replace('{{', '{').replace('}}', '}'))
            parts.append("")
        elif lm_gen and not tool_result:
            # Final fallback: LM generation
            parts.append("LM generation:")
            parts.append("")
            parts.append(lm_gen)
        return '\n'.join(parts)

    # LM generation takes priority when no tool fired (and not a recall query)
    if lm_gen and not tool_result and not _is_recall_query:
        parts.append("Generated continuation:")
        parts.append("")
        parts.append(lm_gen)
        parts.append("")

    # Keyword results (non-synthesis)
    if kw_hits and not _is_recall_query:
        parts.append("Matching OMC code:")
        parts.append("")
        for score, snippet in kw_hits[:3]:
            parts.append(f"  {snippet}")
        return '\n'.join(parts)

    # Synthesize from code context
    if text_hits:
        best_sim, best_entry = text_hits[0]
        raw_content = best_entry.content if isinstance(best_entry.content, str) else str(best_entry.content)
        snippet = raw_content.strip()
        if best_sim > 0.85:
            parts.append(f"High-confidence match (sim={best_sim:.3f}) at {addr_str(best_entry.addr)}:")
            parts.append("")
            parts.append(snippet)
        elif best_sim > 0.5:
            parts.append(f"Related OMC code (sim={best_sim:.3f}) at {addr_str(best_entry.addr)}:")
            parts.append("")
            # First 5 lines of the snippet
            snippet_lines = snippet.splitlines()
            parts.extend(snippet_lines[:5])
            if len(snippet_lines) > 5:
                parts.append(f"  ... ({len(snippet_lines) - 5} more lines)")
        else:
            parts.append(f"Nearest code pattern (sim={best_sim:.3f}):")
            parts.append(snippet.splitlines()[0] if snippet else "(empty)")

    # Reference memory if relevant (not a recall query — handled above)
    if mem_hits:
        best_mem_sim, best_mem_entry = mem_hits[0]
        if best_mem_sim > 0.5:
            role = best_mem_entry.meta.get('role', '?')
            turn_n = best_mem_entry.meta.get('turn', '?')
            raw_mem = best_mem_entry.content if isinstance(best_mem_entry.content, str) else str(best_mem_entry.content)
            parts.append("")
            parts.append(f"(Related to turn {turn_n}: {repr(raw_mem[:80])})")

    if not parts:
        parts.append(
            f"Input fingerprinted and addressed. Query: {repr(user_input[:60])}\n"
            "No high-confidence context retrieved. Try loading the full index."
        )

    return '\n'.join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Index loading
# ─────────────────────────────────────────────────────────────────────────────

def _bulk_insert_dual_index(store: UniversalAddressStore, dual_index: dict) -> int:
    """Fast-path bulk insert fingerprints from a pre-built dual index.

    Bypasses the per-entry Python loop in store.store() by writing directly
    into the store's internal dicts. About 30× faster than calling store() per window.
    """
    windows = dual_index['windows']
    meta = dual_index.get('meta', {})
    # Use the longest scale (most informative fingerprints)
    long_scale = meta.get('seq_len', 89)
    scales_dict = dual_index['scales']
    if long_scale not in scales_dict:
        long_scale = max(scales_dict.keys())
    sub89 = scales_dict[long_scale]
    vecs: torch.Tensor = sub89['vecs']           # [N, 64]
    addrs_list = sub89['addrs']                   # List[HAddr]
    seq_index = dual_index['seq_index']           # List[int]

    N = len(windows)
    type_str = 'text'
    base = store._counters.get(type_str, 0)
    print(f"[BulkInsert] inserting {N:,} windows (T={long_scale}) ...", flush=True)

    t0 = time.time()
    for k in range(N):
        eid = f"{type_str}_{base + k:06d}"
        fp = vecs[k]
        addr = addrs_list[k]

        entry = Entry(
            eid=eid,
            type=type_str,
            content=windows[k],
            fp=fp,
            addr=addr,
            meta={'pos': seq_index[k], 'seq_len': long_scale},
        )
        store._entries[eid] = entry
        store._vecs[eid] = fp
        store._face_index.setdefault(addr.face, []).append(eid)
        store._type_sets.setdefault(type_str, set()).add(eid)
        store._type_face.setdefault(type_str, {}).setdefault(addr.face, []).append(eid)

        if k > 0 and k % 100_000 == 0:
            elapsed = time.time() - t0
            print(f"  [{k:,}/{N:,}]  {elapsed:.1f}s", flush=True)

    store._counters[type_str] = base + N
    elapsed = time.time() - t0
    print(f"[BulkInsert] done: {N:,} entries in {elapsed:.1f}s", flush=True)
    return N


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

_BANNER = """\
╔══════════════════════════════════════════════════════════════════╗
║           OMC Assistant — Addressable Computation REPL          ║
║  Everything lives in one geometric space (dodecahedral, d=64)   ║
║  Type 'help' for usage hints.  Ctrl-C / 'quit' to exit.         ║
╚══════════════════════════════════════════════════════════════════╝"""

_HELP = """\
OMC Assistant commands:
  Any text          → fingerprint → navigate → retrieve → respond
  run: <code>       → execute OMC code via omc_eval
  search: <query>   → keyword + address search in OMC codebase
  complete: <prefix>→ MCP omc_predict function completion
  stats             → show UniversalAddressStore statistics
  skills            → show navigator skill summary
  history           → show all conversation turns
  reload lm [path]  → hot-swap char LM checkpoint (after OMC training completes)
  quit / exit       → leave the REPL

How it works:
  Your input is φ-hashed to a 64-dim dodecahedral substrate vector.
  The MultiskillNavigator predicts the next address.
  The store retrieves the nearest OMC code, memory turns, and tools.
  Intent prefixes (run:, search:, complete:) override geometric routing.
  If lm_bundle is loaded, each response also includes char-generated OMC code.
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OMC Assistant — addressable computation as a conversational agent"
    )
    g = parser.add_mutually_exclusive_group()
    g.add_argument(
        '--no-index',
        action='store_true',
        help='Skip loading the 648K OMC code windows (fast startup, no code retrieval)',
    )
    g.add_argument(
        '--index',
        type=str,
        default=None,
        metavar='PATH',
        help='Path to a pre-built dual index (.pt) to load (default: omc_dual_index.pt)',
    )
    parser.add_argument(
        '--corpus',
        type=str,
        default=None,
        metavar='PATH',
        help='Path to raw corpus text file (used if --index not given and file exists)',
    )
    parser.add_argument(
        '--navigator',
        type=str,
        default=None,
        metavar='PATH',
        help='Path to navigator checkpoint (default: multiskill_navigator_omc_corpus.pt)',
    )
    parser.add_argument(
        '--d-model',
        type=int,
        default=64,
    )
    parser.add_argument(
        '--lm-ckpt',
        type=str,
        default=None,
        metavar='PATH',
        help='Optional: path to a FibRecLMNavigator checkpoint for char generation '
             '(e.g. bloom_checkpoint_model.pt after OMC training completes)',
    )
    parser.add_argument(
        '--gen-chars',
        type=int,
        default=120,
        help='Number of chars to generate per response when --lm-ckpt is given',
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run one hardcoded test query and exit (for smoke testing)',
    )
    args = parser.parse_args()

    d_model = args.d_model

    # ── Load navigator ────────────────────────────────────────────────────────
    nav_path = Path(args.navigator) if args.navigator else _HERE / 'multiskill_navigator_omc_corpus.pt'
    print(f"[Init] Loading navigator from {nav_path} ...", flush=True)
    nav_ckpt = torch.load(str(nav_path), map_location='cpu', weights_only=False)
    nav = MultiskillNavigator(
        d_model=nav_ckpt.get('d_model', d_model),
        n_skills=nav_ckpt.get('n_skills', 4),
    )
    nav.load_state_dict(nav_ckpt['model_state'])
    nav.eval()
    print(nav.skill_summary(), flush=True)

    # ── Load char LM (optional) ───────────────────────────────────────────────
    lm_bundle = None
    _lm_candidates = [
        Path(args.lm_ckpt) if args.lm_ckpt else None,
        _HERE / 'bloom_best.pt',                    # Phase-4 stacked best (chunk233+n6+long); preferred when present
        _HERE / 'bloom_256_curriculum_model.pt',
        _HERE / 'bloom_256_model.pt',
        _HERE / 'bloom_curriculum_model.pt',
        _HERE / 'bloom_checkpoint_model.pt',
    ]
    for _lm_path in _lm_candidates:
        if _lm_path and _lm_path.exists():
            print(f"[Init] Loading LM from {_lm_path} ...", flush=True)
            lm_bundle = _load_lm(_lm_path)
            if lm_bundle:
                _, _, _, lm_seq_len = lm_bundle
                n_lm_params = sum(p.numel() for p in lm_bundle[0].parameters())
                print(f"  LM loaded: {n_lm_params:,} params  seq_len={lm_seq_len}  "
                      f"gen={args.gen_chars} chars/turn")
                break
            else:
                print(f"  [warn] LM failed to load from {_lm_path}")

    # ── Build store ───────────────────────────────────────────────────────────
    store = UniversalAddressStore(d_model=d_model)

    # ── Load code index ───────────────────────────────────────────────────────
    if not args.no_index:
        index_path: Optional[Path] = None
        if args.index:
            index_path = Path(args.index)
        else:
            for _candidate_idx in [
                _HERE / 'omc_dual_index_256.pt',
                _HERE / 'omc_dual_index.pt',
            ]:
                if _candidate_idx.exists():
                    index_path = _candidate_idx
                    break

        if index_path and index_path.exists():
            print(f"[Init] Loading dual index from {index_path} ...", flush=True)
            t0 = time.time()
            dual = torch.load(str(index_path), map_location='cpu', weights_only=False)
            meta = dual.get('meta', {})
            print(
                f"  windows={meta.get('n_windows', '?'):,}  "
                f"seq_len={meta.get('seq_len', '?')}  "
                f"stride={meta.get('stride', '?')}",
                flush=True,
            )
            _bulk_insert_dual_index(store, dual)
            del dual   # free RAM
            print(f"[Init] index loaded in {time.time() - t0:.1f}s", flush=True)
        elif args.corpus or (_HERE / 'omc_corpus.txt').exists():
            corpus_path = Path(args.corpus) if args.corpus else _HERE / 'omc_corpus.txt'
            print(f"[Init] Building index from corpus: {corpus_path}", flush=True)
            print("  (this takes ~30s — use omc_dual_index.pt for instant load)", flush=True)
            with open(corpus_path, 'r', encoding='utf-8', errors='replace') as fh:
                corpus_text = fh.read()
            _lm_seq = lm_bundle[3] if lm_bundle else 89
            store.store_corpus(corpus_text, seq_len=_lm_seq, stride=_lm_seq, verbose=True)
        else:
            print("[Init] No index found — starting without code context.", flush=True)
    else:
        print("[Init] --no-index: skipping code corpus.", flush=True)

    # ── Load name→address registry ────────────────────────────────────────────
    name_reg = None
    _reg_cache = _HERE / 'omc_name_registry.pt'
    _fndefs = _HERE / 'omc_fndefs.txt'
    if _reg_cache.exists() or _fndefs.exists():
        try:
            from name_registry import load_or_build_registry as _load_reg
            name_reg = _load_reg(_reg_cache, _fndefs, d_model=d_model)
            print(f"[Init] Name registry: {len(name_reg):,} entries  "
                  f"(φ(name)→address, threshold-free)", flush=True)
        except Exception as _e:
            print(f"[Init] Name registry unavailable: {_e}", flush=True)

    # ── Register tools ────────────────────────────────────────────────────────
    print("[Init] Registering tools ...", flush=True)

    assistant = OMCAssistant(store=store, navigator=nav, d_model=d_model,
                             lm_bundle=lm_bundle, gen_chars=args.gen_chars,
                             name_reg=name_reg)

    # Bind store/nav into closures so the tool receives them
    def tool_search(query: str) -> str:
        """Search OMC codebase by semantic address proximity. Returns top code snippets."""
        return _tool_search(query, store=store)

    def tool_explain(code: str) -> str:
        """Explain what a piece of OMC code does by retrieving similar patterns."""
        return _tool_explain(code, store=store)

    def tool_eval(code: str) -> str:
        """Execute OMC code (requires omc binary). Returns stdout/stderr output."""
        return _tool_eval(code)

    def tool_predict(text: str) -> str:
        """Predict the next address for a text fragment using the skill navigator."""
        return _tool_predict(text, nav=nav)

    def tool_complete(prefix: str) -> str:
        """Complete an OMC function prefix using the MCP omc_predict tool."""
        try:
            from omc_tools import omc_predict as _omc_predict
            return _omc_predict(prefix, n=3)
        except ImportError:
            return "[tool_complete] omc_tools not available"

    def tool_recall(query: str) -> str:
        """Recall real code from the written-memory datastore (write-don't-train + IVF) with provenance."""
        return _tool_recall(query)

    for fn in [tool_search, tool_explain, tool_eval, tool_predict, tool_complete, tool_recall]:
        entry = assistant.register_tool(fn, name=fn.__name__)
        print(f"  tool: {fn.__name__}  addr={addr_str(entry.addr)}", flush=True)

    # ── Register skill weights ────────────────────────────────────────────────
    with torch.no_grad():
        for i in range(nav.n_skills):
            W = nav.W_bank[i].detach()
            w_entry = assistant.register_weight(W, name=f'skill_{i}')
            print(
                f"  weight: skill_{i}  addr={addr_str(w_entry.addr)}  |W|={W.norm().item():.3f}",
                flush=True,
            )

    print(f"\n{store.stats()}\n", flush=True)

    # ── Smoke test ────────────────────────────────────────────────────────────
    if args.test:
        print("=" * 68)
        print("SMOKE TEST: one hardcoded query")
        print("=" * 68)
        query = "how does tape_matmul work in OMC?"
        print(f"Q: {query}\n")
        resp = assistant.respond(query)
        print(resp)
        print("\n[SMOKE TEST PASSED]")
        return

    # ── REPL ──────────────────────────────────────────────────────────────────
    print(_BANNER)
    print()

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[bye]")
            break

        if not user_input:
            continue

        if user_input.lower() in ('quit', 'exit', 'q', 'bye'):
            print("[bye]")
            break

        if user_input.lower() == 'help':
            print(_HELP)
            continue

        if user_input.lower() == 'stats':
            print(store.stats())
            continue

        if user_input.lower() == 'skills':
            print(nav.skill_summary())
            continue

        if user_input.lower().startswith('reload lm'):
            # Hot-swap the char LM (useful after training completes and overwrites checkpoint)
            if len(user_input.split()) > 2:
                _lm_path = Path(user_input.split()[-1])
            else:
                _lm_path = next(
                    (p for p in [
                        _HERE / 'bloom_best.pt',
                        _HERE / 'bloom_256_curriculum_model.pt',
                        _HERE / 'bloom_256_model.pt',
                        _HERE / 'bloom_curriculum_model.pt',
                        _HERE / 'bloom_checkpoint_model.pt',
                    ] if p.exists()),
                    _HERE / 'bloom_checkpoint_model.pt',
                )
            print(f"Reloading LM from {_lm_path} ...", flush=True)
            new_bundle = _load_lm(_lm_path)
            if new_bundle:
                assistant.lm_bundle = new_bundle
                _, _, _, new_sl = new_bundle
                n_p = sum(p.numel() for p in new_bundle[0].parameters())
                print(f"  LM reloaded: {n_p:,} params  seq_len={new_sl}")
            else:
                print(f"  [warn] LM reload failed from {_lm_path}")
            continue

        if user_input.lower() == 'history':
            mem_type = store._type_sets.get('memory', set())
            turns = sorted(
                ((e.meta.get('turn', 0), e.meta.get('role', '?'), e.content)
                 for eid in mem_type
                 if (e := store._entries.get(eid)) is not None),
                key=lambda x: x[0],
            )
            if not turns:
                print("(no history yet)")
            for turn_n, role, content in turns[-20:]:
                raw = (content if isinstance(content, str) else str(content)).replace('\n', ' ')[:100]
                print(f"  {turn_n:3d} [{role:9s}] {raw}")
            continue

        print()
        t0 = time.time()
        response = assistant.respond(user_input)
        elapsed = time.time() - t0
        print(response)
        print(f"\n[{elapsed:.2f}s]")
        print()


if __name__ == '__main__':
    main()

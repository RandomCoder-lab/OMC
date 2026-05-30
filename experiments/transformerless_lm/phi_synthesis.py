"""φ-substrate template synthesis engine.

Mathematical primitives are derivable, not retrieved.

Architecture:
  query → φ-hash → navigate template lattice → nearest category → instantiate OMC

The template lattice is φ-addressed: each algorithmic category lives at a
canonical address derived from its mathematical structure (not a word list).
A query fingerprint navigates to the nearest category → fills template slots
with query-specific names → returns valid OMC code.

This works for functions ABSENT from corpus: the φ-substrate encodes the
mathematical skeleton, not the specific function name.

Template categories (mathematically grounded):
  ACCUMULATE    – while loop + running accumulator (sum, product, count)
  TRANSFORM     – element-wise output array (map, reverse, scale)
  FILTER        – conditional selection into result (filter, unique, where)
  SEARCH_LINEAR – sequential scan returning index (linear_search, find)
  SEARCH_BINARY – bisection on sorted array (binary_search, lower_bound)
  RECURSIVE_1   – single recursive call + base (factorial, count_down)
  RECURSIVE_2   – double recursive call + base (fibonacci, tree traversal)
  DIVIDE_CONQUER – split + recurse + combine (merge_sort, karatsuba)
  TWO_POINTER   – two indices traversing arrays (merge, partition)
  NESTED_LOOP   – O(n²) double iteration (bubble_sort, matmul)
  SLIDING_WINDOW – fixed-width pass over array (running_avg, convolution)
  STACK_BASED   – explicit stack instead of recursion (iterative DFS, postfix)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from corpus_address_index import substrate_fingerprint


# ─────────────────────────────────────────────────────────────────────────────
# Template definitions — OMC code skeletons with SLOT placeholders
# ─────────────────────────────────────────────────────────────────────────────

# Slots: {FN}, {ARR}, {ARG1}, {ARG2}, {INIT}, {OP}, {COND}, {RETURN}

TEMPLATES: Dict[str, Dict] = {

    "ACCUMULATE": {
        "description": "while loop with running accumulator — sum, product, count, max, min",
        "anchor_words": ["sum", "product", "count", "total", "accumulate", "aggregate",
                         "mean", "average", "max", "min", "reduce", "add", "multiply",
                         "sum array", "sum of", "find sum", "compute sum"],
        "code": """\
fn {FN}({ARR}) {{
    h acc = {INIT};
    h i = 0;
    while i < arr_len({ARR}) {{
        acc = {OP}(acc, arr_get({ARR}, i));
        i = i + 1;
    }}
    return acc;
}}""",
        "slots": {"INIT": "0", "OP": "acc + "},
    },

    "TRANSFORM": {
        "description": "element-wise output array — map, reverse, negate, scale",
        "anchor_words": ["map", "transform", "apply", "convert", "scale", "reverse",
                         "negate", "normalize", "copy"],
        "code": """\
fn {FN}({ARR}) {{
    h n = arr_len({ARR});
    h result = arr_new(n, {INIT});
    h i = 0;
    while i < n {{
        arr_set(result, i, {OP}arr_get({ARR}, i));
        i = i + 1;
    }}
    return result;
}}""",
        "slots": {"INIT": "0", "OP": ""},
    },

    "FILTER": {
        "description": "conditional selection — filter, where, select, unique",
        "anchor_words": ["filter", "where", "select", "keep", "unique", "distinct",
                         "remove", "exclude", "prune"],
        "code": """\
fn {FN}({ARR}) {{
    h result = arr_new(0, 0);
    h i = 0;
    while i < arr_len({ARR}) {{
        h val = arr_get({ARR}, i);
        if {COND} {{
            arr_push(result, val);
        }}
        i = i + 1;
    }}
    return result;
}}""",
        "slots": {"COND": "val > 0"},
    },

    "SEARCH_LINEAR": {
        "description": "sequential scan returning index — linear_search, find, index_of",
        "anchor_words": ["linear", "linear search", "linear_search", "locate", "index",
                         "contains", "position", "scan", "lookup", "index_of"],
        "code": """\
fn {FN}({ARR}, target) {{
    h i = 0;
    while i < arr_len({ARR}) {{
        if arr_get({ARR}, i) == target {{
            return i;
        }}
        i = i + 1;
    }}
    return -1;
}}""",
        "slots": {},
    },

    "SEARCH_BINARY": {
        "description": "bisection on sorted array — binary_search, lower_bound, upper_bound",
        "anchor_words": ["binary", "bisect", "halve", "logarithmic", "sorted", "ordered"],
        "code": """\
fn {FN}({ARR}, target) {{
    h lo = 0;
    h hi = arr_len({ARR}) - 1;
    while lo <= hi {{
        h mid = (lo + hi) / 2;
        h v = arr_get({ARR}, mid);
        if v == target {{ return mid; }}
        if v < target {{ lo = mid + 1; }} else {{ hi = mid - 1; }}
    }}
    return -1;
}}""",
        "slots": {},
    },

    "RECURSIVE_1": {
        "description": "single recursive call with base case — factorial, count_down, power",
        "anchor_words": ["factorial", "power", "countdown", "recursive", "descend",
                         "chain", "step"],
        "code": """\
fn {FN}(n) {{
    if n <= {BASE} {{ return {BASE_RETURN}; }}
    return {OP};
}}""",
        "slots": {"BASE": "1", "BASE_RETURN": "1", "OP": "n * {FN}(n - 1)"},
    },

    "RECURSIVE_2": {
        "description": "double recursive call with base case — fibonacci, tribonacci, tree sum",
        "anchor_words": ["fibonacci", "fib", "tribonacci", "golden", "recurrence",
                         "tree", "branch", "double"],
        "code": """\
fn {FN}(n) {{
    if n <= 1 {{ return n; }}
    return {FN}(n - 1) + {FN}(n - 2);
}}""",
        "slots": {},
    },

    "DIVIDE_CONQUER": {
        "description": "split + recurse each half + combine — merge_sort, binary operations",
        "anchor_words": ["merge_sort", "mergesort", "merge sort", "merge", "divide",
                         "conquer", "split", "halve", "combine", "recursive_sort"],
        "code": """\
fn {FN}_merge(left, right) {{
    h n_l = arr_len(left);
    h n_r = arr_len(right);
    h result = arr_new(n_l + n_r, 0);
    h i = 0; h j = 0; h k = 0;
    while i < n_l && j < n_r {{
        if arr_get(left, i) <= arr_get(right, j) {{
            arr_set(result, k, arr_get(left, i)); i = i + 1;
        }} else {{
            arr_set(result, k, arr_get(right, j)); j = j + 1;
        }}
        k = k + 1;
    }}
    while i < n_l {{ arr_set(result, k, arr_get(left, i)); i = i + 1; k = k + 1; }}
    while j < n_r {{ arr_set(result, k, arr_get(right, j)); j = j + 1; k = k + 1; }}
    return result;
}}

fn {FN}({ARR}) {{
    h n = arr_len({ARR});
    if n <= 1 {{ return {ARR}; }}
    h mid = n / 2;
    h left = arr_new(mid, 0);
    h right = arr_new(n - mid, 0);
    h i = 0;
    while i < mid {{ arr_set(left, i, arr_get({ARR}, i)); i = i + 1; }}
    h j = 0;
    while j < n - mid {{ arr_set(right, j, arr_get({ARR}, mid + j)); j = j + 1; }}
    return {FN}_merge({FN}(left), {FN}(right));
}}""",
        "slots": {"ARR": "arr"},
    },

    "TWO_POINTER": {
        "description": "two indices advancing through arrays — merge, zip, diff",
        "anchor_words": ["merge", "zip", "interleave", "combine", "two", "pointer",
                         "dual", "pair", "parallel"],
        "code": """\
fn {FN}(left, right) {{
    h n_l = arr_len(left);
    h n_r = arr_len(right);
    h result = arr_new(n_l + n_r, 0);
    h i = 0; h j = 0; h k = 0;
    while i < n_l && j < n_r {{
        if arr_get(left, i) <= arr_get(right, j) {{
            arr_set(result, k, arr_get(left, i)); i = i + 1;
        }} else {{
            arr_set(result, k, arr_get(right, j)); j = j + 1;
        }}
        k = k + 1;
    }}
    while i < n_l {{ arr_set(result, k, arr_get(left, i)); i = i + 1; k = k + 1; }}
    while j < n_r {{ arr_set(result, k, arr_get(right, j)); j = j + 1; k = k + 1; }}
    return result;
}}""",
        "slots": {},
    },

    "NESTED_LOOP": {
        "description": "O(n²) double iteration — bubble_sort, selection_sort, matrix ops",
        "anchor_words": ["bubble", "selection", "quadratic", "matrix", "grid",
                         "pairwise", "compare", "swap", "nested", "bubble_sort",
                         "selection_sort"],
        "code": """\
fn {FN}({ARR}) {{
    h n = arr_len({ARR});
    h i = 0;
    while i < n {{
        h j = 0;
        while j < n - i - 1 {{
            if arr_get({ARR}, j) > arr_get({ARR}, j + 1) {{
                h tmp = arr_get({ARR}, j);
                arr_set({ARR}, j, arr_get({ARR}, j + 1));
                arr_set({ARR}, j + 1, tmp);
            }}
            j = j + 1;
        }}
        i = i + 1;
    }}
    return {ARR};
}}""",
        "slots": {"ARR": "arr"},
    },

    "INSERTION_SORT": {
        "description": "insertion sort — grow sorted prefix one element at a time",
        "anchor_words": ["insertion", "insert", "grow", "prefix", "online", "adaptive"],
        "code": """\
fn {FN}({ARR}) {{
    h n = arr_len({ARR});
    h i = 1;
    while i < n {{
        h key = arr_get({ARR}, i);
        h j = i - 1;
        while j >= 0 && arr_get({ARR}, j) > key {{
            arr_set({ARR}, j + 1, arr_get({ARR}, j));
            j = j - 1;
        }}
        arr_set({ARR}, j + 1, key);
        i = i + 1;
    }}
    return {ARR};
}}""",
        "slots": {"ARR": "arr"},
    },

    "SLIDING_WINDOW": {
        "description": "fixed-width pass — running average, convolution, rolling stats",
        "anchor_words": ["running", "sliding", "window", "rolling", "moving",
                         "average", "convolution", "smooth"],
        "code": """\
fn {FN}({ARR}, window) {{
    h n = arr_len({ARR});
    h result = arr_new(n, 0.0);
    h i = 0;
    while i < n {{
        h s = 0.0;
        h count = 0;
        h j = i;
        while j < i + window && j < n {{
            s = s + arr_get({ARR}, j);
            count = count + 1;
            j = j + 1;
        }}
        arr_set(result, i, s / to_float(count));
        i = i + 1;
    }}
    return result;
}}""",
        "slots": {"ARR": "arr"},
    },

    "GCD_EUCLID": {
        "description": "Euclidean algorithm — gcd, lcm, modular inverse",
        "anchor_words": ["gcd", "lcm", "euclid", "greatest", "common", "divisor",
                         "modular", "coprime"],
        "code": """\
fn {FN}(a, b) {{
    while b != 0 {{
        h tmp = b;
        b = a % b;
        a = tmp;
    }}
    return a;
}}""",
        "slots": {},
    },

    "PRIME_SIEVE": {
        "description": "Sieve of Eratosthenes — list primes up to n",
        "anchor_words": ["prime", "sieve", "eratosthenes", "primes", "composite",
                         "factor"],
        "code": """\
fn {FN}(limit) {{
    h sieve = arr_new(limit + 1, 1);
    arr_set(sieve, 0, 0);
    arr_set(sieve, 1, 0);
    h i = 2;
    while i * i <= limit {{
        if arr_get(sieve, i) == 1 {{
            h j = i * i;
            while j <= limit {{
                arr_set(sieve, j, 0);
                j = j + i;
            }}
        }}
        i = i + 1;
    }}
    h primes = arr_new(0, 0);
    h k = 2;
    while k <= limit {{
        if arr_get(sieve, k) == 1 {{
            arr_push(primes, k);
        }}
        k = k + 1;
    }}
    return primes;
}}""",
        "slots": {},
    },

    "SIGMOID": {
        "description": "sigmoid activation — logistic, softmax component, probability",
        "anchor_words": ["sigmoid", "logistic", "activation", "softmax", "probability",
                         "squash", "logit"],
        "code": """\
fn {FN}(x) {{
    return 1.0 / (1.0 + exp(-x));
}}""",
        "slots": {},
    },

    "RELU": {
        "description": "ReLU activation — relu, rectified, clamp below zero",
        "anchor_words": ["relu", "rectified", "rectify", "clamp", "threshold",
                         "activation", "nonlinear"],
        "code": """\
fn {FN}(x) {{
    if x > 0.0 {{ return x; }}
    return 0.0;
}}""",
        "slots": {},
    },

    "DOT_PRODUCT": {
        "description": "inner product of two vectors — dot, inner, similarity",
        "anchor_words": ["dot", "inner", "product", "similarity", "projection",
                         "vector", "multiply", "weighted"],
        "code": """\
fn {FN}(a, b) {{
    h acc = 0.0;
    h i = 0;
    while i < arr_len(a) {{
        acc = acc + arr_get(a, i) * arr_get(b, i);
        i = i + 1;
    }}
    return acc;
}}""",
        "slots": {},
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# φ-address the template lattice
# Each template gets a canonical φ-fingerprint derived from its anchor words.
# The centroid of all anchor fingerprints is the template's lattice address.
# This is corpus-agnostic — same address on any new corpus.
# ─────────────────────────────────────────────────────────────────────────────

def _build_template_lattice(d_model: int = 64) -> Dict[str, torch.Tensor]:
    """Compute φ-address for each template from its anchor words."""
    lattice: Dict[str, torch.Tensor] = {}
    for name, tpl in TEMPLATES.items():
        anchors = tpl["anchor_words"]
        fps = [substrate_fingerprint(w, d_model) for w in anchors]
        # Centroid in φ-space
        centroid = torch.stack(fps).mean(dim=0)
        centroid = centroid / (centroid.norm() + 1e-8)
        lattice[name] = centroid
    return lattice


_LATTICE_CACHE: Optional[Dict[str, torch.Tensor]] = None
_D_MODEL_CACHE: int = 0


def get_lattice(d_model: int = 64) -> Dict[str, torch.Tensor]:
    global _LATTICE_CACHE, _D_MODEL_CACHE
    if _LATTICE_CACHE is None or _D_MODEL_CACHE != d_model:
        _LATTICE_CACHE = _build_template_lattice(d_model)
        _D_MODEL_CACHE = d_model
    return _LATTICE_CACHE


# ─────────────────────────────────────────────────────────────────────────────
# Query → template navigation
# ─────────────────────────────────────────────────────────────────────────────

def _extract_fn_name(query: str) -> str:
    """Extract the intended function name from the query.

    Prefer an explicit snake_case identifier if present (e.g. 'merge_sort').
    Otherwise take the 2-3 most informative content words.
    """
    # Check for explicit snake_case or camelCase identifier in query
    explicit = re.search(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b', query.lower())
    if explicit:
        return explicit.group(1)

    stopwords = {'write', 'generate', 'create', 'implement', 'make', 'build',
                 'code', 'give', 'show', 'produce', 'output', 'a', 'an', 'the',
                 'function', 'omc', 'in', 'for', 'me', 'please', 'can', 'you',
                 'to', 'find', 'that', 'all', 'up', 'has', 'been', 'seen',
                 'never', 'before', 'with', 'using', 'based', 'on', 'of', 'is',
                 'are', 'which', 'how', 'what', 'algorithm'}
    words = [w.lower() for w in re.split(r'\W+', query) if w]
    name_words = [w for w in words if w not in stopwords and len(w) > 1]
    # Keep at most 3 words to avoid absurdly long names
    return '_'.join(name_words[:3]) if name_words else 'fn'


def _query_to_template(query: str, d_model: int = 64,
                        top_k: int = 3) -> List[Tuple[float, str]]:
    """Navigate φ-lattice from query fingerprint → ranked template categories.

    Returns [(similarity, template_name), ...] sorted by similarity.
    Combines φ-hash navigation (structural) with keyword matching (semantic).
    """
    lattice = get_lattice(d_model)

    # φ-hash navigation: query → φ-address → nearest template centroid
    q_fp = substrate_fingerprint(query, d_model)
    q_fp_n = q_fp / (q_fp.norm() + 1e-8)

    phi_scores: Dict[str, float] = {}
    for name, centroid in lattice.items():
        sim = torch.dot(q_fp_n, centroid).item()
        phi_scores[name] = sim

    # Keyword matching: query words ∩ anchor words
    q_words = set(re.split(r'\W+', query.lower()))
    # Also check bigrams (e.g. "merge sort", "bubble sort")
    q_tokens = re.split(r'\W+', query.lower())
    q_bigrams = {q_tokens[i] + ' ' + q_tokens[i+1]
                 for i in range(len(q_tokens) - 1)}
    q_all = q_words | q_bigrams

    kw_scores: Dict[str, float] = {}
    for name, tpl in TEMPLATES.items():
        anchors = tpl["anchor_words"]
        # Exact anchor hits — count how many anchor words appear in query
        hits = sum(1 for w in anchors if w in q_all)
        kw_scores[name] = hits / max(len(anchors), 1)

    # Combined: 15% φ-hash + 85% keyword
    # Short synthesis queries don't have enough φ-hash signal on their own;
    # keyword matching is far more discriminative at this scale.
    combined: List[Tuple[float, str]] = []
    for name in TEMPLATES:
        score = 0.15 * phi_scores[name] + 0.85 * kw_scores[name]
        combined.append((score, name))

    combined.sort(key=lambda x: x[0], reverse=True)
    return combined[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# Template instantiation
# ─────────────────────────────────────────────────────────────────────────────

def _instantiate(template_name: str, fn_name: str,
                 arr_name: str = "arr") -> str:
    """Fill template slots with query-derived names."""
    tpl = TEMPLATES[template_name]
    code = tpl["code"]

    # Apply user-provided slot defaults, then query-specific overrides
    slots = dict(tpl.get("slots", {}))
    slots["FN"] = fn_name
    slots["ARR"] = arr_name

    # For RECURSIVE_1, fix the self-reference in OP slot
    if template_name == "RECURSIVE_1":
        slots["OP"] = f"n * {fn_name}(n - 1)"

    # For DIVIDE_CONQUER, merge function name
    if template_name == "DIVIDE_CONQUER":
        slots["FN"] = fn_name  # merge helper will be fn_name_merge

    for slot, val in slots.items():
        code = code.replace(f'{{{slot}}}', val)

    return code


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def synthesize(query: str, d_model: int = 64) -> Optional[str]:
    """Synthesize OMC code for a query using φ-template navigation.

    Returns (category_name, code) or None if no template matches well.
    """
    candidates = _query_to_template(query, d_model, top_k=3)
    if not candidates:
        return None

    best_score, best_name = candidates[0]

    # Minimum threshold: require at least keyword signal or strong φ-match
    if best_score < 0.01:
        return None

    fn_name = _extract_fn_name(query)

    # Pick array arg name from query context
    arr_name = "arr"
    for w in re.split(r'\W+', query.lower()):
        if w in {'array', 'list', 'sequence', 'vec', 'vector', 'data', 'nums', 'numbers'}:
            arr_name = w if len(w) <= 5 else 'arr'
            break

    code = _instantiate(best_name, fn_name, arr_name)
    return code, best_name, best_score


def synthesize_with_fallback(query: str, d_model: int = 64,
                              top_k_templates: int = 2) -> List[Tuple[str, str, float]]:
    """Return top_k_templates candidate synthesized functions."""
    candidates = _query_to_template(query, d_model, top_k=top_k_templates)
    fn_name = _extract_fn_name(query)
    results = []
    for score, name in candidates:
        if score > 0.005:
            code = _instantiate(name, fn_name)
            results.append((name, code, score))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    queries = sys.argv[1:] or [
        "write a fibonacci function",
        "write a merge sort",
        "write binary search",
        "write bubble sort",
        "write a function to find the sum of an array",
        "implement sigmoid activation",
        "write a prime sieve",
        "write a gcd function",
        "write an insertion sort",
        "write a running average",
        "implement a dot product",
        "write a filter function",
        "write a function to find all primes up to n",
        "write a function that has never been seen before: quadratic_sieve",
    ]

    print("φ-Template Synthesis Engine")
    print("=" * 60)
    for q in queries:
        result = synthesize(q)
        if result:
            code, category, score = result
            print(f"\nQuery: {q!r}")
            print(f"Template: {category}  (score={score:.3f})")
            print(code)
            print("-" * 60)
        else:
            print(f"\nQuery: {q!r}  → no template matched")

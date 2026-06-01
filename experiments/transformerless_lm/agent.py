#!/usr/bin/env python3
"""agent.py — THE TOOL-USING WEB-NATIVE AGENT.

The engine SPEAKS (realize) and THINKS (resolve/bridge); this gives it TOOLS and the judgment to pick
one. Tool selection is itself ADDRESSING: the query's shape/concepts address the tool that answers it —
no LLM, no hand-coded keyword menu beyond the structural signal each tool advertises.

Tools (each grounded, each returns (answer, tool_name, confidence)):
  charcount  — exact char-level counting (char_skills) — what token-LLMs get WRONG; our proven edge.
  compute    — safe arithmetic evaluation (ast) — retrieval can't add; a tool can.
  relate     — two topics → grounded cross-source relation/bridge (create.py).
  recall     — one topic → competence-check → facets + grounded passage (engine.py).
  decline    — no addressable topic → honest "I don't know".

confidence ∈ [0,1] is the agent's own grounded self-estimate (exactness for tools, resolve for recall/
relate) — used by the self-improvement loop to detect its own weak answers.
"""
import re, ast, operator
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

import json
import char_skills
from create import create as create_bridge
from engine import (build_frame, is_frame, WH_ANCHORS, near_categories,
                    relate_pmi, hybrid_realize, heal_surface)
from thinkloop import resolve, concepts_of
from langexec import edge_w

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg, ast.FloorDiv: operator.floordiv}


def _eval_arith(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_arith(node.left), _eval_arith(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_arith(node.operand))
    raise ValueError("unsupported")


# ── tools ──────────────────────────────────────────────────────────────────
def tool_charcount(k, model, query, **kw):
    a = char_skills.answer_char_question(query)
    return (a, "charcount", 1.0) if a else None       # exact by construction


def tool_compute(k, model, query, **kw):
    m = re.search(r"(?:what\s+is\s+|compute\s+|calculate\s+|=\s*)?"
                  r"([0-9][0-9\s\.\+\-\*/%\(\)]*[0-9\)])\s*\??$", query.strip(), re.I)
    if not m or not re.search(r"[\+\-\*/%]", m.group(1)):
        return None
    try:
        v = _eval_arith(ast.parse(m.group(1).strip(), mode="eval").body)
    except Exception:
        return None
    v = int(v) if isinstance(v, float) and v.is_integer() else v
    return (f"{m.group(1).strip()} = {v}", "compute", 1.0)


def _topics(k, query, frame):
    return [c for c in dict.fromkeys(concepts_of(k, query))
            if c in k.nodeset and c not in k.stop and not is_frame(k, c, frame, WH_ANCHORS)]


def tool_memory(k, model, query, frame=None, mem=None, **kw):
    """SELF-IMPROVEMENT PAYOFF: consult the verified-thought memory FIRST. If the engine has previously
    reasoned out + gated this connection, recall it instantly (no re-bridging). This is where the
    overnight self-improvement shows up at query time."""
    if mem is None:
        return None
    topics = _topics(k, query, frame)
    for i, a in enumerate(topics):
        for b in topics[i + 1:]:
            if mem.has(a, b):
                r = mem.get(a, b)
                if r:
                    thought, coh, sup, fields, path = r
                    return (f"{a} ⇄ {b} (recalled verified thought · coh {coh:.2f} sup {sup:.2f} · {fields}):\n"
                            f"   {thought}", "memory", min(1.0, 0.6 + 0.1 * sup))
    return None


def tool_relate(k, model, query, frame=None, **kw):
    topics = _topics(k, query, frame)
    if len(topics) < 2:
        return None
    # pick the most semantically-related pair (robust to leaked scaffolding + no-direct-edge pairs)
    pairs = [(a, b) for i, a in enumerate(topics) for b in topics[i + 1:]]
    a, b = max(pairs, key=lambda p: k.relate(p[0], p[1]))
    ans = create_bridge(k, model, a, b)
    dp = edge_w(k, a, b)
    pmi = k.assoc(a, b, dp) if dp is not None else 0.0
    conf = min(1.0, max(0.0, 0.4 + 0.1 * max(0.0, pmi)))
    return (ans, "relate", conf)


def tool_recall(k, model, query, frame=None, **kw):
    topics = _topics(k, query, frame)
    if not topics:
        return None
    main = min(topics, key=lambda c: k.deg.get(c, 0))
    cats = near_categories(k, main, 6)
    if len(cats) < 3:
        return (f"I have only sparse knowledge around '{main}' ({len(cats)} facets).", "recall", 0.1)
    substance = list(dict.fromkeys(topics + cats[:5]))
    said = heal_surface(hybrid_realize(k, model, substance, k.stop))
    r = resolve(k, [w for w in re.findall(r"[a-z]+(?:'[a-z]+)?", said.lower()) if w in k.nodeset])["pos_pmi"]
    return (f"'{main}' — facets: {', '.join(cats[:5])}.\n   {said}", "recall", round(r, 2))


# Order = specificity. tool_memory first: a verified recall beats re-deriving. charcount/compute next
# (exact). Then relate (bridge) and recall (single topic).
TOOLS = [tool_charcount, tool_compute, tool_memory, tool_relate, tool_recall]


def answer(k, model, query, frame, mem=None):
    """Address the query to the first tool that handles it (ordered by specificity). Returns a dict."""
    for tool in TOOLS:
        res = tool(k, model, query, frame=frame, mem=mem)
        if res is not None:
            ans, name, conf = res
            return dict(query=query, tool=name, answer=ans, confidence=conf)
    return dict(query=query, tool="decline", answer="I don't recognize a topic I hold knowledge about there.",
                confidence=0.0)


DEMO = [
    "how many r's in strawberry",                 # charcount  (token-LLMs fail this)
    "what is 47 * 19 + 3",                         # compute
    "how do light and gravity relate",            # relate (bridge)
    "tell me about napoleon",                      # recall
    "describe the relationship between sleep and disease",  # relate
    "what is 2 ** 10",                             # compute
    "explain the flrbzy of the quux",             # decline
]


def load_memory():
    """Open the verified-thought memory read-only (shared with the overnight self-improver), if present."""
    from pathlib import Path as _P
    p = _P(__file__).parent / "derived.db"
    if not p.exists():
        return None
    try:
        from selfimprove import VerifiedMemory
        return VerifiedMemory(p, read_only=True)
    except Exception:
        return None


def main():
    from engine import open_web
    from fluency import get_fluency_model
    k = open_web()
    mp = k.db.execute("SELECT MAX(pid) FROM passages").fetchone()[0]
    print("[agent] loading fluency model (cached) ...", flush=True)
    model = get_fluency_model(k, 80000, mp)
    frame = build_frame(k)
    mem = load_memory()
    nthoughts = mem.size() if mem else 0
    print(f"[agent] ready — tools: {[t.__name__.replace('tool_','') for t in TOOLS]}  "
          f"(verified-memory: {nthoughts} thoughts)\n", flush=True)
    demo = list(DEMO)
    if mem:                                  # add a query that should HIT verified memory (self-improvement payoff)
        row = mem.db.execute("SELECT a,b FROM thoughts ORDER BY support DESC LIMIT 1").fetchone()
        if row:
            demo.insert(3, f"how do {row[0]} and {row[1]} relate")
    for q in demo:
        r = answer(k, model, q, frame, mem=mem)
        print(f"USER: {q}")
        print(f"  [{r['tool']} · conf {r['confidence']}] {r['answer']}\n")


if __name__ == "__main__":
    main()

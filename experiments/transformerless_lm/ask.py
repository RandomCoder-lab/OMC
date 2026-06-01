#!/usr/bin/env python3
"""ask.py — USE the web: answer a question by ADDRESSING it, not generating about it. Composes the whole
stack into one grounded, cited, honest answer:
  1. address — which of the question's words are concepts in the web
  2. ground  — for each concept pair, the strongest grounded EVIDENCE (quoted source + field + pid)
  3. relate  — each concept's dominant grounded factors (what the web most connects it to)
  4. recall  — what the navigator already DISCOVERED about these concepts (derived memories) + told facts
  5. honest verdict — supported pairs vs unaddressable terms; never asserts what has no address

This is retrieve→stitch→ground, NOT fluent generation. Its guarantee: every line traces to a source or is
flagged. It cannot say what the web can't ground.

  python ask.py "how does gravity relate to light"
  python ask.py "what is energy"
"""
import sys, re, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from factcheck import open_web, FactChecker

TOK = re.compile(r"[a-z][a-z']+")


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    q = " ".join(sys.argv[1:])
    k = open_web()
    fc = FactChecker(k)
    cs = fc.concepts(q)
    raw = [w for w in TOK.findall(q.lower()) if w not in k.stop]
    unaddr = sorted(set(w for w in raw if w not in cs and w not in k.stoi))

    print(f"Q: {q}")
    print(f"⌖ addressed: {', '.join(cs) if cs else '(none)'}" +
          (f"   ·   not in web: {', '.join(unaddr)}" if unaddr else ""))

    # 2. grounded evidence between the question's concepts (cited)
    print("\n── GROUNDED (what the corpus attests, with citations) ──")
    shown = 0
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            ev = fc.evidence(cs[i], cs[j]) or fc.evidence(cs[j], cs[i])
            if ev:
                shown += 1
                print(f"  • {cs[i]}–{cs[j]}  ⟨{ev['src']}, pid {ev['pid']}, ×{ev['w']}⟩")
                print(f"      “…{ev['span']}…”")
    if not shown:
        print("  (no direct grounded co-occurrence between the addressed concepts)")

    # 3. each concept's dominant grounded factors (the structure)
    print("\n── STRUCTURE (what the web most connects each concept to) ──")
    for x in cs:
        fac = fc.factors(x, topk=8)
        if fac:
            print(f"  {x}: " + ", ".join(b for _, _, b, _, _ in fac))

    # 4. recall what the navigator discovered + what was told
    print("\n── RECALLED (the system's own discoveries + told facts about these) ──")
    qset = set(cs)
    rec = 0
    try:
        rows = k.db.execute("SELECT kind,salience,forms,text,concepts FROM memory").fetchall()
    except Exception:
        rows = []
    import json as _j
    hits = []
    for kind, sal, forms, text, cj in rows:
        try:
            mc = set(_j.loads(cj))
        except Exception:
            mc = set()
        ov = len(qset & mc)
        if ov:
            hits.append((ov, sal, kind, forms, text))
    for ov, sal, kind, forms, text in sorted(hits, reverse=True)[:6]:
        rec += 1
        tagline = (forms or kind)
        print(f"  ⟪{kind} sal{sal:.0f} · {tagline}⟫ {text[:96]}")
    if not rec:
        print("  (no prior memories touch these concepts yet)")

    # 5. honest verdict
    print(f"\n── VERDICT ── {shown} grounded pair(s), {len(unaddr)} unaddressable term(s), {rec} recalled. "
          f"Every line above is sourced or flagged; the web does not assert what it cannot ground.")


if __name__ == "__main__":
    main()

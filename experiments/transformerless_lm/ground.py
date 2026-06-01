#!/usr/bin/env python3
"""ground.py — the SELF-GROUNDING LOOP. Before asserting, check. Given a draft answer, this verifies each
sentence's factual claims against the web (factcheck.evidence, scoped to knowledge sources), attaches
citations to what's supported, and flags what isn't — turning a fluent answer into a VERIFIABLE one:
cite-or-abstain. The discipline: the assistant drafts, runs ground(), then asserts only the GROUNDED
sentences (with their citations) and hedges/abstains on the UNVERIFIED ones.

It does NOT make the model truthful by fiat — it makes the model's claims checkable against an inspectable,
cited source, and makes "I can't support this" a first-class output. A claim with no address gets flagged,
not asserted.

  python ground.py "Gravity bends light. The mitochondria is the powerhouse of the cell."
"""
import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from factcheck import open_web, FactChecker, TOK

SENT = re.compile(r"[^.!?]+[.!?]?")


def ground(text):
    fc = FactChecker(open_web())
    sentences = [s.strip() for s in SENT.findall(text) if len(s.strip()) > 3]
    citations = []
    lines = ["=== SELF-GROUNDING REPORT ==="]
    grounded = unver = thin = 0
    for sent in sentences:
        cs = fc.concepts(sent)
        if len(cs) < 2:
            lines.append(f"  [—  no checkable relation] {sent}")
            continue
        ev_pairs = []
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                ev = fc.evidence(cs[i], cs[j]) or fc.evidence(cs[j], cs[i])
                if ev:
                    ev_pairs.append((cs[i], cs[j], ev))
        if ev_pairs:
            # strong if any pair has solid co-occurrence weight, else thin
            strong = any(e[2]["w"] >= 5 for e in ev_pairs)
            grounded += strong; thin += (not strong)
            tags = []
            for a, b, ev in ev_pairs:
                citations.append((len(citations) + 1, a, b, ev))
                tags.append(str(len(citations)))
            mark = "✓ GROUNDED" if strong else "~ THIN (weak co-occurrence)"
            lines.append(f"  [{mark}  cite {','.join(tags)}] {sent}")
        else:
            unver += 1
            lines.append(f"  [✗ UNVERIFIED — assert with care or abstain] {sent}")
    if citations:
        lines.append("\n--- CITATIONS ---")
        for n, a, b, ev in citations:
            lines.append(f"  [{n}] {a}–{b}  ⟨{ev['src']}, pid {ev['pid']}, co×{ev['w']}⟩  “…{ev['span'][:140]}…”")
    rec = ("assert all (well grounded)" if grounded and not unver and not thin else
           "assert grounded sentences with citations; hedge/abstain on UNVERIFIED/THIN")
    lines.append(f"\nSUMMARY: {grounded} grounded · {thin} thin · {unver} unverified  → {rec}")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    print(ground(" ".join(sys.argv[1:])))


if __name__ == "__main__":
    main()

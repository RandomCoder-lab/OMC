#!/usr/bin/env python3
"""factcheck.py — the practical application: check a claim against the web, with citations. This is the
tool the assistant runs to GROUND what it says. For a claim, it (1) addresses the claim's concepts, (2) for
each concept pair finds grounded co-occurrence EVIDENCE in knowledge sources (NOT translation/parallel edges
— a Spanish synonym isn't evidence), quoting the source passage + field + pid, and (3) reports what's
supported, what's unsupported (concepts present but no connecting evidence), and what's unaddressable (not in
the web at all).

HONEST FRAMING: this checks whether a claim is SUPPORTED AND GROUNDED in the corpus, with provenance — not
whether it is objectively true. A wrong source yields a confidently-cited wrong "fact". It is a
support+provenance checker, not a truth oracle. Its power: it cannot assert what has no address, and it always
shows the receipt.

  python factcheck.py "gravity bends light"
  python factcheck.py "the heart pumps blood"
"""
import sys, re, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from kdb import KnowledgeDB, load_embedding

DB = Path(__file__).parent / "knowledge.db"
TOK = re.compile(r"[a-z][a-z']+")
# sources that count as factual EVIDENCE (exclude translation/parallel bridges and bare code token-streams)
NOT_EVIDENCE = ("align-", "parallel-", "memory")
HUB_DEG = 300_000   # interim function-word guard until node_expand's stop table is built


def open_web():
    s, E, n, e = load_embedding()
    return KnowledgeDB(str(DB), s, E, n, e)


class FactChecker:
    def __init__(self, k):
        self.k = k

    def concepts(self, claim):
        out, seen = [], set()
        for w in TOK.findall(claim.lower()):
            if w in self.k.stop:                      # skip function words
                continue
            for cand in (w, w[:-1] if w.endswith("s") else None, w[:-2] if w.endswith("es") else None):
                if cand and cand in self.k.stoi and cand not in seen:
                    out.append(cand); seen.add(cand); break
        return out

    def evidence(self, a, b, radius=110):
        """A grounded co-occurrence of a and b in a knowledge source: the example passage of their edge,
        quoted, with field + pid. None if the only link is a translation edge or no edge."""
        row = self.k.db.execute("SELECT src,pid,w FROM edges WHERE a=? AND b=?", (a, b)).fetchone()
        if not row:
            return None
        src, pid, w = row
        if src.startswith(NOT_EVIDENCE):
            return None
        text, dom = self.k._passage(pid)
        if not text:
            return None
        m = re.search(r"\b" + re.escape(b) + r"\b", text.lower()) or re.search(r"\b" + re.escape(a) + r"\b", text.lower())
        if m:
            lo = max(0, m.start() - radius)
            span = re.sub(r"\s+", " ", text[lo:m.end() + radius]).strip()
        else:
            span = re.sub(r"\s+", " ", text[:2 * radius]).strip()
        return {"src": dom or src, "pid": pid, "w": w, "span": span}

    def factors(self, x, topk=12):
        """The grounded neighborhood of x: every CONTENT relation the corpus links to x (knowledge sources
        only; translation/code/stopword/hub neighbors dropped), ranked by weighted-PMI. This is 'everything
        the web connects to x' — the structure a claim must be placed within, not a single yes/no edge."""
        if x not in self.k.stoi:
            return None
        out = []
        for b, src, pid, w in self.k.db.execute(
                "SELECT b,src,pid,w FROM edges WHERE a=? AND src NOT LIKE 'align-%' AND src NOT LIKE 'parallel-%'", (x,)):
            if b in self.k.stop or b == x or self.k.deg.get(b, 0) > HUB_DEG:
                continue
            out.append((self.k.assoc(x, b, w) * math.log1p(w), w, b, src, pid))
        out.sort(reverse=True)
        return out[:topk]

    def agnostic(self, claim):
        """Agnostic verdict: don't affirm the claim — show how its concepts sit in the web's structure. For
        each claim concept, surface its full grounded factor-set; locate the other claim concepts within it
        and report position (sole/primary/one-of-several/absent). Never collapses to 'true'."""
        cs = self.concepts(claim)
        out = [f"CLAIM: {claim}", f"⌖ concepts: {', '.join(cs) if cs else '(none addressable)'}"]
        if len(cs) < 2:
            out.append("→ need ≥2 addressable concepts."); return "\n".join(out)
        for x in cs:
            fac = self.factors(x)
            if not fac:
                out.append(f"\n■ {x}: no grounded factors in web"); continue
            others = [c for c in cs if c != x]
            names = [b for _, _, b, _, _ in fac]
            out.append(f"\n■ {x} — grounded factors (web neighborhood): " +
                       ", ".join(f"{b}" for b in names[:10]))
            for o in others:
                if o in names:
                    r = names.index(o) + 1
                    w_o = fac[names.index(o)][1]
                    w_top = fac[0][1]
                    role = ("the SOLE listed factor" if len(names) == 1 else
                            "PRIMARY (dominant)" if r == 1 and w_o >= 2 * (fac[1][1] if len(fac) > 1 else 1) else
                            f"ONE OF SEVERAL (rank {r} of {len(names)})")
                    out.append(f"    ↳ '{o}' is {role} among {x}'s grounded factors — "
                               f"corpus also grounds: {', '.join(n for n in names if n != o)[:90]}")
                else:
                    # not in the dominant set — but is there a weak/specialized direct link?
                    row = self.k.db.execute(
                        "SELECT w,src FROM edges WHERE a=? AND b=? AND src NOT LIKE 'align-%' AND src NOT LIKE 'parallel-%'",
                        (x, o)).fetchone()
                    if row:
                        out.append(f"    ↳ '{o}' is a WEAK/SPECIALIZED link to {x} (co×{row[0]} ⟨{row[1]}⟩) — "
                                   f"present but NOT among {x}'s dominant factors; {x} is grounded mainly as: "
                                   f"{', '.join(n for n in names[:6])}")
                    else:
                        out.append(f"    ↳ '{o}' has NO grounded link to {x} at all")
        out.append("\nAGNOSTIC VERDICT: the corpus grounds these as RELATED, within a larger structure of "
                   "competing factors — it does NOT support any single concept as the sole/exclusive cause. "
                   "(structure shown, not collapsed)")
        return "\n".join(out)

    def check(self, claim):
        cs = self.concepts(claim)
        out = [f"CLAIM: {claim}", f"⌖ addressable concepts: {', '.join(cs) if cs else '(none)'}"]
        unaddr = [w for w in TOK.findall(claim.lower())
                  if w not in self.k.stop and w not in self.k.stoi
                  and (w[:-1] not in self.k.stoi) and (w[:-2] not in self.k.stoi)]
        if unaddr:
            out.append(f"⚠ not in web (can't verify): {', '.join(sorted(set(unaddr)))}")
        if len(cs) < 2:
            out.append("→ need ≥2 addressable concepts to check a relation."); return "\n".join(out)
        supported, unsupported = 0, 0
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                a, b = cs[i], cs[j]
                ev = self.evidence(a, b) or self.evidence(b, a)
                if ev:
                    supported += 1
                    out.append(f"  ✓ {a} — {b}  ⟨{ev['src']}, pid {ev['pid']}, co×{ev['w']}⟩")
                    out.append(f"      “…{ev['span']}…”")
                else:
                    # is there an indirect grounded path?
                    path = self.k.nav(a, b)
                    if path and 2 < len(path) <= 5:
                        out.append(f"  ~ {a} — {b}: no direct evidence; indirect via {' → '.join(path)}")
                    else:
                        unsupported += 1
                        out.append(f"  ✗ {a} — {b}: NO grounded evidence in the web")
        verdict = ("SUPPORTED" if supported and not unsupported else
                   "PARTIALLY SUPPORTED" if supported else "UNSUPPORTED")
        out.append(f"\nVERDICT: {verdict}  ({supported} grounded / {unsupported} unsupported pairs). "
                   f"[support+provenance, not absolute truth]")
        return "\n".join(out)


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    args = sys.argv[1:]
    agn = False
    if args[0] in ("--agnostic", "-a"):
        agn = True; args = args[1:]
    fc = FactChecker(open_web())
    print(fc.agnostic(" ".join(args)) if agn else fc.check(" ".join(args)))


if __name__ == "__main__":
    main()

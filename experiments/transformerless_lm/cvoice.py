"""cvoice.py — a voice DERIVED FROM THE CORPUS, not hand-authored templates.

The agnostic-voice principle (user, 2026-05-30): the system should speak using language drawn from what
it READ, not from my English templates. Key realization: when two concepts co-occur, the text BETWEEN
them already states their relationship in the corpus's own words. So the connective phrasing is EXTRACTED
from the evidence passage — never templated. The only authored tokens are structural (arrows, a numeric
score, the concept names — which are themselves corpus entities). No authored interpretation, no
hand-coded register words ("I think"), no hand-coded type lists. Confidence = the derived number.

CorpusNarrator renders: connections (the corpus span linking each hop), discoveries, and likeness
(nearest concepts by learned meaning + a representative corpus quote of the concept).
"""
import re
from pathlib import Path


def link_span(passage: str, a: str, b: str, maxlen: int = 160):
    """The corpus's OWN words linking a and b: the text spanning the closest a-occurrence to the closest
    b-occurrence in this passage. This is the corpus describing the relationship — extracted, not authored."""
    pl = passage.lower()
    ai = [m.start() for m in re.finditer(r"\b" + re.escape(a) + r"\b", pl)]
    bi = [m.start() for m in re.finditer(r"\b" + re.escape(b) + r"\b", pl)]
    if not ai or not bi:
        return None
    best = min(((abs(x - y), x, y) for x in ai for y in bi), key=lambda t: t[0])
    _, x, y = best
    if x <= y:
        lo, hi, hi_ent = x, y, b
    else:
        lo, hi, hi_ent = y, x, a
    span = passage[lo:hi + len(hi_ent)].strip()
    span = re.sub(r"\s+", " ", span)
    if len(span) > maxlen:                                  # keep the head+tail (both entities visible)
        span = span[:maxlen // 2].rstrip() + " … " + span[-maxlen // 2:].lstrip()
    return span


class CorpusNarrator:
    def __init__(self, cs):
        self.cs = cs

    def _span(self, a, b, passage_idx):
        s = link_span(self.cs.passages[passage_idx], a, b)
        if s:
            return s
        # entities present but no clean inter-span (different sentences) — quote the passage as-is
        p = re.sub(r"\s+", " ", self.cs.passages[passage_idx].strip())
        return (p[:150] + "…") if len(p) > 150 else p

    def connection(self, a, b, path, edges, score):
        """Render a grounded connection using the corpus's own linking words per hop. Authored tokens:
        only the arrow, the 'meaning=' label, and the number — all metadata, not asserted content."""
        head = f"{a.title()} → {' → '.join(p.title() for p in path[1:-1] + [b]) if len(path) > 2 else b.title()}"
        head = f"{' → '.join(p.title() for p in path)}   [meaning {a.title()}~{b.title()} = {score:.2f}]"
        lines = [head]
        for (x, y), ei in zip(zip(path, path[1:]), edges):
            lines.append(f"  {x.title()}–{y.title()}:  “…{self._span(x, y, ei)}…”")
        return "\n".join(lines)

    def discovery(self, x, found):
        """found = [(score, c, path, edges)] from cs.discover — render via corpus spans, score as number."""
        lines = [f"{x.title()} — links not stated outright in the text:"]
        for score, c, path, edges in found:
            lines.append(f"\n  {' → '.join(p.title() for p in path)}   [{x.title()}~{c.title()} = {score:.2f}]")
            for (p, q), ei in zip(zip(path, path[1:]), edges):
                lines.append(f"    {p.title()}–{q.title()}:  “…{self._span(p, q, ei)}…”")
        return "\n".join(lines)

    def likeness(self, x, neighbors):
        """neighbors = [(score, concept)]. Pure derived: nearest-by-meaning + one corpus quote of x."""
        quote = None
        for i, pl in enumerate(self.cs.plow):
            if (" " + x + " ") in pl or (" " + x + ",") in pl or (" " + x + ".") in pl:
                p = re.sub(r"\s+", " ", self.cs.passages[i].strip())
                quote = (p[:150] + "…") if len(p) > 150 else p
                break
        lead = ", ".join(f"{c.title()} {s:.2f}" for s, c in neighbors)
        out = f"{x.title()} — nearest by learned meaning: {lead}."
        if quote:
            out += f"\n  in the corpus: “{quote}”"
        return out

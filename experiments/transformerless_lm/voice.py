"""voice.py — give the dot-connector a VOICE: narrate discovered grounded chains like a mind reasoning.

The user's framing (2026-05-30): it doesn't have to be 100% factual. Like a person, it doesn't know
everything — but it understands these are PEOPLE/PLACES, that through some bridge they're somehow connected,
and it says so with CALIBRATED CONFIDENCE (the learned meaning-score worn honestly). A narrator that pieces
things together and signals how sure it is — not a fluent oracle, an honest reasoner.

Design (honest about the tiny-generator ceiling): the VOICE is grounded narration, not free generation.
It infers each concept's TYPE from grammatical context (person/place — structural function-word signal,
NOT a domain dictionary), reads the bridge from the evidence, and states a hedged hypothesis whose
certainty tracks the meaning-score. Every claim is either quoted (grounded) or marked as inference.
"""
import sys, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from connect import ConceptSpace, quote

HERE = Path(__file__).parent

# Closed-class GRAMMATICAL function words (structural English, not domain knowledge — the same tier as
# knowing whitespace/characters). Used only to infer person-vs-place TYPE from context, never as facts.
_PRON = {"he", "she", "him", "her", "his", "hers", "himself", "herself"}
_PLACEPREP = {"at", "in", "to", "from", "near", "of", "towards", "into"}
_SAYS = {"said", "replied", "cried", "answered", "asked", "thought", "felt", "exclaimed"}


class Narrator:
    def __init__(self, cs: ConceptSpace):
        self.cs = cs
        self._type = {}
        self._infer_types()

    def _mentions(self, e, limit=40):
        out = []
        for i, pl in enumerate(self.cs.plow):
            if (" " + e + " ") in pl or (" " + e + ",") in pl or (" " + e + ".") in pl:
                out.append(i)
            if len(out) >= limit:
                break
        return out

    def _infer_types(self):
        """person vs place vs other — from grammatical context statistics (agnostic, structural)."""
        for e in self.cs.ents:
            person = place = n = 0
            for i in self._mentions(e):
                toks = re.findall(r"[a-z']+", self.cs.passages[i].lower())
                if e not in toks:
                    continue
                n += 1
                idx = toks.index(e)
                window = toks[max(0, idx - 4):idx] + toks[idx + 1:idx + 5]
                if any(w in _PRON for w in window) or any(w in _SAYS for w in window):
                    person += 1
                if idx > 0 and toks[idx - 1] in _PLACEPREP:
                    place += 1
            if n == 0:
                self._type[e] = "concept"
            elif person >= max(1, n * 0.25) and person >= place:
                self._type[e] = "person"
            elif place >= max(1, n * 0.3):
                self._type[e] = "place"
            else:
                self._type[e] = "concept"

    def typ(self, e):
        return self._type.get(e, "concept")

    def _confidence(self, score):
        if score >= 0.6:
            return "I'm fairly sure", "are clearly connected"
        if score >= 0.4:
            return "I think", "seem to be connected"
        if score >= 0.25:
            return "I suspect", "may be loosely connected"
        return "I'm reaching here, but", "might be faintly connected"

    def narrate(self, a, c, path, edges, score):
        """A hedged, human-like hypothesis about how a and c connect — grounded, confidence-aware."""
        ta, tc = self.typ(a), self.typ(c)
        bridges = path[1:-1]
        lead, rel = self._confidence(score)
        kind = (f"two {ta}s" if ta == tc and ta != "concept"
                else f"a {ta} and a {tc}" if ta != "concept" or tc != "concept" else "two things")
        lines = []
        lines.append(f"{a.title()} and {c.title()} — {kind}. They never appear together in the text, "
                     f"but {lead.lower()} they {rel}.")
        # the reasoning: walk the bridge
        if len(bridges) == 1:
            b = bridges[0]
            lines.append(f"What links them is {b.title()} (a {self.typ(b)}). "
                         f"{a.title()} connects to {b.title()}, and {b.title()} to {c.title()} — "
                         f"so the thread runs through {b.title()}.")
        else:
            chain = " → ".join(p.title() for p in path)
            lines.append(f"The thread runs {chain} — each step is someone or somewhere they share.")
        # the honest part: how sure, and why
        lines.append(f"(My confidence is {score:.2f} — drawn from how these move in similar company "
                     f"across the text, not from any stated fact. I could be pattern-matching.)")
        # the grounding — let the text speak
        ev = []
        for (x, y), ei in zip(zip(path, path[1:]), edges):
            ev.append(f"    · {x.title()}~{y.title()}: “{quote(self.cs, ei, 110)}”")
        return "\n".join(lines) + "\n  evidence:\n" + "\n".join(ev)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="pride_prejudice.txt")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--per", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cs = ConceptSpace((HERE / args.corpus).read_text(errors="replace"), seed=args.seed)
    nar = Narrator(cs)
    types = {}
    for e in cs.ents:
        types.setdefault(nar.typ(e), []).append(e)
    print(f"[voice] inferred types (agnostic, from grammar): "
          + "  ".join(f"{t}:{len(v)}" for t, v in types.items()), flush=True)

    print(f"\n[voice] === the system narrates the connections it discovered ===", flush=True)
    n = 0
    for a in [e for e in cs.ents if cs.adj[e]][:args.seeds]:
        for r, c, path, edges in cs.discover(a, k=args.per):
            print(f"\n— — —\n{nar.narrate(a, c, path, edges, r)}", flush=True)
            n += 1
    print(f"\n[voice] narrated {n} discovered connections (each grounded + confidence-tagged).", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""mind.py — the dot-connector as ONE thing you talk to.

Integrates the organs built this session into a single conversational agent over a corpus:
  ConceptSpace (connect.py) = WHERE (addressed index) + GROUND (verify-per-hop) + MEANING (learned
  embedding) + REASON (meaning-guided navigation); Narrator (voice.py) = calibrated-confidence voice;
  char_skills = exact char-level answers; shape arithmetic = analogy. Honest throughout: every claim is
  quoted (grounded) or flagged as inference, confidence is worn, and it says when it doesn't know.

You can:
  - ask a letter-count question            ("how many r's in strawberry?")            -> exact
  - ask how two things connect             ("how are Darcy and Wickham connected?")   -> grounded path + voice
  - explore a concept's hidden links       ("what connects to Pemberley?")            -> discover + voice
  - ask what something is like             ("what is Bingley like?")                  -> meaning neighbors / analogy
  - mention a concept it doesn't know      -> it says so, honestly

Usage:  python mind.py [--corpus FILE | --multi]  [--demo]
"""
import sys, re, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import char_skills
from connect import ConceptSpace
from cvoice import CorpusNarrator

HERE = Path(__file__).parent
CORP = HERE / "corpora"


class Mind:
    def __init__(self, cs, knowledge="", log=print):
        self.cs = cs
        self.nar = CorpusNarrator(self.cs)              # voice DERIVED from the corpus, not templated
        self.resolve_group = char_skills.make_corpus_resolver(knowledge + "\n" + " ".join(cs.passages))
        log(f"[mind] ready — {len(self.cs.ents)} concepts in play. Ask me to connect things.")

    # ── map a user's word to the nearest concept it knows (exact, then by meaning) ──
    def resolve(self, word):
        w = word.lower().strip(".,!?;:'\"")
        if w in self.cs.ents:
            return w, 1.0
        v = self.cs.vec(w)
        if v is None:
            return None, 0.0
        best, bs = None, -1.0
        for e in self.cs.ents:
            s = float(v @ self.cs.vec(e))
            if s > bs:
                bs, best = s, e
        return (best, bs) if bs >= 0.45 else (None, bs)   # honest threshold: don't force a match

    def _quote(self, i, w=120):
        s = self.cs.passages[i].strip().replace("\n", " ")
        return (s[:w] + "…") if len(s) > w else s

    # ── capabilities ──
    def _subnote(self, word, resolved, score):
        """Transparency: if we mapped the user's word to a DIFFERENT tracked concept, say so."""
        if resolved and resolved != word.lower().strip(".,!?;:'\"") and score < 0.999:
            return f"(I don't track \"{word}\" directly; nearest concept I know is {resolved.title()}.) "
        return ""

    def connect(self, a_word, b_word):
        a, sa = self.resolve(a_word); b, sb = self.resolve(b_word)
        if a is None or b is None:
            unk = a_word if a is None else b_word
            return f"I don't know \"{unk}\" well enough to place it — it's not a concept I've read about."
        note = self._subnote(a_word, a, sa) + self._subnote(b_word, b, sb)
        if a == b:
            return note + f"Those resolve to the same concept ({a.title()})."
        path, edges = self.cs.nav(a, b)
        r = self.cs.relate(a, b)
        # The meaning-judge is the arbiter: a grounded path through a generic shared token (e.g. "Sir"
        # spanning two books) is spurious when meaning is ~0. Trust the derived score, not the token-path.
        if path and r >= 0.2:
            return note + self.nar.connection(a, b, path, edges, r)
        if path:
            bridge = " → ".join(p.title() for p in path[1:-1]) or "a shared term"
            return note + (f"{a.title()} — {b.title()}: linkable only through {bridge}, but learned "
                           f"meaning = {r:.2f} (near-unrelated) — a generic-token bridge, not a real link.")
        return note + f"{a.title()} — {b.title()}: no grounded path in the text; learned meaning = {r:.2f}."

    def explore(self, x_word, k=3):
        x, sx = self.resolve(x_word)
        if x is None:
            return f"I don't know \"{x_word}\" — not a concept I've read about."
        note = self._subnote(x_word, x, sx)
        found = self.cs.discover(x, k=k)
        if not found:
            return note + f"{x.title()}: every concept it relates to, it already directly co-occurs with."
        return note + self.nar.discovery(x, found)

    def like(self, x_word, k=5):
        x, sx = self.resolve(x_word)
        if x is None:
            return f"I don't know \"{x_word}\"."
        note = self._subnote(x_word, x, sx)
        nbrs = sorted(((self.cs.relate(x, e), e) for e in self.cs.ents if e != x), reverse=True)[:k]
        nbrs = [(s, e) for s, e in nbrs if s > 0.2]
        if not nbrs:
            return note + f"{x.title()}: nothing nearby by learned meaning."
        return note + self.nar.likeness(x, nbrs)

    # ── router (agnostic: parses the request's shape, not domain facts) ──
    def respond(self, q):
        ans = char_skills.answer_char_question(q, self.resolve_group)
        if ans is not None:
            return f"[exact] {ans}"
        ql = q.lower()
        m = re.search(r"(?:connect|relate|link|between|how (?:are|is|do))\b.*?\b([a-z][\w']+)\b.*?\b(?:and|to|with|&)\b.*?\b([a-z][\w']+)\b", ql)
        if m:
            return self.connect(m.group(1), m.group(2))
        m = re.search(r"(?:what|which|anything).*(?:connect|relate|link|tie).*?\bto\b\s+([a-z][\w']+)", ql) \
            or re.search(r"(?:explore|surprise me|hidden|tell me about|what about)\b.*?\b([a-z][\w']+)\b", ql)
        if m:
            return self.explore(m.group(1))
        m = re.search(r"\b([a-z][\w']+)\s+(?:like|similar)\b", ql) \
            or re.search(r"(?:like|similar to|reminiscent of|resembl\w*)\s+([a-z][\w']+)\b", ql)
        if m:
            return self.like(m.group(1))
        # fallback: explore the salient EXACT concept the user named (prefer capitalized, exact match only —
        # never fuzzy-resolve a stopword like "what")
        words = re.findall(r"\b([A-Za-z][\w']{2,})\b", q)
        exact = [w for w in words if w.lower() in self.cs.ents]
        pick = next((w for w in exact if w[0].isupper()), exact[0] if exact else None)
        if pick:
            return "Here's what's interesting about it —\n" + self.explore(pick)
        return ("I can connect concepts I've read about. Try: \"how are X and Y connected?\", "
                "\"what connects to X?\", \"what is X like?\", or a letter-count question.")


def build_space(args):
    """Build a ConceptSpace (single corpus or multi-domain). drop_hubs=False so the protagonist stays queryable."""
    if args.multi:
        labeled = [(f.stem.split("_")[0], f.read_text(errors="replace")) for f in sorted(CORP.glob("*.txt"))]
        return ConceptSpace.from_texts(labeled, embed_steps=args.steps, drop_hubs=False), "multi-domain"
    p = HERE / args.corpus
    return ConceptSpace(p.read_text(errors="replace"), embed_steps=args.steps, drop_hubs=False), args.corpus


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="pride_prejudice.txt")
    ap.add_argument("--multi", action="store_true", help="build over all corpora/ (cross-domain)")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--rebuild", action="store_true", help="force rebuild even if a cached space exists")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    kpath = HERE / "knowledge.txt"
    knowledge = kpath.read_text() if kpath.exists() else ""
    # persistence: cache the built ConceptSpace; instant startup on reuse (addressing = just data).
    label = "multi" if args.multi else Path(args.corpus).stem
    cache = HERE / ".mindcache" / label
    if cache.exists() and not args.rebuild:
        t0 = time.time()
        cs = ConceptSpace.load(cache)
        print(f"[mind] loaded cached space '{label}' ({len(cs.ents)} concepts, {len(cs.passages)} passages, {time.time()-t0:.1f}s)")
    else:
        cs, lbl = build_space(args)
        cs.save(cache)
        print(f"[mind] built + cached space '{label}' ({lbl})")
    mind = Mind(cs, knowledge=knowledge)

    if args.demo:
        for q in ["how many r's in strawberry?",
                  "how are Darcy and Wickham connected?",
                  "what connects to Pemberley?",
                  "what is Bingley like?",
                  "how are Elizabeth and Rome connected?",
                  "tell me about Lydia"]:
            print(f"\nyou> {q}\nmind> {mind.respond(q)}")
        return
    print("\n[mind] talk to me (quit to exit).\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[bye]"); break
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            print("mind>", mind.respond(q), "\n")


if __name__ == "__main__":
    main()

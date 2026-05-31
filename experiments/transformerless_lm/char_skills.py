"""char_skills.py — character-level capabilities that token-based LLMs (incl. 35B) get WRONG.

The user's insight: "how many D's in the days of the week?" — big LLMs miscount because BPE tokenization
makes them blind to characters. A char-level system sees every character, so letter questions are exact.

LAW-CLEAN (universal-substrate law): this file holds NO hand-coded knowledge — no days/months/word lists.
It has only (a) char-level LOGIC (counting), and (b) a pluggable RESOLVER that extracts a named group
("the days of the week") from a CORPUS generically ("X are A, B, C, ..."). Knowledge enters as DATA the
system reads (knowledge.txt / addressed memory), never as a literal in the code path. If a group isn't in
any corpus the system has, it honestly counts only the literal text it was given.
"""
import re

# generic "named group" definition pattern: "<group> are/is/: A, B, C, and D."  (corpus-derived, not a list)
_DEF_RE = lambda core: re.compile(
    r"(?:the\s+)?" + re.escape(core) + r"\s+(?:are|is|:)\s+([^.\n]+)", re.I)


def count_letter(text: str, letter: str, case_sensitive: bool = False) -> int:
    if not case_sensitive:
        text, letter = text.lower(), letter.lower()
    return text.count(letter[:1])


def count_letter_breakdown(words, letter: str):
    per = [(w, count_letter(w, letter)) for w in words]
    return per, sum(c for _, c in per)


def _parse_list(s: str):
    items = re.split(r",\s*(?:and\s+)?|\s+and\s+", s.strip())
    return [it.strip(" .;:'\"") for it in items if it.strip(" .;:'\"")]


def extract_group(name: str, corpus: str):
    """Find a definition of the named group in the corpus and return its members (or None). GENERIC —
    works for days, months, planets, colors, anything the corpus defines as 'X are A, B, C'."""
    core = re.sub(r"^the\s+", "", name.strip(), flags=re.I)
    m = _DEF_RE(core).search(corpus)
    if not m:
        return None
    members = _parse_list(m.group(1))
    return members if len(members) >= 2 else None


def make_corpus_resolver(corpus: str):
    """Return a resolver(name)->members|None backed by a corpus (the agnostic knowledge source)."""
    return lambda name: extract_group(name, corpus)


def answer_char_question(q: str, resolver=None):
    """Route a letter-counting question to an EXACT char-level answer. The group of words to count over
    comes from the RESOLVER (corpus/memory-derived); if unresolved, we count the literal text given —
    never a hard-coded list."""
    m = re.search(r"how many\s+([a-zA-Z])(?:'s|s)?\b.*?\bin\b\s+(.+?)\s*\??$", q.strip(), re.I)
    if not m:
        return None
    letter, subject = m.group(1), m.group(2).strip().rstrip("?.! ")
    members = resolver(subject) if resolver else None
    if members:                                  # corpus-derived group ("the days of the week" -> names)
        words, src = members, f"from corpus across {len(members)} item(s)"
    else:                                        # agnostic fallback: count the literal text we were given
        words = [w for w in re.split(r"[\s,]+", subject) if w]
        src = "in the given text"
    per, total = count_letter_breakdown(words, letter)
    parts = ", ".join(f"{w}={c}" for w, c in per if c) or "none"
    return f"{total}  ('{letter.upper()}' counted char-by-char {src}: {parts})"


if __name__ == "__main__":
    from pathlib import Path
    kpath = Path(__file__).parent / "knowledge.txt"
    corpus = kpath.read_text() if kpath.exists() else ""
    resolve = make_corpus_resolver(corpus)
    print("=== char-level letter counting (knowledge comes from knowledge.txt, NOT hard-coded) ===\n")
    for q in ["How many D's are in the days of the week?",
              "how many r's in strawberry?",                       # literal text -> agnostic fallback
              "How many s's are in the months of the year?",
              "how many e's in the planets?"]:                     # works if knowledge.txt defines planets
        print(f"Q: {q}\nA: {answer_char_question(q, resolve)}\n")
    days = extract_group("the days of the week", corpus)
    print(f"resolver extracted days-of-week from corpus: {days}")
    if days:
        _, tot = count_letter_breakdown(days, "d")
        print(f"verified D-count over corpus-derived days = {tot}  (no hard-coded list in the code)")

#!/usr/bin/env python3
"""engine.py — THE WEB-NATIVE CONVERSATIONAL ENGINE  (the full loop, talking)

Your opening example, wired from the proven parts — no token-prediction LM anywhere:
  USER: "I have a question about quantum physics"
  LM  : (parses the web) does the topic RESOLVE densely?  → competence detector (the resolve oracle)
        if yes → "I'm knowledgeable on that — we could talk about [kask-near categories]"
        then ANSWER = realize (hybrid: template structure + heal) the query's resolved neighborhood
        if no  → honest "I don't have dense knowledge there"

Everything — words, knowledge, categories, fluency — is derived from the one web.

  python engine.py --demo            # scripted exchanges (no TTY)
  python engine.py                   # interactive REPL
"""
import sys
from itertools import combinations
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from langexec import open_web, edge_w
from fluency import get_fluency_model
from thinkloop import resolve, concepts_of
from realize import hybrid_realize, heal_surface
from create import create as create_bridge

ENGINE_TRAIN = 80000


def build_frame(k, topk=1500):
    """The query-FRAME words (question/tell/explain/about…) are high-degree hubs. Strip them (agnostic:
    derived by degree, not a hand-coded list). Degree alone can't fully separate frame from common topics
    (light/war rank above explain), so this pairs with the interrogative-context filter below."""
    return set(w for w, _ in sorted(k.deg.items(), key=lambda x: -x[1])[:topk])


# Two anchor families (measured probes, not classifier lists). INTERROGATIVE markers signal question/
# discourse words (explain/tell/question); RELATIONAL markers signal relate-intent scaffolding
# (relationship/between/connection). Kept SEPARATE because averaging them into one set dilutes the
# signal for words that lean one way but not the other (e.g. 'tell' co-occurs with how/what but NOT with
# relationship/between — a merged mean cancels it out and lets it leak as a topic).
ID_ANCHORS = ["how", "what", "why", "which", "whether", "question", "explain", "describe",
              "define", "discuss", "tell", "compare"]
REL_ANCHORS = ["relationship", "relation", "between", "connection", "difference", "versus", "related"]
WH_ANCHORS = ID_ANCHORS                      # back-compat alias for callers passing a single anchor set

# Relational connectives — a CLOSED-CLASS grammatical set (like prepositions/conjunctions, same category
# as the derived stopwords). MEASURED finding: these can't be separated from content words distributionally
# (relationship rank 6669 is RARER than the topic 'sleep' rank 635; co-occurrence flags genes/disease too).
# Listing a finite grammatical class is NOT hand-coding ontology (which would classify CONTENT/topics) —
# it's the same as a stopword list. These are stripped so a relate-query keys on its TOPICS, not its scaffolding.
RELATIONAL_CONNECTIVES = {
    "relationship", "relationships", "relation", "relations", "related", "relate", "relating",
    "connection", "connections", "connected", "connect", "link", "links", "linked", "linking",
    "association", "associations", "associated", "associate", "correlation", "correlated",
    "between", "versus", "vs", "difference", "differences", "comparison", "compare", "compared",
}
FRAME_WH_THRESH = 0.7                         # interrogative threshold (explain/relate/discuss ~0.8-0.95)
FRAME_REL_THRESH = 1.3                        # relational threshold — HIGHER: true scaffolding (relationship
                                             # 2.22, between 2.13) scores far above a content word merely
                                             # pulled into "relationship between X and <disease>" (0.88).


def frame_score(k, c, anchors):
    """Mean above-/below-chance association of c with the given anchors. High ⇒ lives in those contexts."""
    vals = []
    for w in anchors:
        if w == c:
            continue
        e = edge_w(k, c, w)
        if e is not None:
            vals.append(k.assoc(c, w, e))
    return sum(vals) / len(vals) if vals else -9.0


def is_frame(k, c, frame, anchors=None):
    """A word is FRAME if: (a) derived stopword, OR (b) common AND leans interrogative (in degree-frame
    AND id-score>0 — catches explain/tell/question), OR (c) strongly interrogative regardless of
    frequency (id-score>0.7 — relate/discuss/describe), OR (d) strongly relational (rel-score>0.7 —
    relationship/between/connection). The 'AND id>0' guard in (b) protects common TOPICS (memory/light/
    war lean NON-interrogative → survive). Neither degree nor either anchor-family alone separates frame
    from topic (measured); the composition does. Agnostic — degree + two interrogative-context probes."""
    if c in k.stop or c in RELATIONAL_CONNECTIVES:
        return True
    # INTERROGATIVE family flags discourse verbs (explain/tell/question). Relational scaffolding is handled
    # by the closed-class RELATIONAL_CONNECTIVES set above, NOT by co-occurrence (which wrongly caught
    # science content words like genes/disease that appear in "relationship between …").
    ids = frame_score(k, c, ID_ANCHORS)
    return (c in frame and ids > 0.0) or ids > FRAME_WH_THRESH


def topic_concepts(k, qc, frame):
    """The query's actual subject = its addressed, non-hub, non-stop concepts."""
    return [c for c in qc if c in k.nodeset and c not in frame and c not in k.stop]


def near_categories(k, c, n=6, pool=40):
    """kask-near: the topic's facets. Above-chance (hub-damped) neighbors, THEN re-ranked by mutual
    coherence — real same-domain facets co-occur with EACH OTHER (spinal–cerebral–nerve); scattered
    cross-lingual translations (the multilingual web's parallel-text artifact) don't, so they sink."""
    nb = []
    for b, w in k.db.execute("SELECT b,w FROM edges WHERE a=? ORDER BY w DESC LIMIT 400", (c,)):
        if b == c or b in k.stop:
            continue
        a = k.assoc(c, b, w)
        if a > 0:
            nb.append((a, b))
    nb.sort(reverse=True)
    cand = [b for _a, b in nb[:pool]]

    def coherence(x):                                   # how many other candidates x co-occurs with above chance
        s = 0
        for y in cand:
            if y == x:
                continue
            w = edge_w(k, x, y)
            if w is not None and k.assoc(x, y, w) > 0:
                s += 1
        return s

    return sorted(cand, key=lambda x: -coherence(x))[:n]


def relate_pmi(k, a, b):
    w = edge_w(k, a, b)
    return k.assoc(a, b, w) if w is not None else -1.0


def respond(k, model, query, frame):
    qc = concepts_of(k, query)
    # TOPICS = addressed query concepts that are neither stopwords, degree-hubs, nor interrogative-context
    # frame words. The union frame-filter (degree ∪ wh-context) is what lets "light"/"gravity" survive
    # while "relate"/"question"/"about" are stripped — even though degree alone can't separate them.
    topics = [c for c in dict.fromkeys(qc)
              if c in k.nodeset and c not in k.stop and not is_frame(k, c, frame, WH_ANCHORS)]
    # INTENT ROUTING (agnostic): ≥2 surviving TOPICS = a RELATE query → show their connection. Pick the
    # pair with the highest EMBEDDING similarity relate(a,b) (most semantically sensible to explain) —
    # robust to a leaked scaffolding word (sleep·disease ≫ relationship·sleep) and to pairs with no direct
    # edge (create.py then BRIDGES them). create.py decides direct-relation vs cross-source bridge.
    if len(topics) >= 2:
        a, b = max(combinations(topics, 2), key=lambda p: k.relate(p[0], p[1]))
        return f"  LM [relate '{a}' ⇄ '{b}']:\n{create_bridge(k, model, a, b)}"

    # RECALL intent: a single topic.
    if not topics:
        return "  LM: I don't recognize a specific topic I hold knowledge about there."
    main = min(topics, key=lambda c: k.deg.get(c, 0))       # MOST SPECIFIC surviving topic (not the framing verb)
    cats = near_categories(k, main, 6)
    if len(cats) < 3:
        return (f"  LM: I only have sparse knowledge around '{main}' ({len(cats)} facets) — "
                f"not enough to speak on it confidently.")
    # ANSWER substance = the query's topic concepts + the topic's resolved neighborhood (spreading activation)
    substance = list(dict.fromkeys(topics + cats[:5]))
    said = heal_surface(hybrid_realize(k, model, substance, k.stop))
    r = resolve(k, [w for w in WORDS(said) if w in k.nodeset])["pos_pmi"]
    return (f"  LM: I'm knowledgeable about '{main}' — we could talk about: {', '.join(cats[:5])}.\n"
            f"      A grounded thought: {said}\n"
            f"      [self-check — does my own answer resolve? {r:.2f}]")


def WORDS(s):
    import re
    return re.findall(r"[a-z]+(?:'[a-z]+)?", s.lower())


DEMO = [
    "I have a question about gravity",
    "how do light and gravity relate",
    "tell me about memory and the brain",
    "what do you know about war and empire",
    "explain the flrbzy of the quux",          # nonsense → should honestly decline
]


def main():
    k = open_web()
    max_pid = k.db.execute("SELECT MAX(pid) FROM passages").fetchone()[0]
    print("[engine] loading fluency model (cached) ...", flush=True)
    model = get_fluency_model(k, ENGINE_TRAIN, max_pid)
    frame = build_frame(k)
    print(f"[engine] ready. web nodes={len(k.nodes):,}  frame-hubs stripped={len(frame)}\n", flush=True)

    if "--demo" in sys.argv:
        for q in DEMO:
            print(f"USER: {q}")
            print(respond(k, model, q, frame))
            print()
        return
    print("web-native engine. type a question (ctrl-D to exit).")
    while True:
        try:
            q = input("USER: ").strip()
        except EOFError:
            break
        if q:
            print(respond(k, model, q, frame))


if __name__ == "__main__":
    main()

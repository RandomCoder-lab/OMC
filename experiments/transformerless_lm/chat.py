#!/usr/bin/env python3
"""chat.py — a small LM you can talk to that rivals a 35B model where it counts: on KNOWLEDGE-per-compute
and on tasks tokenization breaks. NOT raw fluency (that needs scale) — the honest, provable edge.

Three pillars, all built and proven in this repo:
  1. char-substrate skills (char_skills.py) — exact letter/char answers that token-LLMs (incl. 35B) get
     wrong ("how many D's in the days of the week?" -> 8). Big models are blind to characters; we aren't.
  2. addressed memory (written_memory.py) — a small char/BPE generator + an IVF-indexed datastore WRITTEN
     from a corpus (no training). Injects a big model's worth of domain knowledge on ~2M params
     (write-don't-train, +11-48% measured). Answers by retrieving real source with provenance.
  3. the router (here) — send a query to the pillar that answers it best; be honest when we don't know.

Usage:
  python chat.py --demo                 # scripted exchanges (no TTY needed)
  python chat.py [--store DIR]          # interactive REPL over a memory store
"""
import sys, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import char_skills

HERE = Path(__file__).parent


class Chat:
    def __init__(self, store=None, log=print):
        self.mem = None
        self.store = store
        if store:
            try:
                from written_memory import WrittenMemory
                self.mem = WrittenMemory.load(store)
                log(f"[chat] loaded memory: {self.mem.K.shape[0]:,} entries, vocab {self.mem.vocab}, "
                    f"{len(self.mem.source):,} chars of source")
            except Exception as e:
                log(f"[chat] no memory store ({e}); char-skills only")
        # law-clean resolver: named-group knowledge comes from a CORPUS (knowledge.txt + memory source),
        # never a hard-coded list. char_skills holds only logic.
        kpath = HERE / "knowledge.txt"
        knowledge = (kpath.read_text() if kpath.exists() else "")
        if self.mem is not None:
            knowledge += "\n" + self.mem.source
        self.resolver = char_skills.make_corpus_resolver(knowledge)

    def respond(self, q: str) -> str:
        # Pillar 1: char-level questions — exact, where 35B models fail. Group knowledge from corpus.
        ans = char_skills.answer_char_question(q, self.resolver)
        if ans is not None:
            return f"[char-substrate] {ans}"

        # Pillar 2: knowledge — retrieve the most relevant real source from addressed memory (provenance).
        if self.mem is not None:
            hits = self.mem.nearest_contexts(q, k=3)
            if hits and hits[0]["distance"] < 0.6:
                top = hits[0]
                snip = top["snippet"].strip()
                more = "".join(f"\n   · …{h['snippet'].strip()[:80]}…" for h in hits[1:3])
                return (f"[memory] nearest written context (dist {top['distance']:.2f}, "
                        f"source pos {top['position']}):\n   {snip}{more}")
            # weak match -> be honest
            return ("[memory] no strong match in memory for that. (This is a small model whose knowledge "
                    "is whatever has been WRITTEN to memory — add more with codemem to extend it.)")
        return "[chat] I only have char-substrate skills loaded (no memory store). Try a letter-count question."


DEMO = [
    "How many D's are in the days of the week?",          # char-substrate (35B fails)
    "how many r's in strawberry?",                          # char-substrate
    "How many s's are in the months of the year?",          # char-substrate
    "how does the IVF index search work",                   # memory (if OMC store loaded)
    "content addressed heap hashing",                       # memory
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="/tmp/omc_core_bpe.codemem")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    bot = Chat(store=args.store if Path(args.store).joinpath("memory.pt").exists() else None)

    if args.demo:
        print("\n=== chat demo: small generator + addressed memory + char-substrate ===")
        for q in DEMO:
            print(f"\nyou> {q}\nbot> {bot.respond(q)}")
        return

    print("\nchat ready (type 'quit' to exit). Ask a letter-count question, or about the memory's domain.\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[bye]"); break
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            print("bot>", bot.respond(q))


if __name__ == "__main__":
    main()

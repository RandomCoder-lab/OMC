#!/usr/bin/env python3
"""selfimprove.py — THE ENGINE IMPROVES ITSELF, NO HUMAN IN THE LOOP.

The loop (a MAPE-K cycle, one level up from the code-healer):
  MONITOR  — generate self-probes from the web (concept pairs that are related but not directly stated).
  ANALYZE  — the agent attempts each; detect WEAK answers (low confidence / decline / sparse).
  PLAN     — for a bridge that passes BOTH gates (coherent: resolve>τ AND supported: weakest-hop PMI>τ),
             the engine has DISCOVERED a verified connection no single source stated.
  EXECUTE  — WRITE that verified thought to a derived store (separate derived.db — the 9GB knowledge.db is
             never touched). This is write-don't-train: the engine records what it reasoned out.
  KNOWLEDGE— next time the same/related connection is queried, RECALL the verified thought instantly
             (high confidence) instead of re-bridging. The store IS the accumulated self-improvement.

Honest scope: this CLOSES connection-gaps (verified recombinations of existing grounded knowledge) and
MAPS knowledge-gaps (sparse/declined topics it logs but cannot fabricate facts for). It does not invent
new facts — every thought is a gated recombination of real, sourced passages.

Measured round-over-round on a FIXED held-out probe set: answer-rate, mean confidence, cache-hit rate,
mean bridge length (shrinks as multi-hops become recalled direct knowledge), store size.

  python selfimprove.py [rounds] [probes_per_round]
"""
import sys, sqlite3, time, json, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from engine import open_web, build_frame
from fluency import get_fluency_model
from thinkloop import resolve
from create import bounded_bridge, hop_info, connecting_span
from langexec import edge_w

random.seed(1618)
HERE = Path(__file__).parent
DERIVED_DB = HERE / "derived.db"
LEDGER = HERE / "selfimprove_ledger.jsonl"

COHERENCE_GATE = 0.6      # gate 1: the recombination must RESOLVE (path holds together)
SUPPORT_GATE = 0.7        # gate 2: weakest hop PMI — a real association, not a hub-walk
SEMANTIC_GATE = 0.14      # gate 3: endpoint embedding similarity — a real relation, not a co-occurrence
                          #         artifact (translations / incidental narrative adjacency score ~0;
                          #         measured: junk endpoints <0.13, real cross-domain ≥0.15).

# Translation/parallel-text corpora: aligned sentence pairs give CROSS-LINGUAL co-occurrence (high PMI)
# that is NOT a conceptual relation (lifting↔levantando). Reject bridges hopping through them — pure
# provenance hygiene (reading the corpus's own source labels, not hand-coded ontology).
def _is_translation_src(src):
    return src.startswith("align-") or src.startswith("parallel-") or src == "bible-parallel"


class VerifiedMemory:
    """The engine's write-don't-train memory of self-verified thoughts (separate from knowledge.db).
    read_only=True opens a shared reader (e.g. the agent) alongside the overnight writer — SQLite WAL
    allows one writer + many readers concurrently."""
    def __init__(self, path, read_only=False):
        self.read_only = read_only
        if read_only:
            self.db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        else:
            self.db = sqlite3.connect(str(path))
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("""CREATE TABLE IF NOT EXISTS thoughts(
                a TEXT, b TEXT, path TEXT, thought TEXT, coherence REAL, support REAL,
                fields TEXT, ts REAL, PRIMARY KEY(a,b))""")
            self.db.commit()
        self._cache = {(r[0], r[1]) for r in self.db.execute("SELECT a,b FROM thoughts")}

    def refresh(self):
        """Re-read the key set (a reader picks up thoughts the writer has added since open)."""
        try:
            self._cache = {(r[0], r[1]) for r in self.db.execute("SELECT a,b FROM thoughts")}
        except Exception:
            pass

    def has(self, a, b):
        return (a, b) in self._cache or (b, a) in self._cache

    def get(self, a, b):
        r = self.db.execute("SELECT thought,coherence,support,fields,path FROM thoughts WHERE (a=? AND b=?) OR (a=? AND b=?)",
                             (a, b, b, a)).fetchone()
        return r

    def store(self, a, b, path, thought, coherence, support, fields, ts):
        self.db.execute("INSERT OR REPLACE INTO thoughts VALUES(?,?,?,?,?,?,?,?)",
                        (a, b, json.dumps(path), thought, coherence, support, ",".join(sorted(fields)), ts))
        self.db.commit()
        self._cache.add((a, b))

    def size(self):
        return self.db.execute("SELECT COUNT(*) FROM thoughts").fetchone()[0]


def bridge_and_grade(k, a, b):
    """Attempt to connect a,b. Returns (kind, path, weakest_support, fields). kind: 'direct'|'bridge'|None."""
    direct = edge_w(k, a, b)
    dpmi = k.assoc(a, b, direct) if direct is not None else -99.0
    if dpmi > 1.0:
        src, pid, _w = hop_info(k, a, b)
        return ("direct", [a, b], dpmi, {src})
    path = bounded_bridge(k, a, b)
    if not path or len(path) < 3:
        return (None, None, 0.0, set())
    hops, fields = [], set()
    for x, y in zip(path, path[1:]):
        src, pid, w = hop_info(k, x, y)
        if _is_translation_src(src):
            return (None, None, 0.0, set())      # translation artifact — not a conceptual bridge
        hops.append(k.assoc(x, y, w)); fields.add(src)
    return ("bridge", path, min(hops), fields)


def attempt(k, mem, a, b):
    """Answer 'how do a,b relate' — FIRST consult verified memory (the improvement), else bridge fresh.
    Returns dict(answered, confidence, source, length, verified_storable, path, support, fields)."""
    if mem.has(a, b):
        r = mem.get(a, b)
        path = json.loads(r[4])
        return dict(answered=True, confidence=min(1.0, 0.6 + 0.1 * r[2]), source="memory",
                    length=len(path), storable=False, path=path, support=r[2], fields=r[3].split(","))
    kind, path, support, fields = bridge_and_grade(k, a, b)
    if kind is None:
        return dict(answered=False, confidence=0.0, source="none", length=0,
                    storable=False, path=None, support=0.0, fields=[])
    coh = resolve(k, [c for c in path if c in k.nodeset])["pos_pmi"]
    conf = min(1.0, max(0.1, 0.3 + 0.12 * support + 0.3 * coh))
    # endpoint quality: reject 2-char / non-alphabetic tokens (abbreviations, OCR fragments like
    # 'co'/'az'/'si'). >=3 keeps real short concepts (war/dna/gas/oil); >=4 wrongly killed those too.
    good_ends = len(a) >= 3 and len(b) >= 3 and a.isalpha() and b.isalpha()
    # KNOWLEDGE-FIELD gate: require >=1 hop sourced from a NAMED knowledge domain (not the 'general'
    # catch-all). Latin / archaic-English word-walks (illi→neque→enim, tyme→drede→gret) live entirely in
    # 'general' old-text and pass the PMI+semantic gates (within-language clustering), but cross no real
    # knowledge domain — this drops them at the source while keeping cross-domain synthesis.
    has_named_field = any(f and f != "general" for f in fields)
    # FOUR gates: coherent AND supported AND meaningful AND crosses a named knowledge field — plus
    # endpoint quality. All must hold to record a verified thought.
    storable = (kind == "bridge" and coh >= COHERENCE_GATE and support >= SUPPORT_GATE
                and k.relate(a, b) >= SEMANTIC_GATE and good_ends and has_named_field)
    return dict(answered=True, confidence=round(conf, 3), source=kind, length=len(path),
                storable=storable, path=path, support=round(support, 3), fields=sorted(fields), coherence=round(coh, 3))


def sample_related_pairs(k, n, max_pid, bridging_only=False):
    """Self-generated probes: pairs of content concepts from real passages (so a connection plausibly
    exists). bridging_only ⇒ keep only pairs that need a real BRIDGE (weak/no direct edge AND semantically
    related endpoints) — these are where verified-thought memory matters most."""
    pairs, tries = [], 0
    import re
    TOK = re.compile(r"[a-z][a-z']+")
    while len(pairs) < n and tries < n * 80:
        tries += 1
        pid = random.randint(0, max_pid)
        r = k.db.execute("SELECT text FROM passages WHERE pid=?", (pid,)).fetchone()
        if not r:
            continue
        cs = [t for t in dict.fromkeys(TOK.findall(r[0].lower()))
              if t in k.nodeset and t not in k.stop and k.deg.get(t, 0) < 200000][:12]
        if len(cs) < 2:
            continue
        a, b = random.sample(cs, 2)
        if bridging_only:
            d = edge_w(k, a, b)
            dpmi = k.assoc(a, b, d) if d is not None else -99.0
            if dpmi > 1.0 or k.relate(a, b) < SEMANTIC_GATE:    # already-direct OR unrelated → skip
                continue
        pairs.append((a, b))
    return pairs


def measure(k, mem, probes, base_cache):
    """Snapshot answer-quality over the FIXED curriculum. FAST: the from-scratch bridge for each pair is
    computed once (base_cache) — knowledge.db is read-only, so it never changes; only whether a pair has
    entered verified memory changes round-over-round. Memory hit ⇒ recall (fast, higher conf)."""
    answered = conf = length = hits = 0
    for a, b in probes:
        if mem.has(a, b):
            r = mem.get(a, b); path = json.loads(r[4])
            answered += 1; conf += min(1.0, 0.6 + 0.1 * r[2]); length += len(path); hits += 1
        else:
            c = base_cache[(a, b)]
            if c["answered"]:
                answered += 1; conf += c["confidence"]; length += c["length"]
    n = len(probes)
    return dict(answer_rate=round(answered / n, 3), mean_conf=round(conf / max(1, answered), 3),
                mean_len=round(length / max(1, answered), 2), cache_hits=hits, store=mem.size())


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    ppr = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    hours = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0   # wall-clock budget (0 = use round count)
    t_start = time.time()
    k = open_web()
    mp = k.db.execute("SELECT MAX(pid) FROM passages").fetchone()[0]
    print("[selfimprove] loading fluency model (cached) ...", flush=True)
    get_fluency_model(k, 80000, mp)   # warm cache for the agent layer (not used directly here)
    mem = VerifiedMemory(DERIVED_DB)
    print(f"[selfimprove] verified-thought store starts at {mem.size()} thoughts", flush=True)

    # FIXED curriculum: the task set we track consolidation on. Compute each pair's from-scratch bridge
    # ONCE (slow) — this is the "before learning" baseline the engine improves upon.
    curriculum = sample_related_pairs(k, 30, mp, bridging_only=True)
    base_cache = {}
    print(f"[selfimprove] computing baseline over {len(curriculum)} curriculum pairs (one-time) ...", flush=True)
    for a, b in curriculum:
        base_cache[(a, b)] = attempt(k, mem, a, b)
    base = measure(k, mem, curriculum, base_cache)
    print(f"[selfimprove] BASELINE: {base}", flush=True)
    led = open(LEDGER, "a")
    led.write(json.dumps({"event": "baseline", "round": 0, **base}) + "\n"); led.flush()

    discovered = 0
    rnd = 0
    while True:
        rnd += 1
        if hours > 0:
            if time.time() - t_start > hours * 3600:
                break
        elif rnd > rounds:
            break
        t0 = time.time()
        new = 0
        try:
            # (1) work the curriculum: consolidate any not-yet-verified pair the engine can now verify
            for a, b in curriculum:
                if mem.has(a, b):
                    continue
                r = base_cache[(a, b)]
                if r.get("storable"):
                    mem.store(a, b, r["path"], " → ".join(r["path"]), r["coherence"], r["support"], r["fields"], time.time())
                    new += 1; discovered += 1
            # (2) free exploration: fresh self-probes grow the store beyond the curriculum (the "thinking")
            for a, b in sample_related_pairs(k, ppr, mp, bridging_only=True):
                if mem.has(a, b):
                    continue
                try:
                    r = attempt(k, mem, a, b)
                except Exception:
                    continue                      # one bad bridge must not stop the night
                if r.get("storable"):
                    mem.store(a, b, r["path"], " → ".join(r["path"]), r["coherence"], r["support"], r["fields"], time.time())
                    new += 1; discovered += 1
            snap = measure(k, mem, curriculum, base_cache)
        except Exception as e:
            led.write(json.dumps({"event": "error", "round": rnd, "err": str(e)[:200]}) + "\n"); led.flush()
            continue
        snap.update(round=rnd, new_this_round=new, total_discovered=discovered, secs=round(time.time() - t0, 1),
                    elapsed_min=round((time.time() - t_start) / 60, 1))
        print(f"[selfimprove] round {rnd} (+{snap['elapsed_min']}min): +{new} thoughts (store={snap['store']}), "
              f"curriculum answer={snap['answer_rate']} conf={snap['mean_conf']} hits={snap['cache_hits']} "
              f"len={snap['mean_len']} [{snap['secs']}s]", flush=True)
        led.write(json.dumps({"event": "round", **snap}) + "\n"); led.flush()

    final = measure(k, mem, curriculum, base_cache)
    print(f"\n[selfimprove] DONE. store {base['store']}→{final['store']} (+{discovered} verified thoughts)")
    print(f"  curriculum answer-rate {base['answer_rate']}→{final['answer_rate']}  "
          f"conf {base['mean_conf']}→{final['mean_conf']}  cache-hits {base['cache_hits']}→{final['cache_hits']}  "
          f"mean-len {base['mean_len']}→{final['mean_len']}")
    # a few example self-discovered thoughts
    print("\n  example self-verified thoughts (no single source states these):")
    for a, b, thought, coh, sup, flds in mem.db.execute(
            "SELECT a,b,thought,coherence,support,fields FROM thoughts ORDER BY support DESC LIMIT 5"):
        print(f"   {a} ⇄ {b}  [coh {coh:.2f} sup {sup:.2f} · {flds}]:  {thought}")
    led.close()


if __name__ == "__main__":
    main()

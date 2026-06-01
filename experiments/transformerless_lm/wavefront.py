#!/usr/bin/env python3
"""wavefront.py — quantum-INSPIRED multi-path diffusion over the web. Not quantum (no qubits, no speedup):
the classical shadow of a path integral / quantum walk. Instead of one greedy step (which drifts into the
densest swamp), a WAVEFRONT spreads from the seed(s) along PMI-weighted edges — every path contributes at
once, as an amplitude distribution — then we COLLAPSE to where amplitude concentrates.

Engine: forward-push Personalized PageRank (Andersen–Chung–Lang). LOCAL — the restart keeps the wavefront
near the seeds, so it only touches the few-thousand reachable nodes, querying neighbors lazily from the DB.
Tractable on 100M edges without loading them.

INTERFERENCE / hub-suppression: raw diffusion (PageRank) rewards generic hubs reachable from everything. We
divide amplitude by a degree penalty — the destructive-interference-with-the-background move — so SPECIFIC
multi-path agreement survives and everything-connects-to-it hubs cancel. (Works even if node_deg is missing,
using neighbor-count degree — so this runs before node_expand finishes.)

Modes:
  pool X        — diffuse from X; collapse to the concepts all paths pool into (where X 'leads').
  bridge A B    — diffuse from A and from B; collapse to where the two wavefronts CONSTRUCTIVELY OVERLAP
                  (pa·pb) = the grounded bridges between them. "Explore all paths from both ends, read the
                  answer where they meet." A robust, drift-proof deep_connect.

  python wavefront.py bridge gravity time
  python wavefront.py pool algorithm
"""
import sys, math
from collections import defaultdict, deque
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from kdb import KnowledgeDB, load_embedding

DB = Path(__file__).parent / "knowledge.db"


class WaveFront:
    def __init__(self, k):
        self.k = k
        self._nbr = {}
        self._deg = {}

    def neighbors_w(self, u):
        """PMI-weighted knowledge neighbors of u (translation/parallel edges and stopwords excluded)."""
        if u in self._nbr:
            return self._nbr[u]
        out = {}
        for b, src, w in self.k.db.execute(
                "SELECT b,src,w FROM edges WHERE a=? AND src NOT LIKE 'align-%' AND src NOT LIKE 'parallel-%'", (u,)):
            if b in self.k.stop or b == u:
                continue
            wt = self.k.assoc(u, b, w) if self.k.deg else float(w)   # PMI if grounded, else raw
            if wt > 0:
                out[b] = max(out.get(b, 0.0), wt)
        self._nbr[u] = out
        return out

    def degree(self, n):
        if n in self._deg:
            return self._deg[n]
        d = self.k.deg.get(n) if self.k.deg else None
        if d is None:
            d = self.k.db.execute("SELECT COUNT(*) FROM edges WHERE a=?", (n,)).fetchone()[0]
        self._deg[n] = d
        return d

    def diffuse(self, seeds, alpha=0.2, eps=1e-5, max_touch=2500):
        """Forward-push PPR: amplitude p over nodes the wavefront reaches from seeds. Local + lazy."""
        p = defaultdict(float); r = defaultdict(float)
        for s in seeds:
            if s in self.k.stoi:
                r[s] = 1.0 / len(seeds)
        work = deque(s for s in seeds if s in self.k.stoi); inq = set(work)
        touched = set()
        while work:
            u = work.popleft(); inq.discard(u)
            ru = r[u]
            if ru < eps:
                continue
            p[u] += alpha * ru; r[u] = 0.0
            if len(touched) >= max_touch:
                continue
            touched.add(u)
            nbrs = self.neighbors_w(u); tot = sum(nbrs.values()) or 1.0
            rem = (1 - alpha) * ru
            for v, wv in nbrs.items():
                r[v] += rem * (wv / tot)
                if r[v] >= eps and v not in inq:
                    work.append(v); inq.add(v)
        return p

    def pool(self, x, topk=12):
        x = x.lower()
        p = self.diffuse([x])
        scored = sorted(((amp / (1 + math.log1p(self.degree(n))), amp, n)
                         for n, amp in p.items() if n != x and n not in self.k.stop), reverse=True)
        return scored[:topk]

    def infer(self, seeds, topk=10):
        """Multi-seed inference: diffuse a wavefront from EACH premise, then collapse to where they ALL
        constructively overlap — the geometric mean of amplitudes (a conclusion must be reached strongly from
        EVERY premise, not just one), hub-suppressed. That node is what the premises JOINTLY ground.
        Honest: this is associative/abductive convergence — 'what do these together point to', grounded — not
        deductive proof of 'what logically follows'."""
        seeds = [s.lower() for s in seeds]
        ps = [(s, self.diffuse([s])) for s in seeds if s in self.k.stoi]
        if len(ps) < 2:
            return [], [s for s in seeds if s not in self.k.stoi]
        common = set(ps[0][1])
        for _, p in ps[1:]:
            common &= set(p)
        out = []
        for n in common:
            if n in seeds or n in self.k.stop:
                continue
            amps = [p[n] for _, p in ps]
            geo = math.exp(sum(math.log(a) for a in amps) / len(amps))   # geometric mean over premises
            score = geo / (1 + math.log1p(self.degree(n)))
            # which premises have a DIRECT grounded edge to the conclusion (vs reached only by path)
            direct = [s for s, _ in ps
                      if self.k.db.execute("SELECT 1 FROM edges WHERE a=? AND b=?", (s, n)).fetchone()]
            out.append((score, n, direct))
        out.sort(reverse=True)
        return out[:topk], [s for s in seeds if s not in self.k.stoi]

    def bridge(self, a, b, topk=10):
        a, b = a.lower(), b.lower()
        pa = self.diffuse([a]); pb = self.diffuse([b])
        both = set(pa) & set(pb)
        scored = []
        for n in both:
            if n in (a, b) or n in self.k.stop:
                continue
            inter = pa[n] * pb[n] / (1 + math.log1p(self.degree(n)))   # constructive overlap, hub-suppressed
            scored.append((inter, n))
        scored.sort(reverse=True)
        return scored[:topk]


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    s, E, n, e = load_embedding()
    wf = WaveFront(KnowledgeDB(str(DB), s, E, n, e))
    if not wf.k.deg:
        print("⚠ node_deg missing — diffusing on raw weights; hub-suppression still active via degree penalty.")
    cmd = sys.argv[1]
    if cmd == "pool" and len(sys.argv) >= 3:
        print(f"WAVEFRONT pool from '{sys.argv[2]}' — where all paths concentrate:")
        for sc, amp, node in wf.pool(sys.argv[2]):
            print(f"   {node:20} amp·{amp:.2e}  score·{sc:.2e}")
    elif cmd == "infer" and len(sys.argv) >= 4:
        seeds = [s.lower() for s in sys.argv[2:]]
        print(f"WAVEFRONT infer — what do premises {seeds} JOINTLY ground:")
        res, missing = wf.infer(seeds)
        if missing:
            print(f"   (not addressable, ignored: {missing})")
        if not res:
            print("   (no joint grounding — premises don't converge on a shared concept)")
        for score, node, direct in res:
            d = f" [direct from: {','.join(direct)}]" if direct else " [via paths]"
            print(f"   ⇒ {node:18} joint·{score:.2e}{d}")
        # auto-remember the discovery (only when grounded — never deposit blind reasoning)
        if res and wf.k.deg:
            from memory import Memory
            top = res[0][1]
            mid, act = Memory(wf.k).remember_discovery(
                seeds + [top], f"premises {'+'.join(seeds)} jointly ground → {top}", "derived")
            print(f"   · auto-remembered discovery #{mid} ({act}): {'+'.join(seeds)} ⇒ {top}")
    elif cmd == "bridge" and len(sys.argv) >= 4:
        a, b = sys.argv[2], sys.argv[3]
        print(f"WAVEFRONT bridge '{a}' ⟷ '{b}' — where the two wavefronts constructively meet:")
        res = wf.bridge(a, b)
        if not res:
            print("   (no overlap — the wavefronts don't meet within range)")
        for inter, node in res:
            # show the grounded two-sided link
            ea = wf.k.db.execute("SELECT pid FROM edges WHERE a=? AND b=?", (a.lower(), node)).fetchone()
            eb = wf.k.db.execute("SELECT pid FROM edges WHERE a=? AND b=?", (node, b.lower())).fetchone()
            tag = []
            if ea: tag.append("⟨" + wf.k._passage(ea[0])[1] + "⟩")
            print(f"   {node:20} interfere·{inter:.2e}   {a}–{node}{'·direct' if ea else ''}  {node}–{b}{'·direct' if eb else ''}")
        # auto-remember the bridge discovery (only when grounded)
        if res and wf.k.deg:
            from memory import Memory
            top = res[0][1]
            mid, act = Memory(wf.k).remember_discovery(
                [a.lower(), b.lower(), top], f"{a} ⟷ {b} bridged by → {top}", "derived")
            print(f"   · auto-remembered discovery #{mid} ({act}): {a}⟷{b} via {top}")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()

"""connect.py — THE LOOP CLOSED: the learned meaning-judge drives the grounded navigator.

The four primitives, integrated into one system that connects dots by its OWN judgment:
  WHERE      addressed inverted index (entity -> postings)            [recall 1.00, cheap]
  GROUND     follow an edge only if a real passage co-mentions both   [verify per hop, quoted]
  MEANING    distributional embedding learned from the corpus         [AUC 0.83, zero labels]
  REASON     meaning-GUIDED best-first walk over the grounded graph   [steered by meaning, not blind]

Two capabilities:
  - directed(A, B): connect two concepts by the most MEANINGFUL grounded path (best-first steered by
    learned relatedness to the goal), vs blind BFS-shortest. Metric: path coherence = mean learned
    relatedness of intermediate nodes to the goal.
  - discover(A): surface concepts that are MEANINGFULLY related to A (learned) but DON'T directly
    co-occur (non-obvious), each justified by a grounded path. "The dots you didn't consider" — found
    and verified by the system itself.

All agnostic / corpus-derived (law-clean). The system decides what's meaningful; grounding keeps it honest.
"""
import sys, re, time, json, math, random, heapq
from collections import deque
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from nl_ground import clean_text, split_passages, extract_entities

HERE = Path(__file__).parent
WORD = re.compile(r"[a-z]+")


def train_embedding(text, dim=64, window=5, max_vocab=6000, steps=4000, neg=5, seed=0, strip_top=1,
                    return_matrix=False):
    """Distributional meaning from the corpus's own co-occurrence (skip-gram neg-sampling). Returns
    (vec_fn, stoi), or (vec_fn, stoi, E) when return_matrix (E = the normalized embedding, for persistence).
    strip_top removes dominant common directions (all-but-the-top) to de-hub."""
    torch.manual_seed(seed)
    toks = WORD.findall(text.lower())
    freq = {}
    for t in toks:
        freq[t] = freq.get(t, 0) + 1
    vocab = [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:max_vocab]]
    stoi = {w: i for i, w in enumerate(vocab)}; V = len(vocab)
    ids = torch.tensor([stoi[t] for t in toks if t in stoi], dtype=torch.long)
    cnt = torch.zeros(V)
    for i in ids:
        cnt[i] += 1
    negp = (cnt ** 0.75); negp /= negp.sum()
    ein, eout = nn.Embedding(V, dim), nn.Embedding(V, dim)
    nn.init.uniform_(ein.weight, -0.5 / dim, 0.5 / dim); nn.init.zeros_(eout.weight)
    opt = torch.optim.Adam(list(ein.parameters()) + list(eout.parameters()), lr=2e-3)
    n = len(ids); g = torch.Generator().manual_seed(seed); BS = 1024
    for _ in range(steps):
        c0 = torch.randint(window, n - window, (BS,), generator=g)
        off = torch.randint(1, window + 1, (BS,), generator=g) * (torch.randint(0, 2, (BS,), generator=g) * 2 - 1)
        c, o = ids[c0], ids[c0 + off]
        negs = torch.multinomial(negp, BS * neg, replacement=True, generator=g).view(BS, neg)
        vc, vo, vn = ein(c), eout(o), eout(negs)
        loss = -(F.logsigmoid((vc * vo).sum(1)) + F.logsigmoid(-(vn * vc.unsqueeze(1)).sum(2)).sum(1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    E = ein.weight.detach()
    E = E / (E.norm(dim=1, keepdim=True) + 1e-9)
    if strip_top > 0:
        mu = E.mean(0); Ec = E - mu
        _, _, Vt = torch.linalg.svd(Ec, full_matrices=False)
        top = Vt[:strip_top]
        E = Ec - (Ec @ top.t()) @ top
        E = E / (E.norm(dim=1, keepdim=True) + 1e-9)
    vecfn = lambda w: E[stoi[w]] if w in stoi else None
    return (vecfn, stoi, E) if return_matrix else (vecfn, stoi)


class ConceptSpace:
    """Unified: addressed grounded graph + learned meaning. The closed loop lives in nav()/discover().
    Build from one corpus (__init__) or many domains (from_texts); save/load to skip rebuild (persistence).
    Edge evidence = the RICHEST passage (where the two concepts are CLOSEST = tightest relational quote)."""

    def __init__(self, corpus_text=None, top_n=45, embed_steps=4000, seed=0, log=print, drop_hubs=True):
        if corpus_text is None:
            return                                             # empty shell for load()
        text = clean_text(corpus_text)
        passages = split_passages(text)
        ents = [e.lower() for e in extract_entities(text, top_n=top_n)]
        self._build(passages, [None] * len(passages), ents, text, embed_steps, seed, drop_hubs, log)

    @classmethod
    def from_texts(cls, labeled_texts, top_n_each=30, embed_steps=6000, seed=0, log=print, drop_hubs=False):
        """Multi-domain: split EACH text separately (passages never span a book boundary), union the
        per-text entities, train ONE shared meaning-space over all. Cross-domain links come from meaning/
        shape (no shared passage across books) — honest by construction."""
        self = cls.__new__(cls)
        passages, pass_dom, parts, ents, seen = [], [], [], [], set()
        for label, raw in labeled_texts:
            t = clean_text(raw)
            ps = split_passages(t)
            passages += ps; pass_dom += [label] * len(ps); parts.append(t)
            for e in extract_entities(t, top_n=top_n_each):
                el = e.lower()
                if el not in seen:
                    seen.add(el); ents.append(el)
            log(f"[connect] {label}: {len(ps)} passages")
        self._build(passages, pass_dom, ents, "\n".join(parts), embed_steps, seed, drop_hubs, log)
        return self

    def _build(self, passages, pass_dom, ents, embed_text, embed_steps, seed, drop_hubs, log):
        self.passages = passages
        self.pass_dom = pass_dom
        self.plow = [" " + p.lower() + " " for p in passages]
        # addressed grounded graph: for each edge keep the RICHEST passage (concepts closest together).
        self.adj = {e: {} for e in ents}
        best_dist = {}
        for i, p in enumerate(passages):
            plo = p.lower()
            pos = {}
            for e in ents:
                if e in plo:                                   # cheap gate before regex
                    occ = [m.start() for m in re.finditer(r"\b" + re.escape(e) + r"\b", plo)]
                    if occ:
                        pos[e] = occ
            present = list(pos)
            for a in present:
                for b in present:
                    if a == b:
                        continue
                    d = min(abs(x - y) for x in pos[a] for y in pos[b])
                    if b not in self.adj[a] or d < best_dist.get((a, b), 1 << 30):
                        self.adj[a][b] = i; best_dist[(a, b)] = d
        n = len(ents)
        hubs = [e for e in ents if len(self.adj[e]) > 0.92 * (n - 1)] if drop_hubs else []
        ents = [e for e in ents if e not in hubs]
        self.adj = {e: {b: i for b, i in self.adj[e].items() if b not in hubs} for e in ents}
        log(f"[connect] {len(passages)} passages, {len(ents)} concepts (dropped hubs {hubs}); training meaning...")
        t0 = time.time()
        _, self.stoi, self.E = train_embedding(embed_text, steps=embed_steps, seed=seed, return_matrix=True)
        self.ents = [e for e in ents if self.vec(e) is not None]
        log(f"[connect] meaning learned ({time.time()-t0:.0f}s). Loop ready.")

    def vec(self, w):                                          # method (not a closure) so it survives save/load
        i = self.stoi.get(w)
        return self.E[i] if i is not None else None

    def save(self, path):
        import json
        p = Path(path); p.mkdir(parents=True, exist_ok=True)
        torch.save(self.E, p / "E.pt")
        (p / "space.json").write_text(json.dumps(dict(
            stoi=self.stoi, passages=self.passages, ents=self.ents, adj=self.adj,
            pass_dom=getattr(self, "pass_dom", None))))

    @classmethod
    def load(cls, path):
        import json
        p = Path(path)
        self = cls.__new__(cls)
        self.E = torch.load(p / "E.pt")
        d = json.loads((p / "space.json").read_text())
        self.stoi = d["stoi"]; self.passages = d["passages"]; self.ents = d["ents"]
        self.adj = {a: {b: int(i) for b, i in dd.items()} for a, dd in d["adj"].items()}
        self.pass_dom = d.get("pass_dom")
        self.plow = [" " + s.lower() + " " for s in self.passages]
        return self

    def relate(self, a, b):                                    # learned meaning-relatedness
        va, vb = self.vec(a), self.vec(b)
        return float(va @ vb) if va is not None and vb is not None else -1.0

    # ── blind BFS shortest grounded path (the unguided baseline) ──
    def bfs(self, start, goal):
        prev = {start: (None, None)}; q = deque([start])
        while q:
            c = q.popleft()
            if c == goal:
                break
            for nb, evi in self.adj[c].items():
                if nb not in prev:
                    prev[nb] = (c, evi); q.append(nb)
        if goal not in prev:
            return None, None
        path, edges, x = [goal], [], goal
        while prev[x][0] is not None:
            edges.append(prev[x][1]); x = prev[x][0]; path.append(x)
        return path[::-1], edges[::-1]

    # ── THE CLOSED LOOP: meaning-GUIDED best-first grounded navigation ──
    def nav(self, start, goal, max_expand=4000):
        """Best-first walk steered by LEARNED MEANING (heuristic = relatedness to goal), every hop GROUNDED
        (real co-mention, evidence kept). The meaning-judge decides where the navigator goes."""
        # priority = -relatedness(node, goal): expand most-meaningful-toward-goal first
        h0 = -self.relate(start, goal)
        pq = [(h0, 0, start, [start], [])]; best = {start: 0}; seen = 0
        while pq and seen < max_expand:
            _, g, node, path, edges = heapq.heappop(pq); seen += 1
            if node == goal:
                return path, edges
            for nb, evi in self.adj[node].items():
                ng = g + 1
                if nb not in best or ng < best[nb]:
                    best[nb] = ng
                    pri = ng - 2.0 * self.relate(nb, goal)     # cost-so-far minus meaning-pull to goal
                    heapq.heappush(pq, (pri, ng, nb, path + [nb], edges + [evi]))
        return None, None

    def coherence(self, path):
        """Mean learned relatedness of each INTERMEDIATE node to the goal — how 'on-theme' the path is."""
        if not path or len(path) < 3:
            return None
        goal = path[-1]
        mids = path[1:-1]
        return sum(self.relate(m, goal) for m in mids) / len(mids)

    # ── DISCOVER: meaningful-but-non-obvious connections, each grounded by a path ──
    def discover(self, a, k=6):
        """Concepts MEANINGFULLY related to `a` (learned) but with NO direct co-occurrence (non-obvious),
        that are nonetheless reachable by a GROUNDED path. The dots the text never put together."""
        direct = set(self.adj.get(a, {}))
        cands = []
        for c in self.ents:
            if c == a or c in direct:
                continue                                       # skip self + the obvious (already adjacent)
            r = self.relate(a, c)
            cands.append((r, c))
        cands.sort(reverse=True)
        out = []
        for r, c in cands:
            path, edges = self.nav(a, c)
            if path and len(path) >= 3:                        # grounded but indirect (>=2 hops)
                out.append((r, c, path, edges))
            if len(out) >= k:
                break
        return out


def quote(cs, i, w=130):
    s = cs.passages[i].strip().replace("\n", " ")
    return (s[:w] + "…") if len(s) > w else s


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="pride_prejudice.txt")
    ap.add_argument("--pairs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cs = ConceptSpace((HERE / args.corpus).read_text(errors="replace"), seed=args.seed)
    rng = random.Random(args.seed)

    # ── (1) meaning-guided vs blind BFS: path coherence on distance>=2 pairs ──
    def dist(a):
        dd = {a: 0}; q = deque([a])
        while q:
            c = q.popleft()
            for nb in cs.adj[c]:
                if nb not in dd:
                    dd[nb] = dd[c] + 1; q.append(nb)
        return dd
    far = []
    for a in cs.ents:
        dd = dist(a)
        for b, d in dd.items():
            if a < b and d >= 2:
                far.append((a, b))
    rng.shuffle(far); far = far[:args.pairs]
    coh_nav, coh_bfs, both = 0.0, 0.0, 0
    for a, b in far:
        pn, _ = cs.nav(a, b); pb, _ = cs.bfs(a, b)
        cn, cb = cs.coherence(pn), cs.coherence(pb)
        if cn is not None and cb is not None:
            coh_nav += cn; coh_bfs += cb; both += 1
    if both:
        print(f"\n[connect] === meaning-GUIDED nav vs blind BFS (path coherence, {both} pairs dist>=2) ===", flush=True)
        print(f"[connect] meaning-guided path coherence = {coh_nav/both:.3f}", flush=True)
        print(f"[connect] blind BFS-shortest coherence  = {coh_bfs/both:.3f}", flush=True)
        print(f"[connect] -> guided paths are {'MORE on-theme (loop adds value)' if coh_nav>coh_bfs+1e-3 else 'not more on-theme (honest)'}", flush=True)

    # ── (2) DISCOVER: the autonomous dot-connector ──
    print(f"\n[connect] === DISCOVER: meaningful-but-non-obvious connections (system's own judgment) ===", flush=True)
    seeds = [e for e in cs.ents if cs.adj[e]][:4]
    shown = 0
    for a in seeds:
        found = cs.discover(a, k=2)
        for r, c, path, edges in found:
            print(f"\n[connect] {a}  ⤳meaning {r:.2f}⤳  {c}   (never co-occur; grounded via {' → '.join(path)})", flush=True)
            for (x, y), ei in zip(zip(path, path[1:]), edges):
                print(f"[connect]   {x}~{y}: \"{quote(cs, ei)}\"", flush=True)
            shown += 1
            if shown >= 5:
                break
        if shown >= 5:
            break

    (HERE / "results_connect.json").write_text(json.dumps(dict(
        corpus=args.corpus, concepts=len(cs.ents), passages=len(cs.passages),
        coherence_guided=round(coh_nav / both, 3) if both else None,
        coherence_bfs=round(coh_bfs / both, 3) if both else None), indent=2))
    print("\n[connect] wrote results_connect.json", flush=True)


if __name__ == "__main__":
    main()

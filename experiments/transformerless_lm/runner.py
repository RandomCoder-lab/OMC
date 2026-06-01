#!/usr/bin/env python3
"""runner.py — THE RUNNER: an active re-grounding walk over the web, with two gears and a trail.

The web is the static layered map; this is the traveler. Each step it ASSESSES grounding (strength × number
of distinct frames — degree-and-frame, never a true/false stamp), picks a next focus via one of two gears,
RE-GROUNDS the link (navigates the deeper grounded context, quoted), and MOVES — recording every step to a
TRAIL it can trace back, resume, and branch from.

GEARS (the compass — a deliberate, inspectable, content-neutral EPISTEMIC value, not a fact bias):
  • coherence — stay in the origin's basin and go DEEPER. Scores by PMI-association with the origin (hard
    gate: off-theme refused so a big raw link can't drown the theme), specific link to focus, a new-frame
    bonus, minus a degree penalty (boilerplate/metadata hubs demoted). Avoids the boilerplate swamp.
  • explore — LEAP to a distant-but-GROUNDED concept in a NEW field, reachable from the focus by a real
    multi-hop grounded chain (deep_connect). The serendipity gear: surprising, but never ungrounded.
  • alternate — deepen, then leap, then deepen … the mind-like rhythm.

TRAIL: every step → a record (focus, gear, grounding, frontier, why, re-grounding chain), persisted to
.kwebcache/trails/. `--trace FILE` prints the walked map; `--resume FILE` continues without re-walking;
launch from any listed node to branch. Truth treated as degree-and-frame; nothing deleted.

NOTE: the compass needs its ground — if node_deg is missing (mid-rebuild), PMI degrades to raw co-occurrence
and it warns "RUNNING BLIND". Let node_expand rebuild the stats for real navigation.

  python runner.py gravity --steps 6 --mode coherence
  python runner.py gravity --steps 8 --mode alternate
  python runner.py --trace .kwebcache/trails/gravity_alternate.json
"""
import sys, math, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from factcheck import open_web, FactChecker

TRAILDIR = Path(__file__).parent / ".kwebcache" / "trails"
THEME_MIN = 1.0


class Runner:
    def __init__(self, k):
        self.k = k
        self.fc = FactChecker(k)
        self.visited = set()
        self.trail = []

    def coh(self, a, b):
        row = self.k.db.execute(
            "SELECT w FROM edges WHERE a=? AND b=? AND src NOT LIKE 'align-%' AND src NOT LIKE 'parallel-%'",
            (a, b)).fetchone()
        return self.k.assoc(a, b, row[0]) if row else max(0.0, self.k.relate(a, b))

    def grounding(self, x):
        fac = self.fc.factors(x, topk=15)
        if not fac:
            return 0.0, []
        strong = [f for f in fac if f[1] >= 4]
        return len(strong) + 2.0 * len({f[3] for f in strong}), fac

    def reground(self, a, b):
        chain = self.k.deep_connect(a, b)
        return chain.split("\n")[0] if chain else f"{a}–{b}: no deeper grounding found"

    # ── gear 1: coherence (deepen on-theme) ──
    def coherence_move(self, focus, origin, seen_frames):
        _, fac = self.grounding(focus)
        best, best_s, why = None, 0.0, None
        for wpmi, w, b, src, pid in fac[:25]:
            if b in self.visited or b in self.k.stop:
                continue
            theme = self.coh(b, origin)
            if theme < THEME_MIN:                       # HARD GATE: off-theme refused
                continue
            spec = self.k.assoc(focus, b, w)
            novelty = 0.6 if src not in seen_frames else 0.0
            deg_pen = 0.20 * math.log1p(self.k.deg.get(b, 1))
            s = 1.4 * theme + spec + novelty - deg_pen
            if s > best_s:
                best, best_s, why = b, s, f"coherence: theme·{theme:.2f}+spec·{spec:.2f}" + (" +frame" if novelty else "") + f" ⟨{src}⟩"
        return best, why

    # ── gear 2: explore (leap to distant-but-grounded, new field) ──
    def explore_move(self, focus, origin, seen_frames):
        _, fac = self.grounding(focus)
        # 2-hop candidates in NEW frames, reachable by a real grounded bridge
        best, best_s, why = None, -1e9, None
        for _, w1, mid, src1, _ in fac[:8]:
            if mid in self.k.stop:
                continue
            _, fac2 = self.grounding(mid)
            for _, w2, b, src2, _ in fac2[:8]:
                if b in self.visited or b in self.k.stop or b == focus or b == origin:
                    continue
                if src2 in seen_frames:                 # want a NEW field (the leap)
                    continue
                g, _f = self.grounding(b)
                if g < 6:                               # must land somewhere well-grounded
                    continue
                distance = -self.coh(b, origin)         # reward DISTANCE from origin (surprise)
                s = g + 1.5 * (src2 not in seen_frames) + 0.3 * distance
                if s > best_s:
                    best, best_s, why = b, s, f"explore→{src2}: via '{mid}' (grounding {g:.0f}, new frame)"
        return best, why

    def run(self, start, steps=6, mode="coherence"):
        origin = focus = start.lower()
        seen_frames = set()
        out = [f"=== RUNNER: '{origin}' — mode={mode} ==="]
        if not self.k.deg:
            out.append("⚠ RUNNING BLIND: node_deg missing — PMI degraded to raw co-occurrence, degree-penalty "
                       "off. Results unreliable until node_expand rebuilds the stats.")
        for i in range(steps):
            self.visited.add(focus)
            score, fac = self.grounding(focus)
            seen_frames |= {f[3] for f in fac[:6]}
            gear = ("explore" if mode == "explore" else
                    "coherence" if mode == "coherence" else
                    ("coherence" if i % 2 == 0 else "explore"))
            top = ", ".join(b for _, w, b, _, _ in fac[:6]) if fac else "(none)"
            out.append(f"\n[{i+1}] FOCUS '{focus}'  grounding={score:.0f}  ({len({f[3] for f in fac})} frames)  via {gear}")
            out.append(f"     factors: {top}")
            nxt, why = (self.coherence_move if gear == "coherence" else self.explore_move)(focus, origin, seen_frames)
            if not nxt:
                out.append(f"     {gear}: no qualifying next concept — switching/ending.")
                # try the other gear once before giving up
                other = self.explore_move if gear == "coherence" else self.coherence_move
                nxt, why = other(focus, origin, seen_frames)
                if not nxt:
                    out.append("     walk converged."); break
            rg = self.reground(focus, nxt)
            out.append(f"     re-ground: {rg}")
            out.append(f"     → '{nxt}'   [{why}]")
            self.trail.append({"step": i + 1, "from": focus, "to": nxt, "gear": gear, "grounding": score,
                               "why": why, "reground": rg})
            focus = nxt
        out.append(f"\n— walked {len(self.visited)} concepts ({mode}). trail saved; nothing deleted.")
        self.save(origin, mode)
        return "\n".join(out)

    def save(self, origin, mode):
        TRAILDIR.mkdir(parents=True, exist_ok=True)
        f = TRAILDIR / f"{origin}_{mode}.json"
        f.write_text(json.dumps({"origin": origin, "mode": mode,
                                 "visited": sorted(self.visited), "trail": self.trail}, indent=1))
        return f


def trace(path):
    d = json.loads(Path(path).read_text())
    out = [f"=== TRAIL MAP: origin '{d['origin']}' (mode {d['mode']}) — {len(d['trail'])} steps ==="]
    for r in d["trail"]:
        out.append(f"  [{r['step']}] {r['from']} →({r['gear']})→ {r['to']}")
        out.append(f"        why: {r['why']}")
        out.append(f"        reground: {r['reground']}")
    out.append(f"\n  places visited (relaunch from any to branch): {', '.join(d['visited'])}")
    return "\n".join(out)


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__); return
    if args[0] == "--trace":
        print(trace(args[1])); return
    steps, mode, resume = 6, "coherence", None
    for flag, attr in (("--steps", "steps"), ("--mode", "mode"), ("--resume", "resume")):
        if flag in args:
            i = args.index(flag); val = args[i + 1]; args = args[:i] + args[i + 2:]
            if attr == "steps":
                steps = int(val)
            elif attr == "mode":
                mode = val
            else:
                resume = val
    r = Runner(open_web())
    if resume:
        d = json.loads(Path(resume).read_text())
        r.visited = set(d["visited"]); r.trail = d["trail"]
        start = d["trail"][-1]["to"] if d["trail"] else d["origin"]
        print(f"(resuming from '{start}', {len(r.visited)} places already walked)")
    else:
        start = " ".join(args)
    print(r.run(start, steps, mode))


if __name__ == "__main__":
    main()

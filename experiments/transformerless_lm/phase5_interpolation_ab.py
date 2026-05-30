"""Phase 5.2 / capability C6 — PRE-REGISTERED A/B: approximate compute by interpolation.

QUESTION: can you skip computing f(x) by reusing the nearest *cached* (addressed) result, and
can a substrate signal tell you WHEN that's safe (the divergence gate)?

PRE-REGISTERED PREDICTIONS:
  P1. 1-NN (nearest-cached) approximation error is LOW for SMOOTH functions (near inputs -> near
      outputs) and HIGH for DISCRETE/chaotic functions (near inputs -> unrelated outputs).
  P2. A LOCAL-smoothness signal (mean |f(x+1)-f(x)|) separates the two -> a valid gate.
  P3. The @dualband snap-to-Fibonacci dissonance also separates them (the gate we shipped).
KILL CRITERION: if smooth approx_err is not clearly < discrete approx_err, interpolation is dead.
HONEST EXPECTATION: P1 holds (it's the falsified-substrate-as-compute wall, restated); P2 should
hold; P3 is the open question — snapping to the SPARSE Fibonacci lattice moves inputs FAR, so it may
measure lattice-coherence rather than local smoothness and predict POORLY. Report it either way.
"""
import math, random

FIBS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]
def snap(x):                      # nearest Fibonacci attractor (the @dualband substrate band)
    return min(FIBS, key=lambda f: abs(f - x))
def harmony(a, b):               # substrate agreement: 1 when equal, ->0 as |a-b| grows
    return 1.0 if a == b else 1.0 / (1.0 + math.log1p(abs(a - b)))

DOMAIN = list(range(1, 1000))
random.seed(0)
cache_xs = sorted(random.sample(DOMAIN, 100))     # 100 cached (addressed) points
test_xs  = random.sample(DOMAIN, 200)             # 200 unseen queries
probe    = random.sample(DOMAIN, 100)

def rng_of(f):
    vals = [f(x) for x in DOMAIN]
    return (max(vals) - min(vals)) or 1.0

def approx_err(f):               # 1-NN in input space, error as fraction of range
    R = rng_of(f); cache = {x: f(x) for x in cache_xs}; cxs = sorted(cache)
    return sum(abs(cache[min(cxs, key=lambda c: abs(c - x))] - f(x)) for x in test_xs) / len(test_xs) / R
def local_var(f):                # mean local step, normalized -> LOW = smooth (P2 gate)
    R = rng_of(f); return sum(abs(f(x + 1) - f(x)) for x in probe) / len(probe) / R
def snap_dissonance(f):          # the @dualband gate: f(x) vs f(snap(x)) (P3 gate)
    return sum(1.0 - harmony(f(x), f(snap(x))) for x in probe) / len(probe)

FUNCS = {
    "smooth  : 2x+3":          (lambda x: 2 * x + 3,          True),
    "smooth  : x*x//7":        (lambda x: x * x // 7,         True),
    "discrete: x % 7":         (lambda x: x % 7,              False),
    "chaotic : (x*7919)%97":   (lambda x: (x * 7919) % 97,    False),
    "discrete: gcd(x,84)":     (lambda x: math.gcd(x, 84),    False),
}

print("[5.2] interpolation A/B  (cache=100, test=200 unseen, domain 1..999)")
print(f"  {'function':24s} {'approx_err':>10s} {'local_var':>10s} {'snap_diss':>10s}")
rows = []
for name, (f, smooth) in FUNCS.items():
    e, lv, sd = approx_err(f), local_var(f), snap_dissonance(f)
    rows.append((smooth, e, lv, sd))
    print(f"  {name:24s} {e:10.3f} {lv:10.3f} {sd:10.3f}")

def avg(sel, i):
    xs = [r[i] for r in rows if r[0] == sel];  return sum(xs) / len(xs)
se, de = avg(True, 1), avg(False, 1)
sl, dl = avg(True, 2), avg(False, 2)
ss, dssn = avg(True, 3), avg(False, 3)
print(f"\n  smooth   : approx_err={se:.3f}  local_var={sl:.3f}  snap_diss={ss:.3f}")
print(f"  discrete : approx_err={de:.3f}  local_var={dl:.3f}  snap_diss={dssn:.3f}")
print(f"\n  P1 interpolation: smooth {se:.3f} {'<' if se < de else '>='} discrete {de:.3f}  -> "
      f"{'VIABLE only for smooth (C6 wall confirmed)' if se < de * 0.5 else 'NO clean separation'}")
print(f"  P2 local_var gate : {'SEPARATES' if dl > sl * 2 else 'does NOT separate'} "
      f"(discrete {dl:.3f} vs smooth {sl:.3f}) -> {'valid router' if dl > sl * 2 else 'weak'}")
print(f"  P3 snap-diss gate : {'SEPARATES' if dssn > ss * 1.5 else 'does NOT separate'} "
      f"(discrete {dssn:.3f} vs smooth {ss:.3f}) -> "
      f"{'the shipped @dualband gate predicts it' if dssn > ss * 1.5 else 'shipped gate is the WRONG signal for interpolation; needs local_var'}")

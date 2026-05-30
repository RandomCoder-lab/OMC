# OMC Substrate Primitives (v1.8.x)

This document describes the substrate capabilities promoted into the OMC core language in the
v1.8.x series — content-addressing, an addressable heap, content-similarity, verify-gated
self-modification, correct-by-construction synthesis, and HBit dual-band computation. Every claim
here is backed by a test or a runnable demo; honest limits are stated alongside.

The one idea underneath all of it: **in OMC a value's identity can be its content, not its
location.** `address = f(content)`, in an address space that is *uniform* (equal-area, χ²≈9 on
uniform points) and has *locality* (similar content lands near). That inversion is what makes
memoization a property of identity, equality O(1), programs navigable, and self-modification safe.

---

## 1. Content-addressing — `haddr`

Uniform dodecahedral address of any string/value. The 12 face normals are the icosahedral
vertices `(0, ±1, ±φ)` and cyclic permutations — equal solid angles, so a decorrelated hash lands
on each face with equal probability.

```
haddr(text)            -> {face: 0..11, sub_face: 0..2, zeck: [Fibonacci values]}
haddr_face(text)       -> int 0..11
haddr_distance(a, b)   -> float   (a, b are address dicts or strings)
```

**Verified:** face χ² = 9.16 on uniform sphere points, 4.90 on 20k hashed strings (uniform
expectation ≈ 11; the old sin/cos fingerprint scored ≈216). `haddr` is for **exact keys / uniform
buckets** — *not* similarity (see §3).

---

## 2. The addressable heap + `@memo`

A value's content hash is its address. Compute something once; find it by content forever.

```
value_addr(v)   -> dodecahedral address of any value (structural)
value_hash(v)   -> content key (string)
same_value(a,b) -> bool    O(1) semantic equality (structural, provenance-independent)
cas_put(v)      -> key     store in the content-addressed heap (persists to ~/.omc/cas)
cas_get(key)    -> value   retrieve (memory, then disk)
cas_has(key)    -> bool
```

`@memo` is transparent, content-addressed, **persistent across runs** memoization. The cache key
includes a hash of the function body, so editing the function invalidates stale results.

```omc
@memo
fn fib(n) { if n < 2 { return n; } return fib(n-1) + fib(n-2); }
print(fib(90));   // 2880067194370816120 — instant; naive recursion is ~2.9e18 calls
```

**Honest limits:** `@memo`/`@dualband` require purity (the body must not call I/O, random, time,
etc.) — impure functions are refused at definition. Persistence is a plain on-disk pool under
`~/.omc/cas`; share it via your own sync if you want cross-machine.

---

## 3. Content-similarity — `locality_fp`

`haddr` is uniform but has **no** content locality (a one-character change scrambles the address).
For "find the most similar X", use the locality fingerprint: a normalized byte histogram, so
similar content → similar vector.

```
locality_fp(text, [bigram])           -> float[]   (unigram 256-dim; bigram=1 → 4096-dim)
locality_sim(a, b, [bigram])          -> float in [0,1]
locality_nearest(query, candidates)   -> index of the most similar candidate
nearest_fn(need)                      -> name of the closest-by-locality defined function
call_nearest(need, args)              -> dispatch to that function and call it
```

**Verified:** on a corrupted-retrieval task, recall@1 = 0.99 (locality) vs 0.02 (φ/haddr).
**Two fingerprints, two jobs:** `haddr` for keys, `locality_fp` for similarity.

**Honest limit:** locality matches *character distribution*, so it is typo/variant-tolerant
(`"quicksrt"` → `quicksort`) but is **not** semantic NL→code (`"greatest common divisor"` will not
find `gcd`) — that needs a learned encoder.

---

## 4. Verify-gated self-modification

The interpreter is the gate. A candidate is installed and tested in a sandbox; it is kept only if
it passes, otherwise rolled back. Nothing that fails its spec is ever accepted.

```
fn_swap_verified(name, new_source, test_source) -> {accepted, error, result}
fns_on_face(face)                               -> functions bucketed by name-address
```

```omc
fn slow(n) { return 0-1; }              // a stub to improve
h good = "fn slow(n) { return n*n; }";
h test = "slow(5) == 25";
print(fn_swap_verified("slow", good, test));  // {accepted: true, ...}
```

---

## 5. Correct-by-construction synthesis

A generator that emits only grammar-legal structure, so **every program parses**, and tracks
declared variables + guards division + bounds loops, so (almost) every program runs.

```
gen_omc([seed])           -> a valid-by-construction OMC program string
gen_at(address_or_text)   -> same address/need → the same valid program (deterministic)
```

**Verified:** parse-rate 1.000, run-rate 1.000 over 300 seeds, checked by the real
parser+interpreter. Pair with `code_parse_check` / `eval_omc` / `fn_swap_verified` for a
generate → verify → accept loop. Standard LMs cannot guarantee parse-rate 1.0.

**Honest limit:** the generator covers the executable core (functions, declarations, assignment,
if/else, while, for, return, print, arithmetic, calls). It does not emit every construct (e.g.
try/match/class) yet.

---

## 6. HBit dual-band computation

Two bands run together: **α** (the exact value) and **β** (its harmonic *shadow* — the
"what if we'd stayed on the Fibonacci attractor lattice" companion). α is always the exact answer;
β only records how far a computation has drifted from the lattice. The drift is the *gate*: trust
a fast/addressed path while in tune, fall back to exact when dissonant.

### Per-value (pervasive)
Ordinary values are single-band and behave exactly as before. `phi_shadow` attaches β; it then
rides through arithmetic.

```
phi_shadow(v)        -> v with β = nearest Fibonacci attractor of α
bands(v)             -> [α, β]
harmony(v)           -> 0..1000 (1000 = in tune; reads the carried bands)
value_divergence(v)  -> 0..1000 (0 = on the lattice, high = dissonant)
hbit_harmony(a, b)   -> 0..1000   two-band resonance of explicit a, b
hbit_divergence(a,b) -> 0..1000   the gate value (0 = in tune)
```

```omc
h s = phi_shadow(10);        // bands [10, 8]
h t = (s + 1) * 3;           // α: 33 ; β: (8+1)*3 = 27
print(bands(t));             // [33, 27]
print(value_divergence(t));  // drift of 33 vs 27
```

### Per-function (opt-in)
```omc
@dualband
fn sq(n) { return n*n; }
print(sq(8));                    // 64 (exact α — always)
print(band_divergence("sq"));    // 0 on-lattice, high when dissonant
print(band_route("sq"));         // fast-substrate / cached-exact / linear
```

`@dualband` also takes the exact-memo fast path (the A→Z skip) when an exact result is cached.

**Honest limits:** today the dual band is a *coherence monitor and exact-skip router* — α is always
computed as ground truth, so a strict speedup comes from the exact-memo hit, not yet from skipping
α on the strength of the gate. The snap-to-Fibonacci gate measures **lattice-coherence**, which is
the right signal for "is this on the harmonic lattice" but (measured) **not** a predictor of
interpolation safety on arbitrary functions. Approximate skipping is viable only on *smooth*
domains (near inputs → near outputs); on discrete/chaotic functions it is not (this is a measured
result, not a hope).

---

## How it scales (and why it's CPU, not GPU)

A dense transformer gains capability by adding parameters → more FLOPs/query → GPU. The substrate
gains capability by adding **addressed content** + composition + a **verify** step — all CPU.

Measured (1896-function corpus, real interpreter as oracle):
- correctness rises with coverage `0.04 → 1.00`,
- per-query exact-key retrieval is flat `0.059µs → 0.060µs` across 100× more content,
- the verify step is constant (one interpreter run, independent of store size),
- the O(N) similarity scan is what addressing (O(1)) removes.

So capability scales at flat per-query CPU cost. The "ceiling" is coverage + composition, both
CPU-scalable. (Scope: verified code synthesis over a corpus; generalizing beyond stored content is
bounded by generator quality — but that too is CPU.)

---

## Reproduce

```
cargo test -p omnimcode-core --lib            # 172 tests incl. address/cas/locality/synth
cargo build -p omnimcode-cli --release
./target/release/omnimcode-standalone experiments/transformerless_lm/valueband_demo.omc
```

Full evidence ledger: `experiments/transformerless_lm/AUTONOMOUS_LOG.md` and
`experiments/transformerless_lm/SUBSTRATE_INTEGRATION_ROADMAP.md`.

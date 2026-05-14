# OMNIcode

**A self-hosting harmonic computing language with a self-healing compiler.**

OMNIcode (OMC) is an experimental programming language built around a single architectural premise: **φ-math (Fibonacci resonance, value-danger, harmonic alignment) is not decoration — it is a decidable, cheap-to-compute substrate the compiler can reason against.** That substrate makes possible two things conventional languages structurally cannot do without external tooling:

1. **Self-hosting at the back-end level.** The OMC compiler is written in OMC, and the bytecode the compiler produces for its own source is byte-identical to the bytecode the host-language tree-walker would produce. See `examples/self_hosting_v9b.omc` — `gen2 == gen3` of a real compiler-as-function.
2. **Self-healing at compile time AND runtime.** The compiler can detect a working class of bugs (numeric off-by-one against Fibonacci attractors, identifier typos, dynamic divide-by-singularity, array-index out-of-bounds, missing braces / parens / semicolons) and rewrite the program — using the language's own φ-math primitives, not a hand-written rule table. See `examples/self_healing_h5.omc`.

This is a research artifact. It is not a production runtime. But the architectural claims above are **demonstrable, reproducible, and run on the binary in this repository.**

---

## Two ways to read this project

### For language designers and researchers

OMC sits in a small category that includes Lisp, Smalltalk, and Forth: languages where the entire toolchain (read, parse, emit, execute, analyze, repair) lives inside the language itself, on a single coherent semantic substrate. What's distinctive about OMC is the **substrate**: not S-expressions, not a stack-VM convention, but **φ-math** — a value-semantic lattice with the property that "is this number on the Fibonacci geodesic?" is decidable in O(log N) and correlates with what the project calls *harmonic alignment*.

The interesting consequence is that **the compiler's static analysis pass is one-line-per-check**. `is_fibonacci(n) == 0 && |Δ| ≤ 3 → flag` catches off-by-one harmonic violations. `value_danger(b) > 0.5 → rewrite a/b as safe_divide(a,b)` catches dynamic divide-by-zero. The heavy lifting lives in the math — `is_fibonacci`, `value_danger`, `harmony_value`, `fold_escape` — not in the rules. Adding a new diagnostic class is composing existing primitives, not authoring a parser pass.

The math is documented in `PHI_PI_FIB_ALGORITHM.md` and the type system in `ARCHITECTURE.md`. The complete self-hosting proof is `examples/self_hosting_v9b.omc`.

### For developers and engineers

What you can do with OMC right now, with the binary in this repo:

- **Run a Turing-complete language.** Recursion, functions, strings, arrays, mutating builtins, while loops, if/else. ~100 host primitives across strings, arrays, file I/O, type introspection, math, harmonic-math, and self-healing — full reference in `STDLIB.md`. See `examples/fibonacci.omc`, `array_ops.omc`, `strings.omc`, `stdlib_expansion.omc`.
- **Compile OMC to bytecode AND execute that bytecode** — both stages of which can be written in OMC. The bytecode VM is faithful to the tree-walker: byte-identical output across both paths for any program in the supported feature surface.
- **Feed broken code into a self-healing compiler.** A program with a missing semicolon, a missing `}`, a typo'd function name, an off-by-one Fibonacci constant, and a `/0` runtime crash will be **rewritten and executed to a finite answer** — no try/catch, no defensive guards. The math is the error handling.

What this is **not**: a fast runtime, a production toolchain, a stable API, a deployment target. The tree-walker is in Rust (fast); the bytecode VM is in OMC running on the tree-walker (slow, for science). Single-developer experimental codebase. The point is the architectural pattern, not the throughput numbers.

---

## What's proven right now

| Claim | File to run | What you'll see |
|---|---|---|
| Bytecode VM = tree-walker (semantic fixpoint) | `examples/self_hosting_v8.omc` | `✓ FIXPOINT REACHED` on two demos |
| The compiler is a fixed point under self-application | `examples/self_hosting_v9b.omc` | `✓✓✓ ALL THREE FIXPOINTS REACHED` |
| Self-healing across two stages (token + AST), 5 bugs healed in one source | `examples/self_healing_h3.omc` | All four demos converge; `safe(8) → 8` on the integrated case |
| User-declared runtime self-healing via `safe` keyword | `examples/self_healing_h4.omc` | `compute(144, 0) → 144` — runtime crash converted to finite answer on attractor |
| Array-bounds healing — out-of-bounds reads become attractor-landing | `examples/self_healing_h5.omc` | Loop walking 8 indices off a 5-element array; every output has `φ=1.000` |
| Host-level `safe` keyword — works in any OMC program, not just the self-healing demos | `examples/safe_keyword_host.omc` | `safe 89/0 → 89`, `safe arr_get(xs, 999) → 20`, `safe arr_set(xs, 999, 99)` mutates xs[1] |
| Python-tier standard library: 16 new built-ins added 2026-05-14 | `examples/stdlib_expansion.omc` | `str_split`, `arr_sort`, `read_file`/`write_file`, `type_of`, `gcd`, `now_ms` and more — see `STDLIB.md` for the full reference |

Run any of these with the binary built from this repo:

```bash
./target/release/omnimcode-standalone examples/self_hosting_v9b.omc
./target/release/omnimcode-standalone examples/self_healing_h4.omc
```

---

## The arc

How the project reached the claims above. Each phase has a working demo file you can run today — every entry in this section is a real artifact in this repo, not a roadmap item.

### Phase V — Self-hosting (front-end through back-end)

The arc that closed the bootstrap loop.

- **V.1 / V.2 — Lexer in OMC** (`examples/self_hosting_lexer.omc`, `self_hosting_lexer_v2.omc`)
  An OMC program that tokenizes OMC source. Multi-char operators, float literals, escape decoding.
- **V.3 — Parser in OMC** (`examples/self_hosting_parser.omc`)
  Recursive-descent parser; produces nested-tagged-array AST. Surfaced the type-aware-equality bug; fixed at host level.
- **V.4 — Codegen in OMC** (`examples/self_hosting_codegen.omc`)
  AST → OMC source pretty-printer. Closes the parse↔emit half of the loop.
- **V.5 — Lex/parse/print fixpoint** (`examples/self_hosting_fixpoint.omc`)
  `source → tokens → AST → source' → tokens' → AST'` with `AST == AST'`. 6/6 tests pass.
- **V.6 — Bytecode encoder + stack-VM executor in OMC** (`examples/self_hosting_bytecode.omc`)
  Integer arithmetic, while, if/else. Discovered OMC's pass-by-value array semantics; encoder uses return-and-rebind with relative jumps.
- **V.7 — Functions, recursion, call frames** (`examples/self_hosting_v7.omc`)
  `fib(10) = 55` via 177 recursive CALL/RETURN cycles on the OMC-written executor.
- **V.7b — Strings, arrays, builtin dispatch** (`examples/self_hosting_v7b.omc`)
  `LOAD_STR`, `MAKE_ARR`, `CALL_BUILTIN` open the bytecode VM to non-numeric data.
- **V.7c — Mutating builtins via named-store opcodes** (`examples/self_hosting_v7c.omc`)
  `ARR_PUSH_NAMED`, `ARR_SET_NAMED` close the pass-by-value trap.
- **V.8 — Round-trip fixpoint** (`examples/self_hosting_v8.omc`)
  Same OMC source, two paths (tree-walk vs OMC-bytecode-VM), byte-identical output.
- **V.8b — Full compiler subset round-trips** (`examples/self_hosting_v8b.omc`)
  `#` comments, `-> type` annotations, `break` — all the syntax the compiler source itself uses.
- **V.9 — UTF-8 safety via `str_chars`** (`examples/self_hosting_v9.omc`)
  Host-side fix to the byte-vs-char-index mismatch that broke V.8b's lexer on non-ASCII source.
- **V.9b — Gen2 == Gen3 of a compiler** (`examples/self_hosting_v9b.omc`)
  A real mini-compiler-in-OMC (`mini_enc`: NUM/VAR/BIN expression encoder), run two ways on the same hardcoded input AST. Both produce identical bytecode arrays. **The textbook self-hosting closure property at compiler-bootstrap level.**

### Phase H — Self-Healing Compiler

Built on top of the Phase V self-hosting stack. The compiler now uses φ-math to detect AND repair a working class of bugs.

- **H.1 — Harmonic + typo diagnostics** (`examples/self_healing_compiler.omc`)
  AST walker. `is_fibonacci(n) == 0 && |Δ| ≤ 3 → flag and rewrite to nearest Fibonacci`. Levenshtein-distance typo correction against the defined-name table. Demo input has 2 bugs, both fixed; output lands on Fibonacci attractor.
- **H.2 — Iterative loop + divide-by-singularity** (`examples/self_healing_h2.omc`)
  `heal_until_fixpoint`. `value_danger(b) > 0.5 → rewrite a/b as safe_divide(a, b)`. `numerator / 0` becomes a working program returning 8 (attractor).
- **H.3 — Parse-level recovery** (`examples/self_healing_h3.omc`)
  Token-level repair: missing braces, parens, semicolons. The integrated demo handles **five bugs across two stages** (token + AST) in one source. Output: `safe(8) → 8` on attractor.
- **H.4 — `safe` keyword** (`examples/self_healing_h4.omc`)
  User-declared runtime self-healing. `safe count / mod` unconditionally rewrites to `safe_divide(count, mod)` even when `mod` is a variable the static healer can't reach. `compute(144, 0)` returns 144 instead of crashing.
- **H.5 — Array-bounds healing** (`examples/self_healing_h5.omc`)
  Extends `safe` to array accesses. `safe arr_get(xs, idx)` rewrites to `safe_arr_get`, which folds the index onto the nearest Fibonacci attractor and modulos by `arr_len(xs)`. Out-of-bounds reads become total, deterministic, attractor-landing finite values. Demo: a loop reading 8 indices off a 5-element array — every value has `φ=1.000`.

The full design rationale, milestone-by-milestone, is in `CHANGELOG.md`.

---

## Quick start

### Build

```bash
git clone https://github.com/RandomCoder-lab/OMC.git
cd OMC
cargo build --release
```

### Run a program

```bash
./target/release/omnimcode-standalone examples/fibonacci.omc
```

### Interactive REPL

```bash
./target/release/omnimcode-standalone
```

### See the headline demos

```bash
# The compiler is a fixed point under self-application:
./target/release/omnimcode-standalone examples/self_hosting_v9b.omc

# Five bugs (token-level + AST-level) healed in one source, runs to completion:
./target/release/omnimcode-standalone examples/self_healing_h3.omc

# `safe` keyword: runtime guard against dynamic singularities:
./target/release/omnimcode-standalone examples/self_healing_h4.omc
```

---

## Try the language

A taste of OMC syntax. The grammar is defined in `omnimcode-core/src/parser.rs`. The complete standard library — ~100 host primitives organized by category — is in `STDLIB.md`.

### Hello world

```omnicode
print("Hello, harmonic world!");
```

### Recursive Fibonacci

```omnicode
fn fib(n) {
    if n < 2 { return n; }
    return fib(n - 1) + fib(n - 2);
}
print(fib(13));   # → 233 (itself on the Fibonacci attractor)
```

### Self-healing semantics

```omnicode
fn compute(numerator, divisor) {
    return safe numerator / divisor;   # `safe` opts into runtime guards
}
print(compute(144, 0));   # → 144, not a crash
```

### Harmonic primitives

```omnicode
print(is_fibonacci(144));    # 1 — on the φ-geodesic
print(is_fibonacci(145));    # 0 — close miss, off by 1
print(harmony_value(89));    # 1.0 — Fibonacci attractor
print(value_danger(0));      # 1.0 — singularity (exp(-|0|) = 1)
print(fold_escape(0));       # 1 — escape from the singularity
```

---

## What this doesn't do yet

This is a research codebase. Honest list of things that are NOT done:

- **No fast bytecode runtime.** The bytecode VM is written in OMC and executes on the tree-walker. It's correct (byte-identical to tree-walk) but slow. A native bytecode VM in Rust is future work.
- **Phase H limits:**
  - Naive brace placement in token-level repair appends missing `}` at EOF — fine for end-of-file mistakes, will fold mid-source statements into function bodies if the missing brace is conceptually mid-source. Indentation-aware repair (H.3.1) is logged.
  - The healer's identifier-correction has no semantic check beyond edit-distance. A typo that resolves to ANOTHER typo would stabilize but not be correct.
  - The `stuck` and `exhausted` outcomes of `heal_until_fixpoint` are designed but unexercised — no current demo triggers them.
- **No production deployment target.** No package manager. No formatter. No LSP. No debugger. The standard library is real (~100 host primitives covering strings, arrays, file I/O, type introspection, math, φ-math, and self-healing — see `STDLIB.md`), but it's not Python-tier — no first-class functions, no formatters, no module ecosystem.
- **Adversarial cases untested.** The healer's correctness has been demonstrated on the demo inputs in `examples/self_healing_*.omc`. Fuzz testing, malicious inputs, and pathological edge cases have not been done.
- **Single-developer experiment.** The codebase has not had external review. There are likely bugs we don't know about.

The architectural claims at the top of this README are demonstrable today. The above limits are real and undersold nowhere. If you're considering OMC for production, the answer is "not yet."

---

## Architecture

The full breakdown is in `ARCHITECTURE.md`. The short version:

```
OMC source
    ↓
omnimcode-core/src/parser.rs       Lexer + recursive-descent parser
    ↓
omnimcode-core/src/ast.rs          AST node definitions
    ↓
omnimcode-core/src/interpreter.rs  Tree-walk interpreter + host primitives
    ↓
omnimcode-core/src/vm.rs           Stack VM (used by bytecode path)
    ↓
omnimcode-core/src/value.rs        HInt with φ-resonance, HArray, HFloat, HSingularity
```

OMC files implementing the language itself, in the language itself, live under `examples/self_hosting_*.omc` and `examples/self_healing_*.omc`. They run on the Rust interpreter above; this is what makes the gen2==gen3 claim verifiable.

### Key supporting documents

- `CHANGELOG.md` — milestone-by-milestone account of the V and H phases (this is where the design history lives in real detail)
- `ARCHITECTURE.md` — type system, interpreter, VM internals
- `BUILD.md` — build instructions including cross-compilation and optimization flags
- `BENCHMARKS.md` — criterion benchmarks comparing tree-walk vs VM vs VM+optimizer
- `PHI_PI_FIB_ALGORITHM.md` — mathematical foundation
- `OMC_STRATEGIC_PLAN.md` — where the project is headed
- `00-START-HERE.md` and `READING_ORDER.md` — recommended traversal for new readers

---

## Implications

For language design, OMC demonstrates that a coherent value-semantic substrate (φ-math, in our case) enables a class of compile-time analyses that are **cheap, decidable, and one-line-per-check**. Most modern languages would benefit from having ANY such substrate they could check against, instead of the patchwork of lint rules they currently maintain. OMC is one existence proof.

For LLM-generated code specifically: the failure modes of language-model-generated programs cluster around three classes — typos and naming drift, off-by-one numeric constants, and unguarded edge cases. Phase H handles all three. A self-healing target language reduces the defensive-coding burden on the generator: it doesn't need to write defensively because the compiler does the defense automatically. Whether this generalizes beyond OMC's specific aesthetic is open.

For the broader question of whether OMC will grow beyond a research artifact: that depends on whether the φ-math substrate resonates outside the people who already think this way. The science is real. The tooling is small. The substrate is reusable. What it becomes is downstream of what gets built on top.

---

## Contributing

This is a single-developer research codebase. Issues, observations, and PRs are welcome but please read `00-START-HERE.md` first to calibrate.

If you want to write OMC code: look at any `examples/self_healing_*.omc` — they exercise nearly every language feature.

If you want to extend the host: `omnimcode-core/src/interpreter.rs` is the place to add new primitives. New host builtins automatically become available to the bytecode VM through `vm_call_builtin → call_function`.

If you want to extend the self-healer: `examples/self_healing_h4.omc` Part V is the template. Add a diagnostic class by composing the existing φ-math primitives (`is_fibonacci`, `value_danger`, `harmony_value`, `fold_escape`).

---

## License

MIT. See `LICENSE` if present, or `Cargo.toml` workspace metadata.

---

## Credits

OMNIcode is a research project by The Architect, with substantial co-development assistance from Claude (Anthropic). The φ-math foundation, harmonic-integer type system, and overall language design are The Architect's. The Phase V (self-hosting) and Phase H (self-healing compiler) implementations were co-authored across an extended series of sessions.

The work builds on a long lineage of self-hosting language research — Lisp, Smalltalk, Forth, and the broader bootstrapping-compiler tradition. OMC's contribution to that tradition is a specific kind of architectural completeness: the toolchain (lex, parse, emit, execute, analyze, repair) lives inside the language on a single mathematical substrate.

---

**Built around φ (1.618…). The substrate is the architecture.**

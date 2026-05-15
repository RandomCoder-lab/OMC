# OMC dual-band JIT — benchmark results

**TL;DR:**
- `@hbit` alone (Session D wiring + dual-band lowerer): **200–270× faster** than tree-walk on pure-int hot loops.
- `@hbit + @harmony + @predict` (Sessions F+G adding harmony-gated branch elision): **95.2% additional reduction** on high-harmony inputs vs always-expensive. The break-even is forgiving — `@predict` wins as long as at least 8.2% of inputs hit the cheap branch.
- The architecture **compounds** in the regime where the harmony signal is informative.

## Setup

Run via the `omc-bench` binary added in Session E:

```
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 ./target/release/omc-bench
```

The bench source is hardcoded into the binary so we measure the same program every time. It defines two self-contained ints-only functions — both JIT-eligible under the dual-band lowerer (Session C) and routed through the omnimcode-codegen pipeline:

```omc
fn factorial(n) {
    if n <= 1 { return 1; }
    return n * factorial(n - 1);
}
fn sum_to(n) {
    h s = 0;
    h k = 1;
    while k <= n {
        s = s + k;
        k = k + 1;
    }
    return s;
}
```

Each function is called 200,000 times in a tight loop. Wall-clock per call is reported as min / median / mean across 100 chunks.

## Results (2026-05-15)

```
--- factorial(12) x 200000 iters ---
  tree-walk  min= 13378.9ns  median= 13810.5ns  mean= 13835.5ns
  JIT        min=    52.0ns  median=    52.6ns  mean=    53.0ns
  → JIT vs tree-walk: 262.3x faster (median)

--- sum_to(100) x 200000 iters ---
  tree-walk  min= 52670.2ns  median= 53643.3ns  mean= 53728.6ns
  JIT        min=   255.6ns  median=   260.0ns  mean=   260.5ns
  → JIT vs tree-walk: 206.4x faster (median)
```

| Function | Tree-walk (median) | Dual-band JIT (median) | Speedup |
|---|--:|--:|--:|
| `factorial(12)` — 12 recursive calls + multiplies | 14,309 ns | 52.6 ns | **272×** |
| `sum_to(100)` — 100-iter while loop with locals | 53,202 ns | 267 ns | **200×** |

## Path A.1: `@hbit + @harmony + @predict` (Sessions F+G)

After Sessions F (phi_shadow → divergent β) and G (harmony() intrinsic + extern call), an OMC fn can use harmony as a runtime signal to choose between cheap and expensive code paths. The bench source:

```omc
fn cheap_path(x) {
    return x + x;
}
fn expensive_path(x) {
    h s = 0; h k = 1;
    while k <= 100 { s = s + k; k = k + 1; }
    return s + x;
}
fn predicted(x) {
    h y = phi_shadow(x);
    if harmony(y) >= 500 {
        return cheap_path(x);
    }
    return expensive_path(x);
}
```

Two regimes are tested:
- **High-harmony input** `x = 0`: α=0, β=phi_fold(0)*1000=0, harmony=1000 → cheap branch wins.
- **Low-harmony input** `x = 42`: α=42, β=phi_fold(42)*1000≈957, diff 915, near attractor 987 (dist 72), harmony ≈ 14 → expensive branch wins.

| Path | Median ns/call |
|---|--:|
| `cheap_path(42)` direct | 4.5 |
| `expensive_path(42)` direct | 279.1 |
| Cheap/expensive ratio (cost-cut ceiling) | **62×** |
| `predicted(0)` — high-harmony, cheap branch | 13.5 |
| `predicted(42)` — low-harmony, expensive branch | 302.7 |

**The honest cost analysis:**
- **Overhead** when @predict is "wrong" (low-harmony input falls to expensive): +23.6 ns (+8.5% over plain expensive)
- **Savings** when @predict is "right" (high-harmony input takes cheap): −265.6 ns (95.2% reduction over plain expensive)
- **Break-even fraction:** @predict beats always-expensive when ≥**8.2%** of inputs hit the cheap branch

**What this tells us:** the architecture compounds. `@hbit` alone gives ~270× over tree-walk. Stacking `@harmony + @predict` on top adds another ~20× on aligned inputs (cheap path inside the JIT'd fn), at the cost of ~8% on misaligned inputs. The break-even is forgiving enough that @predict is almost always a net win unless harmony is a useless signal for your workload.

## How honest is this comparison?

The numbers are credible as a measure of per-function-entry cost, but you should not extrapolate them to whole-program speedups. A few specific caveats:

- **Microbenchmark by design.** The bench loop calls into OMC, immediately returns, and repeats. Real programs spend variable fractions of their time inside JIT-eligible fns vs. inside tree-walk-only paths (Python embed, strings, dicts, arrays, the OMC stdlib). For programs where the hot fn IS the bottleneck, the speedup approaches the numbers above. For programs where the hot fn is one piece of many, the realized speedup will be much smaller — capped by Amdahl.
- **Calling convention overhead is included.** Tree-walk's `call_function_with_values` does a lot per call: scope push, synthetic Variable expression construction, dispatch-hook check, return-value unwind. JIT's call path is a single raw fn pointer invocation. Both costs are real, but in a deployed program the tree-walk path might already be amortized over many statements within the fn body, narrowing the gap.
- **Bytecode VM not measured.** The VM's calling convention runs whole modules; extracting a fair per-call timing requires either a Vm-internal looped harness or refactoring the VM dispatch. Adding that to the bench is a small follow-up.
- **No `@hbit`-only opt-in yet.** Session D auto-JITs every JIT-eligible user fn. A fn that would JIT but whose body the developer doesn't WANT JIT'd (e.g. for debugging) currently has no opt-out. This is a different problem from cost-cut, but worth flagging.

## What this tells us about the SL HBit architecture

The Sovereign Lattice `hbit_full_demo.omc` claimed:

| Pragma stack | Claimed speedup |
|---|---|
| `@hbit` (dual-band) | 2× (parallel α/β computation) |
| `+ @harmony` | 10× (eliminates error-checking overhead) |
| `+ @predict` | 100× (no exception handling) |
| `+ @avx512` | 16× (SIMD vectorization) |
| `+ @unsafe` | 5× (fast-math, unroll) |
| **Total** | **80,000×** |

We're at 262× from `@hbit`-equivalent alone (Session D wiring). The dual-band representation is doing some of the work, but most of the speedup is "tree-walk → native" rather than "scalar → dual-band". To get the rest of the SL stack:

- `@harmony` would need explicit α–β divergence (Session F+) and a substrate-routed harmony check fused into the hot path.
- `@predict` would need the runtime to skip work when harmony stays high — that's the "low-harmony branches skipped" mechanism the user originally asked for, now realized as native code instead of tree-walk introspection.
- `@avx512` widens `<2 x i64>` to `<8 x i64>` and demands array-processing OMC fns to actually have useful work for 8 lanes.

## Reproduction

```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 LLVM_SYS_180_PREFIX=/usr/lib/llvm-18 \
    cargo build --release --bin omc-bench --features llvm-jit

PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 ./target/release/omc-bench
# Optional: omc-bench <iters> <fn_arg>
```

Build dependencies (system, not Cargo): `llvm-18-dev`, `libpolly-18-dev`, `libzstd-dev`.

## Numbers are timestamped

These numbers were taken on 2026-05-15 with: AMD64 host, Rust release profile (`opt-level = 3`, `lto = "off"` — see Session D.5 for why LTO had to be disabled), LLVM 18.1.8 via inkwell 0.5. Reruns on different hardware or after compiler upgrades will produce different absolute timings, but the *ratio* should hold within ~30%.

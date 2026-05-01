# DEVELOPER GUIDE - OMNIcode Architecture & Extension

**Document Version**: 1.1  
**Last Updated**: April 30, 2026  
**Target Audience**: Rust developers, AI researchers, system designers

---

## TABLE OF CONTENTS

1. [Architecture Overview](#architecture-overview)
2. [Module Breakdown](#module-breakdown)
3. [Circuit DSL Grammar](#circuit-dsl-grammar)
4. [Compiler Pipeline](#compiler-pipeline)
5. [Adding New Features](#adding-new-features)
6. [Testing Strategy](#testing-strategy)
7. [Performance Tuning](#performance-tuning)
8. [Common Pitfalls](#common-pitfalls)

---

## ARCHITECTURE OVERVIEW

### Three-Layer Design

```
┌─────────────────────────────────────────────────┐
│  Layer 1: Source Language                       │
│  OMNIcode (.omc) + Circuit DSL                  │
└──────────────────────────────────────────────────┘
                     ▼
┌──────────────────────────────────────────────────┐
│  Layer 2: Parser & Representation                │
│  ├─ Lexer (Tokenization)                         │
│  ├─ Parser (AST generation)                     │
│  └─ Type System (Value enum)                     │
└──────────────────────────────────────────────────┘
                     ▼
┌──────────────────────────────────────────────────┐
│  Layer 3: Execution Engine                       │
│  ├─ Interpreter (Tree-walk evaluation)          │
│  ├─ Circuit Evaluator (Hard/Soft)               │
│  ├─ Genetic Operators (Evolution)               │
│  └─ Built-in Functions (Stdlib)                 │
└──────────────────────────────────────────────────┘
```

### Data Flow Example

```
Program:
  h c = circuit_new(2);
  h result = circuit_eval_hard(c, [true, false]);
  print(result);

Execution:
  Lexer → Tokens([h, c, =, circuit_new, ...])
    ↓
  Parser → AST([VarDecl(...), Assignment(...), Print(...)])
    ↓
  Interpreter:
    VarDecl: execute stmt → eval circuit_new(2) → Value::Circuit
    Assignment: execute stmt → eval circuit_eval_hard(...) → Value::Bool(false)
    Print: output false
```

---

## MODULE BREAKDOWN

### 1. `src/main.rs` - Entry Point (127 lines)

**Responsibility**: Program entry, REPL, file execution

**Key Functions**:
- `main()` - Route to file or REPL mode
- `execute_program()` - Parse and run .omc file
- `repl()` - Interactive prompt loop

**Extension Points**:
- Add command-line flags (--debug, --benchmark, --compile)
- Implement REPL completion
- Add interactive circuit builder

---

### 2. `src/parser.rs` - Lexer & Parser (850 lines)

**Responsibility**: Text → AST conversion

**Architecture**:
```
Lexer::tokenize()
  Reads chars, produces Token stream
    ↓
Parser::parse()
  Consumes tokens, builds AST
  Uses recursive descent with operator precedence
    ↓
AST nodes (Statement, Expression enums)
```

**Key Types**:
```rust
pub enum Token {
    // Keywords: Harmonic, If, Else, While, For, Fn, ...
    // Operators: Plus, Minus, Star, Slash, ...
    // Literals: Number(i64), String(String), Ident(String)
    // Delimiters: LParen, RParen, LBrace, RBrace, ...
}

pub enum Statement {
    VarDecl { name, value, is_harmonic },
    Assignment { name, value },
    If { condition, then_body, elif_parts, else_body },
    // ... more variants
}

pub enum Expression {
    Number(i64),
    String(String),
    Variable(String),
    Add(Box<Expression>, Box<Expression>),
    Call { name, args },
    // ... more variants
}
```

**Operator Precedence** (lowest to highest):
```
or (logical OR)
  ↓
and (logical AND)
  ↓
not (logical NOT)
  ↓
== != < > <= >= (comparison)
  ↓
+ - (addition/subtraction)
  ↓
* / % (multiplication/division)
  ↓
Primary (literals, variables, function calls)
```

**Extension Points**:
- Add infix circuit notation (a & b, a | b, !a)
- Add macro definitions (@macro name = expr)
- Add type annotations (param: bool)
- Add generics (fn<T> func(x: T) → T)

---

### 3. `src/ast.rs` - Type Definitions (120 lines)

**Responsibility**: AST node types for parser output

**Key Types**:
```rust
pub enum Statement { ... }  // 12 variants
pub enum Expression { ... } // 15+ variants
pub enum ForIterable { ... } // Range or Array
```

**Design Pattern**:
- Boxed recursive types (`Box<Expression>`)
- Enum-based pattern matching
- No circular references (DAG structure)

**Extension Points**:
- Add `CircuitDef { name, body }` statement
- Add `CircuitExpr { gates, output }` expression
- Add type annotations to parameters

---

### 4. `src/value.rs` - Type System (250 lines)

**Responsibility**: Runtime value representation

**Key Types**:
```rust
pub enum Value {
    HInt(HInt),           // Harmonic integers
    String(String),       // Text
    Bool(bool),           // Boolean
    Array(HArray),        // Collections
    Circuit(Circuit),     // Genetic circuits (NEW)
    Null,
}

pub struct HInt {
    pub value: i64,           // Integer value
    pub resonance: f64,       // φ-alignment (0-1)
    pub him_score: f64,       // Harmonic Integer Map
    pub is_singularity: bool, // Division-by-zero marker
}

pub struct HArray {
    pub items: Vec<Value>,    // Heterogeneous elements
}
```

**φ-Mathematics**:
- Resonance: How close to nearest Fibonacci number
- HIM: Harmonic Integer Map via golden ratio
- Singularity: Special marker for undefined operations

**Type Conversions**:
- `to_int()` - Any value → integer
- `to_bool()` - Any value → boolean
- `to_string()` - Display representation

**Extension Points**:
- Add `Function` value type (closures)
- Add `Range` value for iteration
- Add `Module` for namespacing

---

### 5. `src/circuits.rs` - Genetic Circuits (540 lines)

**Responsibility**: Logic gates, circuit evaluation, DAG operations

**Key Types**:
```rust
pub enum Gate {
    XAnd { inputs: Vec<GateId> },      // AND gate
    XOr { inputs: Vec<GateId> },       // XOR gate (odd parity)
    XIf { condition, then_gate, else_gate }, // Conditional
    XElse { default_value: bool },      // Fallback
    Input { index: usize },              // Circuit input reference
    Constant { value: bool },            // Hardcoded value
    Not { input: GateId },              // Negation
}

pub struct Circuit {
    pub gates: Vec<Gate>,        // All gates in DAG
    pub output: GateId,          // Output gate ID
    pub num_inputs: usize,       // Input count
}

pub type GateId = usize;  // Index into gates vector
```

**Evaluation Modes**:
```rust
// Hard (Boolean) evaluation
pub fn eval_hard(&self, inputs: &[bool]) -> bool {
    // Recursive evaluation with memoization
}

// Soft (Probabilistic) evaluation
pub fn eval_soft(&self, inputs: &[f64]) -> f64 {
    // Continuous evaluation: AND=product, OR=balanced, IF=weighted
}
```

**Circuit Analysis**:
- `validate()` - DAG check, bounds check
- `to_dot()` - Graphviz export
- `metrics()` - Depth, gate count, histograms

**Extension Points**:
- Add `Latch`, `Memory` for sequential logic
- Add `Multiplexer`, `Decoder` for combinational primitives
- Add custom gate types via plugin system

---

### 6. `src/evolution.rs` - Genetic Operators (360 lines)

**Responsibility**: Mutation, crossover, fitness, GA framework

**Key Functions**:
```rust
pub fn evaluate_fitness(circuit: &Circuit, test_cases: &[TestCase]) -> f64 {
    // Fitness: proportion of correct outputs
}

pub fn mutate_circuit(circuit: &Circuit, mutation_rate: f64) -> Circuit {
    // Random gate type flips, input changes, constant flips
}

pub fn crossover(parent1: &Circuit, parent2: &Circuit) 
    -> (Circuit, Circuit) {
    // Swap gates at random crossover points
}

pub fn evolve_circuits(
    initial_circuit: &Circuit,
    test_cases: &[TestCase],
    config: &EvolutionConfig,
) -> EvolutionResult {
    // Full GA: selection, breeding, mutation, elite preservation
}
```

**GA Configuration**:
```rust
pub struct EvolutionConfig {
    pub population_size: usize,    // 50
    pub num_generations: usize,    // 100
    pub mutation_rate: f64,        // 0.1
    pub crossover_rate: f64,       // 0.7
    pub elite_size: usize,         // 5
}
```

**Test Case Format**:
```rust
pub type TestCase = (Vec<bool>, bool);
// (inputs, expected_output)
```

**Extension Points**:
- Add multi-objective fitness (Pareto front)
- Add speciation (niching) for diversity
- Add adaptive mutation rates
- Implement parallel population evaluation

---

### 7. `src/interpreter.rs` - Execution Engine (520 lines)

**Responsibility**: AST traversal, scope management, function calls

**Key Methods**:
```rust
impl Interpreter {
    pub fn execute(&mut self, statements: Vec<Statement>) -> Result<(), String> {
        // Execute top-level statements
    }

    fn execute_stmt(&mut self, stmt: &Statement) -> Result<(), String> {
        // Route statement to handler
    }

    fn eval_expr(&mut self, expr: &Expression) -> Result<Value, String> {
        // Evaluate expression to value
    }

    fn call_function(&mut self, name: &str, args: &[Expression]) 
        -> Result<Value, String> {
        // Dispatch to built-in or user-defined function
    }
}
```

**Scope Management**:
```
globals: HashMap<String, Value>  // Global variables
functions: HashMap<String, (Vec<String>, Vec<Statement>)> // Defined functions
locals: Vec<HashMap<String, Value>>  // Stack of scopes
```

Each function call pushes a scope, pops on return.

**Built-in Functions** (68+):
- Math: `fibonacci(n)`, `is_fibonacci(x)`
- Strings: `str_len`, `str_concat`, `str_uppercase`, ...
- Arrays: `arr_new`, `arr_push`, `arr_sum`, ...
- Circuits: `circuit_new`, `circuit_eval_hard`, `circuit_mutate`, ...
- Evolution: `evolve_circuits`, `create_random_circuit`, ...

**Extension Points**:
- Add async/await for background execution
- Implement tail call optimization
- Add try/catch for error handling
- Implement lazy evaluation

---

### 8. `src/runtime/stdlib.rs` - Standard Library (309 lines)

**Responsibility**: Built-in function implementations

**Organization**:
```rust
// String functions (30+)
pub fn str_len(s: &str) -> HInt { ... }
pub fn str_concat(s1: &str, s2: &str) -> String { ... }
// ...

// Array functions (35+)
pub fn arr_new(size: usize, default: Value) -> HArray { ... }
pub fn arr_sum(arr: &HArray) -> HInt { ... }
// ...

// Math functions
pub fn fibonacci(n: i64) -> i64 { ... }
pub fn is_fibonacci(x: i64) -> bool { ... }

// Circuit functions (NEW in v1.1)
pub fn circuit_new(num_inputs: usize) -> Circuit { ... }
pub fn circuit_eval_hard(c: &Circuit, inputs: &[bool]) -> bool { ... }
// ...
```

**Design Pattern**:
- Each function takes fully evaluated arguments (already `Value`)
- Returns `Result<Value, String>` for error handling
- No side effects (pure functions)

**Extension Points**:
- Add I/O functions (file read/write)
- Add random number generation
- Add advanced math (trig, statistics)
- Add string regex operations

---

## CIRCUIT DSL GRAMMAR

### Current Grammar (v1.0)

```
program := statement*

statement := var_decl | assignment | print_stmt | if_stmt | while_stmt | for_stmt | fn_def | return_stmt | expr_stmt

var_decl := "h" NAME "=" expr ";"

expr := logical_or

logical_or := logical_and ("or" logical_and)*

logical_and := logical_not ("and" logical_not)*

logical_not := ("not")? comparison

comparison := arith_expr (("==" | "!=" | "<" | ">" | "<=" | ">=") arith_expr)*

arith_expr := term (("+" | "-") term)*

term := factor (("*" | "/" | "%") factor)*

factor := atom | function_call | index_access

atom := NUMBER | STRING | NAME | array_literal | "(" expr ")"

function_call := NAME "(" [expr ("," expr)*] ")"
```

### Planned Extensions (v1.2+)

```
circuit_def := "circuit" NAME "{" gate_expr_list "}"

gate_expr_list := gate_expr (";" gate_expr)* [";"]

gate_expr := 
    | NAME "=" gate_expr
    | gate_expr "&" gate_expr              # Infix AND
    | gate_expr "|" gate_expr              # Infix OR  (actually XOR for now)
    | "!" gate_expr                        # NOT
    | "xAND" "(" gate_expr ("," gate_expr)+ ")"
    | "xOR" "(" gate_expr ("," gate_expr)+ ")"
    | "xIF" "(" gate_expr ")" "{" gate_expr "}" "else" "{" gate_expr "}"
    | NAME "(" [gate_expr ("," gate_expr)*] ")"  # Macro call
    | NAME                                 # Input reference

macro_def := "@macro" NAME "(" [NAME ("," NAME)*] ")" "=" gate_expr ";"

Example DSL (planned):
@macro xor(a, b) = (a & !b) | (!a & b);
@macro majority(a, b, c) = (a & b) | (b & c) | (a & c);

circuit adder {
    sum = xor(a, b);
    carry = (a & b);
}

circuit multiply {
    // 2-bit multiplier
    p00 = (a[0] & b[0]);
    p01 = (a[0] & b[1]);
    p10 = (a[1] & b[0]);
    p11 = (a[1] & b[1]);
}
```

---

## COMPILER PIPELINE

### Current Pipeline (v1.1)

```
Source .omc file
    ↓ Lexer
Token stream
    ↓ Parser
AST (Statement/Expression tree)
    ↓ Interpreter
Evaluate statements
    ├─ Variable bindings (locals/globals)
    ├─ Function calls (built-in or user-defined)
    ├─ Circuit operations (eval_hard, eval_soft)
    ├─ Evolution operations (mutate, crossover)
    └─ I/O (print)
    ↓
Output / Results
```

### Planned Improvements (Tiers 2-4)

**Tier 2 (Advanced Transpiler)**:
```
Source .omc + Circuit DSL
    ↓ Macro Expansion
Expanded AST
    ↓ Linting & Analysis
Warnings (unused vars, dead code, cycles)
    ↓ Normalization
Canonical AST form
    ↓ (Continue to Tier 3)
```

**Tier 3 (Optimizing Compiler)**:
```
Canonical AST
    ↓ Constant Folding
Circuit with constants pre-evaluated
    ↓ Algebraic Simplification
xAND(x, x) → x, xOR(x, x) → 0
    ↓ Dead Code Elimination
Unused gates removed
    ↓ Common Subexpression Elimination
Repeated subexpressions cached
    ↓ Bytecode Compilation
Compact instruction set
    ↓ (Frozen circuits ready for deployment)
```

**Tier 4 (Performance)**:
```
Bytecode or Frozen AST
    ↓ [If Multithreading]
Parallel Fitness Evaluation (4-8× speedup)
    ↓ [If Memory Pooling]
Arena-allocated gates (2× mutation speed)
    ↓ [If AOT Compilation]
Generate Rust → Compile to .so/.dll → Load dynamically
    ↓ Native execution (zero-overhead)
```

---

## ADDING NEW FEATURES

### Add a New Built-in Function

**Example**: Add `circuit_print_stats(circuit) → String`

**Step 1**: Add test in `src/circuits.rs`
```rust
#[test]
fn test_circuit_stats() {
    let mut c = Circuit::new(2);
    let i0 = c.add_gate(Gate::Input { index: 0 });
    let i1 = c.add_gate(Gate::Input { index: 1 });
    c.output = c.add_gate(Gate::XAnd { inputs: vec![i0, i1] });
    
    let metrics = c.metrics();
    assert_eq!(metrics.num_gates, 3);
}
```

**Step 2**: Implement in `src/circuits.rs`
```rust
pub fn print_stats(&self) -> String {
    let m = self.metrics();
    format!("Circuit: {} gates, depth {}, inputs {}",
        m.num_gates, m.depth, m.num_inputs)
}
```

**Step 3**: Add function handler in `src/interpreter.rs`
```rust
fn call_function(&mut self, name: &str, args: &[Expression]) 
    -> Result<Value, String> {
    // ...existing code...
    match name {
        "circuit_print_stats" => {
            if args.len() != 1 { return Err("...".into()); }
            if let Value::Circuit(c) = self.eval_expr(&args[0])? {
                Ok(Value::String(c.print_stats()))
            } else {
                Err("Expected circuit".into())
            }
        }
        // ...
    }
}
```

**Step 4**: Test in OMNIcode
```omnicode
h c = circuit_new(2);
h stats = circuit_print_stats(c);
print(stats);
```

**Step 5**: Rebuild
```bash
cargo build --release
```

### Add a New Gate Type

**Example**: Add `Multiplexer { selector: GateId, options: Vec<GateId> }`

**Step 1**: Update `src/circuits.rs` Gate enum
```rust
pub enum Gate {
    // ...existing...
    Multiplexer { 
        selector: GateId, 
        options: Vec<GateId> 
    }, // NEW
}
```

**Step 2**: Implement evaluation
```rust
fn eval_gate_hard(&self, gate_id: GateId, ...) -> bool {
    match &self.gates[gate_id] {
        // ...existing...
        Gate::Multiplexer { selector, options } => {
            let sel_val = self.eval_gate_hard(*selector, ...);
            let sel_idx = if sel_val { 1 } else { 0 };
            if sel_idx < options.len() {
                self.eval_gate_hard(options[sel_idx], ...)
            } else {
                false
            }
        }
    }
}

fn eval_gate_soft(&self, gate_id: GateId, ...) -> f64 {
    match &self.gates[gate_id] {
        // ...existing...
        Gate::Multiplexer { selector, options } => {
            let sel_val = self.eval_gate_soft(*selector, ...);
            let mut result = 0.0;
            for (i, &option_id) in options.iter().enumerate() {
                let weight = if i == 0 { 1.0 - sel_val } else { sel_val };
                result += weight * self.eval_gate_soft(option_id, ...);
            }
            result
        }
    }
}
```

**Step 3**: Update validation, to_dot, metrics
**Step 4**: Add tests
**Step 5**: Rebuild and test

---

## TESTING STRATEGY

### Unit Tests

Located in each module's `#[cfg(test)]` section:

```bash
# Run all tests
cargo test

# Run specific test
cargo test circuit_and

# Run tests with output
cargo test -- --nocapture

# Run tests in release mode
cargo test --release
```

### Integration Tests

Example test (.omc file):
```omnicode
# tests/evolution_xor.omc
h test_cases = [
    [0, 0, 0],  # Input 0, 1, expected output
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
];

h circuit = circuit_new(2);
h result = evolve_circuits(circuit, test_cases, 100);

if result_fitness > 0.9 {
    print("XOR evolution: PASS");
} else {
    print("XOR evolution: FAIL");
}
```

Run: `./standalone.omc tests/evolution_xor.omc`

### Property-Based Testing

For fuzzing circuit operations:
```rust
#[test]
fn prop_circuit_eval_hard_vs_soft_convergence() {
    // For any circuit, soft eval with inputs [0, 1] 
    // should produce values in [0, 1]
    for _ in 0..100 {
        let c = create_random_circuit(3, 15);
        let soft_result = c.eval_soft(&[0.5, 0.5, 0.5]);
        assert!(soft_result >= 0.0 && soft_result <= 1.0);
    }
}
```

### Regression Tests

Keep golden outputs for complex operations:
```
tests/golden/
  ├── xor_circuit.dot        # Expected Graphviz output
  ├── adder_circuit.dot
  └── evolved_multiplier.json
```

Compare against: `./standalone.omc tests/regressions.omc`

---

## PERFORMANCE TUNING

### Profiling

Use `perf` on Linux:
```bash
# Compile with debug info
cargo build

# Profile
perf record -g ./target/debug/standalone examples/benchmark.omc

# Analyze
perf report
```

Or use `flamegraph`:
```bash
cargo install flamegraph
cargo flamegraph --bin standalone -- examples/benchmark.omc
```

### Hotspots to Watch

1. **Circuit Evaluation** (57% in current benchmark)
   - Solution: Bytecode compilation (Tier 3)
   - Could add: Memoization, caching, SIMD

2. **Fitness Calculation** (Loop bottleneck)
   - Solution: Parallel evaluation (Tier 4)
   - Use `rayon` for data parallelism

3. **Mutation/Crossover** (12% + 8%)
   - Solution: In-place operations, arena allocation
   - Avoid cloning large circuits

4. **Memory Allocation** (Hidden overhead)
   - Solution: Pre-allocate pools, reuse buffers
   - Use `Vec::with_capacity()`

### Optimization Checklist

- [ ] Use `--release` for 10-100× speedup
- [ ] Profile before optimizing (find real hotspots)
- [ ] Measure improvements (criterion.rs)
- [ ] Avoid premature optimization
- [ ] Prefer algorithm improvements over micro-optimizations
- [ ] Keep code readable (let the compiler optimize)

---

## COMMON PITFALLS

### 1. Circuit Cycles

❌ **Mistake**: Creating gates with circular references
```rust
let g1 = circuit.add_gate(Gate::Input { index: 0 });
let g2 = circuit.add_gate(Gate::Input { index: 1 });
// ... somehow g1 depends on g2, and g2 depends on g1
```

✅ **Solution**: Always call `circuit.validate()` after construction
```rust
circuit.validate()?;  // Returns error if cycles detected
```

### 2. Type Mismatches

❌ **Mistake**: Wrong types in function arguments
```omnicode
h c = circuit_new("2");  # Should be number, not string
```

✅ **Solution**: Runtime type checking in functions
```rust
match self.eval_expr(&args[0])? {
    Value::HInt(h) => { /* use h.value */ }
    _ => Err("Expected integer".into()),
}
```

### 3. Unbounded Evolution

❌ **Mistake**: Evolution with no convergence check
```omnicode
# This could run forever if fitness never reaches 1.0
h result = evolve_circuits(c, test_cases, 1000000);
```

✅ **Solution**: Set reasonable limits, check convergence
```rust
let config = EvolutionConfig {
    num_generations: 100,  // Fixed limit
    population_size: 50,
    // ...
};
```

### 4. Soft Evaluation Precision

❌ **Mistake**: Comparing soft eval results with ==
```rust
if c.eval_soft(&[0.5, 0.5]) == 0.5 { ... }  // May fail due to rounding
```

✅ **Solution**: Use approximate comparison
```rust
if (c.eval_soft(&[0.5, 0.5]) - 0.5).abs() < 0.01 { ... }
```

### 5. Memory Leaks in Crossover

❌ **Mistake**: Cloning entire population on each generation
```rust
let mut new_pop = Vec::new();
for circuit in &population {
    new_pop.push(circuit.clone());  // O(n²) memory in total
}
```

✅ **Solution**: Reuse allocation, swap instead of clone
```rust
let mut new_pop = Vec::with_capacity(population.len());
for (parent1, parent2) in parent_pairs {
    let (c1, c2) = crossover(parent1, parent2);
    new_pop.push(c1);
    new_pop.push(c2);
}
population = new_pop;  // Reuse allocation
```

### 6. Missing Error Handling

❌ **Mistake**: Ignoring validation errors
```rust
let c = create_random_circuit(4, 20);
// May contain cycles or invalid references!
```

✅ **Solution**: Always validate before use
```rust
let c = create_random_circuit(4, 20);
c.validate()?;  // Propagate error if invalid
```

---

## DEBUGGING TECHNIQUES

### Print Debugging

```rust
eprintln!("Gate {}: {:?}", gate_id, &self.gates[gate_id]);
eprintln!("Eval result: {}", result);
```

### Visual Debugging

Export to Graphviz:
```omnicode
h c = circuit_new(2);
# ... build circuit ...
h dot_string = circuit_to_dot(c);
print(dot_string);
```

Save and render:
```bash
./standalone.omc debug_circuit.omc > circuit.dot
dot -Tpng circuit.dot -o circuit.png
```

### Unit Test Isolation

Test individual gates:
```rust
#[test]
fn test_xand_gate_only() {
    let mut c = Circuit::new(2);
    let i0 = c.add_gate(Gate::Input { index: 0 });
    let i1 = c.add_gate(Gate::Input { index: 1 });
    c.output = c.add_gate(Gate::XAnd { inputs: vec![i0, i1] });
    
    assert_eq!(c.eval_hard(&[true, true]), true);
    assert_eq!(c.eval_hard(&[false, true]), false);
}
```

### LLDB Debugger (Advanced)

```bash
rust-lldb ./target/debug/standalone -- examples/debug.omc
(lldb) break set --name main
(lldb) run
(lldb) print circuit
```

---

## CONCLUSION

This guide covers the essentials of extending OMNIcode:
- Module organization and responsibilities
- Data flow through the pipeline
- Grammar and syntax
- Testing and profiling strategies
- Common mistakes to avoid

**Next Steps**:
1. Study `src/circuits.rs` to understand gate types
2. Implement a simple new built-in function
3. Write tests for your changes
4. Profile and optimize hotspots
5. Document your extensions

**For Questions**:
- Review the code's inline comments
- Check the test cases for usage examples
- Refer to the IMPROVEMENT_PLAN.md for architectural roadmap
- Study the BENCHMARKS.md for performance insights

**Happy coding!** 🚀


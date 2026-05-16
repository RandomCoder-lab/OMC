# OMNIcode Shared Library for GDExtension

## Summary

Successfully converted OMNIcode from a binary executable to a shared library (`libomnimcode_ffi.so`) that can be called from GDScript at native speed.

## Key Files

| File | Purpose |
|------|---------|
| `/home/thearchitect/OMC/omnimcode-ffi/src/lib.rs` | C FFI bindings with `evaluate()` and `vm_*` functions |
| `/home/thearchitect/OMC/target/release/libomnimcode_ffi.so` | Compiled shared library (528KB) |
| `/home/thearchitect/OMC/omnimcode-gdextension/include/omnimcode.h` | C header for GDExtension |
| `/home/thearchitect/OMC/omnimcode-gdextension/src/omnimcode_extension.cpp` | GDExtension binding class wrappers |

## Exposed C API

```c
// OMC Code Execution
int omnicode_evaluate(const char* source);       // Stateless execute
OmnimcodeVM* omnicode_vm_new(void);                // Create VM for stateful execution
int omnicode_vm_execute(OmnimcodeVM* vm, const char* source);
int omnicode_vm_reset(OmnimcodeVM* vm);
void omnicode_vm_free(OmnimcodeVM* vm);

// Circuit API
OmnimcodeCircuit* omnicode_circuit_new(uint32_t inputs);
bool omnicode_circuit_eval(OmnimcodeCircuit* circuit, const bool* inputs, size_t n);
void omnicode_circuit_free(OmnimcodeCircuit* circuit);

// Evolution API
OmnimcodeEvolver* omnicode_evolver_new(uint32_t pop_size);
void omnicode_evolver_step(OmnimcodeEvolver* evolver);
uint32_t omnicode_evolver_generation(OmnimcodeEvolver* evolver);
double omnicode_evolver_best_fitness(OmnimcodeEvolver* evolver);
void omnicode_evolver_free(OmnimcodeEvolver* evolver);
```

## GDExtension Classes

- **OmnimcodeVMRef** - Execute OMC code with persistent state
- **OmnimcodeCircuitRef** - Genetic logic circuit evaluation
- **OmnimcodeEvolverRef** - Run genetic algorithms at native speed

## Usage from GDScript

```gdscript
# Simple one-shot execution
var result = omnicode_evaluate("print(fibonacci(10));")

# VM for persistent state
var vm = OmnimcodeVMRef.new()
vm.execute("x = 42; print(x);")
vm.execute("print(x + 1);")  # x persists
vm.reset()

# Circuits for genetic logic
var circuit = OmnimcodeCircuitRef.new()
# circuit.eval([true, false]) -> bool output
```
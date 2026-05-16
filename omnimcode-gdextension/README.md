# OMNIcode GDExtension

This module provides GDExtension bindings for the OMNIcode harmonic computing language library.

## Building the Shared Library

```bash
cd /home/thearchitect/OMC
cargo build --release -p omnimcode-ffi
```

This produces `target/release/libomnimcode_ffi.so`.

## Library Functions

### Circuit API (Genetic Logic Circuits)
- `omnicode_circuit_new(inputs)` - Create a circuit with N inputs
- `omnicode_circuit_eval(circuit, inputs, count)` - Evaluate circuit with boolean inputs
- `omnicode_circuit_free(circuit)` - Free circuit memory

### Evolution API
- `omnicode_evolver_new(pop_size)` - Create an evolver
- `omnicode_evolver_step(evolver)` - Run one evolution step
- `omnicode_evolver_generation(evolver)` - Get current generation
- `omnicode_evolver_best_fitness(evolver)` - Get best fitness score
- `omnicode_evolver_free(evolver)` - Free evolver

### OMC Code Execution API
- `omnicode_evaluate(source)` - Execute OMC code (stateless, new interpreter each call)
- `omnicode_vm_new()` - Create a VM context with persistent state
- `omnicode_vm_execute(vm, source)` - Execute OMC code in VM context
- `omnicode_vm_reset(vm)` - Reset VM state
- `omnicode_vm_free(vm)` - Free VM

## GDExtension Classes

### OmnimcodeVMRef
GDScript usage:
```gdscript
var vm = OmnimcodeVMRef.new()
vm.execute("print(42);")
vm.reset()
```

### OmnimcodeCircuitRef
```gdscript
var circuit = OmnimcodeCircuitRef.new()
# circuit.initialize(2) # 2 inputs
# circuit.evaluate([true, false])
```

### OmnimcodeEvolverRef
```gdscript
var evolver = OmnimcodeEvolverRef.new()
# evolver.initialize(100) # population 100
# evolver.step()
# var gen = evolver.get_generation()
# var fitness = evolver.get_best_fitness()
```

## OMC Language Examples

```omc
# Basic print
print(42);

# Variables
x = 10;
y = 20;
print(x + y);

# Loops
for i in range(0, 5) {
    print(i);
}

# Functions
fn add(a, b) -> a + b;
print(add(3, 4));

# Decision evolution (XOR problem)
# The evolver functions handle genetic algorithms natively
```
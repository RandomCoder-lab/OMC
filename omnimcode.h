// omnimcode.h - C FFI Header for OMNIcode
// Generated from omnimcode-ffi Rust library
// Version: 1.0.0

#ifndef OMNIMCODE_H
#define OMNIMCODE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ===== Types =====

/// Opaque handle to a Circuit
/// Use omnicode_circuit_new to create, omnicode_circuit_free to destroy
typedef struct OmnimcodeCircuit OmnimcodeCircuit;

/// Opaque handle to an Evolver
/// Use omnicode_evolver_new to create, omnicode_evolver_free to destroy
typedef struct OmnimcodeEvolver OmnimcodeEvolver;

// ===== Functions =====

/// Create a new circuit with given number of inputs
/// @param inputs Number of boolean inputs to the circuit
/// @return Pointer to new circuit, or NULL on error
/// @note Must be freed with omnicode_circuit_free
OmnimcodeCircuit* omnicode_circuit_new(uint32_t inputs);

/// Evaluate a circuit with given boolean inputs
/// @param circuit Valid circuit pointer
/// @param inputs Array of boolean inputs
/// @param input_count Length of inputs array
/// @return Boolean output of the circuit
bool omnicode_circuit_eval(
    OmnimcodeCircuit* circuit,
    const bool* inputs,
    uintptr_t input_count
);

/// Free a circuit and release associated resources
/// @param circuit Pointer to circuit created with omnicode_circuit_new
/// @note After this call, circuit pointer is invalid
void omnicode_circuit_free(OmnimcodeCircuit* circuit);

/// Create a new evolver
/// @param population_size Number of circuits to evolve simultaneously
/// @return Pointer to new evolver, or NULL on error
/// @note Must be freed with omnicode_evolver_free
OmnimcodeEvolver* omnicode_evolver_new(uint32_t population_size);

/// Free an evolver and release associated resources
/// @param evolver Pointer to evolver created with omnicode_evolver_new
/// @note After this call, evolver pointer is invalid
void omnicode_evolver_free(OmnimcodeEvolver* evolver);

/// Get version string
/// @return Pointer to version string (e.g., "1.0.0")
const char* omnicode_version(void);

#ifdef __cplusplus
}
#endif

#endif // OMNIMCODE_H

// omnimcode-gdextension/include/omnimcode.h
// C FFI header for OMNIcode shared library

#ifndef OMNIICODE_H
#define OMNIICODE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

// ============= Circuit Functions =============

typedef struct OmnimcodeCircuit OmnimcodeCircuit;

OmnimcodeCircuit* omnicode_circuit_new(uint32_t inputs);
bool omnicode_circuit_eval(OmnimcodeCircuit* circuit, const bool* inputs, size_t input_count);
void omnicode_circuit_free(OmnimcodeCircuit* circuit);

// ============= Evolver Functions =============

typedef struct OmnimcodeEvolver OmnimcodeEvolver;

OmnimcodeEvolver* omnicode_evolver_new(uint32_t population_size);
void omnicode_evolver_free(OmnimcodeEvolver* evolver);
void omnicode_evolver_step(OmnimcodeEvolver* evolver);
uint32_t omnicode_evolver_generation(OmnimcodeEvolver* evolver);
double omnicode_evolver_best_fitness(OmnimcodeEvolver* evolver);

// ============= OMC Code Execution =============

const char* omnicode_version(void);
int omnicode_evaluate(const char* source);

typedef struct OmnimcodeVM OmnimcodeVM;

OmnimcodeVM* omnicode_vm_new(void);
int omnicode_vm_execute(OmnimcodeVM* vm, const char* source);
int omnicode_vm_reset(OmnimcodeVM* vm);
void omnicode_vm_free(OmnimcodeVM* vm);

#ifdef __cplusplus
}
#endif

#endif // OMNIICODE_H
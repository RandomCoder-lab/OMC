// omnimcode-ffi/src/lib.rs
// C FFI bindings for OMNIcode

use omnimcode_core::circuits::Circuit;
use omnimcode_core::evolution::{evaluate_fitness, TestCase, EvolutionConfig};
use std::ffi::CStr;
use std::os::raw::c_char;

/// Opaque handle to a Circuit
#[repr(C)]
pub struct OmnimcodeCircuit {
    inner: Box<Circuit>,
}

/// Opaque handle to an Evolver
#[repr(C)]
pub struct OmnimcodeEvolver {
    config: EvolutionConfig,
    population: Vec<Circuit>,
    test_cases: Vec<TestCase>,
}

/// Create a new circuit with given number of inputs
/// # Safety
/// Caller must ensure the pointer is freed with omnicode_circuit_free
#[no_mangle]
pub unsafe extern "C" fn omnicode_circuit_new(inputs: u32) -> *mut OmnimcodeCircuit {
    let circuit = Box::new(Circuit::new(inputs as usize));
    Box::into_raw(Box::new(OmnimcodeCircuit {
        inner: circuit,
    }))
}

/// Evaluate a circuit with given inputs
/// # Safety
/// Circuit pointer must be valid and inputs must point to valid bool array of correct length
#[no_mangle]
pub unsafe extern "C" fn omnicode_circuit_eval(
    circuit: *mut OmnimcodeCircuit,
    inputs: *const bool,
    input_count: usize,
) -> bool {
    if circuit.is_null() || inputs.is_null() {
        return false;
    }
    
    let circuit = &(*circuit).inner;
    let input_slice = std::slice::from_raw_parts(inputs, input_count);
    circuit.eval_hard(input_slice)
}

/// Free a circuit
/// # Safety
/// Pointer must be valid and not used after this call
#[no_mangle]
pub unsafe extern "C" fn omnicode_circuit_free(circuit: *mut OmnimcodeCircuit) {
    if !circuit.is_null() {
        let _ = Box::from_raw(circuit);
    }
}

/// Create a new evolver
/// # Safety
/// Caller must ensure the pointer is freed with omnicode_evolver_free
#[no_mangle]
pub unsafe extern "C" fn omnicode_evolver_new(population_size: u32) -> *mut OmnimcodeEvolver {
    let config = EvolutionConfig {
        population_size: population_size as usize,
        num_generations: 100,
        mutation_rate: 0.05,
        crossover_rate: 0.8,
        elite_size: 2,
    };
    
    Box::into_raw(Box::new(OmnimcodeEvolver {
        config,
        population: Vec::new(),
        test_cases: Vec::new(),
    }))
}

/// Free an evolver
/// # Safety
/// Pointer must be valid and not used after this call
#[no_mangle]
pub unsafe extern "C" fn omnicode_evolver_free(evolver: *mut OmnimcodeEvolver) {
    if !evolver.is_null() {
        let _ = Box::from_raw(evolver);
    }
}

/// Get the version string
#[no_mangle]
pub extern "C" fn omnicode_version() -> *const c_char {
    const VERSION: &str = "1.0.0\0";
    VERSION.as_ptr() as *const c_char
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_creation() {
        unsafe {
            let circuit = omnicode_circuit_new(2);
            assert!(!circuit.is_null());
            omnicode_circuit_free(circuit);
        }
    }
}

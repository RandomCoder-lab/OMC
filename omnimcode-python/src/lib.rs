// omnimcode-python/src/lib.rs
// Python bindings for OMNIcode using PyO3

use pyo3::prelude::*;
use omnimcode_core::circuits::Circuit;
use omnimcode_core::evolution::{evaluate_fitness, TestCase};

/// A Python wrapper around the OMNIcode Circuit
#[pyclass]
pub struct OmnimcodeCircuit {
    inner: Circuit,
}

#[pymethods]
impl OmnimcodeCircuit {
    /// Create a new circuit with given number of inputs
    #[new]
    fn new(inputs: usize) -> Self {
        OmnimcodeCircuit {
            inner: Circuit::new(inputs),
        }
    }

    /// Evaluate the circuit with given boolean inputs
    fn eval(&self, inputs: Vec<bool>) -> bool {
        self.inner.eval_hard(&inputs)
    }

    /// Get the number of gates in the circuit
    fn gate_count(&self) -> usize {
        self.inner.gates.len()
    }

    /// Get string representation
    fn __repr__(&self) -> String {
        format!("OmnimcodeCircuit(gates={})", self.inner.gates.len())
    }
}

/// A Python wrapper around fitness evaluation
#[pyfunction]
fn evaluate_circuit_fitness(
    circuit: &OmnimcodeCircuit,
    test_cases: Vec<(Vec<bool>, bool)>,
) -> PyResult<f64> {
    // TestCase is a type alias: (Vec<bool>, bool)
    Ok(evaluate_fitness(&circuit.inner, &test_cases))
}

/// Module initialization
#[pymodule]
fn omnimcode(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<OmnimcodeCircuit>()?;
    m.add_function(wrap_pyfunction!(evaluate_circuit_fitness, m)?)?;
    m.add("__version__", "1.0.0")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_creation() {
        let circuit = OmnimcodeCircuit::new(2);
        // Circuits start with one constant gate by default
        assert_eq!(circuit.gate_count(), 1);
    }
}

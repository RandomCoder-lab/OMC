using System;
using System.Collections.Generic;

namespace SovereignLattice.OMNIcode
{
    /// <summary>
    /// Managed wrapper around an OMNIcode Circuit
    /// </summary>
    public class OmnimcodeCircuit : IDisposable
    {
        private NativeBindings.CircuitHandle _handle;
        private uint _inputCount;
        private bool _disposed = false;

        /// <summary>
        /// Number of boolean inputs this circuit accepts
        /// </summary>
        public uint InputCount => _inputCount;

        /// <summary>
        /// Create a new circuit with specified number of inputs
        /// </summary>
        /// <param name="numInputs">Number of boolean inputs</param>
        /// <exception cref="InvalidOperationException">If native library fails</exception>
        public OmnimcodeCircuit(uint numInputs)
        {
            _inputCount = numInputs;
            _handle = NativeBindings.omnicode_circuit_new(numInputs);
            
            if (_handle.IsInvalid)
            {
                throw new InvalidOperationException("Failed to create OMNIcode circuit");
            }
        }

        /// <summary>
        /// Evaluate the circuit with given boolean inputs
        /// </summary>
        /// <param name="inputs">Array of boolean inputs (length must match InputCount)</param>
        /// <returns>Boolean output of circuit</returns>
        /// <exception cref="ArgumentException">If input array length doesn't match</exception>
        /// <exception cref="ObjectDisposedException">If circuit is disposed</exception>
        public bool Evaluate(bool[] inputs)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(OmnimcodeCircuit));

            if (inputs == null)
                throw new ArgumentNullException(nameof(inputs));

            if ((uint)inputs.Length != _inputCount)
                throw new ArgumentException(
                    $"Expected {_inputCount} inputs but got {inputs.Length}",
                    nameof(inputs)
                );

            return NativeBindings.omnicode_circuit_eval(_handle, inputs, _inputCount);
        }

        /// <summary>
        /// Evaluate the circuit with given boolean inputs (convenience overload)
        /// </summary>
        public bool Evaluate(params bool[] inputs) => Evaluate(inputs);

        /// <summary>
        /// Free the circuit resources
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                _handle?.Dispose();
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }

        ~OmnimcodeCircuit()
        {
            Dispose();
        }
    }

    /// <summary>
    /// Managed wrapper around an OMNIcode Evolver for genetic algorithm evolution
    /// </summary>
    public class OmnimcodeEvolver : IDisposable
    {
        private NativeBindings.EvolverHandle _handle;
        private bool _disposed = false;

        /// <summary>
        /// Create a new evolver with specified population size
        /// </summary>
        /// <param name="populationSize">Number of circuits in population</param>
        /// <exception cref="InvalidOperationException">If native library fails</exception>
        public OmnimcodeEvolver(uint populationSize)
        {
            _handle = NativeBindings.omnicode_evolver_new(populationSize);
            
            if (_handle.IsInvalid)
            {
                throw new InvalidOperationException("Failed to create OMNIcode evolver");
            }
        }

        /// <summary>
        /// Current generation number
        /// </summary>
        public uint Generation
        {
            get
            {
                if (_disposed)
                    throw new ObjectDisposedException(nameof(OmnimcodeEvolver));
                return NativeBindings.omnicode_evolver_generation(_handle);
            }
        }

        /// <summary>
        /// Best fitness found so far
        /// </summary>
        public double BestFitness
        {
            get
            {
                if (_disposed)
                    throw new ObjectDisposedException(nameof(OmnimcodeEvolver));
                return NativeBindings.omnicode_evolver_best_fitness(_handle);
            }
        }

        /// <summary>
        /// Run one generation of evolution
        /// </summary>
        /// <exception cref="ObjectDisposedException">If evolver is disposed</exception>
        public void Step()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(OmnimcodeEvolver));

            NativeBindings.omnicode_evolver_step(_handle);
        }

        /// <summary>
        /// Get the best circuit evolved so far
        /// </summary>
        /// <returns>New OmnimcodeCircuit instance (caller must dispose)</returns>
        /// <exception cref="ObjectDisposedException">If evolver is disposed</exception>
        public OmnimcodeCircuit GetBestCircuit()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(OmnimcodeEvolver));

            // This is a placeholder - real implementation would need to track input count
            throw new NotImplementedException("GetBestCircuit requires additional state tracking");
        }

        /// <summary>
        /// Run evolution for specified number of generations
        /// </summary>
        /// <param name="generations">Number of generations to evolve</param>
        public void EvolveForGenerations(uint generations)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(OmnimcodeEvolver));

            for (uint i = 0; i < generations; i++)
            {
                Step();
            }
        }

        /// <summary>
        /// Free evolver resources
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                _handle?.Dispose();
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }

        ~OmnimcodeEvolver()
        {
            Dispose();
        }
    }
}

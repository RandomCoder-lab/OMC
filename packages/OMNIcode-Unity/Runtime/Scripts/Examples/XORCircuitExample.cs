using UnityEngine;
using SovereignLattice.OMNIcode;

namespace SovereignLattice.OMNIcode.Examples
{
    /// <summary>
    /// Simple example: Evolve a circuit to solve XOR problem
    /// </summary>
    public class XORCircuitExample : MonoBehaviour
    {
        public uint populationSize = 100;
        public uint generationsPerFrame = 10;
        public uint targetGenerations = 1000;

        private OmnimcodeEvolver _evolver;
        private uint _currentGeneration = 0;

        void Start()
        {
            _evolver = new OmnimcodeEvolver(populationSize);
            Debug.Log("OMNIcode XOR Evolution Started");
        }

        void Update()
        {
            if (_evolver == null || _currentGeneration >= targetGenerations)
                return;

            // Run generationsPerFrame steps
            _evolver.EvolveForGenerations(generationsPerFrame);
            _currentGeneration += generationsPerFrame;

            // Log progress every 100 generations
            if (_currentGeneration % 100 == 0)
            {
                Debug.Log($"Generation: {_currentGeneration}, Best Fitness: {_evolver.BestFitness:F4}");
            }

            // Check if converged
            if (_currentGeneration >= targetGenerations)
            {
                Debug.Log($"Evolution complete! Final fitness: {_evolver.BestFitness:F4}");
            }
        }

        void OnDestroy()
        {
            _evolver?.Dispose();
        }
    }
}

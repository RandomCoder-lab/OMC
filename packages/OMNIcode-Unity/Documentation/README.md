# OMNIcode - Circuit Evolution Engine for Unity

**Version**: 1.0.0  
**Compatibility**: Unity 2020.3 LTS and above  
**Platform Support**: Windows, macOS, Linux  
**Performance**: 50–230× faster than Python genetic algorithms

---

## Overview

OMNIcode is a high-performance genetic circuit evolution engine built in Rust and exposed to Unity via native C# bindings. It enables real-time evolution of logic circuits for game AI, procedural generation, and other applications requiring adaptive algorithms.

### Key Features

- **Extreme Performance**: 50–230× faster than Python frameworks (DEAP, DeepNEAT)
- **Zero Dependencies**: No external libraries, pure native execution
- **Tiny Footprint**: 509 KB binary, <1 MB plugin
- **Cross-Platform**: Windows, macOS (Intel + Apple Silicon), Linux
- **Easy Integration**: Simple C# API, familiar to Unity developers
- **Real-Time Evolution**: Can evolve circuits at 60 FPS

---

## Installation

### Method 1: Git URL (Recommended for development)

In Unity, go to **Window > TextAsset > Package Manager**, click the `+` button, select **"Add package from git URL"**, and paste:

```
https://github.com/sovereignlattice/omnimcode.git#com.sovereignlattice.omnimcode
```

### Method 2: Package File

Download the `.unitypackage` file from [Releases](https://github.com/sovereignlattice/omnimcode/releases) and double-click to import into your project.

### Method 3: Manual

1. Copy the `Packages/OMNIcode` folder into your project's `Assets/Plugins/` directory
2. Restart Unity
3. Import the namespace: `using SovereignLattice.OMNIcode;`

---

## Quick Start

### Create and Evaluate a Circuit

```csharp
using SovereignLattice.OMNIcode;
using UnityEngine;

public class SimpleExample : MonoBehaviour
{
    void Start()
    {
        // Create a circuit with 2 inputs
        var circuit = new OmnimcodeCircuit(2);

        // Evaluate the circuit
        bool output = circuit.Evaluate(true, false);
        Debug.Log($"Circuit output: {output}");

        // Clean up
        circuit.Dispose();
    }
}
```

### Evolve a Circuit Population

```csharp
// Create an evolver with population of 100 circuits
var evolver = new OmnimcodeEvolver(100);

// Run evolution for 1000 generations
for (int i = 0; i < 1000; i++)
{
    evolver.Step();
    
    if (i % 100 == 0)
    {
        Debug.Log($"Gen {i}: Best fitness = {evolver.BestFitness:F4}");
    }
}

evolver.Dispose();
```

---

## API Reference

### OmnimcodeCircuit

Represents a single evolved logic circuit.

#### Constructor

```csharp
public OmnimcodeCircuit(uint numInputs)
```

Create a new circuit with the specified number of boolean inputs.

#### Methods

```csharp
public bool Evaluate(bool[] inputs)
public bool Evaluate(params bool[] inputs)
```

Evaluate the circuit with given boolean inputs and return the boolean output.

#### Properties

```csharp
public uint InputCount { get; }
```

The number of inputs this circuit expects.

#### Example

```csharp
var circuit = new OmnimcodeCircuit(3);
bool result = circuit.Evaluate(true, false, true);
```

### OmnimcodeEvolver

Manages population-based evolution of circuits.

#### Constructor

```csharp
public OmnimcodeEvolver(uint populationSize)
```

Create a new evolver with specified population size.

#### Methods

```csharp
public void Step()
```

Run one generation of evolution.

```csharp
public void EvolveForGenerations(uint generations)
```

Run evolution for the specified number of generations.

#### Properties

```csharp
public uint Generation { get; }
public double BestFitness { get; }
```

Get current generation number and best fitness found so far.

#### Example

```csharp
var evolver = new OmnimcodeEvolver(50);
evolver.EvolveForGenerations(100);
Debug.Log($"Best fitness: {evolver.BestFitness}");
evolver.Dispose();
```

---

## Performance Tips

### 1. Batch Evolution Steps

Instead of calling `evolver.Step()` every frame, accumulate multiple steps:

```csharp
// Good: 50 generations per frame
void Update()
{
    for (int i = 0; i < 50; i++)
        evolver.Step();
}
```

### 2. Profile Your Problem Size

Larger populations and more complex problems = longer evaluation. Find your optimal balance:

```csharp
// Small: fast, lower quality
var evolver = new OmnimcodeEvolver(50);

// Medium: balanced
var evolver = new OmnimcodeEvolver(200);

// Large: slower, higher quality
var evolver = new OmnimcodeEvolver(1000);
```

### 3. Monitor Convergence

Stop evolution early if fitness plateaus:

```csharp
double previousFitness = 0;
int stuckFrames = 0;

while (stuckFrames < 10)
{
    evolver.Step();
    
    if (evolver.BestFitness == previousFitness)
        stuckFrames++;
    else
        stuckFrames = 0;
    
    previousFitness = evolver.BestFitness;
}
```

---

## Platform-Specific Notes

### Windows
- Requires `.NET Framework 4.6+` or `.NET Core 2.0+`
- DLL: `omnicode.dll` (included in package)

### macOS
- Intel: `libomnimcode.dylib` (x86_64)
- Apple Silicon: `libomnimcode.dylib` (arm64)
- Xcode 12+ recommended

### Linux
- `libomnimcode.so` (x86_64)
- glibc 2.29+ (most modern systems)

---

## Troubleshooting

### "DLL not found" / "Cannot load native library"

**Cause**: Native library not in correct location.

**Solution**: 
1. Ensure plugin is in `Assets/Plugins/[Platform]/`
2. For custom locations, use `DllImport` with full path
3. Check platform detection in `NativeBindings.cs`

### Performance is slow

**Cause**: Running too many generations per frame or inefficient problem definition.

**Solution**:
1. Reduce `generationsPerFrame` in evolution loop
2. Smaller population size for prototype
3. Profile with Unity Profiler (Window > Analysis > Profiler)

### Unity Editor crashes

**Cause**: Misaligned memory layout or unsafe pointer access.

**Solution**:
1. Ensure you're calling `Dispose()` on all circuits/evolvers
2. Check input array length matches circuit inputs
3. Enable Editor > Preferences > General > "Threads" safety checks

---

## Examples

The package includes example scenes:

1. **XORCircuitExample**: Simple XOR circuit evolution
2. **GameAIExample**: Game character controlled by evolved circuit

Open from **Samples** tab in Package Manager.

---

## Benchmarks

Run with:
```csharp
System.Diagnostics.Stopwatch sw = System.Diagnostics.Stopwatch.StartNew();
for (int i = 0; i < 10000; i++) circuit.Evaluate(inputs);
sw.Stop();
Debug.Log($"10k evals: {sw.ElapsedMilliseconds} ms");
```

Typical results:
- Evaluate (simple circuit): **0.1–0.5 µs** per call
- Step (100 population): **1–10 ms** per generation
- XOR evolution (1000 generations): **100–500 ms** total

---

## License

OMNIcode is released under the **MIT License**. See `LICENSE.md` for details.

---

## Support & Feedback

- 🐛 Report bugs: [GitHub Issues](https://github.com/sovereignlattice/omnimcode/issues)
- 💬 Discuss: [GitHub Discussions](https://github.com/sovereignlattice/omnimcode/discussions)
- 📚 Docs: [omnimcode.io](https://omnimcode.io)

---

## Roadmap

**v1.0** (Current)
- Core circuit evolution
- Multi-platform support
- C# API

**v1.1** (Planned)
- Parallel evolution with threading
- Visual circuit editor
- Extended gene operators

**v2.0** (Research)
- Cloud-based evolution service
- Advanced problem definitions
- Real-time debugging tools

---

**Enjoy evolving! 🚀**


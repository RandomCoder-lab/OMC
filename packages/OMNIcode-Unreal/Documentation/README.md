# OMNIcode - Unreal Engine Plugin

**Version**: 1.0.0  
**Engine Compatibility**: Unreal Engine 5.0+ (Windows, Linux, macOS)  
**Performance**: 50–230× faster than Python + genetic libraries

---

## Installation

1. **Download** the plugin from Unreal Marketplace or GitHub Releases
2. **Extract** to your project: `Plugins/OMNIcode/`
3. **Restart** Unreal Engine
4. **Enable** the plugin in Edit → Plugins → Search "OMNIcode" → Enable
5. **Restart** the editor again

---

## Quick Start

### Blueprint Setup

1. Create a Blueprint actor that uses `UOmnimcodeEvolver`
2. In `BeginPlay()`:
   ```cpp
   UOmnimcodeEvolver* Evolver = NewObject<UOmnimcodeEvolver>();
   Evolver->Create(64);  // 64 population size
   ```

3. Define test cases (truth table):
   ```
   0 0 -> 1  (XOR training)
   0 1 -> 1
   1 0 -> 1
   1 1 -> 0
   ```

4. Call `Evolve()`:
   ```cpp
   TArray<FString> TestCases = {
       "00->1", "01->1", "10->1", "11->0"
   };
   Evolver->Evolve(TestCases, 100);  // 100 generations
   ```

5. Get the result:
   ```cpp
   UOmnimcodeCircuit* BestCircuit = Evolver->GetBestCircuit();
   bool Result = BestCircuit->Evaluate({false, true});
   ```

---

## API Reference

### UOmnimcodeCircuit

#### Create(NumInputs: int32)
Create a new circuit with specified boolean inputs.
- **Parameters**: NumInputs (2-8 recommended)
- **Returns**: void

#### Evaluate(Inputs: TArray<bool>): bool
Evaluate the circuit with given boolean inputs.
- **Parameters**: Array of boolean values
- **Returns**: Boolean output value
- **Performance**: 200-700 ns typical

#### GetGateCount(): int32
Get the number of logic gates in the circuit.
- **Returns**: Gate count

#### GetComplexity(): int32
Get circuit complexity metric.
- **Returns**: Complexity value

---

### UOmnimcodeEvolver

#### Create(PopulationSize: int32)
Create a new evolver with specified population size.
- **Parameters**: PopulationSize (32-256 typical)
- **Returns**: void

#### Evolve(TestCases: TArray<FString>, NumGenerations: int32)
Run genetic evolution on the population.
- **Parameters**:
  - TestCases: Array of strings like "01010->1"
  - NumGenerations: Number of evolution steps
- **Returns**: void

#### GetBestCircuit(): UOmnimcodeCircuit*
Get the best-evolved circuit from the population.
- **Returns**: UOmnimcodeCircuit object (may be null)

---

## Use Cases

### Game AI Training

Train opponent AI circuits offline:

```cpp
// Training phase
UOmnimcodeEvolver* Trainer = NewObject<UOmnimcodeEvolver>();
Trainer->Create(128);
TArray<FString> GameRules = GenerateGameRulesFromTraining();
Trainer->Evolve(GameRules, 500);
UOmnimcodeCircuit* AICircuit = Trainer->GetBestCircuit();
SaveAIToFile(AICircuit);

// Runtime
AICircuit->Evaluate(CurrentGameState);
```

### Procedural Generation

Evolve circuits for procedural content generation:

```cpp
// Generate building layout rules
UOmnimcodeEvolver* GenEvolver = NewObject<UOmnimcodeEvolver>();
GenEvolver->Create(64);
GenEvolver->Evolve(LayoutConstraints, 200);
UOmnimcodeCircuit* LayoutLogic = GenEvolver->GetBestCircuit();
```

### Real-Time Decision Making

Evaluate evolved circuits for frame-rate-safe decisions:

```cpp
// Per tick - circuit evaluation is <1 microsecond
bool ShouldAttack = AICircuit->Evaluate({PlayerNearby, LowHealth, CanReach});
```

---

## Troubleshooting

### Plugin fails to load

- **Check**: `Window → Developer Tools → Output Log` for error messages
- **Verify**: Plugin enabled in Edit → Plugins
- **Ensure**: Engine version 5.0 or later
- **Re-extract**: Fresh plugin files

### Compilation errors

- **Clean**: Delete `Intermediate/` folder
- **Rebuild**: File → Refresh Visual Studio Project
- **Regenerate**: Delete `.sln` and regenerate

### Evaluation returns unexpected values

- **Verify**: Inputs array length matches circuit inputs
- **Check**: Test cases are correctly formatted
- **Review**: Evolution converged (check generation count)

---

## Performance Notes

- **Circuit evaluation**: 200-700 ns per call
- **Evolution speed**: 4.6M-1.4M evaluations/second
- **Memory**: ~50 KB per population member
- **CPU**: Single-threaded (default), parallelizable

---

## License

MIT License - See LICENSE.md in plugin directory

---

## Support

- **Documentation**: See `/Documentation/` folder
- **Issues**: GitHub Issues
- **Email**: support@sovereignlattice.io


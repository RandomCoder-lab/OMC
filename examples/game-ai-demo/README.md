# OMNIcode Game AI Demo

**Real-time evolved neural circuits controlling game AI**

## Overview

This Unity demo shows how OMNIcode circuits can control non-player characters (NPCs) in games. Watch evolved logic make intelligent decisions in real-time.

**Key Features**:
- ✅ Real-time AI training scene
- ✅ Playable game against evolved opponent
- ✅ Live performance metrics
- ✅ Circuit visualization
- ✅ Easy circuit loading from JSON/binaries

## Scenes

### Training Scene
Evolve circuits to make better game decisions.

**UI Elements**:
- "Run Evolution" button - Start/stop evolution
- Fitness display - Current best performance
- Generation counter - How many generations evolved
- Progress bar - Evolution progress (0-100%)

**What Happens**:
1. Population of 32 AI agents spawn randomly
2. Each agent has an evolved circuit controlling its decisions
3. Fitness evaluated based on game performance
4. Best circuits selected for next generation
5. Live metrics update every frame

**Controls**:
- Button: Run/Stop evolution
- Display: Fitness score, generation count

### Play Scene
Play against the evolved AI with human controls.

**UI Elements**:
- Score display - Player vs AI wins
- Level indicator - Current difficulty
- Back button - Return to training

**Controls** (customizable):
- WASD - Move player
- Space - Attack
- Mouse - Look around

**Gameplay**:
- Simple deathmatch arena
- Evolved AI learns to attack, dodge, and defend
- Win condition: First to 10 kills

## C# Scripts

### OmnimcodeCircuit.cs
Wrapper for circuit evaluation.

```csharp
public bool Evaluate(bool[] inputs)
{
    // Returns circuit output for given 3 boolean inputs
}
```

**Inputs**:
- `inputs[0]` - Can see target?
- `inputs[1]` - Obstacle ahead?
- `inputs[2]` - Health low?

**Output**:
- `true` - Attack mode
- `false` - Defensive mode

### EvolvedAIAgent.cs
Represents one AI-controlled character.

```csharp
public class EvolvedAIAgent : MonoBehaviour
{
    public void SetTarget(Transform newTarget);
    public void TakeDamage(float damage);
    public float GetHealth();
    public bool IsAttacking();
    public bool CanSeeTarget();
}
```

### TrainingSceneManager.cs
Manages the training loop.

- Spawns population of agents
- Evaluates fitness each generation
- Updates UI with progress
- Allows manual evolution control

### PlaySceneManager.cs
Manages the playable game scene.

- Spawns player and AI opponent
- Tracks score
- Manages difficulty/levels
- Handles scene transitions

## Project Setup

### Requirements
- Unity 2020.3 LTS or newer
- OMNIcode C# bindings (included in package)

### Installation

1. **Copy to Unity Project**:
```bash
cp -r examples/game-ai-demo Assets/OMNIcode-GameAI
```

2. **Open in Unity**:
```bash
unity -projectPath . -sceneList Assets/OMNIcode-GameAI/Assets/Scenes/TrainingScene.unity
```

3. **Run**:
   - Click "Play" in Unity Editor
   - Button to start evolution

### Scene Setup

**Training Scene** (`TrainingScene.unity`):
```
TrainingScene/
├── Canvas
│   ├── FitnessText
│   ├── GenerationText
│   ├── ProgressSlider
│   └── EvolveButton
├── GameManager (TrainingSceneManager.cs)
├── Camera
└── Agents (spawned at runtime)
```

**Play Scene** (`PlayScene.unity`):
```
PlayScene/
├── Canvas
│   ├── ScoreText
│   ├── LevelText
│   └── BackButton
├── GameManager (PlaySceneManager.cs)
├── Camera
├── Player (PlayerController.cs)
└── AIAgent (EvolvedAIAgent.cs)
```

## Integration with OMNIcode

### Loading Evolved Circuits

**From JSON** (exported by Modding Tool):
```csharp
OmnimcodeCircuit circuit = gameObject.AddComponent<OmnimcodeCircuit>();
circuit.LoadFromFile("path/to/circuit.json");
```

**From Binary** (exported by Circuit Trainer):
```csharp
// Load evolved circuit binary
circuit.LoadFromBinary("path/to/circuit.bin");
```

### Exporting Results

After training, export evolved AI:
```csharp
circuit.ExportToJSON("evolved_ai_circuit.json");
```

Use in other projects or games!

## Usage Workflow

### Scenario 1: Evolve New AI

1. Open Training Scene
2. Click "Run Evolution"
3. Watch fitness increase (should reach 0.8+ in 100 generations)
4. Click "Stop Evolution"
5. Export best circuit (right-click → Export)
6. Use in Play Scene

### Scenario 2: Play Against AI

1. Open Play Scene
2. AI opponent already loaded with trained circuit
3. Press Space to attack
4. Move with WASD
5. Try to beat the evolved opponent!

### Scenario 3: Compare Strategies

1. Train multiple evolved circuits (different populations)
2. Load different circuits into Play Scene
3. Measure win rates
4. Identify best strategy

## Customization

### Change Fitness Function
Edit `TrainingSceneManager.EvaluateAgentFitness()`:

```csharp
private float EvaluateAgentFitness(EvolvedAIAgent agent)
{
    // Your custom fitness logic
    // Return 0.0 to 1.0
}
```

### Add More Inputs

Modify `OmnimcodeCircuit`:
```csharp
bool[] inputs = new bool[] { 
    canSeeTarget, 
    obstacleAhead, 
    healthLow,
    enemyNearby,        // Add more...
    hasAmmo,
    isReloading
};
```

### Adjust Difficulty

Edit `TrainingSceneManager`:
```csharp
[SerializeField] private int populationSize = 32;     // Larger = better AI
[SerializeField] private int generationsPerUpdate = 10; // More = faster evolution
```

## Performance Tips

### Optimization Checklist
- [ ] Batch evaluate circuits (don't evaluate every frame)
- [ ] Use object pooling for agents
- [ ] Disable AI when off-screen
- [ ] Cache raycast results
- [ ] Profile with Unity Profiler

### Typical Performance
- Training: 1000-5000 agents/sec evaluation
- Play: 60 FPS with 4-8 AI opponents
- Binary size: ~500 KB (OMNIcode library)

## Examples

### Example 1: Simple Attack Logic
```
Inputs:  CanSeeTarget, HealthLow
Output:  ShouldAttack
Logic:   Attack if can see target AND health > 50%
```

### Example 2: Strategic Defense
```
Inputs:  CanSeeTarget, ObstacleAhead, HealthLow
Output:  ShouldAttack
Logic:   Attack if target visible, no obstacles, health good
         Otherwise retreat and hide
```

### Example 3: Resource Management
```
Inputs:  CanSeeTarget, HasAmmo, HealthLow
Output:  ShouldAttack
Logic:   Only attack if armed and healthy
         Flee if low on ammo or health
```

## Troubleshooting

### "Fitness stuck at 0.5"
- Population too small (try 64+)
- Fitness function not rewarding good behavior
- Evolution rate too high (reduce mutation rate)

### "AI not responding"
- Check circuit inputs are correct (order matters!)
- Verify circuit loaded successfully
- Debug output with Debug.Log()

### "Performance too slow"
- Reduce population size
- Evaluate less frequently
- Disable AI rendering when off-screen

### "Can't load circuit file"
- Check file path is correct
- Verify JSON format matches spec
- Use absolute paths during development

## Advanced Topics

### Multi-Objective Optimization

Evolve multiple traits simultaneously:
- Aggressiveness vs Survivability
- Speed vs Accuracy
- Solo vs Team play

### Transfer Learning

Train circuits in one game, use in another:
1. Evolve in simple test environment
2. Export best circuits
3. Load into complex game
4. Fine-tune with additional evolution

### Circuit Visualization

See what the evolved circuit is "thinking":
```csharp
// Display decision tree
circuit.DrawDebugInfo();
```

## Next Steps

1. **Train Your Own AI** - Run training scene and evolve
2. **Play the Game** - Challenge your evolved opponent
3. **Customize Logic** - Modify fitness function for your game
4. **Export Results** - Save evolved circuits for reuse
5. **Integrate** - Use OMNIcode in your own projects

## Resources

- **Circuit Trainer**: `examples/circuit-trainer/` - Learn how evolution works
- **Modding Tool**: `examples/modding-tool/` - Create custom circuits
- **Documentation**: See parent README for API reference
- **Tutorials**: Check `docs/tutorials/` for guides

## Performance Benchmarks

Typical metrics on modest hardware:

| Metric | Value |
|--------|-------|
| Agents/sec evaluated | 2,000-5,000 |
| Circuit eval time | 215-700 ns |
| Training generations/sec | 10-50 |
| Play scene FPS (8 agents) | 55-60 |
| Binary size | 500 KB |

## License

MIT - See parent LICENSE

## Support

Questions or issues? Check the tutorials or open an issue on GitHub.

---

**Ready to evolve your game AI!** 🚀


# Circuit Trainer - Interactive Evolution Demo

**Learn how genetic algorithms evolve solutions**

## Overview

Circuit Trainer is an interactive command-line tool that demonstrates how OMNIcode evolves circuits to solve logical problems. Watch in real-time as generations progress from random circuits to solutions.

**Perfect for**:
- Students learning about genetic algorithms
- Educators teaching evolution concepts
- Developers understanding OMNIcode capabilities

## Installation

### From Source
```bash
cd examples/circuit-trainer
cargo build --release
```

Binary location: `target/release/circuit-trainer`

### Prebuilt
Download from GitHub Releases (when available)

## Quick Start

```bash
./circuit-trainer

# Choose a problem:
# 1. Custom problem (enter your own truth table)
# 2. XOR (classic boolean problem)
# 3. AND-OR combination
# 4. 3-bit Majority
# 5. Exit

# Example: Solve XOR
Choose (1-5): 2
```

### Custom Problem Example

```
Choose (1-5): 1
Enter number of inputs (2-6): 2

Enter truth table (2 inputs, binary + space + output):
Example: 0010 1 (means: input 0010 should output 1)
Enter empty line when done:

> 00 0
> 01 1
> 10 1
> 11 0

Starting evolution...
```

## Predefined Problems

### 1. XOR Gate (2 inputs)
```
Input → Output
00    → 0
01    → 1
10    → 1
11    → 0
```
**Difficulty**: Medium | **Solution**: Usually <100 generations

### 2. AND-OR Combination (3 inputs)
```
(A AND B) OR C
```
**Difficulty**: Hard | **Solution**: Usually 200-400 generations

### 3. 3-bit Majority (3 inputs)
```
Output 1 if majority of inputs are 1
```
**Difficulty**: Hard | **Solution**: Usually 300-500 generations

## Understanding the Output

```
Gen | Fitness | Gates | Time    | Status
────┼─────────┼───────┼─────────┼──────────────────────
 1  | 0.25    |  3    | 12ms    | 🔄 Searching...
 10 | 0.50    |  4    | 120ms   | 🔄 Searching...
 50 | 0.75    |  5    | 600ms   | ⚡ Good progress
100 | 0.95    |  3    | 1.2s    | 🎯 Converging...
```

**Columns**:
- **Gen**: Generation number
- **Fitness**: Percentage of test cases the best circuit passes (0.00-1.00)
- **Gates**: Number of logic gates in the best circuit
- **Time**: Elapsed time for this generation
- **Status**: Progress indicator

## Performance Insights

The tool displays:
- **Evaluations/second**: How many circuits are tested per second
- **Time per evaluation**: Typically 215-700 nanoseconds
- **Comparison to Python**: OMNIcode is typically 50-230× faster

```
Speed: 4,600,000 evals/sec
Evaluation time: 217 ns/circuit
vs Python: OMNIcode is ~115× faster
```

## What You'll Learn

### 1. How Genetic Algorithms Work
- Population of random circuits
- Fitness evaluation
- Selection of best performers
- Mutation and recombination

### 2. Convergence Patterns
- Early progress is fast
- Later generations hit diminishing returns
- Some problems have multiple solutions
- Population size affects speed

### 3. Circuit Complexity
- Simple problems: 2-4 gates
- Medium problems: 4-6 gates
- Complex problems: 6-10+ gates

## Tips & Tricks

### Faster Convergence
- Smaller problems (fewer inputs/outputs)
- Clear patterns in the truth table
- Population size (default 128 is good)

### Harder Problems
- More inputs (4-6)
- Complex logical functions
- Rare output patterns (e.g., mostly 1s with one 0)

### Understanding Results
- If fitness reaches 1.0: Perfect solution found ✅
- If fitness plateaus: Evolution hit local optimum
- Gates increasing: Overfitting to training data

## Common Problems to Try

### Easy (2-3 gates usually)
```
NOT gate:
00 -> 1
10 -> 0
```

### Medium (4-5 gates)
```
XOR gate:
00 -> 0
01 -> 1
10 -> 1
11 -> 0
```

### Hard (6-8 gates)
```
Majority-3:
000 -> 0
001 -> 0
010 -> 0
011 -> 1
100 -> 0
101 -> 1
110 -> 1
111 -> 1
```

## Troubleshooting

### "Fitness plateaus at 0.5"
- The problem might be harder than expected
- Try running longer (up to 500 generations)
- Check your truth table for errors

### "Gets stuck finding simple gates"
- Population might be too small
- Try rerunning (random starting point matters)
- Complex problems need more generations

### Performance seems slow
- Running in debug mode? Use release: `cargo build --release`
- System under heavy load? Close other apps
- Expected: ~100 generations should take <2 seconds

## Example Output

```
╔════════════════════════════════════════════════════════════╗
║          OMNIcode - Circuit Evolution Trainer             ║
║     Learn how genetic algorithms discover solutions       ║
╚════════════════════════════════════════════════════════════╝

Problem: XOR Gate
════════════════════════════════════════════════════════════

Test cases: 4
Inputs per test: 2

Starting evolution...
Population: 128 circuits
Max generations: 500

Gen | Fitness | Gates | Time    | Status
────┼─────────┼───────┼─────────┼──────────────────────
  1 | 0.25    |  3    | 12ms    | 🔄 Searching...
 10 | 0.50    |  5    | 145ms   | 🔄 Searching...
 50 | 0.75    |  4    | 695ms   | ⚡ Good progress
100 | 0.95    |  3    | 1.4s    | 🎯 Converging...

Final Statistics:
  Generations:        127
  Time elapsed:       1.45s
  Best fitness:       100.00% (matches 4 of 4 test cases)
  Circuit gates:      3
  Population size:    128
  Evaluations:        ~16,256
  Speed:              11,200,000 evals/sec
  Evaluation time:    89 ns/circuit
  vs Python:          OMNIcode is ~180× faster

Solution found? ✅ YES
```

## Next Steps

After trying Circuit Trainer:

1. **Read the tutorials**: `docs/tutorials/`
2. **Try custom problems**: Design your own logic functions
3. **Explore the code**: `src/` shows how evolution works
4. **Build a game**: See `examples/game-ai-demo/`
5. **Create your own**: Use OMNIcode library in your projects

## License

MIT - See LICENSE in parent directory

## Questions?

- Documentation: See `docs/` folder
- Issues: GitHub Issues
- Email: support@sovereignlattice.io


# OMNIcode Modding Tool

**Multi-format circuit evolution and export**

## Overview

The Modding Tool lets designers and modders create evolved circuits without writing code. Input a logic problem, let it evolve a solution, and export in multiple formats (JSON, Rust, C).

## Quick Start

### Interactive Mode
```bash
./modding-tool
```

### File Mode
```bash
./modding-tool problems/xor.json
```

## Input Format (JSON)

```json
{
  "name": "XOR Gate",
  "inputs": 2,
  "cases": [
    {"input": "00", "output": 0},
    {"input": "01", "output": 1},
    {"input": "10", "output": 1},
    {"input": "11", "output": 0}
  ]
}
```

## Supported Export Formats

### JSON Export
Portable circuit description suitable for any language.

```json
{
  "name": "XOR_Gate",
  "inputs": 2,
  "fitness": 0.98,
  "gates": 5,
  "test_cases": [...]
}
```

### Rust Export
Drop-in Rust code with tests.

```rust
pub fn create_xor_circuit() -> Circuit {
    let mut circuit = Circuit::new(2);
    // ... evolved gates ...
    circuit
}
```

### C Export
Callable C function suitable for game engines.

```c
bool eval_xor(const bool inputs[2]) {
    bool gate_xor_0 = inputs[0] ^ inputs[1];
    // ...
    return gate_or_2;
}
```

## Workflow

1. **Define Problem**: Enter truth table (interactive or JSON file)
2. **Evolution**: Auto-optimizes circuit (128 population, 500 generations max)
3. **Export**: Choose format(s) and save

## Tips

- **JSON input**: Great for batch processing, version control
- **Interactive**: Fast experimentation and learning
- **Exports**: Reuse circuits across projects in any language

## Performance

Typical evolution times:
- 2-3 input: 500ms
- 4 input: 1.5s
- 5+ input: 3-5s

## Examples

Pre-made problems in `examples/` folder:
- `xor.json` - 2-bit XOR
- `and_or.json` - 3-bit AND-OR
- `majority.json` - 3-bit Majority vote

## Troubleshooting

### "Export failed"
- Check file permissions in current directory
- Ensure filename has no invalid characters

### "Fitness stuck at 50%"
- Problem might be NP-hard
- Try more generations or different initial population

## Next Steps

- Use exported circuits in your game
- Integrate with C FFI for real-time evolution
- Combine with Unreal or Unity plugins


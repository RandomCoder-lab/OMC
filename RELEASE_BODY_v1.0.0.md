## OMNIcode v1.0.0

### Added
- Initial release of OMNIcode circuit evolution engine
- C FFI layer (`omnimcode-ffi` crate)
- Python bindings (`omnimcode-python` with PyO3)
- Unity package with C# wrappers and examples
- Unreal Engine plugin with C++ wrappers
- Circuit Trainer CLI demo (368 KB standalone binary)
- Modding Tool demo (387 KB standalone binary)
- Game AI demo for Unity
- 5 comprehensive tutorials (22.5K words total)
- GitHub Actions CI/CD workflows

### Performance
- 509 KB binary size (zero external dependencies)
- 215-693 ns per circuit evaluation
- 4.64M-1.44M evals/sec throughput
- 51/51 tests passing

### Build System
- Rust workspace with 3 crates: omnimcode-core, omnimcode-ffi, omnimcode-python
- LTO and opt-level=3 for minimal size
- Cross-compilation support (Linux, Windows, macOS)

## Installation

### Cargo
```bash
cargo install omnimcode-core
```

### Unity
Import `OMNIcode-Unity.unitypackage` into your Unity project.

### Unreal
Copy the `OMNIcode-Unreal` plugin to your project's `Plugins/` directory.

### Python
```bash
pip install omnimcode
```

## Performance
- Binary size: 509 KB (zero dependencies)
- Circuit evaluation: 215-693 ns
- Throughput: 4.64M-1.44M evals/sec

## Links
- [Documentation](https://github.com/RandomCoder-lab/OMC/wiki)
- [Crate](https://crates.io/crates/omnimcode-core)
- [Unity Asset Store](https://assetstore.unity.com/)

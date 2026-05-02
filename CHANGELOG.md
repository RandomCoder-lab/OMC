# Changelog

All notable changes to OMNIcode will be documented in this file.

## [Unreleased]

## [1.0.0] - 2026-05-02
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

## [0.9.0-beta] - 2026-04-15
### Added
- Beta release with core circuit evolution
- XOR problem solving via genetic algorithms
- Basic C FFI exports

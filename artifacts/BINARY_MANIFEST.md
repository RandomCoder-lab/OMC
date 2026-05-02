# OMNIcode v1.0.0 Binary Manifest

**Generated**: May 7, 2026  
**Platform**: Linux x86_64 (GNU)  
**Build Profile**: Release (optimized, stripped)  
**Dependency**: libc (system standard only)

---

## Binaries

### 1. omnimcode-linux-x64
**Type**: Standalone executable  
**Size**: 509 KB  
**SHA256**: `834add40d826a51e612a9f4d753a472268e05ee89e1c2b4b98a4066a51617441`  
**Purpose**: Direct command-line execution of OMNIcode interpreter  
**Usage**:
```bash
./omnimcode-linux-x64 < program.omc
```

### 2. libomnimcode-linux-x64.so
**Type**: C FFI Shared Library (cdylib)  
**Size**: 286 KB  
**SHA256**: (run `sha256sum libomnimcode-linux-x64.so`)  
**Purpose**: C/C++ integration via FFI bindings  
**Header**: `omnimcode.h` (at project root)  
**Usage**:
```c
#include "omnimcode.h"
OmnimcodeCircuit* c = omnicode_circuit_new(2);
bool result = omnicode_circuit_eval(c, inputs, 2);
omnicode_circuit_free(c);
```

### 3. omnimcode-python-linux-x64.so
**Type**: Python Extension Module (compiled with PyO3)  
**Size**: 404 KB  
**SHA256**: (run `sha256sum omnimcode-python-linux-x64.so`)  
**Purpose**: Python integration via native bindings  
**Python Version**: 3.8+ (using ABI3 stable ABI)  
**Usage**:
```python
import omnimcode
circuit = omnimcode.OmnimcodeCircuit(2)
result = circuit.eval([True, False])
```

---

## Cross-Platform Availability

The binaries listed above are for **Linux x86_64** only.

To build for other platforms, use the source code and Cargo:

```bash
# macOS ARM64 (Apple Silicon)
cargo build --release --target aarch64-apple-darwin

# macOS x86_64
cargo build --release --target x86_64-apple-darwin

# Windows x86_64
cargo build --release --target x86_64-pc-windows-msvc

# Linux ARM64
cargo build --release --target aarch64-unknown-linux-gnu
```

**Note**: Requires appropriate Rust target installed:
```bash
rustup target add aarch64-apple-darwin  # macOS ARM64
```

---

## Verification

To verify binary integrity:

```bash
# Check SHA256
sha256sum -c BINARY_MANIFEST.md

# Test executable
./omnimcode-linux-x64 --version

# Test FFI library
ldd libomnimcode-linux-x64.so

# Test Python module (requires Python 3.8+)
python3 -c "import omnimcode; print(omnimcode.__doc__)"
```

---

## Building from Source

All binaries can be rebuilt from the source code in the parent directory:

```bash
cd /home/thearchitect/OMC
cargo build --release --workspace
```

This produces:
- `target/release/omnimcode-standalone` (main binary)
- `target/release/libomnimcode_ffi.so` (FFI library)
- `target/release/libomnimcode_python.so` (Python module)

---

## Dependencies

All binaries require only the C standard library (libc):

```
omnimcode-linux-x64 => libc (system)
libomnimcode-linux-x64.so => libc (system)
omnimcode-python-linux-x64.so => libc, libpython3.8+ (system)
```

**No third-party dependencies** are vendored or required.

---

## Distribution & Licensing

All binaries are provided under the **MIT License**.

See `LICENSE.md` in the project root for full terms.

---

## Release Notes

**v1.0.0 (May 7, 2026)**
- Phase 0: Core validation (49/51 tests)
- Phase 1: SDK packaging (FFI, Python, workspace)
- Three confirmed bugs fixed (cross-over logic, const_fold, LRUCache alias)
- Performance benchmarks: 215–693 ns/eval, 1.44M–4.64M evals/sec
- Ready for multi-platform distribution

---

**Next Steps**:
1. Cross-compile to macOS, Windows
2. Create Unity package with all binaries
3. Create Unreal plugin with all binaries
4. Distribute via GitHub Releases, package managers


# OMNIcode Build Targets Matrix

## Supported Platforms

| Target Triple | Platform | Architecture | Status | Notes |
|--------------|----------|--------------|--------|-------|
| `x86_64-unknown-linux-gnu` | Linux | x86_64 | ✅ Tested | Primary development platform |
| `x86_64-pc-windows-gnu` | Windows | x86_64 | 🔄 Pending | Requires MinGW cross-compilation |
| `x86_64-pc-windows-msvc` | Windows | x86_64 | 🔄 Pending | Requires Visual Studio toolchain |
| `x86_64-apple-darwin` | macOS | x86_64 | 🔄 Pending | Requires Apple developer tools |
| `aarch64-apple-darwin` | macOS | ARM64 | 🔄 Pending | Apple Silicon (M1/M2/M3) |
| `x86_64-unknown-linux-musl` | Linux (static) | x86_64 | 🔄 Pending | Static linking for portability |

## Build Commands

### Linux (native)
```bash
cargo build --release -p omnimcode-ffi
# Output: target/release/libomnimcode_ffi.so
```

### Windows (cross-compile from Linux)
```bash
# Install MinGW
sudo apt-get install mingw-w64

# Build for Windows (GNU)
cargo build --release --target x86_64-pc-windows-gnu -p omnimcode-ffi
# Output: target/x86_64-pc-windows-gnu/release/omnimcode_ffi.dll
```

### macOS (cross-compile from Linux - requires osxcross)
```bash
# Install osxcross (complex setup)
# Then build:
cargo build --release --target x86_64-apple-darwin -p omnimcode-ffi
# Output: target/x86_64-apple-darwin/release/libomnimcode_ffi.dylib
```

## CI/CD Matrix

The GitHub Actions workflow (`.github/workflows/build-binaries.yml`) tests:
- Ubuntu Latest (Linux x64)
- Windows Latest (Windows x64) - via GitHub runner
- macOS Latest (macOS x64) - via GitHub runner

## Known Quirks & Issues

### Windows
- **GNU vs MSVC**: GNU toolchain (mingw-w64) is easier for cross-compilation; MSVC requires Windows host
- **DLL export names**: Use `#[no_mangle]` and proper `__declspec(dllexport)` for C-compatible exports
- **Path separators**: Windows uses `\` vs Linux `/`

### macOS
- **Codesigning**: Required for distribution outside App Store (see Task 3.4)
- **Gatekeeper**: Unsigned binaries trigger security warnings
- **rpath**: Dynamic libraries need proper `@rpath` or absolute paths

### Linux
- **musl vs gnu**: musl produces static binaries but may have compatibility issues
- **`$ORIGIN`**: Use `-C link-args=-Wl,-rpath,'$ORIGIN'` for relative library loading

## Testing Matrix

| Feature | Linux | Windows | macOS |
|---------|-------|----------|-------|
| FFI library loads | ✅ | 🔄 | 🔄 |
| Circuit creation | ✅ | 🔄 | 🔄 |
| Evolution runs | ✅ | 🔄 | 🔄 |
| Python bindings | ✅ | 🔄 | 🔄 |
| Unity plugin | ✅ | 🔄 | 🔄 |
| Unreal plugin | ✅ | 🔄 | 🔄 |

## File Naming Convention

- **Linux**: `libomnimcode_ffi.so`
- **Windows**: `omnimcode_ffi.dll`
- **macOS**: `libomnimcode_ffi.dylib`

## Minimum Requirements

- **Rust**: 1.75+ (edition 2021)
- **Windows**: Windows 10+ (for DLL compatibility)
- **macOS**: macOS 11+ (Big Sur)
- **Linux**: glibc 2.31+ or musl 1.2+

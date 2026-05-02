# Unity Plugin Binary Installation

This directory will contain the OMNIcode native binaries for each platform.

## Structure

```
Binaries/
├── Windows/
│   └── x64/
│       └── omnicode.dll
├── macOS/
│   ├── x64/
│   │   └── libomnimcode.dylib
│   └── arm64/
│       └── libomnimcode.dylib
└── Linux/
    └── x64/
        └── libomnimcode.so
```

## Installation

Run the build script from the project root:

```bash
./scripts/build-unity-binaries.sh
```

This will:
1. Build native libraries for all platforms
2. Copy to appropriate directories
3. Generate metadata files

## Manual Installation

### Linux x64
```bash
cargo build --release -p omnimcode-ffi
cp target/release/libomnimcode_ffi.so Binaries/Linux/x64/libomnimcode.so
```

### macOS x64
```bash
rustup target add x86_64-apple-darwin
cargo build --release --target x86_64-apple-darwin -p omnimcode-ffi
cp target/x86_64-apple-darwin/release/libomnimcode_ffi.dylib Binaries/macOS/x64/libomnimcode.dylib
```

### macOS ARM64 (Apple Silicon)
```bash
rustup target add aarch64-apple-darwin
cargo build --release --target aarch64-apple-darwin -p omnimcode-ffi
cp target/aarch64-apple-darwin/release/libomnimcode_ffi.dylib Binaries/macOS/arm64/libomnimcode.dylib
```

### Windows x64
```bash
rustup target add x86_64-pc-windows-msvc
cargo build --release --target x86_64-pc-windows-msvc -p omnimcode-ffi
cp target/x86_64-pc-windows-msvc/release/omnicode_ffi.dll Binaries/Windows/x64/omnicode.dll
```

## Platform Detection

Unity will automatically detect which binary to use based on build target via the NativeBindings.cs platform #if directives.


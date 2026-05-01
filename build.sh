#!/bin/bash
# build.sh - Build and verify OMNIcode standalone executable

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         OMNIcode Standalone Executable Builder                ║"
echo "║                   Rust Native Compiler                         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo

# Check for Rust
if ! command -v cargo &> /dev/null; then
    echo "❌ Error: Rust not found. Install from https://rustup.rs/"
    exit 1
fi

echo "✅ Rust toolchain found:"
rustc --version
echo

# Build
echo "📦 Building OMNIcode standalone (release mode)..."
echo "   This may take 4-5 seconds on first build..."
echo

cd "$(dirname "$0")" || exit 1
cargo build --release

# Copy binary
echo
echo "✅ Build complete!"
echo

cp target/release/standalone standalone.omc
echo "📋 Binary details:"
ls -lh standalone.omc
file standalone.omc
echo

# Run tests
echo "🧪 Running test suite..."
echo

test_count=0
pass_count=0

for test_file in examples/*.omc; do
    test_count=$((test_count + 1))
    echo "  Test $test_count: $(basename "$test_file")..."
    if ./standalone.omc "$test_file" > /dev/null 2>&1; then
        echo "    ✅ PASS"
        pass_count=$((pass_count + 1))
    else
        echo "    ❌ FAIL"
    fi
done

echo
echo "╔════════════════════════════════════════════════════════════════╗"
if [ $pass_count -eq $test_count ]; then
    echo "║  ✅ ALL TESTS PASSED ($pass_count/$test_count)                     ║"
else
    echo "║  ⚠️  SOME TESTS FAILED ($pass_count/$test_count)                     ║"
fi
echo "║                                                                ║"
echo "║  Ready to use: ./standalone.omc <program.omc>                ║"
echo "║  Or start REPL: ./standalone.omc                             ║"
echo "╚════════════════════════════════════════════════════════════════╝"

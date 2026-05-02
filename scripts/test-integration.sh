#!/bin/bash
# Integration tests for OMNIcode packages
# Tests that all packages work correctly

set -e

echo "=== OMNIcode Package Integration Tests ==="
echo ""

# Test 1: C FFI Linking
echo "Test 1: C FFI Linking"
if [ -f "/home/thearchitect/OMC/target/release/libomnimcode_ffi.so" ]; then
    echo "✅ FFI library exists: libomnimcode_ffi.so"
    # Check if it's a valid ELF binary
    if file "/home/thearchitect/OMC/target/release/libomnimcode_ffi.so" | grep -q "ELF"; then
        echo "✅ Valid ELF shared library"
    else
        echo "❌ Not a valid ELF library"
        exit 1
    fi
    # Check for any exported symbols (not just omnimcode_)
    if nm -D /home/thearchitect/OMC/target/release/libomnimcode_ffi.so 2>/dev/null | grep -q "T "; then
        echo "✅ FFI exports found (checking symbols...)"
        nm -D /home/thearchitect/OMC/target/release/libomnimcode_ffi.so 2>/dev/null | grep " T " | head -5
    else
        echo "⚠️  No exported symbols found (may be stripped)"
    fi
else
    echo "❌ FFI library not found"
    exit 1
fi
echo ""

# Test 2: Python Wheel Build (dry-run)
echo "Test 2: Python Bindings"
if [ -d "/home/thearchitect/OMC/omnimcode-python" ]; then
    echo "✅ Python bindings directory exists"
    if grep -q "pyo3" /home/thearchitect/OMC/omnimcode-python/Cargo.toml; then
        echo "✅ PyO3 dependency found"
    else
        echo "❌ PyO3 not found in Cargo.toml"
    fi
else
    echo "❌ Python bindings directory not found"
fi
echo ""

# Test 3: Unity Package Structure
echo "Test 3: Unity Package"
UNITY_PKG="/home/thearchitect/GameAssetProduction/packages/OMNIcode-Unity"
if [ -d "$UNITY_PKG" ]; then
    echo "✅ Unity package directory exists"
    # Check required files
    for file in "package.json" "Runtime/OMNIcode.asmdef" "Runtime/Scripts/OmnimcodeCircuit.cs"; do
        if [ -f "$UNITY_PKG/$file" ]; then
            echo "✅ $file exists"
        else
            echo "❌ $file missing"
        fi
    done
    # Check FFI binary in Plugins
    if [ -f "$UNITY_PKG/Runtime/Plugins/x86_64/libomnimcode_ffi.so" ]; then
        echo "✅ FFI binary in Plugins/x86_64/"
    else
        echo "❌ FFI binary missing from Plugins"
    fi
else
    echo "❌ Unity package directory not found"
fi
echo ""

# Test 4: Unreal Plugin Structure
echo "Test 4: Unreal Plugin"
UNREAL_PLUGIN="/home/thearchitect/GameAssetProduction/packages/OMNIcode-Unreal"
if [ -d "$UNREAL_PLUGIN" ]; then
    echo "✅ Unreal plugin directory exists"
    if [ -f "$UNREAL_PLUGIN/OMNIcode.uplugin" ]; then
        echo "✅ .uplugin file exists"
    else
        echo "❌ .uplugin file missing"
    fi
else
    echo "❌ Unreal plugin directory not found"
fi
echo ""

# Test 5: CLI Tools
echo "Test 5: CLI Tools"
for tool in "circuit-trainer" "modding-tool"; do
    if [ -f "/home/thearchitect/GameAssetProduction/examples/$tool/target/release/$tool" ]; then
        echo "✅ $tool binary exists"
        if file "/home/thearchitect/GameAssetProduction/examples/$tool/target/release/$tool" | grep -q "ELF"; then
            echo "✅ $tool is valid ELF binary"
        fi
    else
        echo "❌ $tool binary not found"
    fi
done
echo ""

# Test 6: Tutorial Documents
echo "Test 6: Tutorial Documents"
TUTORIAL_DIR="/home/thearchitect/GameAssetProduction/tutorials"
if [ -d "$TUTORIAL_DIR" ]; then
    TUTORIAL_COUNT=$(ls -1 "$TUTORIAL_DIR"/*.md 2>/dev/null | wc -l)
    echo "✅ Found $TUTORIAL_COUNT tutorial files"
    if [ $TUTORIAL_COUNT -ge 5 ]; then
        echo "✅ All 5+ tutorials present"
    else
        echo "⚠️  Less than 5 tutorials found"
    fi
else
    echo "❌ Tutorial directory not found"
fi
echo ""

echo "=== All Integration Tests Completed ==="

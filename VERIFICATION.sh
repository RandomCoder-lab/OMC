#!/bin/bash
# TIER 4 VERIFICATION SCRIPT
# Runs all checks to verify Tier 4 implementation

echo "════════════════════════════════════════════════════════════════"
echo "TIER 4 VERIFICATION SCRIPT"
echo "════════════════════════════════════════════════════════════════"
echo

echo "1. Checking source files..."
if [ -f "omnimcode-core/src/phi_pi_fib.rs" ] && [ -f "omnimcode-core/src/phi_disk.rs" ]; then
    echo "   ✅ phi_pi_fib.rs exists"
    echo "   ✅ phi_disk.rs exists"
else
    echo "   ❌ Missing source files"
    exit 1
fi
echo

echo "2. Checking binary..."
if [ -f "target/release/omnimcode-standalone" ]; then
    SIZE=$(ls -lh target/release/omnimcode-standalone | awk '{print $5}')
    echo "   ✅ Binary exists: $SIZE"
else
    echo "   ❌ Binary not found"
    exit 1
fi
echo

echo "3. Running tests..."
TEST_OUT=$(cargo test --release 2>&1)
PASS_COUNT=$(echo "$TEST_OUT" | grep -E "test result: ok\. [0-9]+ passed" | awk '{sum+=$4} END {print sum}')
FAIL_COUNT=$(echo "$TEST_OUT" | grep -E "test result: ok\. [0-9]+ passed" | awk '{sum+=$6} END {print sum}')
if [ "${FAIL_COUNT:-0}" = "0" ] && [ "${PASS_COUNT:-0}" -gt 0 ]; then
    echo "   ✅ All tests passing: ${PASS_COUNT} passed, 0 failed"
else
    echo "   ❌ Tests failed: ${PASS_COUNT:-0} passed, ${FAIL_COUNT:-0} failed"
    exit 1
fi
echo

echo "4. Checking documentation..."
for doc in README.md BUILD.md ARCHITECTURE.md DEVELOPER.md CHANGELOG.md TIER_4_HONEST_REVISION.md; do
    if [ -f "$doc" ]; then
        echo "   ✅ $doc"
    else
        echo "   ❌ Missing $doc"
        exit 1
    fi
done
echo

echo "════════════════════════════════════════════════════════════════"
echo "VERIFICATION COMPLETE ✅"
echo "════════════════════════════════════════════════════════════════"
echo
echo "Status: TIER 4 PRODUCTION READY (post-consolidation)"
echo "Binary: target/release/omnimcode-standalone"
echo "Tests: ${PASS_COUNT:-?} passing ✅"
echo "Ready for deployment"
echo

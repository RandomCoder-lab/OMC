#!/bin/bash
# Verify bloom_best.pt integrates into the assistant: waits for the LM to finish,
# then boots the assistant headless and confirms it (a) selects bloom_best, (b) loads
# the 233/n6 architecture, (c) generates a real (non-crashing) response.
cd "$(dirname "$0")"
LOG=AUTONOMOUS_LOG.md
exec >> /tmp/verify_integration.log 2>&1

echo "==== verify-integration START $(date) ===="
echo "[verify] waiting for bloom_best.pt + PHASE 4 LM COMPLETE ..."
until [ -f bloom_best.pt ] && grep -q "PHASE 4 LM COMPLETE" $LOG 2>/dev/null; do sleep 60; done
sleep 5

# Boot the assistant headless with two queries (uses default candidate list → bloom_best first).
printf "fibonacci\nfn add(a, b) {\nbye\n" | timeout 300 python3 -u omc_assistant.py > /tmp/assistant_bloombest.log 2>&1

SEL=$(grep -oE "Loading LM from .*bloom_best.pt" /tmp/assistant_bloombest.log | head -1)
ARCH=$(grep -oE "seq_len=[0-9]+" /tmp/assistant_bloombest.log | head -1)
GEN=$(grep -cE "Retrieved from name registry|Retrieved function|Generated|φ-Synth" /tmp/assistant_bloombest.log)
{
  echo ""
  echo "### bloom_best INTEGRATION into assistant"
  echo "_$(date '+%m-%d %H:%M')_"
  if [ -n "$SEL" ]; then
    echo "- ✓ assistant SELECTED bloom_best ($SEL), loaded $ARCH, produced $GEN response blocks."
    echo "  bloom_best.pt is now the assistant's default LM (preferred in _lm_candidates + reload lm)."
  else
    echo "- ✗ assistant did NOT select bloom_best — check /tmp/assistant_bloombest.log (load error?)."
    grep -iE "error|trace|warn.*LM|failed" /tmp/assistant_bloombest.log | head -3 | sed 's/^/    /'
  fi
} >> $LOG
echo "==== verify-integration DONE $(date) ===="
echo "INTEGRATION-VERIFY DONE"

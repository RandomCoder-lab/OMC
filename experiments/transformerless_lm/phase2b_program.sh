#!/bin/bash
# ============================================================================
# PHASE 2b — REVISIONS of the 5 "failures" (per user: they may be proxy/bug
# artifacts, like K-shrink was, not real failures). Faithful re-implementations.
#   R1 Self-Witness: real quality-selected bloom centroid (not ungated EMA proxy)
#   R2 Spells XIII-XVIII: attenuated/tanh-clamped (not raw additive) — tested
#      separately on the current model; result folded into the ledger.
# Chained after PHASE 2 COMPLETE. Robust: failures don't abort.
# ============================================================================
cd "$(dirname "$0")"
LOG=AUTONOMOUS_LOG.md
exec >> /tmp/phase2b_master.log 2>&1

echo "==== PHASE 2b START $(date) ===="
echo "[p2b] waiting for PHASE 2 COMPLETE ..."
until grep -q "PHASE 2 COMPLETE" $LOG 2>/dev/null; do sleep 60; done
sleep 10

{ echo ""; echo "### PHASE 2b — REVISIONS (faithful re-implementations of the '5 failures')"; echo "_$(date '+%m-%d %H:%M')_"; } >> $LOG

# R1 — Self-Witness with a real bloom centroid (revises B.1)
echo "[p2b] R1 Self-Witness (real bloom centroid) @ $(date)"
python3 -u train_r1_self_witness.py > /tmp/auto_r1.log 2>&1 || echo "  [R1 FAILED]"
R1C=$(grep '\[R1\] control' /tmp/auto_r1.log | grep -oE 'gen_validity=0\.[0-9]+' | head -1)
R1W=$(grep '\[R1\] witness' /tmp/auto_r1.log | grep -oE 'gen_validity=0\.[0-9]+' | head -1)
R1V=$(grep '\[R1\] VERDICT' /tmp/auto_r1.log | sed 's/\[R1\] VERDICT: //')
{
  echo "- R1 Self-Witness REVISED (real frozen bloom centroid vs no-SW, n=8):"
  echo "  control $R1C  |  witness $R1W"
  echo "  verdict: ${R1V:-see /tmp/auto_r1.log}"
  echo "  (B.1 used an ungated EMA-of-mean-hidden proxy; this uses the derivation's actual b.)"
} >> $LOG

echo "==== PHASE 2b COMPLETE $(date) ===="
{ echo ""; echo "### PHASE 2b COMPLETE"; echo "_$(date '+%m-%d %H:%M')_"; } >> $LOG

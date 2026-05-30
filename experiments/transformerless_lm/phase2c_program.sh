#!/bin/bash
# PHASE 2c — PARAMETERS AS ADDRESSES (user's sifted-out idea). Capability test:
# can weights fetched/shared by address reduce param count while keeping quality?
# Uses the resurrected addressed_param_store + addressed_linear (now with an
# n_buckets sharing knob). Chained after PHASE 2b. Robust: failure non-fatal.
cd "$(dirname "$0")"
LOG=AUTONOMOUS_LOG.md
exec >> /tmp/phase2c_master.log 2>&1

echo "==== PHASE 2c START $(date) ===="
echo "[p2c] waiting for PHASE 2b COMPLETE ..."
until grep -q "PHASE 2b COMPLETE" $LOG 2>/dev/null; do sleep 60; done
sleep 10

{ echo ""; echo "### PHASE 2c — Parameters-as-addresses capability test"; echo "_$(date '+%m-%d %H:%M')_"; } >> $LOG
echo "[p2c] running addressed-params sweep @ $(date)"
python3 -u train_addressed_params.py > /tmp/auto_addrparams.log 2>&1 || echo "  [addr-params FAILED]"
# fold the result rows into the ledger
grep -E "params=.*validity=" /tmp/auto_addrparams.log | sed 's/^\[addr\] /- /' >> $LOG
grep "VERDICT" /tmp/auto_addrparams.log | sed 's/^\[addr\] /- /' >> $LOG

echo "==== PHASE 2c COMPLETE $(date) ===="
{ echo ""; echo "### PHASE 2c COMPLETE"; echo "_$(date '+%m-%d %H:%M')_"; } >> $LOG

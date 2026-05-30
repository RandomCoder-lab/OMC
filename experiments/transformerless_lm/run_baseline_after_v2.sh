#!/bin/bash
# Conventional baseline, chained after the v2-capacity fast run. Produces the
# three-way verdict: FibRec-small vs FibRec-capacity vs vanilla-Transformer,
# all ~333K params / 200 steps / same corpus.
cd "$(dirname "$0")"
LOG=/tmp/baseline.log
exec > >(tee "$LOG") 2>&1

echo "[base] waiting for v2-capacity run to finish ..."
until grep -q "ALL DONE\|Error\|Traceback" /tmp/cap_v2.log 2>/dev/null; do sleep 30; done
echo "[base] v2 done — training vanilla transformer baseline @ $(date)"

python3 -u conventional_baseline.py

echo ""
echo "[base] ===================  THREE-WAY VERDICT (200 steps, ~333K params)  ==================="
echo "[base] FibRec results (from the fast A/B):"
grep -E "gen_validity" /tmp/cap_v2.log | sed 's/^/[base]   /'
echo "[base] Transformer result:"
grep -E "BASELINE transformer gen_validity" /tmp/baseline.log | sed 's/^/[base]   /'
echo "[base] (higher validity = more corpus-like OMC; this tells us if substrate is competitive as a generator)"
echo "[base] ALL DONE @ $(date)"

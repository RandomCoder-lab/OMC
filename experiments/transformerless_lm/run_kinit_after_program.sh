#!/bin/bash
# CORRECTED capacity test (EXP-2 revealed K-shrink is a no-op; capacity is fixed by
# K_init). Decisive question: is the ~0.49 plateau set by K_init=89? Test K_init=144
# (next Fibonacci up — the only untested higher capacity; risks the documented NaN).
#   NaN        → plateau is HARD-CAPPED at K=89 (can't scale the substrate up)
#   beats 0.53 → capacity IS the lever (scale K_init)
#   matches    → plateau is deeper than capacity
# Chained after the autonomous program completes.
cd "$(dirname "$0")"
LOG=AUTONOMOUS_LOG.md
exec >> /tmp/autonomous_master.log 2>&1

echo "[kinit] waiting for autonomous program to complete ..."
until grep -q "PROGRAM COMPLETE" $LOG 2>/dev/null || ! pgrep -f autonomous_program.sh >/dev/null 2>&1; do sleep 60; done
echo "[kinit] program done — running K_init=144 decisive capacity test @ $(date)"

python3 -u train_address_navigator.py --steps 500 --seq-len 256 \
  --K-init 144 --K-min 144 --d-model 64 --n-blocks 4 \
  --batch-size 24 --lr 3e-4 --corpus omc_corpus.txt --out kinit_144.pt --device cpu \
  > /tmp/auto_kinit_144.log 2>&1 || echo "  [kinit_144 train errored]"

CHAR=$(grep BEST /tmp/auto_kinit_144.log 2>/dev/null | tail -1 | grep -oE 'char=[0-9.]+|char=nan')
VAL=$(python3 -u measure_validity.py kinit_144.pt 2>/dev/null | grep -oE '0\.[0-9]+' | head -1)
NAN=$(grep -ic "nan" /tmp/auto_kinit_144.log)
{
  echo ""
  echo "### EXP-Kinit  K_init=144 decisive capacity test"
  echo "_$(date '+%m-%d %H:%M')_"
  echo "- K_init=144  ${CHAR:-char=?}  validity=${VAL:-NA}  (nan_lines=$NAN)"
  echo "- vs K_init=89 baseline ~0.49-0.53. Verdict: NaN=hard-capped; >0.53=capacity-lever; ~0.49=deeper-plateau."
} >> $LOG
echo "[kinit] ALL DONE @ $(date)"

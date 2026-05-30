#!/bin/bash
# PHASE 4 — DEVELOP THE NEW LM: stack the proven wins into one best substrate LM.
# Combines EXP-9 (chunk-scale 233 = paragraph context, best), EXP-3/7 (depth n6,
# ~free via recurrence), EXP-4 (long training breaks the plateau), EXP-1 (lr 3e-4).
# Prior isolated bests were ~0.55-0.58; this is the first time they're COMBINED.
# Chained after the n=5 confirm run (free CPU). Robust: non-fatal on error.
cd "$(dirname "$0")"
LOG=AUTONOMOUS_LOG.md
exec >> /tmp/phase4_master.log 2>&1

echo "==== PHASE 4 (best stacked LM) START $(date) ===="
echo "[p4] waiting for n=5 confirm to finish (free CPU) ..."
until grep -q "PHASE3-CONFIRM DONE" /tmp/phase3_confirm.log 2>/dev/null || ! pgrep -f phase3_confirm.py >/dev/null 2>&1; do sleep 60; done
sleep 10

{ echo ""; echo "### PHASE 4 — best stacked substrate LM (chunk233 + depth6 + long + lr3e-4)"; echo "_$(date '+%m-%d %H:%M')_"; } >> $LOG

# Train the stacked-best FibRec LM. seq_len 233 (paragraph chunk), n_blocks 6, K_init 89,
# static full K (K-shrink collapses — EXP-6), lr 3e-4, 1800 steps (EXP-4 plateau-breaker).
python3 -u train_address_navigator.py --steps 1800 --seq-len 233 \
  --K-init 89 --K-min 89 --d-model 64 --n-blocks 6 --no-k-shrink \
  --batch-size 24 --lr 3e-4 --corpus omc_corpus.txt --out bloom_best.pt --device cpu \
  > /tmp/auto_best_lm.log 2>&1 || echo "  [best-LM train FAILED]"

CHAR=$(grep BEST /tmp/auto_best_lm.log 2>/dev/null | tail -1 | grep -oE 'char=[0-9.]+')
VAL=$(python3 -u measure_validity.py bloom_best.pt 2>/dev/null)
{
  echo "- best stacked LM (seq233/n6/staticK89/lr3e-4/1800): ${CHAR:-char=?}"
  echo "  $VAL"
  echo "  (vs prior isolated bests: EXP-4 1800@seq256=0.583 unguarded; EXP-7 n6@seq256 guarded 0.473;"
  echo "   chunk233 alone guarded 0.572. This is the first COMBINED config.)"
} >> $LOG

echo "==== PHASE 4 (best stacked LM) COMPLETE $(date) ===="
{ echo ""; echo "### PHASE 4 LM COMPLETE"; echo "_$(date '+%m-%d %H:%M')_"; } >> $LOG

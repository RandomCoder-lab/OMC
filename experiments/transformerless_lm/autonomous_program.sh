#!/bin/bash
# ============================================================================
# 12-HOUR AUTONOMOUS RESEARCH PROGRAM  (budgeted ~9-10h @ ~7s/step, margin to 12h)
# Chained pre-registered A/B experiments on the φ-substrate LM. Robust: each
# stage logs to AUTONOMOUS_LOG.md; failures don't abort the rest.
#
# Theme: the substrate ARCHITECTURE beats a transformer on sample-efficiency
# (0.495 vs 0.291). Push *training-time* levers toward corpus-fidelity and map
# how the win scales. Metric = corpus 4-gram gen-validity, ~330K params, scratch.
# ============================================================================
cd "$(dirname "$0")"
LOG=AUTONOMOUS_LOG.md
exec > >(tee -a /tmp/autonomous_master.log) 2>&1

note()   { { echo ""; echo "### $1"; echo "_$(date '+%m-%d %H:%M')_"; } >> $LOG; }
result() { echo "- $1" >> $LOG; }
val()    { python3 -u measure_validity.py "$1" 2>/dev/null | grep -oE '0\.[0-9]+' | head -1; }
cbest()  { grep BEST "/tmp/auto_${1}.log" 2>/dev/null | tail -1 | grep -oE 'char=[0-9.]+'; }
# train <tag> <steps> <Kinit> <Kmin> <d> <nblk> <lr>
train()  {
  python3 -u train_address_navigator.py --steps "$2" --seq-len 256 \
    --K-init "$3" --K-min "$4" --d-model "$5" --n-blocks "$6" \
    --batch-size 24 --lr "$7" --corpus omc_corpus.txt --out "${1}.pt" --device cpu \
    > "/tmp/auto_${1}.log" 2>&1 || echo "  [train $1 FAILED]"
}

echo "==== AUTONOMOUS PROGRAM START $(date) ===="

# Wait for the in-flight crossover pipeline (don't contend with it).
echo "[auto] waiting for in-flight crossover pipeline ..."
until grep -q "ALL DONE\|Error" /tmp/xover.log 2>/dev/null; do sleep 60; done
note "EXP-0 Crossover (in-flight pipeline)"
result "FibRec @200=$(grep 'fast_v1.pt: gen' /tmp/cap_v2.log|grep -oE '0\.[0-9]+'|head -1)  @1000=$(grep 'fresh1k.pt: gen' /tmp/data_exposure.log|grep -oE '0\.[0-9]+'|head -1)"
result "Transformer @200=0.291  @1000=$(grep 'BASELINE transformer gen' /tmp/xover.log|grep -oE '0\.[0-9]+'|head -1)"

# EXP-1  LR sweep — which lr for FibRec? (300 steps each; 3 pts)  ~1.7h
note "EXP-1 LR sweep (300 steps, d64/n4/K_min13)"
for lr in 1e-4 3e-4 1e-3; do
  train "lr_${lr}" 300 89 13 64 4 $lr
  result "lr=$lr  $(cbest lr_${lr})  validity=$(val lr_${lr}.pt)"
done

# EXP-2  K-min sweep — retained Fibonacci bands (the substrate lever). 500 steps.  ~3.9h
note "EXP-2 K-min sweep (500 steps, d64/n4, lr3e-4)"
for km in 8 13 21 34; do
  train "km_${km}" 500 89 $km 64 4 3e-4
  result "K_min=$km  $(cbest km_${km})  validity=$(val km_${km}.pt)"
done

# EXP-3  Depth scan — FibRec depth is ~free via recurrence. 500 steps, {2,6}.  ~2h
note "EXP-3 Depth scan (500 steps, d64/K_min13, lr3e-4)"
for nb in 2 6; do
  train "nb_${nb}" 500 89 13 64 $nb 3e-4
  result "n_blocks=$nb  $(cbest nb_${nb})  validity=$(val nb_${nb}.pt)"
done

# EXP-4  The scale ladder — best small config, LONG.  ~3.5h (1800 steps)
note "EXP-4 Scale ladder (1800 steps, d64/n4/K_min13, lr3e-4)"
train "ladder_1800" 1800 89 13 64 4 3e-4
result "1800 steps  $(cbest ladder_1800)  validity=$(val ladder_1800.pt)"

# EXP-5  Transformer matched long control (fast, ~1s/step).  ~33m
note "EXP-5 Transformer long control (2000 steps)"
python3 -u conventional_baseline.py 2000 > /tmp/auto_tx2k.log 2>&1 || echo "  [tx2k FAILED]"
result "Transformer@2000 validity=$(grep 'BASELINE transformer gen' /tmp/auto_tx2k.log|grep -oE '0\.[0-9]+'|head -1)"

echo "==== AUTONOMOUS PROGRAM COMPLETE $(date) ===="
note "PROGRAM COMPLETE — see per-stage results above"

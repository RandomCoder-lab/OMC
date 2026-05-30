#!/bin/bash
# FAST capacity A/B — 200 steps each, early-convergence read (NOT asymptotic).
# Faithful to the queued configs: K_init=89 (same 332K-seed scale as v1, so results
# are comparable to v1's 0.547), differing in K_min (13 vs 34) + width/depth.
# Caveat: K-shrink is rushed over 200 steps (both configs equally); this is a fast
# early-convergence signal, where the project's substrate priors historically help.
cd "$(dirname "$0")"
LOG=/tmp/capacity_fast.log
exec > >(tee "$LOG") 2>&1

echo "[fast] === v1-config: d64 n4 K_init89 K_min13, 200 steps @ $(date) ==="
python3 -u train_address_navigator.py \
    --steps 200 --seq-len 256 --K-init 89 --K-min 13 \
    --d-model 64 --n-blocks 4 --batch-size 24 --lr 3e-4 \
    --corpus omc_corpus.txt --out fast_v1.pt --device cpu

echo "[fast] === v2-capacity: d128 n6 K_init89 K_min34, 200 steps @ $(date) ==="
python3 -u train_address_navigator.py \
    --steps 200 --seq-len 256 --K-init 89 --K-min 34 \
    --d-model 128 --n-blocks 6 --batch-size 24 --lr 3e-4 \
    --corpus omc_corpus.txt --out fast_v2.pt --device cpu

echo "[fast] === 200-STEP EARLY-CONVERGENCE COMPARISON @ $(date) ==="
echo "--- v1 (d64 n4 K_min13) ---";     python3 -u measure_validity.py fast_v1.pt
echo "--- v2 (d128 n6 K_min34) ---";    python3 -u measure_validity.py fast_v2.pt
echo "[fast] reminder: 200-step = early signal only; asymptotic verdict needs full run"
echo "[fast] ALL DONE @ $(date)"

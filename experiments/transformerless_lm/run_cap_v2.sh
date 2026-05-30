#!/bin/bash
# Resume the fast capacity A/B after restart: v1 already done (fast_v1.pt, char@199=2.742).
# Re-run v2-capacity (200 steps) then compare both.
cd "$(dirname "$0")"
LOG=/tmp/cap_v2.log
exec > >(tee "$LOG") 2>&1

echo "[v2] === v2-capacity: d128 n6 K_init89 K_min34, 200 steps @ $(date) ==="
python3 -u train_address_navigator.py \
    --steps 200 --seq-len 256 --K-init 89 --K-min 34 \
    --d-model 128 --n-blocks 6 --batch-size 24 --lr 3e-4 \
    --corpus omc_corpus.txt --out fast_v2.pt --device cpu

echo "[v2] === FAST CAPACITY COMPARISON (200 steps each) @ $(date) ==="
echo "--- v1 (d64 n4 K_min13, char@199=2.742) ---"; python3 -u measure_validity.py fast_v1.pt
echo "--- v2 (d128 n6 K_min34) ---";                python3 -u measure_validity.py fast_v2.pt
echo "[v2] reminder: 200-step early-convergence signal, not asymptotic"
echo "[v2] ALL DONE @ $(date)"

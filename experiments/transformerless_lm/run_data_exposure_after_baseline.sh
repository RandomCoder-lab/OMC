#!/bin/bash
# Data-exposure test: is the base undertrained or architecture-limited?
# Fresh d64/n4/K_init89/K_min13 (same config+lr as fast_v1's 200-step run) but
# 1000 steps. The validity trend 200→1000 distinguishes "needs more data" from
# "plateaus". Chained after the transformer baseline.
cd "$(dirname "$0")"
LOG=/tmp/data_exposure.log
exec > >(tee "$LOG") 2>&1

echo "[data] waiting for baseline to finish ..."
until grep -q "ALL DONE\|Error\|Traceback" /tmp/baseline.log 2>/dev/null; do sleep 30; done
echo "[data] baseline done — fresh 1000-step run @ $(date)"

python3 -u train_address_navigator.py \
    --steps 1000 --seq-len 256 --K-init 89 --K-min 13 \
    --d-model 64 --n-blocks 4 --batch-size 24 --lr 3e-4 \
    --corpus omc_corpus.txt --out bloom_256_fresh1k.pt --device cpu

echo "[data] === DATA-EXPOSURE TREND (fresh, same config/lr) @ $(date) ==="
echo "--- fresh @200 steps (fast_v1) ---"; python3 -u measure_validity.py fast_v1.pt
echo "--- fresh @1000 steps ---";          python3 -u measure_validity.py bloom_256_fresh1k.pt
echo "[data] climbing 200→1000 = undertrained; flat = architecture plateau"
echo "[data] ALL DONE @ $(date)"

#!/bin/bash
# Fair baseline for the B.5 capacity test: train the v1 config (d=64, n=4,
# K_min=13) FROM SCRATCH at the SAME 2500-step / lr / batch budget as the
# capacity model v2. Makes v2-capacity vs v1-fair apples-to-apples (both cold
# start, same budget) so the capacity verdict isn't confounded by v1's warm start.
# Chained AFTER B.5.
cd "$(dirname "$0")"
LOG=/tmp/v1fair.log
exec > >(tee "$LOG") 2>&1

echo "[v1fair] waiting for B.5 to finish ..."
until grep -q "ALL DONE\|Error\|Traceback" /tmp/b5_base_v2.log 2>/dev/null; do
    sleep 30
done
echo "[v1fair] B.5 done — starting fair small-config baseline at $(date)"

python3 -u train_address_navigator.py \
    --steps 2500 --seq-len 256 --K-init 89 --K-min 13 \
    --d-model 64 --n-blocks 4 --batch-size 24 --lr 3e-4 \
    --corpus omc_corpus.txt \
    --out bloom_256_v1fair.pt --device cpu

echo "[v1fair] done at $(date) — FINAL CAPACITY COMPARISON (gen-validity):"
echo "--- v1-warm  (d64 n4, warm-started, the original base) ---"; python3 -u measure_validity.py bloom_256_model.pt
echo "--- v1-fair  (d64 n4, FROM SCRATCH, 2500 steps) ---";        python3 -u measure_validity.py bloom_256_v1fair.pt
echo "--- v2-cap   (d128 n6 K_min34, FROM SCRATCH, 2500 steps) ---"; python3 -u measure_validity.py bloom_256_cap_v2.pt
echo "[v1fair] CAPACITY VERDICT: capacity helps iff v2-cap > v1-fair (both cold, same budget)"
echo "[v1fair] ALL DONE at $(date)"

#!/bin/bash
# B.5 — train a HIGHER-CAPACITY base (width + depth + more frequency bands),
# from scratch, to test whether capacity (not just data exposure) beats the
# small base v1 (d_model 64, n_blocks 4, 332K params, gen-validity 0.547).
# Chained AFTER D.1 to avoid CPU contention.
cd "$(dirname "$0")"
LOG=/tmp/b5_base_v2.log
exec > >(tee "$LOG") 2>&1

echo "[B.5] waiting for D.1 to finish (avoid CPU contention) ..."
until grep -q "BLOOM HELPS\|BLOOM NEUTRAL\|Error\|Traceback" /tmp/bloom_cycle_ab.log 2>/dev/null; do
    sleep 30
done
echo "[B.5] D.1 done — starting CAPACITY base training at $(date)"

# Capacity scaling — FibRec-aware. KEY FINDING: d_model/n_blocks barely change params
# (weights are generated from 4K² seeds; blocks≥2 recurse). The real capacity lever is
# K (Fibonacci frequency count). So: K_min 13→34 = retain 34 active frequency bands in
# the FINAL model vs 13 (2.6× the deployed weight expressivity) — this is the dominant
# capacity increase. K_init stays 89 (K>89 risks NaN: astronomical Fibonacci freqs).
# Plus d_model 64→128 (2× embed/basis width) and n_blocks 4→6 (free depth via recurrence).
# From scratch (shapes differ from v1). CRT-PE/substrate-attention already baked in.
python3 -u train_address_navigator.py \
    --steps 2500 --seq-len 256 --K-init 89 --K-min 34 \
    --d-model 128 --n-blocks 6 --batch-size 24 --lr 3e-4 \
    --corpus omc_corpus.txt \
    --out bloom_256_cap_v2.pt --device cpu

echo "[B.5] training done at $(date) — measuring generation validity v1 vs v2"
echo "--- v1 (small base, d=64 n=4) ---"; python3 -u measure_validity.py bloom_256_model.pt
echo "--- v2 (capacity base, d=128 n=6) ---"; python3 -u measure_validity.py bloom_256_cap_v2.pt
echo "[B.5] ALL DONE at $(date)"

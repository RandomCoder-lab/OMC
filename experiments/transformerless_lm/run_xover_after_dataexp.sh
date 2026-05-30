#!/bin/bash
# Crossover test: does FibRec's sample-efficiency win persist at 1000 steps, or
# does the transformer catch up (project predicted crossover ~step 500-900)?
# data-exposure already gives FibRec@1000 (bloom_256_fresh1k). Here we add
# transformer@1000, then print the 4-point picture.
cd "$(dirname "$0")"
LOG=/tmp/xover.log
exec > >(tee "$LOG") 2>&1

echo "[xover] waiting for data-exposure (FibRec@1000) to finish ..."
until grep -q "ALL DONE\|Error" /tmp/data_exposure.log 2>/dev/null; do sleep 30; done
echo "[xover] data-exposure done — training transformer @1000 steps @ $(date)"

python3 -u conventional_baseline.py 1000

echo ""
echo "[xover] ===============  CROSSOVER PICTURE (gen-validity, ~330K, from scratch)  ==============="
echo "[xover]               @200 steps        @1000 steps"
FV1=$(grep 'fast_v1.pt: gen'      /tmp/cap_v2.log       | grep -oE '0\.[0-9]+' | head -1)
FV1K=$(grep 'bloom_256_fresh1k.pt: gen' /tmp/data_exposure.log | grep -oE '0\.[0-9]+' | head -1)
TX2=0.291
TX1K=$(grep 'BASELINE transformer gen_validity' /tmp/xover.log | grep -oE '0\.[0-9]+' | head -1)
echo "[xover]  FibRec        $FV1            $FV1K"
echo "[xover]  Transformer   $TX2            $TX1K"
echo "[xover] FibRec wins persist if FibRec@1000 still > Transformer@1000"
echo "[xover] ALL DONE @ $(date)"

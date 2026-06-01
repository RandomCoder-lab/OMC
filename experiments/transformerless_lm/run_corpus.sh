#!/bin/bash
cd /home/thearchitect/OMC/experiments/transformerless_lm || exit 1
for L in russian chinese german french spanish; do
  echo "==================== $L : corpus ingest ===================="
  OMP_NUM_THREADS=2 python3 -u bible_corpus.py --name "$L" || { echo "$L ingest failed, skip"; continue; }
  echo "==================== $L : IBM Model 1 align + write ===================="
  OMP_NUM_THREADS=4 python3 -u align.py --lang "$L" --iters 8 --write
done
echo "==================== ALL CORPUS LANGUAGES DONE ===================="

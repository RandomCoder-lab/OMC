#!/bin/bash
cd /home/thearchitect/OMC/experiments/transformerless_lm || exit 1
BOOKS="GEN,JHN,EST,1KI,DAN"
for L in russian chinese; do
  echo "==================== $L : add_language ===================="
  OMP_NUM_THREADS=2 python3 -u add_language.py --name "$L" --pivot dra --books "$BOOKS"
  echo "==================== $L : IBM Model 1 align + write ===================="
  OMP_NUM_THREADS=4 python3 -u align.py --lang "$L" --iters 8 --write
done
echo "==================== ALL LANGUAGES DONE ===================="

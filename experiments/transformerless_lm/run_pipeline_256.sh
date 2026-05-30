#!/bin/bash
# Pipeline: wait for bloom_256_model.pt, then chain index build → multiskill → curriculum
set -e
cd "$(dirname "$0")"

LOG=/tmp/pipeline_256.log
exec > >(tee -a "$LOG") 2>&1

echo "[pipeline] started at $(date)"

# ── Step 1: wait for training to finish ──────────────────────────────────────
echo "[pipeline] waiting for bloom_256_model.pt ..."
while true; do
    if [ -f bloom_256_model.pt ]; then
        # Wait for the file to stop growing (training done writing)
        SZ1=$(stat -c%s bloom_256_model.pt 2>/dev/null || echo 0)
        sleep 10
        SZ2=$(stat -c%s bloom_256_model.pt 2>/dev/null || echo 0)
        if [ "$SZ1" = "$SZ2" ] && [ "$SZ1" -gt 100000 ]; then
            echo "[pipeline] bloom_256_model.pt stable at ${SZ1} bytes — training done"
            break
        fi
    fi
    sleep 30
done

echo "[pipeline] ── Step 2: build dual index at seq_len=256 ──"
python3 -u corpus_address_index.py \
    --corpus-file omc_corpus.txt \
    --seq-len 256 \
    --stride 256 \
    --d-model 64 \
    --dual \
    --scales 21,256 \
    --rebuild \
    --out omc_dual_index_256.pt \
    2>&1 | tee /tmp/index_256.log
echo "[pipeline] dual index done at $(date)"

echo "[pipeline] ── Step 3: retrain multiskill navigator at seq_len=256 ──"
python3 -u train_multiskill.py \
    --corpus omc_corpus.txt \
    --seq-len 256 \
    --d-model 64 \
    --n-skills 4 \
    --steps 600 \
    --batch-size 24 \
    --lr 3e-4 \
    --device cpu \
    2>&1 | tee /tmp/multiskill_256.log
echo "[pipeline] multiskill done at $(date)"

echo "[pipeline] ── Step 4: curriculum fine-tune at seq_len=256 ──"
python3 -u finetune_curriculum.py \
    --checkpoint bloom_256_model.pt \
    --corpus omc_fndefs.txt \
    --steps 800 \
    --batch 24 \
    --lr 3e-5 \
    --out bloom_256_curriculum_model.pt \
    --device cpu \
    2>&1 | tee /tmp/curriculum_256.log
echo "[pipeline] curriculum fine-tune done at $(date)"

echo "[pipeline] ══ ALL DONE at $(date) ══"
echo "[pipeline] Models ready:"
echo "  bloom_256_model.pt"
echo "  omc_dual_index_256.pt"
echo "  bloom_256_curriculum_model.pt"

#!/bin/bash
# growloop2.sh — DISK-BACKED knowledge growth worker (RAM-flat, unbounded, crash-proof).
# v3: feeds the SQLite store (knowledge.db) via dbstack.py — streaming INSERTs, NO whole-web RAM load,
# NO full rebuild. Ceiling is disk, not RAM. Ingests Gutenberg (--seq) + any modern texts dropped in
# library/ (arXiv etc.). Stop: touch /tmp/STOP_GROW. Logs /tmp/growloop.log.
cd /home/thearchitect/OMC/experiments/transformerless_lm || exit 1
LOG=/tmp/growloop.log
HOURS=${1:-12}
END=$(($(date +%s) + HOURS*3600))
echo "[growloop v3 DISK-BACKED] START $(date) deadline +${HOURS}h" >> "$LOG"
while [ "$(date +%s)" -lt "$END" ]; do
    [ -f /tmp/STOP_GROW ] && { echo "[growloop] STOP $(date)" >> "$LOG"; break; }
    AVD=$(df --output=avail /home 2>/dev/null | tail -1)
    if [ "${AVD:-0}" -lt 2000000 ]; then echo "[growloop] disk<2G pause $(date)" >> "$LOG"; sleep 300; continue; fi
    OMP_NUM_THREADS=4 python3 ingest.py --seq 80 >> /tmp/ingest.log 2>&1   # Gutenberg classics (unlimited)
    OMP_NUM_THREADS=4 python3 dbstack.py >> /tmp/dbstack.log 2>&1          # RAM-flat stream into knowledge.db
    N=$(ls library/*.txt 2>/dev/null | wc -l)
    echo "[growloop] disk-cycle $(date): library $N texts, disk $(df -h /home 2>/dev/null|awk 'NR==2{print $4}') free, RAM $(free -h|awk '/Mem:/{print $7}') avail" >> "$LOG"
    sleep 20
done
echo "[growloop] ENDED $(date)" >> "$LOG"

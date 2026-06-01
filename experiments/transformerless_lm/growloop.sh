#!/bin/bash
# growloop.sh — autonomous knowledge-web growth worker (STACKING-ONLY, crash-safe).
# v2 after the 09:48 crash: the full integrate rebuild thrashed 30G RAM into swap and took the box down.
# FIX: NO full rebuilds here — grow by incremental stacking only (light: load web, add edges, save).
# The embedding is already current (last integrate completed); stacking only adds grounded edges.
# RAM guard pauses if memory gets tight (stack load grows with the web). Disk guard too.
# Stop: touch /tmp/STOP_GROW. Logs: /tmp/growloop.log.
cd /home/thearchitect/OMC/experiments/transformerless_lm || exit 1
LOG=/tmp/growloop.log
HOURS=${1:-12}
END=$(($(date +%s) + HOURS*3600))
echo "[growloop v2 STACK-ONLY] START $(date) deadline +${HOURS}h" >> "$LOG"

while [ "$(date +%s)" -lt "$END" ]; do
    [ -f /tmp/STOP_GROW ] && { echo "[growloop] STOP file — exiting $(date)" >> "$LOG"; break; }
    # RAM guard: stacking loads the whole (growing) web; need headroom. Pause if avail < 8G.
    AVKB=$(free | awk '/Mem:/{print $7}')
    if [ "${AVKB:-0}" -lt 8000000 ]; then
        echo "[growloop] RAM avail $((AVKB/1024))MB <8G — pause 3m $(date)" >> "$LOG"; sleep 180; continue
    fi
    # Disk guard: stop if free < 2G.
    AVD=$(df --output=avail /home 2>/dev/null | tail -1)
    if [ "${AVD:-0}" -lt 2000000 ]; then
        echo "[growloop] disk <2G — pause 5m $(date)" >> "$LOG"; sleep 300; continue
    fi
    OMP_NUM_THREADS=4 python3 ingest.py --seq 60 >> /tmp/ingest.log 2>&1
    OMP_NUM_THREADS=4 python3 stack.py >> /tmp/stack.log 2>&1   # incremental: append edges, no retrain
    N=$(ls library/*.txt 2>/dev/null | wc -l)
    echo "[growloop] stack-cycle $(date): library $N texts, RAM $((AVKB/1024))MB avail, disk $(df -h /home 2>/dev/null|awk 'NR==2{print $4}') free" >> "$LOG"
    sleep 30
done
echo "[growloop] ENDED $(date)" >> "$LOG"

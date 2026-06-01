#!/bin/bash
# feed.sh — ONE gentle rotation of all MODERN connectors (arXiv/Wikipedia/PubMed/code+RFC).
# Called periodically by the all-source worker. Code (GitHub, 60/hr) only every 6th feed; RFCs every time.
cd /home/thearchitect/OMC/experiments/transformerless_lm || exit 1
IDX=$(cat /tmp/feed_idx 2>/dev/null || echo 0); IDX=$((IDX+1)); echo $IDX > /tmp/feed_idx
OMP_NUM_THREADS=2 python3 arxiv_ingest.py --per-cat 12   >> /tmp/arxiv.log  2>&1
OMP_NUM_THREADS=2 python3 wiki_ingest.py --random 60 --batch 10 >> /tmp/wiki.log 2>&1
OMP_NUM_THREADS=2 python3 pubmed_ingest.py --per-term 40 >> /tmp/pubmed.log 2>&1
if [ $((IDX % 6)) -eq 0 ]; then
  OMP_NUM_THREADS=2 python3 code_ingest.py --repos --files-per-repo 10 --rfc 30 >> /tmp/code.log 2>&1
else
  OMP_NUM_THREADS=2 python3 code_ingest.py --rfc 30 >> /tmp/code.log 2>&1
fi

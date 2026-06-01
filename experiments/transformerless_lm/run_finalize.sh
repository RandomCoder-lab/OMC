#!/bin/bash
cd /home/thearchitect/OMC/experiments/transformerless_lm || exit 1
echo "[finalize] waiting for all fetches + language expansion to finish ..."
while pgrep -f "run_more_langs|arxiv_ingest.py|se_dump.py|bible_corpus.py|align.py|dbstack.py" >/dev/null 2>&1; do sleep 30; done
echo "[finalize] all clear $(date) — folding new library (code+arxiv+SE) into DB ..."
OMP_NUM_THREADS=4 python3 -u dbstack.py
echo "[finalize] fold done — running node_expand (stopwords + new-term addressability) ..."
OMP_NUM_THREADS=4 python3 -u node_expand.py
echo "[finalize] DONE $(date)"
python3 -c "import sqlite3;d=sqlite3.connect('knowledge.db');print('passages',d.execute('SELECT COUNT(*) FROM passages').fetchone()[0],'edges',d.execute('SELECT COUNT(*) FROM edges').fetchone()[0],'stopwords',d.execute('SELECT COUNT(*) FROM stop').fetchone()[0])"

#!/bin/bash
# wait for fold_subset to finish, then run code-node expansion (OMC+code addressable as a codebase),
# then verify OMC self-knowledge. Serial → no DB write-collision.
cd /home/thearchitect/OMC/experiments/transformerless_lm || exit 1
while pgrep -f fold_subset.py >/dev/null 2>&1; do sleep 20; done
echo "[chain] fold done $(date) — running code-node expansion (codebase addressable) ..."
OMP_NUM_THREADS=4 python3 -u code_nodes.py
echo "[chain] verify OMC self-knowledge ..."
python3 -c "
import sys,math;sys.path.insert(0,'.')
from factcheck import open_web,FactChecker
k=open_web();fc=FactChecker(k)
print('omc passages w/ substrate:', k.db.execute(\"SELECT COUNT(*) FROM passages WHERE dom LIKE 'omc%' AND lower(text) LIKE '%substrate%'\").fetchone()[0])
for c in ['substrate','addressing','harmony','fn','impl','malloc','struct']:
    fac=fc.factors(c,topk=8)
    print(' ',c,'→',', '.join(b for _,_,b,_,_ in fac) if fac else '(not a node)')
" 2>/dev/null
echo "[chain] DONE $(date)"

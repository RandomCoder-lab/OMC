import sys, json; sys.path.insert(0,'.')
from kdb import KnowledgeDB, load_embedding
from navigator import coherent_path
s,E,n,e=load_embedding(); k=KnowledgeDB("knowledge.db",s,E,n,e)
d=k.db; purged=0
for mid,cj in d.execute("SELECT id,concepts FROM memory WHERE kind='derived'").fetchall():
    cs=json.loads(cj)
    if len(cs)>=3 and not coherent_path(k,cs):
        d.execute("DELETE FROM memory WHERE id=?",(mid,)); purged+=1
d.commit()
print("purged",purged,"incoherent | remaining derived:",d.execute("SELECT COUNT(*) FROM memory WHERE kind='derived'").fetchone()[0])

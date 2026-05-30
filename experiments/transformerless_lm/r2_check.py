import statistics
from pathlib import Path
from omc_assistant import _load_lm
from grimoire_spells import spelled_generate, build_spell_tables, ALL_SPELLS, NO_SPELLS
K=4; corpus=Path('omc_corpus.txt').read_text(errors='replace')
grams=set(corpus[i:i+K] for i in range(0,min(len(corpus),20_000_000)-K))
def distinct(s): return len(set(s))/max(len(s),1)
def rawval(s): return 0.0 if len(s)<K else sum(1 for i in range(len(s)-K) if s[i:i+K] in grams)/(len(s)-K)
def val(s): return 0.0 if distinct(s)<0.15 else rawval(s)
m,stoi,itos,sl=_load_lm(Path('bloom_256_curriculum_model.pt'))
tab=build_spell_tables(itos)
prompts=["fn ","h x = ","fn add(a, b) {","return "]
def agg(spells,att):
    return statistics.mean(val(spelled_generate(m,stoi,itos,sl,pr,140,spells=spells,tables=tab,seed=s,attenuate=att)) for s in range(4) for pr in prompts)
def w(line):
    print(line,flush=True); open('/tmp/r2_result.txt','a').write(line+'\n')
open('/tmp/r2_result.txt','w').close()
w(f"OFF              {agg(NO_SPELLS,0.0):.3f}")
w(f"RAW(old proxy)   {agg(ALL_SPELLS,0.0):.3f}")
for a in (0.1,0.236,0.5): w(f"ATTENUATED a={a:<5} {agg(ALL_SPELLS,a):.3f}")
w("R2 DONE")

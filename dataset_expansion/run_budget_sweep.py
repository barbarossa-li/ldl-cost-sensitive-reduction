"""Common-budget ranking sweep on the informative expansion datasets.

For SJAFFE, fbp5500, RAF-ML (three repeats each), sweep acquisition budget caps
{1,2,5,10,20,50} and record the budget-constrained ranking outputs of two
selection rules, mirroring the original common-budget ablation protocol:
  Prediction ranking: largest positive inner-loss improvement per unit cost;
  Dependency guidance: among positive-improvement candidates, prefer positive
  granular dependency gain per unit cost (fallback to prediction rank).
No deletion/exchange refinement. Same frozen splits, folds, predictor, costs as
the expansion protocol. Selections frozen before audit evaluation.

Reproduction: python run_budget_sweep.py  -> budget_sweep_results.csv
"""
from pathlib import Path
import sys,json,csv,time,gc,math
import numpy as np
from scipy.spatial.distance import cdist

here=Path(__file__).resolve().parent
root=here.parent
sys.path.insert(0,str(root));sys.path.insert(0,str(root/'stability'))
from granulation_reference import TrainingGeometry
from fast_granulate import granulate
from run_ablations import load,blocks_for,nearest5

DATASETS=['SJAFFE','fbp5500','RAF_ML']
BUDGETS=[1,2,5,10,20,50]

def safe(stem):return stem.replace('-','_')

def main():
 out=here/'budget_sweep';out.mkdir(exist_ok=True)
 rows=[]
 for stem in DATASETS:
  ds=safe(stem)
  for rep in range(1,4):
   job=out/f'{ds}_repeat_{rep}'
   if (job/'results.json').exists():rows.extend(json.loads((job/'results.json').read_text()));continue
   job.mkdir(parents=True,exist_ok=True)
   d=load(stem,rep)
   X,L,costs,m=d['X'],d['L'],d['costs'],d['m']
   n=len(X)
   # kNN inner loss with fold-standardized blocks
   blocks=blocks_for(d);cache={}
   def risk(P):
    key=tuple(sorted(P))
    if key not in cache:
     ss=0.
     for mat,fit,v in blocks:
      pred=L[fit][nearest5(mat[list(key)].sum(0))].mean(1) if key else np.tile(L[fit].mean(0),(len(v),1))
      ss+=np.sum((L[v]-pred)**2)
     cache[key]=float(ss/n)
    return cache[key]
   model=TrainingGeometry(X,L,kinds=['numerical']*m);dcache={}
   def dep(P):
    key=tuple(sorted(P))
    if key not in dcache:
     dd,k=model.subset(key);dcache[key]=granulate(dd,L,k)[2]
    return dcache[key]
   fullref=risk(range(m));cref=costs.mean()
   local=[]
   for cap in BUDGETS:
    for method in ['Prediction ranking','Dependency guidance']:
     P=set();spent=0
     while spent<cap and len(P)<m:
      cands=[a for a in range(m) if a not in P and spent+costs[a]<=cap]
      if not cands:break
      base=risk(P);gains={a:base-risk(P|{a}) for a in cands}
      positive=[a for a in gains if gains[a]>0]
      if positive:
       if method=='Dependency guidance':
        d0=dep(P);dg={a:dep(P|{a})-d0 for a in positive}
        both=[a for a in positive if dg[a]>0]
        if both:a=min(both,key=lambda a:(-(math.log(dg[a])-math.log(costs[a]/cref)),costs[a],a))
        else:a=min(positive,key=lambda a:(-(math.log(gains[a])-math.log(costs[a]/cref)),costs[a],a))
       else:
        a=min(positive,key=lambda a:(-(math.log(gains[a])-math.log(costs[a]/cref)),costs[a],a))
      else:break
      P.add(a);spent+=costs[a]
      if risk(P)<=fullref:break
     Pl=list(P)
     # audit evaluation
     lo=X.min(0);span=X.max(0)-lo;span[span==0]=1.;z=(X-lo)/span;qa=(d['auditX']-lo)/span
     if Pl:
      pred=L[np.argsort(cdist(qa[:,Pl],z[:,Pl],'sqeuclidean'),axis=1,kind='stable')[:,:5]].mean(1)
     else:pred=np.tile(L.mean(0),(len(d['auditL']),1))
     audit=float(np.mean(np.sum((d['auditL']-pred)**2,1)))
     local.append({'dataset':stem,'repeat':rep,'cap':cap,'method':method,'attributes':len(Pl),
      'spent':int(spent),'audit_brier':audit,'feasible':bool(risk(P)<=fullref)})
   (job/'results.json').write_text(json.dumps(local,indent=2));rows.extend(local)
   print(ds,rep,'done',len(local),'rows',flush=True)
   del model,blocks,dcache,cache;gc.collect()
 with (here/'budget_sweep_results.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('COMPLETE',len(rows),'rows')

if __name__=='__main__':main()

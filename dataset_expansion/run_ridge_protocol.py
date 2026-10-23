"""Full nested-audit protocol with a ridge-regression predictor on the datasets
revived by the ridge probe (Plan C, upgraded).

Datasets: the nine expansion datasets on which the ridge full-set inner loss beat
the ridge global-mean loss on at least two of three repeats (Yeast-diau, Yeast-dtt,
Yeast-heat, Yeast-spoem, FG-Net, Painting, SCUT-FBP, Movie, MOR-PH).

Protocol: identical to run_expansion.py (frozen splits, 3 inner folds, granular
dependency guidance, pure prediction-only search, 200-evaluation exchange
refinement, costs, audit reservation) with one change: the evaluation predictor
inside risk(), the guard, and the audit evaluation use ridge regression with the
alpha frozen per split (selected by inner CV on the full attribute set, training
data only -- recorded in ridge/ by run_ablations.py). This isolates the effect of
the predictor while keeping every other rule frozen.

Reproduction: python run_ridge_protocol.py   (checkpoints per repeat; resumable)
Writes ridge_protocol/ results and ridge_protocol_results.csv.
"""
from pathlib import Path
import sys,json,csv,time,gc
import numpy as np

here=Path(__file__).resolve().parent
root=here.parent
sys.path.insert(0,str(root));sys.path.insert(0,str(root/'stability'));sys.path.insert(0,str(root/'predictive_revision'));sys.path.insert(0,str(root/'refinement_pilot'))
from granulation_reference import TrainingGeometry
from fast_granulate import granulate
from reduction_reference import heuristic as pure
from revised_search import heuristic as guided
from refine import refine
from run_ablations import load,ALPHAS

REVIVED=['Yeast_diau','Yeast_dtt','Yeast_heat','Yeast_spoem','FG-Net','Painting','SCUT_FBP','Movie','MOR-PH']

def safe(stem):return stem.replace('-','_')

def ridge_fit(X,L,alpha):
 n,m=X.shape
 return np.linalg.solve(X.T@X+alpha*np.eye(m),X.T@L)

def main():
 out=here/'ridge_protocol';out.mkdir(exist_ok=True)
 rows=[]
 for stem in REVIVED:
  ds=safe(stem)
  if not (here/'data'/f'{ds}_X.npy').exists():print(stem,'no data, skip');continue
  for rep in range(1,4):
   job=out/f'{ds}_repeat_{rep}'
   job.mkdir(parents=True,exist_ok=True)
   if (job/'complete.json').exists():
    for r in json.loads((job/'audit_results.json').read_text()):rows.append(r)
    continue
   # frozen alpha from the probe (training-only selection)
   probe=json.loads((here/'ridge'/f'{ds}_repeat_{rep}'/'results.json').read_text())[0]
   alpha=probe['alpha']
   d=load(stem,rep)
   X,L,auditX,auditL=d['X'],d['L'],d['auditX'],d['auditL']
   n,m=X.shape;costs=d['costs']
   fold_n=n//3
   folds=[np.arange(j*fold_n,(j+1)*fold_n) for j in range(3)]
   blocks=[]
   for v in folds:
    fit=np.setdiff1d(np.arange(n),v)
    mu=X[fit].mean(0);sd=X[fit].std(0);sd[sd==0]=1.
    z=(X[fit]-mu)/sd;q=(X[v]-mu)/sd
    blocks.append((z,fit,v,q,L[fit].mean(0),mu))
   def risk(P):
    """3-fold inner CV ridge loss on subset P (per-fold standardization)."""
    key=tuple(sorted(P))
    P=list(key)
    ss=0.
    for z,fit,v,q,lm,mu in blocks:
     if P:
      W=ridge_fit(z[:,P],L[fit],alpha)
      pred=q[:,P]@W+(lm-mu[P]@W)
     else:
      pred=np.tile(lm,(len(v),1))
     ss+=np.sum((L[v]-pred)**2)
    return float(ss/n)
   model=TrainingGeometry(X,L,kinds=['numerical']*m);dcache={}
   def dep(P):
    key=tuple(sorted(P))
    if key not in dcache:
     dd,k=model.subset(key);dcache[key]=granulate(dd,L,k)[2]
    return dcache[key]
   (job/'split_meta.json').write_text(json.dumps({'dataset':stem,'repeat':rep,'alpha':alpha,'costs':costs.tolist()},indent=2))
   started=time.perf_counter();results={}
   for method in ['Prediction only','Guided']:
    cache={}
    def riskc(P):
     key=tuple(sorted(P))
     if key not in cache:cache[key]=risk(P)
     return cache[key]
    limit=riskc(range(m))
    if method=='Prediction only':res=pure(costs,lambda P:0.,prediction_loss=riskc)
    else:res=guided(costs,dep,riskc)
    P=res[0];initial_trace=res[3];t0=time.perf_counter()
    assert riskc(P)<=limit and all(riskc(set(P)-{a})>limit for a in P)
    Q,log=refine(P,costs,riskc,limit,200)
    assert sum(costs[list(Q)])<=sum(costs[list(P)]) and riskc(Q)<=limit
    results[method]=dict(before=list(P),after=list(Q),results=dict(
     method=method,before_attributes=len(P),after_attributes=len(Q),
     before_cost=int(costs[list(P)].sum()),after_cost=int(costs[list(Q)].sum()),
     full_cost=int(costs.sum()),full_inner=limit,after_loss=riskc(Q),
     evaluations=log['evaluations'],accepted=len(log['accepted']),
     initial_seconds=time.perf_counter()-t0))
    (job/(method.replace(' ','_')+'.json')).write_text(json.dumps(results[method],indent=2))
    print(ds,method,rep,'before',len(P),costs[list(P)].sum(),'after',len(Q),costs[list(Q)].sum(),flush=True)
   # audit evaluation with ridge predictor (after both selections frozen)
   ck={k:results[k] for k in results}
   selected={'Full':list(range(m)),'Empty':[]}
   for method,q in ck.items():selected[method+' after']=q['after']
   mu=X.mean(0);sd=X.std(0);sd[sd==0]=1.;z=(X-mu)/sd;qa=(auditX-mu)/sd;qt=(d['Xt']-mu)/sd;truth=d['Lt'];lm=L.mean(0)
   arows=[]
   for method,P in selected.items():
    if P:
     W=ridge_fit(z[:,P],L,alpha)
     pa=qa[:,P]@W+(lm-mu[P]@W);pt=qt[:,P]@W+(lm-mu[P]@W)
    else:
     pa=np.tile(lm,(len(auditL),1));pt=np.tile(lm,(len(truth),1))
    audit=float(np.mean(np.sum((auditL-pa)**2,1)))
    pt=np.maximum(pt,1e-12);pt/=pt.sum(1,keepdims=True)
    arows.append({'dataset':stem,'repeat':rep,'alpha':alpha,'method':method,'attributes':len(P),
     'cost':int(costs[P].sum()),'audit_brier':audit,
     'test_brier':float(np.mean(np.sum((truth-pt)**2,1)))})
   (job/'audit_results.json').write_text(json.dumps(arows,indent=2))
   (job/'complete.json').write_text(json.dumps({'elapsed_seconds':time.perf_counter()-started},indent=2))
   rows.extend(arows)
   print(ds,'AUDIT',rep,[(r['method'],r['attributes'],r['cost'],round(r['audit_brier'],4)) for r in arows],'%.0fs'%(time.perf_counter()-started),flush=True)
   del model,blocks,dcache;gc.collect()
 if rows:
  with (here/'ridge_protocol_results.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  print('COMPLETE',len(rows),'rows')

if __name__=='__main__':main()

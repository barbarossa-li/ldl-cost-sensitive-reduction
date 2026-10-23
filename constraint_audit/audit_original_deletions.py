"""Frozen training-only comparisons; reuse 15 existing splits. Not published-method replicas."""
from pathlib import Path
import sys,json,csv,time,gc
import numpy as np
from scipy.spatial.distance import cdist
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from reduction_reference import heuristic
out=Path(__file__).resolve().parent;sys.path.insert(0,str(root/'stability'));
from granulation_reference import TrainingGeometry
from fast_granulate import granulate
X=np.load(root/'pilot/data/feature.npy');L=np.load(root/'pilot/data/label.npy');costs=np.array(json.loads((root/'stability/protocol.json').read_text())['costs']);m=X.shape[1]
def nearest5(d):
 # Exact stable-argsort nearest five, using partial selection and explicit boundary ties.
 idx=np.argpartition(d,4,axis=1)[:,:5];vals=np.take_along_axis(d,idx,axis=1);threshold=vals.max(1)
 for i in range(len(d)):
  below=np.flatnonzero(d[i]<threshold[i]);ties=np.flatnonzero(d[i]==threshold[i]);v=np.r_[below,ties[:5-len(below)]]
  idx[i]=v[np.lexsort((v,d[i,v]))]
 return idx

protocol={'scope':'constraint-only audit on identical frozen budget-bridge prefixes','rule':'For both versions choose lowest-cost prefix satisfying inner loss <= full reference. On version additionally requires Gamma >= full reference. Same highest-cost-first deletion/restart rule and budget. No feasible prefix reported as failure; never substitute a different search or a violating subset.','note':'Same path isolates feasibility checks, not directed geometry ranking or exhaustive global feasibility. Existing test data exploratory only.'}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2));rows=[]
for rep in range(1,6):
 for n in [192,384,576]:
  job=out/f'repeat_{rep}_n{n}';job.mkdir(exist_ok=True)
  if (job/'original_deletion_events.json').exists():continue
  source=root/'stability'/job.name;split=json.loads((source/'split.json').read_text());old=json.loads((source/'results.json').read_text());joint=next(r for r in old if r['method']=='Joint guard');ref=next(r for r in old if r['method']=='All attributes');gref=ref['gamma'];rref=ref['inner_brier'];budget=joint['cost']
  trace=json.loads((root/'baselines'/job.name/'bridge_selection_before_test.json').read_text())['trace']
  train=np.array(split['train_indices']);test=np.array(split['test_indices']);x=X[train];lab=L[train];blocks=[];cache={};dcache={};model=TrainingGeometry(x,lab,kinds=['numerical']*m)
  for val in map(np.array,split['fold_validation_positions']):
   fit=np.setdiff1d(np.arange(n),val);a=x[fit];b=x[val];lo=a.min(0);span=a.max(0)-lo;span[span==0]=1.;a=(a-lo)/span;b=(b-lo)/span;blocks.append((((b[:,None,:]-a[None,:,:])**2).transpose(2,0,1).copy(),fit,val))
  def risk(P):
   key=tuple(sorted(P))
   if key not in cache:
    total=0.
    for mat,fit,val in blocks:
     pred=lab[fit][nearest5(mat[list(key)].sum(0))].mean(1) if key else np.tile(lab[fit].mean(0),(len(val),1))
     total+=np.sum((lab[val]-pred)**2)
    cache[key]=float(total/n)
   return cache[key]
  def dep(P):
   key=tuple(sorted(P))
   if key not in dcache:
    d,k=model.subset(key);dcache[key]=granulate(d,lab,k)[2]
   return dcache[key]
  assert abs(risk(tuple(range(m)))-rref)<1e-12 and abs(dep(tuple(range(m)))-gref)<1e-12
  events=[]
  P=set(joint['selected'])
  for aa in sorted(P,key=lambda a:(-costs[a],a)):
   C=tuple(sorted(P-{aa}));rp=risk(C)<=rref;gp=dep(C)>=gref
   events.append({'removed':int(aa),'selected':list(C),'cost':int(sum(costs[list(C)])),'inner_brier':risk(C),'gamma':dep(C),'risk_pass':rp,'gamma_pass':gp,'gamma_only_block':rp and not gp})
  (job/'original_deletion_events.json').write_text(json.dumps(events,indent=2))
  print('DIAG',rep,n,'gamma-only blocks',sum(e['gamma_only_block'] for e in events),flush=True)
  del model,blocks,cache,dcache;gc.collect()

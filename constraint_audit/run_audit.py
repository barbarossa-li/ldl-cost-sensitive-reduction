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
  if (job/'results.json').exists():rows.extend(json.loads((job/'results.json').read_text()));continue
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
  prefixes=[()];P=set()
  for a,g,v in trace:
   P.add(a);key=tuple(sorted(P));prefixes.append(key);assert abs(risk(key)-v)<1e-12
  events=[]
  for idx,P in enumerate(prefixes):
   events.append({'phase':'common prefix','index':idx,'selected':list(P),'cost':int(sum(costs[list(P)])),'inner_brier':risk(P),'gamma':dep(P),'risk_pass':risk(P)<=rref,'gamma_pass':dep(P)>=gref})
  selected={};counts={}
  for mode,on in [('Constraint off',False),('Constraint on',True)]:
   feasible=[P for P in prefixes if risk(P)<=rref and (not on or dep(P)>=gref)]
   counts[mode]=len(feasible)
   if not feasible:selected[mode]=None;continue
   P=set(min(feasible,key=lambda P:(sum(costs[list(P)]),len(P),P)))
   while True:
    removed=False
    for a in sorted(P,key=lambda a:(-costs[a],a)):
     C=P-{a};rp=risk(C)<=rref;gp=dep(C)>=gref
     events.append({'phase':'deletion','mode':mode,'removed':a,'selected':sorted(C),'cost':int(sum(costs[list(C)])),'inner_brier':risk(C),'gamma':dep(C),'risk_pass':rp,'gamma_pass':gp,'accepted':rp and (gp or not on)})
     if rp and (gp or not on):P=C;removed=True;break
    if not removed:break
   selected[mode]=tuple(sorted(P));assert sum(costs[list(P)])<=budget
  # All choices and constraint decisions fixed before test evaluation.
  (job/'selection_before_test.json').write_text(json.dumps({'selected':selected,'budget':budget,'gamma_reference':gref,'inner_reference':rref,'feasible_prefix_counts':counts},indent=2));(job/'events.json').write_text(json.dumps(events,indent=2))
  lo=x.min(0);span=x.max(0)-lo;span[span==0]=1.;a=(x-lo)/span;b=(X[test]-lo)/span;truth=L[test];local=[]
  for mode,P in selected.items():
   row={'repeat':rep,'train_n':n,'mode':mode,'success':P is not None,'attributes':len(P) if P is not None else None,'cost':int(sum(costs[list(P)])) if P is not None else None,'selected':list(P) if P is not None else None,'inner_brier':risk(P) if P is not None else None,'gamma':dep(P) if P is not None else None,'test_brier':None,'test_kl':None}
   if P is not None:
    pred=lab[nearest5(cdist(b[:,P],a[:,P],'sqeuclidean'))].mean(1) if P else np.tile(lab.mean(0),(len(test),1));pred=np.maximum(pred,1e-12);pred/=pred.sum(1,keepdims=True);row['test_brier']=float(np.mean(np.sum((truth-pred)**2,axis=1)));row['test_kl']=float(np.mean(np.sum(truth*np.log(np.maximum(truth,1e-300)/pred),axis=1)))
   local.append(row)
  (job/'results.json').write_text(json.dumps(local,indent=2));rows.extend(local)
  with (out/'results.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  print('DONE',rep,n,'feasible',counts,'same',selected['Constraint off']==selected['Constraint on'],'extra rejected',sum(e['risk_pass'] and not e['gamma_pass'] for e in events if e['phase']=='common prefix'),flush=True)
  del model,blocks,cache,dcache;gc.collect()

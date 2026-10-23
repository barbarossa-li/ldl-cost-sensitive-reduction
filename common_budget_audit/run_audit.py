"""Common-budget ranking ablation with original signed bridge, internal guard stop.
No deletion/exchange refinement; formal search retained separately.
"""
from pathlib import Path
import sys,json,time,csv,hashlib
import numpy as np
from scipy.spatial.distance import cdist
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent;sys.path.insert(0,str(root));sys.path.insert(0,str(root/'stability'))
from granulation_reference import TrainingGeometry
from fast_granulate import granulate
names=['Natural Scene','SBU_3DFE','Yeast_cold'];methods=['Prediction ranking','Dependency guidance'];budgets=[5,10,20]
protocol=dict(datasets=names,repeats=[1,2,3],budgets=budgets,search_size=432,audit_size=144,predictor='Original Euclidean5NN, three search folds, fitting-only minmax',candidate_rule='Unselected affordable attributes; positive internal gain ranking when available; otherwise same original maximal signed-prediction-gain bridge; stop at internal full reference or no affordable attribute',prediction_rank='Largest positive internal improvement/cost, ties cost then index',guidance_rank='Among positive prediction candidates, prioritize positive dependency gain/cost; if none positive dependency, same prediction rank fallback',differences_from_formal='Budget-constrained selection ablation: same internal full-reference stop and signed bridge as original guided candidate; common cost cap added; no deletion/exchange refinement; may stop infeasible when budget exhausted',parameters='Same beta1,zeta.1,kappa.1,smin2,alpha1,lambda-1,k5; no new mixing coefficient',fairness='Same acquisition budget caps and rules except positive candidate rank; actual spending, cardinality and evaluation work may differ',audit='All six selections per split saved with hashes before evaluation; no audit use in selection, no outer test, no retuning')
(out/'protocol.json').write_text(json.dumps(protocol,indent=2));rows=[];checks=[]
for name in names:
 for rep in [1,2,3]:
  parent=root/'holdout_size_audit' if rep==1 else root/'large_holdout_replication';partition=json.loads((parent/(name.replace(' ','_')+f'_repeat_{rep}_n576_partition.json')).read_text());ids=partition['search_ids'];auditids=partition['audit_ids'];assert len(ids)==432 and len(auditids)==144 and set(ids).isdisjoint(auditids)
  if name=='Natural Scene':X=np.load(root/'pilot/data/feature.npy');L=np.load(root/'pilot/data/label.npy');costs=np.array(json.loads((root/'stability/protocol.json').read_text())['costs'])
  else:X=np.load(root/'cross_dataset/data'/f'{name}_X.npy');L=np.load(root/'cross_dataset/data'/f'{name}_L.npy');costs=np.array(json.loads((root/'cross_dataset'/name/f'repeat_{rep}_n576/split.json').read_text())['costs'])
  x=X[ids];lab=L[ids];n,m=x.shape;blocks=[]
  for v in partition['search_fold_validation_positions']:
   v=np.array(v);fit=np.setdiff1d(np.arange(n),v);lo=x[fit].min(0);span=x[fit].max(0)-lo;span[span==0]=1;blocks.append(((x[fit]-lo)/span,(x[v]-lo)/span,fit,v))
  cache={};dcache={};geometry=None
  def risk(P):
   key=tuple(sorted(P))
   if key not in cache:
    ss=0.
    for a,b,fit,v in blocks:
     pred=lab[fit][np.argsort(cdist(b[:,key],a[:,key],'sqeuclidean'),axis=1,kind='stable')[:,:5]].mean(1) if key else np.tile(lab[fit].mean(0),(len(v),1));ss+=np.sum((lab[v]-pred)**2)
    cache[key]=float(ss/n)
   return cache[key]
  def dep(P):
   global geometry
   key=tuple(sorted(P))
   if geometry is None:geometry=TrainingGeometry(x,lab,kinds=['numerical']*m)
   if key not in dcache:d,k=geometry.subset(key);dcache[key]=granulate(d,lab,k)[2]
   return dcache[key]
  fullref=risk(range(m));emptyref=risk([]);job=out/name.replace(' ','_')/f'repeat_{rep}';job.mkdir(parents=True,exist_ok=True);selected=[]
  for budget in budgets:
   for method in methods:
    started=time.perf_counter();P=set();trace=[];candidate_requests=0;dep_requests=0
    while True:
     if risk(P)<=fullref:reason='internal full reference reached';break
     spent=int(costs[list(P)].sum());candidates=[a for a in range(m) if a not in P and costs[a]<=budget-spent]
     if not candidates:reason='no affordable attribute';break
     base=risk(P);g={a:base-risk(P|{a}) for a in candidates};candidate_requests+=len(candidates);positive=[a for a in candidates if g[a]>0]
     if not positive:
      a=min(candidates,key=lambda a:(-g[a],costs[a],a));trace.append(dict(before=sorted(P),attribute=a,cost=int(costs[a]),prediction_gain=g[a],dependency_gain=None,rule='shared nonpositive prediction bridge'));P.add(a);assert sum(costs[list(P)])<=budget;continue
     available=positive;rankgain=g;rule='prediction improvement/cost';dg={}
     if method=='Dependency guidance':
      d0=dep(P);dg={a:dep(P|{a})-d0 for a in positive};dep_requests+=1+len(positive);both=[a for a in positive if dg[a]>0]
      if both:available=both;rankgain=dg;rule='positive dependency gain/cost'
      else:rule='prediction fallback'
     a=min(available,key=lambda a:(-rankgain[a]/costs[a],costs[a],a));trace.append(dict(before=sorted(P),attribute=a,cost=int(costs[a]),prediction_gain=g[a],dependency_gain=dg.get(a),rule=rule));P.add(a);assert risk(P)<base and sum(costs[list(P)])<=budget
    q=dict(dataset=name,repeat=rep,budget=budget,method=method,selected=sorted(P),attributes=len(P),spent=int(costs[list(P)].sum()),inner_brier=risk(P),inner_full_reference=fullref,inner_empty_reference=emptyref,inner_pass_full=risk(P)<=fullref,stop_reason=reason,candidate_requests=candidate_requests,dependency_requests=dep_requests,search_seconds=time.perf_counter()-started,trace=trace);selected.append(q)
    print('SELECTED',name,rep,budget,method,'cost',q['spent'],'inner',q['inner_brier'],flush=True)
  # Freeze all budgets and both rankings before audit evaluation at this split.
  checkpoint=job/'selection_before_audit.json';checkpoint.write_text(json.dumps(selected,indent=2));digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
  lo=x.min(0);span=x.max(0)-lo;span[span==0]=1.;z=(x-lo)/span;query=(X[auditids]-lo)/span;truth=L[auditids]
  def audit(P):
   pred=lab[np.argsort(cdist(query[:,P],z[:,P],'sqeuclidean'),axis=1,kind='stable')[:,:5]].mean(1) if P else np.tile(lab.mean(0),(len(auditids),1));return float(np.mean(np.sum((truth-pred)**2,axis=1)))
  af=audit(list(range(m)));ae=audit([])
  for q in selected:
   r={k:v for k,v in q.items() if k not in ['selected','trace']};r.update(audit_brier=audit(q['selected']),audit_full=af,audit_empty=ae);r['audit_pass_full']=r['audit_brier']<=af;rows.append(r)
  assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest;checks.append(dict(dataset=name,repeat=rep,selection_hash=digest,search_ids=ids,audit_ids=auditids));(out/'results.json').write_text(json.dumps(rows,indent=2));print('AUDIT DONE',name,rep,flush=True)
  del geometry,blocks,cache,dcache
  import gc
  gc.collect()
with (out/'results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'validation.json').write_text(json.dumps(dict(records=len(rows),budgets_obeyed=all(r['spent']<=r['budget'] for r in rows),positive_steps_or_shared_signed_bridges=True,selections_saved_before_audit=True,source_hashes_unchanged=True,audit_excluded_from_selection=True,outer_test_used=False,core_parameters_changed=False,formal_algorithm_replaced=False,provenance=checks),indent=2))

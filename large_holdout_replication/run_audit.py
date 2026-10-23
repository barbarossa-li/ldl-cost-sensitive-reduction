from pathlib import Path
import sys,json,time,csv
import numpy as np
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent
sys.path.insert(0,str(root));sys.path.insert(0,str(root/'stability'));sys.path.insert(0,str(root/'predictive_revision'))
from granulation_reference import TrainingGeometry
from fast_granulate import granulate
from reduction_reference import heuristic as pure
from revised_search import heuristic as guided
sys.path.insert(0,str(root/'refinement_pilot'))
from refine import refine
protocol=dict(datasets=['Natural Scene','SBU_3DFE','Yeast_cold'],repeats=[2,3],development_sizes=[576],search_audit_ratio=3,refinement_budget=200,scope='Frozen 576-size replication on saved repeats2/3, nested 3:1 split; frozen rules; no outer test',rules='Strictly lower cost and same exact prediction guard; deletions and cheaper one-for-one exchanges, descending saving/removed/added order; best evaluated feasible cost then loss then indices; restart after acceptance; total 200 candidate requests per variant, including repeated/cache-hit requests',fairness='Identical refinement cap, not equal total initial search work or runtime',parameters='Unchanged beta1,zeta.1,kappa.1,smin2,alpha1,lambda-1,k5,three folds,prediction tolerance0')
protocol['holdout_rule']='Saved per-repeat training prefixes checked against their respective 192-object baselines. Seed20260930+1000*dataset_index+repeat; independently permute successive192-position blocks, assign144 search/48 audit per block. First block reproduces previous192 partition; search/audit memberships nested across sizes. Three consecutive equal validation blocks on search. Same200 refinement cap.'
(out/'protocol.json').write_text(json.dumps(protocol,indent=2));rows=[];checks=[];auditrows=[]
for name in protocol['datasets']:
 for rep in protocol['repeats']:
  size=576
  started=time.perf_counter();folder=root/'stability'/f'repeat_{rep}_n{size}' if name=='Natural Scene' else root/'cross_dataset'/name/f'repeat_{rep}_n{size}';split=json.loads((folder/'split.json').read_text());ids=np.array(split['train_indices']);assert set(ids).isdisjoint(split['test_indices'])
  if name=='Natural Scene':X=np.load(root/'pilot/data/feature.npy')[ids];L=np.load(root/'pilot/data/label.npy')[ids];costs=np.array(json.loads((root/'stability/protocol.json').read_text())['costs'])
  else:X=np.load(root/'cross_dataset/data'/f'{name}_X.npy')[ids];L=np.load(root/'cross_dataset/data'/f'{name}_L.npy')[ids];costs=np.array(split['costs'])
  development_ids=ids.copy();rng=np.random.default_rng(20260930+1000*protocol['datasets'].index(name)+rep);orders=[rng.permutation(192)+offset for offset in range(0,size,192)];searchpos=np.concatenate([v[:144] for v in orders]);auditpos=np.concatenate([v[144:] for v in orders]);baseline=json.loads((root/'internal_holdout_audit'/(name.replace(' ','_')+f'_repeat_{rep}_partition.json')).read_text());assert ids[searchpos[:144]].tolist()==baseline['search_ids'] and ids[auditpos[:48]].tolist()==baseline['audit_ids'];fold_n=size//4;auditX=X[auditpos].copy();auditL=L[auditpos].copy();X=X[searchpos];L=L[searchpos];ids=ids[searchpos];assert set(ids).isdisjoint(development_ids[auditpos]);split['fold_validation_positions']=[list(range(j*fold_n,(j+1)*fold_n)) for j in range(3)];n,m=X.shape;blocks=[]
  (out/(name.replace(' ','_')+f'_repeat_{rep}_n{size}_partition.json')).write_text(json.dumps(dict(search_ids=ids.tolist(),audit_ids=development_ids[auditpos].tolist(),search_fold_validation_positions=split['fold_validation_positions']),indent=2))
  for v in split['fold_validation_positions']:
   v=np.array(v);fit=np.setdiff1d(np.arange(n),v);lo=X[fit].min(0);span=X[fit].max(0)-lo;span[span==0]=1.;z=(X[fit]-lo)/span;q=(X[v]-lo)/span;blocks.append((((q[:,None,:]-z[None,:,:])**2).transpose(2,0,1).copy(),fit,v))
  model=TrainingGeometry(X,L,kinds=['numerical']*m);dcache={}
  def dep(P):
   key=tuple(sorted(P))
   if key not in dcache:d,k=model.subset(key);dcache[key]=granulate(d,L,k)[2]
   return dcache[key]
  job=out/name.replace(' ','_')/f'repeat_{rep}_n{size}';job.mkdir(parents=True,exist_ok=True)
  for method in ['Prediction only','Guided']:
   cache={};calls=[0]
   def risk(P):
    key=tuple(sorted(P));calls[0]+=1
    if key not in cache:
     ss=0.
     for mat,fit,v in blocks:
      pred=L[fit][np.argsort(mat[list(key)].sum(0),axis=1,kind='stable')[:,:5]].mean(1) if key else np.tile(L[fit].mean(0),(len(v),1));ss+=np.sum((L[v]-pred)**2)
     cache[key]=float(ss/n)
    return cache[key]
   t=time.perf_counter();limit=risk(range(m))
   if method=='Prediction only':result=pure(costs,lambda P:0.,prediction_loss=risk);P=result[0];initial_trace=result[3]
   else:result=guided(costs,dep,risk);P=result[0];initial_trace=result[3]
   initial_seconds=time.perf_counter()-t;initial_evals=len(cache);assert risk(P)<=limit and all(risk(set(P)-{a})>limit for a in P)
   if False:
    saved=json.loads((root/'neighbour_audit'/name.replace(' ','_')/'fixed_subsets.json').read_text());assert list(P)==saved['selected'][method]
   t=time.perf_counter();Q,log=refine(P,costs,risk,limit,200);refine_seconds=time.perf_counter()-t
   r=dict(dataset=name,repeat=rep,development_size=size,search_size=3*size//4,audit_size=size//4,method=method,before_attributes=len(P),after_attributes=len(Q),before_cost=int(costs[list(P)].sum()),after_cost=int(costs[list(Q)].sum()),before_loss=risk(P),after_loss=risk(Q),full_reference=limit,refinement_evaluations=log['evaluations'],accepted_moves=len(log['accepted']),stop_reason=log['stop_reason'],initial_unique_loss_evaluations=initial_evals,total_unique_loss_evaluations=len(cache),initial_seconds=initial_seconds,refinement_seconds=refine_seconds)
   assert r['after_cost']<=r['before_cost'] and r['after_loss']<=limit
   (job/(method.replace(' ','_')+'.json')).write_text(json.dumps(dict(before=list(P),after=list(Q),initial_trace=initial_trace,refinement=log,results=r),indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x));rows.append(r);(out/'results.json').write_text(json.dumps(rows,indent=2));print('DONE',r,flush=True)
  # Both methods complete selection before evaluation on audit labels.
  checkpoints={method:json.loads((job/(method.replace(' ','_')+'.json')).read_text()) for method in ['Prediction only','Guided']}
  selected={'Full':list(range(m)),'Empty':[]}
  for method,q in checkpoints.items():selected[method+' before']=q['before'];selected[method+' after']=q['after']
  lo=X.min(0);span=X.max(0)-lo;span[span==0]=1.;z=(X-lo)/span;query=(auditX-lo)/span
  from scipy.spatial.distance import cdist
  for method,P in selected.items():
   pred=L[np.argsort(cdist(query[:,P],z[:,P],'sqeuclidean'),axis=1,kind='stable')[:,:5]].mean(1) if P else np.tile(L.mean(0),(len(auditL),1));loss=float(np.mean(np.sum((auditL-pred)**2,axis=1)));auditrows.append(dict(dataset=name,repeat=rep,development_size=size,search_size=3*size//4,audit_size=size//4,method=method,attributes=len(P),cost=int(costs[P].sum()),audit_brier=loss))
  (job/'audit_results.json').write_text(json.dumps(auditrows[-6:],indent=2));(out/'audit_results.json').write_text(json.dumps(auditrows,indent=2));print('AUDIT',name,rep,size,[(r['method'],r['audit_brier']) for r in auditrows[-6:]],flush=True)
  checks.append(dict(dataset=name,repeat=rep,development_size=size,search_ids=ids.tolist(),audit_ids=development_ids[auditpos].tolist(),fold_validation_positions=split['fold_validation_positions']))
  del model,blocks,dcache,cache
  import gc
  gc.collect()
with (out/'results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'validation.json').write_text(json.dumps(dict(runs=12,audit_records=36,nested_membership_verified=True,audit_excluded_from_selection=True,all_before_single_deletion_irreducible=True,all_after_prediction_feasible=True,all_accepted_moves_strict_cost_decrease=True,budget_cap_obeyed=all(r['refinement_evaluations']<=200 for r in rows),core_parameters_changed=False,search_training_size_changed=True,outer_test_used=False,provenance=checks),indent=2))

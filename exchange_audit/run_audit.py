"""Fixed-base cost-reducing one-for-one exchanges; Euclidean original predictor.
Exhaustive local diagnostic plus deterministic equal-budget prefixes, no new search.
"""
from pathlib import Path
import json,csv,sys,time
import numpy as np
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent
sys.path.insert(0,str(root));sys.path.insert(0,str(root/'stability'))
from granulation_reference import TrainingGeometry
from fast_granulate import granulate
previous=list(csv.DictReader(open(root/'neighbour_audit/results.csv')));rows=[];summary=[];checks=[]
for name in ['Natural Scene','SBU_3DFE','Yeast_cold']:
 folder=root/'stability/repeat_1_n192' if name=='Natural Scene' else root/'cross_dataset'/name/'repeat_1_n192';split=json.loads((folder/'split.json').read_text());ids=np.array(split['train_indices']);saved=json.loads((root/'neighbour_audit'/name.replace(' ','_')/'fixed_subsets.json').read_text());costs=np.array(saved['costs']);bases={b:saved['selected'][b] for b in ['Prediction only','Guided']}
 if name=='Natural Scene':X=np.load(root/'pilot/data/feature.npy')[ids];L=np.load(root/'pilot/data/label.npy')[ids]
 else:X=np.load(root/'cross_dataset/data'/f'{name}_X.npy')[ids];L=np.load(root/'cross_dataset/data'/f'{name}_L.npy')[ids]
 n,m=X.shape;blocks=[]
 for v in split['fold_validation_positions']:
  v=np.array(v);fit=np.setdiff1d(np.arange(n),v);assert set(ids[fit]).isdisjoint(ids[v]) and set(ids).isdisjoint(split['test_indices']);lo=X[fit].min(0);span=X[fit].max(0)-lo;span[span==0]=1;z=(X[fit]-lo)/span;q=(X[v]-lo)/span;blocks.append((((q[:,None,:]-z[None,:,:])**2).transpose(2,0,1).copy(),fit,v))
 cache={}
 def risk(P):
  key=tuple(sorted(P))
  if key not in cache:
   ss=0
   for mat,fit,v in blocks:
    pred=L[fit][np.argsort(mat[list(key)].sum(0),axis=1,kind='stable')[:,:5]].mean(1) if key else np.tile(L[fit].mean(0),(len(v),1));ss+=np.sum((L[v]-pred)**2)
   cache[key]=float(ss/n)
  return cache[key]
 ref=float(next(r['inner_brier'] for r in previous if r['dataset']==name and r['subset']=='Full' and r['learner']=='Euclidean'));assert abs(risk(range(m))-ref)<1e-12
 candidates={b:sorted([(int(costs[a]-costs[t]),a,t) for a in P for t in range(m) if t not in P and costs[t]<costs[a]],key=lambda x:(-x[0],x[1],x[2])) for b,P in bases.items()};budget=min(map(len,candidates.values()));model=TrainingGeometry(X,L,kinds=['numerical']*m)
 def gamma(P):d,k=model.subset(P);return granulate(d,L,k)[2]
 for b,P in bases.items():
  base=risk(P);old=float(next(r['inner_brier'] for r in previous if r['dataset']==name and r['subset']==b and r['learner']=='Euclidean'));assert abs(base-old)<1e-12 and base<=ref
  local=[]
  for i,(saving,a,t) in enumerate(candidates[b]):
   C=sorted((set(P)-{a})|{t});r=risk(C);record=dict(dataset=name,base=b,rank=i+1,removed=a,added=t,saving=saving,base_cost=int(costs[P].sum()),candidate_cost=int(costs[P].sum()-saving),base_brier=base,candidate_brier=r,prediction_gain=base-r,full_reference=ref,feasible=r<=ref,matched_budget=i<budget,subset=json.dumps(C));rows.append(record);local.append(record)
  def best(q):
   feasible=[r for r in q if r['feasible']]
   if not feasible:return None
   r=min(feasible,key=lambda r:(r['candidate_cost'],r['candidate_brier'],r['removed'],r['added']));result=dict(r);result['candidate_gamma']=gamma(json.loads(r['subset']));return result
  s=dict(dataset=name,base=b,selected=P,base_cost=int(costs[P].sum()),base_brier=base,base_gamma=gamma(P),exchanges=len(local),feasible_exchanges=sum(r['feasible'] for r in local),matched_budget=budget,matched_feasible=sum(r['feasible'] for r in local[:budget]),best_exhaustive=best(local),best_matched=best(local[:budget]));summary.append(s);print('DONE',json.dumps(s),flush=True)
 checks.append(dict(dataset=name,training_ids=ids.tolist(),unique_loss_evaluations=len(cache)))
with (out/'candidates.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'validation.json').write_text(json.dumps(dict(base_losses_reproduced=6,full_references_reproduced=3,tolerance=1e-12,parameters_changed=False,outer_test_used=False,new_reduct_search=False,training_provenance=checks),indent=2))
(out/'protocol.json').write_text(json.dumps(dict(scope='All strictly cost-reducing one-for-one exchanges from fixed saved bases; no sequential adoption',predictor='Original Euclidean 5NN, saved three folds, fitting-only minmax',feasibility='Same full-set internal Brier reference, exact <=, prediction tolerance 0',ordering='Descending cost saving, then removed attribute index, then added index; independent of labels and candidate loss',matched_budget='Per dataset minimum candidate count across two bases; prefix comparisons; report exhaustion separately',choice='Feasible minimum cost, then loss, then indices; diagnostic best only',limitations='Development reused; no independent generalization, no global optimality; Yeast empty has no exchange'),indent=2))

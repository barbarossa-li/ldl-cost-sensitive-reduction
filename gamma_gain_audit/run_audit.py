"""Actual search-scale training Gamma versus saved conditional inner loss.
No new selection, test evaluation, parameter change or fold averaging of Gamma.
"""
from pathlib import Path
import sys,json,csv
import numpy as np
from scipy.stats import spearmanr
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent
sys.path.insert(0,str(root));sys.path.insert(0,str(root/'stability'))
from granulation_reference import TrainingGeometry,granulate as reference
from fast_granulate import granulate
old=list(csv.DictReader(open(root/'conditional_audit/candidates.csv')));rows=[];summ=[];checks=[]
for name in ['Natural Scene','SBU_3DFE','Yeast_cold']:
 folder=root/'stability/repeat_1_n192' if name=='Natural Scene' else root/'cross_dataset'/name/'repeat_1_n192'
 split=json.loads((folder/'split.json').read_text());ids=split['train_indices'];assert set(ids).isdisjoint(split['test_indices'])
 if name=='Natural Scene':X=np.load(root/'pilot/data/feature.npy')[ids];L=np.load(root/'pilot/data/label.npy')[ids]
 else:X=np.load(root/'cross_dataset/data'/f'{name}_X.npy')[ids];L=np.load(root/'cross_dataset/data'/f'{name}_L.npy')[ids]
 saved=json.loads((root/'neighbour_audit'/name.replace(' ','_')/'fixed_subsets.json').read_text());model=TrainingGeometry(X,L,kinds=['numerical']*X.shape[1]);cache={}
 def dep(P):
  key=tuple(sorted(P))
  if key not in cache:
   d,k=model.subset(key);leaves,accepted,g,h=granulate(d,L,k);p=L.mean(0);v0=np.mean(np.sum((L-p)**2,axis=1));pred=np.tile(p,(len(L),1))
   for B in accepted:pred[B]=L[B].mean(0)
   identity=1-np.mean(np.sum((L-pred)**2,axis=1))/v0 if v0 else 0
   assert abs(g-identity)<1e-12
   cache[key]=g
  return cache[key]
 for b in ['Prediction only','Guided']:
  P=saved['selected'][b];base=dep(P);d,k=model.subset(P);assert abs(base-reference(d,L,k)[2])<1e-12
  if b=='Guided':assert abs(base-json.loads((root/'predictive_revision'/name.replace(' ','_')/'selection_before_test.json').read_text())['gamma'])<1e-12
  for a in range(X.shape[1]):
   C=sorted(set(P)^{a});g=dep(C);delta=g-base
   for r in old:
    if r['dataset']==name and r['base']==b and int(r['attribute'])==a:
     q=dict(r);q.update(base_gamma=base,candidate_gamma=g,gamma_change=delta);rows.append(q)
  for method in ['Euclidean','Manhattan','Incoming directed']:
   local=[r for r in rows if r['dataset']==name and r['base']==b and r['learner']==method and r['operation']=='add'];gx=np.array([float(r['gamma_change']) for r in local]);gy=np.array([float(r['gain']) for r in local]);pos=gx>1e-12;neg=gx < -1e-12;zero=~(pos|neg)
   corr=float(spearmanr(gx,gy).statistic) if np.ptp(gx)>0 else None
   s=dict(dataset=name,base=b,learner=method,base_gamma=base,additions=len(local),spearman_gamma_vs_prediction=corr,positive_gamma=int(pos.sum()),negative_gamma=int(neg.sum()),zero_gamma=int(zero.sum()),positive_both=int((pos&(gy>1e-12)).sum()),positive_gamma_prediction_worse=int((pos&(gy < -1e-12)).sum()),negative_gamma_prediction_better=int((neg&(gy>1e-12)).sum()),zero_gamma_prediction_better=int((zero&(gy>1e-12)).sum()))
   summ.append(s)
  print(name,b,[s for s in summ if s['dataset']==name and s['base']==b and s['learner']=='Euclidean'],flush=True)
 checks.append(dict(dataset=name,unique_gamma_evaluations=len(cache),training_ids=ids))
with (out/'candidates.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'summary.json').write_text(json.dumps(summ,indent=2));(out/'validation.json').write_text(json.dumps(dict(contexts=18,records=len(rows),base_reference_checks=6,all_training_risk_identities_verified=True,maximum_identity_tolerance=1e-12,training_provenance=checks,outer_test_used=False,new_selection=False,parameters_changed=False),indent=2))
(out/'protocol.json').write_text(json.dumps(dict(gamma='192-object development training Gamma, exactly the search evaluator; not validation Gamma',prediction='Saved three-fold fitting-only conditional losses, no recomputation',fixed_bases=True,tolerance=1e-12,limitations='Bases and development folds reused; descriptive, correlated candidate records; identical empty Yeast bases duplicated'),indent=2))

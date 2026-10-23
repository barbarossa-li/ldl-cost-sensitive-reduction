"""Frozen development selections evaluated on reused saved outer tests.
No rule or subset changes, no use of test labels in fitting.
"""
from pathlib import Path
import json,csv,hashlib
import numpy as np
from scipy.spatial.distance import cdist
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent
protocol=dict(datasets=['Natural Scene','SBU_3DFE','Yeast_cold'],repeats=list(range(1,6)),train_n=192,test_n=256,methods=['Full','Empty','Prediction only before','Prediction only after','Guided before','Guided after'],predictor='Original Euclidean 5NN, full-training minmax, stable fitting-row-order ties, arithmetic mean label vectors',metrics='Brier primary; TV, KL, JS secondary; lower is better; natural logarithms for KL/JS; prediction floor1e-12 and renormalization matches prior test reports',status='Exploratory reused-test evaluation, not new confirmation; checkpoint all selections before predictions; no retuning or subset updates')
(out/'protocol.json').write_text(json.dumps(protocol,indent=2));manifest=[]
for name in protocol['datasets']:
 for rep in protocol['repeats']:
  parent=root/'refinement_pilot' if rep<=3 else root/'refinement_replication';job=parent/name.replace(' ','_')/f'repeat_{rep}';selected={};source={}
  for method in ['Prediction only','Guided']:
   p=job/(method.replace(' ','_')+'.json');q=json.loads(p.read_text());source[method]=dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest());selected[method+' before']=q['before'];selected[method+' after']=q['after']
  folder=root/'stability'/f'repeat_{rep}_n192' if name=='Natural Scene' else root/'cross_dataset'/name/f'repeat_{rep}_n192';split=json.loads((folder/'split.json').read_text());manifest.append(dict(dataset=name,repeat=rep,split=str(folder.relative_to(root)/'split.json'),selected=selected,sources=source))
(out/'selection_checkpoint_before_test.json').write_text(json.dumps(manifest,indent=2))
rows=[];validation=[]
for entry in manifest:
 name=entry['dataset'];rep=entry['repeat'];folder=root/Path(entry['split']).parent;split=json.loads((folder/'split.json').read_text());train=split['train_indices'];test=split['test_indices'];assert len(train)==192 and len(test)==256 and set(train).isdisjoint(test)
 if name=='Natural Scene':X=np.load(root/'pilot/data/feature.npy');L=np.load(root/'pilot/data/label.npy');costs=np.array(json.loads((root/'stability/protocol.json').read_text())['costs']);prior=json.loads((root/'baselines'/f'repeat_{rep}_n192/results.json').read_text())
 else:X=np.load(root/'cross_dataset/data'/f'{name}_X.npy');L=np.load(root/'cross_dataset/data'/f'{name}_L.npy');costs=np.array(split['costs']);prior=json.loads((folder/'results.json').read_text())
 x=X[train];lab=L[train];lo=x.min(0);span=x.max(0)-lo;span[span==0]=1.;z=(x-lo)/span;query=(X[test]-lo)/span;truth=L[test];selected={'Full':list(range(x.shape[1])),'Empty':[],**entry['selected']};cache={}
 for method in protocol['methods']:
  P=selected[method];key=tuple(P)
  if key not in cache:
   raw=lab[np.argsort(cdist(query[:,P],z[:,P],'sqeuclidean'),axis=1,kind='stable')[:,:5]].mean(1) if P else np.tile(lab.mean(0),(len(test),1));pred=np.maximum(raw,1e-12);pred/=pred.sum(1,keepdims=True);mid=(truth+pred)/2
   kl=lambda a,b:np.sum(np.where(a>0,a*np.log(np.maximum(a,1e-300)/b),0),axis=1)
   cache[key]=dict(test_brier=float(np.mean(np.sum((truth-pred)**2,axis=1))),test_tv=float(np.mean(.5*np.abs(truth-pred).sum(1))),test_kl=float(np.mean(kl(truth,pred))),test_js=float(np.mean(.5*kl(truth,mid)+.5*kl(pred,mid))))
  r=dict(dataset=name,repeat=rep,method=method,attributes=len(P),cost=int(costs[P].sum()),**cache[key]);rows.append(r)
  oldname={'Full':'All attributes','Empty':'Global mean','Prediction only before':'Prediction-only'}.get(method)
  if oldname:
   old=next((v for v in prior if v['method']==oldname),None)
   if old:
    assert abs(r['test_brier']-old['test_brier'])<1e-12;validation.append(dict(dataset=name,repeat=rep,method=method,brier_error=abs(r['test_brier']-old['test_brier'])))
  if method=='Guided before' and rep==1:
   old=json.loads((root/'predictive_revision'/name.replace(' ','_')/'results.json').read_text())[-1];assert abs(r['test_brier']-old['test_brier'])<1e-12;validation.append(dict(dataset=name,repeat=rep,method=method,brier_error=abs(r['test_brier']-old['test_brier'])))
 # Ensure all immutable source checkpoints remain unchanged.
 for source in entry['sources'].values():assert hashlib.sha256((root/source['path']).read_bytes()).hexdigest()==source['sha256']
 print('DONE',name,rep,[(r['method'],r['test_brier']) for r in rows if r['dataset']==name and r['repeat']==rep],flush=True)
with (out/'results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'results.json').write_text(json.dumps(rows,indent=2));(out/'validation.json').write_text(json.dumps(dict(records=len(rows),saved_brier_checks=validation,all_split_train_test_disjoint=True,source_checkpoints_unchanged=True,fit_uses_training_only=True,test_used_for_evaluation_only=True,retuning=False,new_selection=False),indent=2))

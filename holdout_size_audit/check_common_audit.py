"""Same original48 audit objects at each training scale; frozen checkpoints."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.distance import cdist
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent;rows=[]
for name in ['Natural Scene','SBU_3DFE','Yeast_cold']:
 first=json.loads((root/'internal_holdout_audit'/(name.replace(' ','_')+'_repeat_1_partition.json')).read_text());audit=first['audit_ids']
 if name=='Natural Scene':X=np.load(root/'pilot/data/feature.npy');L=np.load(root/'pilot/data/label.npy')
 else:X=np.load(root/'cross_dataset/data'/f'{name}_X.npy');L=np.load(root/'cross_dataset/data'/f'{name}_L.npy')
 for size in [192,384,576]:
  parent=root/'internal_holdout_audit' if size==192 else out;file=name.replace(' ','_')+'_repeat_1'+('' if size==192 else f'_n{size}')+'_partition.json';partition=json.loads((parent/file).read_text());ids=partition['search_ids'];assert set(ids).isdisjoint(audit) and partition['audit_ids'][:48]==audit
  job=parent/name.replace(' ','_')/('repeat_1' if size==192 else f'repeat_1_n{size}');selected={'Full':list(range(X.shape[1])),'Empty':[]}
  for method in ['Prediction only','Guided']:selected[method+' after']=json.loads((job/(method.replace(' ','_')+'.json')).read_text())['after']
  x=X[ids];lab=L[ids];lo=x.min(0);span=x.max(0)-lo;span[span==0]=1;z=(x-lo)/span;query=(X[audit]-lo)/span;truth=L[audit]
  for method,P in selected.items():
   pred=lab[np.argsort(cdist(query[:,P],z[:,P],'sqeuclidean'),axis=1,kind='stable')[:,:5]].mean(1) if P else np.tile(lab.mean(0),(48,1));rows.append(dict(dataset=name,development_size=size,search_size=len(ids),audit_size=48,method=method,brier=float(np.mean(np.sum((truth-pred)**2,axis=1)))))
(out/'common48_results.json').write_text(json.dumps(rows,indent=2));print('Common audit records',len(rows))

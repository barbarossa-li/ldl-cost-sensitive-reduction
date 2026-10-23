from pathlib import Path
import json
import numpy as np
from scipy.spatial.distance import cdist
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent
rows=[dict(v,repeat=1) for v in json.loads((root/'holdout_size_audit/common48_results.json').read_text()) if v['development_size'] in [192,576]]
for name in ['Natural Scene','SBU_3DFE','Yeast_cold']:
 if name=='Natural Scene':X=np.load(root/'pilot/data/feature.npy');L=np.load(root/'pilot/data/label.npy')
 else:X=np.load(root/'cross_dataset/data'/f'{name}_X.npy');L=np.load(root/'cross_dataset/data'/f'{name}_L.npy')
 for rep in [2,3]:
  old=root/'internal_holdout_audit';base=json.loads((old/(name.replace(' ','_')+f'_repeat_{rep}_partition.json')).read_text());audit=base['audit_ids']
  for size in [192,576]:
   parent=old if size==192 else out;partfile=name.replace(' ','_')+f'_repeat_{rep}'+('' if size==192 else '_n576')+'_partition.json';partition=json.loads((parent/partfile).read_text());ids=partition['search_ids'];assert set(ids).isdisjoint(audit) and partition['audit_ids'][:48]==audit
   job=parent/name.replace(' ','_')/(f'repeat_{rep}' if size==192 else f'repeat_{rep}_n576');selected={'Full':list(range(X.shape[1])),'Empty':[]}
   for method in ['Prediction only','Guided']:selected[method+' after']=json.loads((job/(method.replace(' ','_')+'.json')).read_text())['after']
   x=X[ids];lab=L[ids];lo=x.min(0);span=x.max(0)-lo;span[span==0]=1.;z=(x-lo)/span;query=(X[audit]-lo)/span
   for method,P in selected.items():
    pred=lab[np.argsort(cdist(query[:,P],z[:,P],'sqeuclidean'),axis=1,kind='stable')[:,:5]].mean(1) if P else np.tile(lab.mean(0),(48,1));rows.append(dict(dataset=name,repeat=rep,development_size=size,search_size=len(ids),audit_size=48,method=method,brier=float(np.mean(np.sum((L[audit]-pred)**2,axis=1)))))
(out/'common48_results.json').write_text(json.dumps(rows,indent=2));print('common48 semantic records',len(rows))

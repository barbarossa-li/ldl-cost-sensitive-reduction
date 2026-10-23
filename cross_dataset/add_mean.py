from pathlib import Path
import json,csv
import numpy as np
p=Path(__file__).resolve().parent;allrows=[]
for name in ['SBU_3DFE','Yeast_cold']:
 L=np.load(p/'data'/f'{name}_L.npy')
 for rep in range(1,6):
  for n in [192,384,576]:
   job=p/name/f'repeat_{rep}_n{n}';assert (job/'complete.json').exists();rows=json.loads((job/'results.json').read_text())
   if not any(r['method']=='Global mean' for r in rows):
    s=json.loads((job/'split.json').read_text());lab=L[s['train_indices']];truth=L[s['test_indices']];cp={'selected':[],'rule':'fixed zero-feature control, no selection/tuning'};(job/'global_mean_before_test.json').write_text(json.dumps(cp,indent=2));risk=0.
    for val in map(np.array,s['fold_validation_positions']):
     fit=np.setdiff1d(np.arange(n),val);risk+=np.sum((lab[val]-lab[fit].mean(0))**2)
    pred=np.tile(lab.mean(0),(256,1));pred=np.maximum(pred,1e-12);pred/=pred.sum(1,keepdims=True);mid=(truth+pred)/2
    row={'dataset':name,'repeat':rep,'train_n':n,'method':'Global mean','attributes':0,'cost':0,'inner_brier':float(risk/n),'gamma':0.,'coverage':0.,'test_brier':float(np.mean(np.sum((truth-pred)**2,axis=1))),'test_kl':float(np.mean(np.sum(truth*np.log(np.maximum(truth,1e-300)/pred),axis=1))),'test_tv':float(np.mean(.5*np.abs(truth-pred).sum(1))),'test_js':float(np.mean(.5*np.sum(truth*np.log(np.maximum(truth,1e-300)/mid),axis=1)+.5*np.sum(pred*np.log(pred/mid),axis=1)))};rows.append(row);(job/'results.json').write_text(json.dumps(rows,indent=2))
   allrows.extend(rows)
with (p/'results.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)

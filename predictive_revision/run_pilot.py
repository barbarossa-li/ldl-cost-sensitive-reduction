"""Three predeclared repeat-1 n192 saved splits; development diagnostic only."""
from pathlib import Path
import sys,json,time,gc
import numpy as np
from scipy.spatial.distance import cdist
from revised_search import heuristic
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent
sys.path.insert(0,str(root));sys.path.insert(0,str(root/'stability'))
from granulation_reference import TrainingGeometry
from fast_granulate import granulate

def nearest5(d):
 return np.argsort(d,axis=1,kind='stable')[:,:5]
protocol={'scope':'Development pilot on three already examined saved repeat-1 n192 splits; no fresh confirmatory test data','datasets':['Natural Scene','SBU_3DFE','Yeast_cold'],'train_n':192,'repeat':1,'fixed_rule':'Positive predictive-gain candidates; among them prefer positive dependency gain/cost. Otherwise positive predictive gain/cost. Nonpositive predictive bridge if none. Prediction-only stop and deletion.','lambda':-1,'prediction_tolerance':0,'inner_folds':3,'k':5,'geometry_parameters':'Original beta=1,zeta=.1,kappa=.1,smin=2,alpha=1','empty_subset':'Training global mean; accepted if inner guard passes','comparisons':'Saved original Joint guard and Prediction-only outputs; all use same splits, scaling, folds and costs','test_role':'Recorded after the selection checkpoint; descriptive only, not independent evidence or a tuning signal'}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2));rows=[]
for name in protocol['datasets']:
 started=time.perf_counter();folder=root/'stability/repeat_1_n192' if name=='Natural Scene' else root/'cross_dataset'/name/'repeat_1_n192'
 split=json.loads((folder/'split.json').read_text());train=split['train_indices'];test=split['test_indices']
 if name=='Natural Scene':
  X=np.load(root/'pilot/data/feature.npy');L=np.load(root/'pilot/data/label.npy');costs=np.array(json.loads((root/'stability/protocol.json').read_text())['costs']);prior=json.loads((root/'baselines/repeat_1_n192/results.json').read_text())
 else:
  X=np.load(root/'cross_dataset/data'/f'{name}_X.npy');L=np.load(root/'cross_dataset/data'/f'{name}_L.npy');costs=np.array(split['costs']);prior=json.loads((folder/'results.json').read_text())
 x=X[train];lab=L[train];n,m=x.shape;blocks=[];cache={};dcache={};job=out/name.replace(' ','_');job.mkdir(exist_ok=True)
 for v in split['fold_validation_positions']:
  v=np.array(v);fit=np.setdiff1d(np.arange(n),v);a=x[fit];b=x[v];lo=a.min(0);span=a.max(0)-lo;span[span==0]=1.;a=(a-lo)/span;b=(b-lo)/span;blocks.append((((b[:,None,:]-a[None,:,:])**2).transpose(2,0,1).copy(),fit,v))
 def risk(P):
  key=tuple(sorted(P))
  if key not in cache:
   ss=0.
   for mat,fit,v in blocks:
    pred=lab[fit][nearest5(mat[list(key)].sum(0))].mean(1) if key else np.tile(lab[fit].mean(0),(len(v),1));ss+=np.sum((lab[v]-pred)**2)
   cache[key]=float(ss/n)
  return cache[key]
 model=TrainingGeometry(x,lab,kinds=['numerical']*m)
 def dep(P):
  key=tuple(sorted(P))
  if key not in dcache:
   d,k=model.subset(key);dcache[key]=granulate(d,lab,k)[2]
  return dcache[key]
 full=next(r for r in prior if r['method']=='All attributes');assert abs(risk(range(m))-full['inner_brier'])<1e-12
 sel,c,limit,trace=heuristic(costs,dep,risk);g=dep(sel)
 (job/'selection_before_test.json').write_text(json.dumps({'selected':sel,'cost':c,'inner_brier':risk(sel),'inner_reference':limit,'gamma':g,'original_full_gamma':dep(range(m)),'trace':trace,'split_source':str(folder.relative_to(root))},indent=2))
 # Selection is completed and checkpointed before saved-test predictions below.
 lo=x.min(0);span=x.max(0)-lo;span[span==0]=1.;a=(x-lo)/span;b=(X[test]-lo)/span
 pred=lab[nearest5(cdist(b[:,sel],a[:,sel],'sqeuclidean'))].mean(1) if sel else np.tile(lab.mean(0),(len(test),1))
 pred=np.maximum(pred,1e-12);pred/=pred.sum(1,keepdims=True);loss=float(np.mean(np.sum((L[test]-pred)**2,axis=1)))
 local=[]
 for method in ['All attributes','Joint guard','Prediction-only']:
  r=next(v for v in prior if v['method']==method);local.append({'dataset':name,'method':method,'attributes':r['attributes'],'cost':r['cost'],'inner_brier':r['inner_brier'],'test_brier':r['test_brier']})
 local.append({'dataset':name,'method':'Predictive dependency guidance','attributes':len(sel),'cost':c,'inner_brier':risk(sel),'test_brier':loss,'gamma':g})
 (job/'results.json').write_text(json.dumps(local,indent=2));(job/'runtime.json').write_text(json.dumps({'seconds':time.perf_counter()-started,'loss_evaluations':len(cache),'dependency_evaluations':len(dcache)},indent=2));rows.extend(local)
 (out/'results.json').write_text(json.dumps(rows,indent=2));print('DONE',name,local[-1], 'time',time.perf_counter()-started,flush=True)
 del model,blocks;gc.collect()

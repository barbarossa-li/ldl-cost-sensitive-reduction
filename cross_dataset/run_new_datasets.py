"""Prepare new PyLDL datasets (Human_Gene, M2B) and run the frozen cross-dataset protocol.

Stage 1 (prepare): convert .mat to {ds}_X.npy / {ds}_L.npy with rows normalized to
  sum one (matching the existing cross_dataset/data conventions; verify against
  SBU_3DFE and Yeast_cold .mat files).
Stage 2 (splits + joint method): mirror cross_dataset/run_cross.py exactly, writing
  to new_datasets/<ds>/repeat_*/... with per-split checkpoints.
Reproduction: python run_new_datasets.py   (stage 1 runs first, then stage 2 loops)
"""
from pathlib import Path
import sys,json,csv,time,gc
import numpy as np
from scipy.spatial.distance import cdist

root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root));sys.path.insert(0,str(root/'stability'))
here=Path(__file__).resolve().parent
datadir=root/'cross_dataset'/'data'
NEW=['Human_Gene','M2B']

def prepare():
 import scipy.io as sio
 # Verify convention on existing files first
 for ds in ['SBU_3DFE','Yeast_cold']:
  mat=sio.loadmat(datadir/f'{ds}.mat');Xm=mat['features'];Lm=mat['labels']
  Xn=np.load(datadir/f'{ds}_X.npy');Ln=np.load(datadir/f'{ds}_L.npy')
  assert Xm.shape==Xn.shape and Lm.shape==Ln.shape,(ds,Xm.shape,Xn.shape)
  assert np.allclose(Xm,Xn) and np.allclose(Lm,Ln),(ds,'values differ')
  print(ds,'convention OK: features/labels keys, shapes',Xm.shape,Lm.shape)
 for ds in NEW:
  src=datadir/f'{ds}.mat'
  if not src.exists():print(ds,'.mat missing, skip');continue
  if (datadir/f'{ds}_X.npy').exists():print(ds,'npy exists, skip');continue
  mat=sio.loadmat(src);X=mat['features'];L=mat['labels']
  # normalize rows of L to sum one (safety; PyLDL distributions should already)
  s=L.sum(1,keepdims=True);s[s==0]=1.;L=L/s
  np.save(datadir/f'{ds}_X.npy',X);np.save(datadir/f'{ds}_L.npy',L)
  print(ds,'prepared',X.shape,L.shape)

def nearest5(d):
 idx=np.argpartition(d,4,axis=1)[:,:5];vals=np.take_along_axis(d,idx,axis=1);th=vals.max(1)
 order=np.lexsort((idx,vals),axis=1);idx=np.take_along_axis(idx,order,axis=1)
 for i in np.flatnonzero((d==th[:,None]).sum(1)>1):
  below=np.flatnonzero(d[i]<th[i]);ties=np.flatnonzero(d[i]==th[i]);v=np.r_[below,ties[:5-len(below)]];idx[i]=v[np.lexsort((v,d[i,v]))]
 return idx

def run(ds):
 from granulation_reference import TrainingGeometry
 from fast_granulate import granulate
 from reduction_reference import heuristic
 out=here/ds
 X=np.load(datadir/f'{ds}_X.npy');L=np.load(datadir/f'{ds}_L.npy');m=X.shape[1]
 costs=np.random.default_rng(20260931).integers(1,11,m)
 for rep in range(1,6):
  seed=20261300+rep;order=np.random.default_rng(seed).permutation(len(X));test=order[:256]
  for n in [192,384,576]:
   job=out/f'repeat_{rep}_n{n}';job.mkdir(parents=True,exist_ok=True)
   if (job/'complete.json').exists():continue
   started=time.perf_counter();train=order[256:256+n];x=X[train];lab=L[train];folds=np.array_split(np.random.default_rng(seed+10000+n).permutation(n),3);cache={};dcache={};blocks=[];model=TrainingGeometry(x,lab,kinds=['numerical']*m)
   (job/'split.json').write_text(json.dumps({'seed':seed,'train_indices':train.tolist(),'test_indices':test.tolist(),'fold_validation_positions':[v.tolist() for v in folds],'costs':costs.tolist()},indent=2))
   assert set(train).isdisjoint(test)
   for val in folds:
    fit=np.setdiff1d(np.arange(n),val);a=x[fit];b=x[val];lo=a.min(0);span=a.max(0)-lo;span[span==0]=1;a=(a-lo)/span;b=(b-lo)/span;mat=((b[:,None,:]-a[None,:,:])**2).transpose(2,0,1).copy();blocks.append((mat,fit,val))
   def risk(P):
    key=tuple(sorted(P))
    if key not in cache:
     total=0
     for mat,fit,val in blocks:
      if key:
       dist=mat[list(key)].sum(0);nn=nearest5(dist);pred=lab[fit][nn].mean(1)
      else:pred=np.tile(lab[fit].mean(0),(len(val),1))
      total+=np.sum((lab[val]-pred)**2)
     cache[key]=float(total/n)
    return cache[key]
   def dep(P):
    key=tuple(sorted(P))
    if key not in dcache:
     d,k=model.subset(key);leaves,eligible,g,h=granulate(d,lab,k);dcache[key]=(g,sum(map(len,eligible))/n)
    return dcache[key][0]
   full=tuple(range(m));gref=dep(full);rref=risk(full);selected={'All attributes':list(full)};traces={}
   for name,d in [('Dependency-only',dep),('Joint guard',dep),('Prediction-only',lambda P:0.)]:
    began=time.perf_counter();h=heuristic(costs,d,prediction_loss=risk)
    selected[name]=list(h[0]);traces[name]={'trace':h[3],'seconds':time.perf_counter()-began}
   lo=x.min(0);span=x.max(0)-lo;span[span==0]=1;a=(x-lo)/span;b=(X[test]-lo)/span;truth=L[test]
   def evaluate(P):
    P=tuple(P)
    if P:
     pred=lab[nearest5(cdist(b[:,P],a[:,P],'sqeuclidean'))].mean(1)
    else:pred=np.tile(lab.mean(0),(len(test),1))
    pred=np.maximum(pred,1e-12);pred/=pred.sum(1,keepdims=True);mid=(truth+pred)/2
    return {'inner_brier':risk(P),'test_brier':float(np.mean(np.sum((truth-pred)**2,axis=1))),'test_kl':float(np.mean(np.sum(truth*np.log(np.maximum(truth,1e-300)/pred),axis=1))),'test_tv':float(np.mean(.5*np.abs(truth-pred).sum(1))),'test_js':float(np.mean(.5*np.sum(truth*np.log(np.maximum(truth,1e-300)/mid),axis=1)+.5*np.sum(pred*np.log(pred/mid),axis=1)))}
   rows=[]
   for name,P in selected.items():
    P=tuple(P);rows.append({'dataset':ds,'repeat':rep,'train_n':n,'method':name,'attributes':len(P),'cost':int(sum(costs[list(P)])),**evaluate(P)})
   (job/'results.json').write_text(json.dumps(rows,indent=2));(job/'selection_before_test.json').write_text(json.dumps({k:v for k,v in selected.items()},indent=2));(job/'search_traces.json').write_text(json.dumps(traces,indent=2))
   (job/'complete.json').write_text(json.dumps({'elapsed_seconds':time.perf_counter()-started},indent=2))
   print(ds,'DONE',rep,n,[(r['method'],r['attributes'],round(r['test_brier'],4)) for r in rows],'%.0fs'%(time.perf_counter()-started),flush=True)
   del blocks,cache,dcache,model;gc.collect()

if __name__ == '__main__':
 prepare()
 for ds in NEW:
  if (datadir/f'{ds}_X.npy').exists():run(ds)
  else:print(ds,'not prepared; run stage 1 first')

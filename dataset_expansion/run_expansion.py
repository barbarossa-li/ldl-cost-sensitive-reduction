"""Dataset expansion for the ASC submission: run the frozen nested audit protocol
on every additional native LDL dataset available from the PyLDL repository.

Stage 1 (prepare): convert each .mat in dataset_expansion/data/ to {stem}_X.npy /
  {stem}_L.npy with label rows normalized to sum one (established convention,
  verified in run_new_datasets.py). Human_Gene and M2B are taken from
  cross_dataset/data so the arrays match the published-baseline records.
Stage 2 (run): for each dataset, three frozen repeats of the nested protocol:
  - development subset: 576 objects (three 192-blocks, 144 search / 48 audit each)
    when the dataset has >= 832 objects; otherwise a single 192-block
    (144 search / 48 audit), matching the internal-holdout pilot design.
  - attributes acquired on a held-out test partition are recorded descriptively
    after the audit checkpoint (never during selection).
  - methods: Prediction-only and Guided constructive search under the identical
    prediction guard, each followed by the frozen 200-evaluation exchange
    refinement (P+ / G+), plus Full and Empty references.
Reproduction: python run_expansion.py            (stage 1, then stage 2)
              python run_expansion.py runonly    (skip stage 1)
Per-job complete.json checkpoints make the sweep resumable.
"""
from pathlib import Path
import sys,json,csv,time,gc
import numpy as np

here=Path(__file__).resolve().parent
root=here.parent
sys.path.insert(0,str(root));sys.path.insert(0,str(root/'stability'));sys.path.insert(0,str(root/'predictive_revision'));sys.path.insert(0,str(root/'refinement_pilot'))
datadir=here/'data'

# Datasets in evaluation order. Human_Gene / M2B / Yeast_spo complete the set that
# the manuscript already references; the rest are new additions.
NEW=['Yeast_spo','Yeast_alpha','Yeast_cdc','Yeast_diau','Yeast_dtt','Yeast_elu','Yeast_heat','Yeast_spo5','Yeast_spoem',
     'FG-Net','SJAFFE','Music','Painting','SCUT_FBP','fbp5500','emotion6','RAF_ML','Movie','Flickr','Twitter','MOR-PH',
     'Human_Gene','M2B']

def safe(stem):return stem.replace('-','_')

def prepare():
 import scipy.io as sio
 import shutil
 # Bring the two cross_dataset .mat files into the expansion data folder.
 for ds in ['Human_Gene','M2B']:
  src=root/'cross_dataset'/'data'/f'{ds}.mat'
  dst=datadir/f'{ds}.mat'
  if src.exists() and not dst.exists():shutil.copy2(src,dst)
 for stem in NEW:
  src=datadir/f'{stem}.mat'
  if not src.exists():print(stem,'.mat missing, skip');continue
  if (datadir/f'{safe(stem)}_X.npy').exists():print(stem,'npy exists, skip');continue
  mat=sio.loadmat(src);X=mat['features'];L=mat['labels']
  if X.ndim!=2 or L.ndim!=2 or len(X)!=len(L):print(stem,'unexpected shapes',X.shape,L.shape,'skip');continue
  s=L.sum(1,keepdims=True);s[s==0]=1.;L=L/s
  np.save(datadir/f'{safe(stem)}_X.npy',X);np.save(datadir/f'{safe(stem)}_L.npy',L)
  print(stem,'prepared',X.shape,L.shape)

def run(stem):
 from granulation_reference import TrainingGeometry
 from fast_granulate import granulate
 from reduction_reference import heuristic as pure
 from revised_search import heuristic as guided
 from refine import refine
 from scipy.spatial.distance import cdist
 ds=safe(stem);out=here/ds
 X=np.load(datadir/f'{ds}_X.npy');L=np.load(datadir/f'{ds}_L.npy');N,m=X.shape
 costs=np.random.default_rng(20260931).integers(1,11,m)
 dev_n=576 if N>=832 else 192
 test_n=min(256,N-dev_n)
 for rep in range(1,4):
  job=out/f'repeat_{rep}';job.mkdir(parents=True,exist_ok=True)
  if (job/'complete.json').exists():print(ds,'repeat',rep,'already complete');continue
  started=time.perf_counter()
  seed=20261300+rep
  order=np.random.default_rng(seed).permutation(N)
  test=order[:test_n];dev=order[test_n:test_n+dev_n]
  # Nested 3:1 search/audit within each 192-block (run_audit convention).
  rng=np.random.default_rng(seed+7000)
  blocks=[rng.permutation(192)+off for off in range(0,dev_n,192)]
  searchpos=np.concatenate([b[:144] for b in blocks]);auditpos=np.concatenate([b[144:] for b in blocks])
  Xd=X[dev];Ld=L[dev];Xs=Xd[searchpos];Ls=Ld[searchpos];auditX=Xd[auditpos];auditL=Ld[auditpos]
  n=len(Xs);fold_n=n//3
  folds=[list(range(j*fold_n,(j+1)*fold_n)) for j in range(3)]
  (job/'split.json').write_text(json.dumps({'dataset':stem,'repeat':rep,'seed':seed,'dev_n':dev_n,'test_n':test_n,
   'dev_indices':dev.tolist(),'test_indices':test.tolist(),'search_positions':searchpos.tolist(),'audit_positions':auditpos.tolist(),
   'fold_validation_positions':folds,'costs':costs.tolist()},indent=2))
  assert set(np.arange(dev_n)[searchpos]).isdisjoint(np.arange(dev_n)[auditpos])
  blocks2=[]
  for v in folds:
   v=np.array(v);fit=np.setdiff1d(np.arange(n),v)
   lo=Xs[fit].min(0);span=Xs[fit].max(0)-lo;span[span==0]=1.
   z=(Xs[fit]-lo)/span;q=(Xs[v]-lo)/span
   blocks2.append((((q[:,None,:]-z[None,:,:])**2).transpose(2,0,1).copy(),fit,v))
  model=TrainingGeometry(Xs,Ls,kinds=['numerical']*m);dcache={}
  def dep(P):
   key=tuple(sorted(P))
   if key not in dcache:
    d,k=model.subset(key);dcache[key]=granulate(d,Ls,k)[2]
   return dcache[key]
  rows=[]
  for method in ['Prediction only','Guided']:
   cache={}
   def risk(P):
    key=tuple(sorted(P))
    if key not in cache:
     ss=0.
     for mat,fit,v in blocks2:
      pred=Ls[fit][np.argsort(mat[list(key)].sum(0),axis=1,kind='stable')[:,:5]].mean(1) if key else np.tile(Ls[fit].mean(0),(len(v),1))
      ss+=np.sum((Ls[v]-pred)**2)
     cache[key]=float(ss/n)
    return cache[key]
   t=time.perf_counter();limit=risk(range(m))
   if method=='Prediction only':result=pure(costs,lambda P:0.,prediction_loss=risk)
   else:result=guided(costs,dep,risk)
   P=result[0];initial_trace=result[3];initial_seconds=time.perf_counter()-t;initial_evals=len(cache)
   assert risk(P)<=limit and all(risk(set(P)-{a})>limit for a in P)
   t=time.perf_counter();Q,log=refine(P,costs,risk,limit,200);refine_seconds=time.perf_counter()-t
   assert sum(costs[list(Q)])<=sum(costs[list(P)]) and risk(Q)<=limit
   (job/(method.replace(' ','_')+'.json')).write_text(json.dumps(dict(before=list(P),after=list(Q),initial_trace=initial_trace,refinement=log,
    results=dict(method=method,before_attributes=len(P),after_attributes=len(Q),before_cost=int(costs[list(P)].sum()),
    after_cost=int(costs[list(Q)].sum()),before_loss=risk(P),after_loss=risk(Q),full_reference=limit,
    refinement_evaluations=log['evaluations'],accepted_moves=len(log['accepted']),stop_reason=log['stop_reason'],
    initial_unique_loss_evaluations=initial_evals,total_unique_loss_evaluations=len(cache),
    initial_seconds=initial_seconds,refinement_seconds=refine_seconds)),indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x))
   del cache;gc.collect()
  # Audit evaluation happens strictly after both methods' selections are frozen.
  ck={method:json.loads((job/(method.replace(' ','_')+'.json')).read_text()) for method in ['Prediction only','Guided']}
  selected={'Full':list(range(m)),'Empty':[]}
  for method,q in ck.items():selected[method+' before']=q['before'];selected[method+' after']=q['after']
  lo=Xs.min(0);span=Xs.max(0)-lo;span[span==0]=1.;z=(Xs-lo)/span;qa=(auditX-lo)/span;qt=(X[test]-lo)/span;truth=L[test]
  for method,P in selected.items():
   P=list(P)
   pred=Ls[np.argsort(cdist(qa[:,P],z[:,P],'sqeuclidean'),axis=1,kind='stable')[:,:5]].mean(1) if P else np.tile(Ls.mean(0),(len(auditL),1))
   audit_brier=float(np.mean(np.sum((auditL-pred)**2,1)))
   tpred=Ls[np.argsort(cdist(qt[:,P],z[:,P],'sqeuclidean'),axis=1,kind='stable')[:,:5]].mean(1) if P else np.tile(Ls.mean(0),(len(truth),1))
   tpred=np.maximum(tpred,1e-12);tpred/=tpred.sum(1,keepdims=True)
   rows.append({'dataset':stem,'repeat':rep,'dev_n':dev_n,'test_n':test_n,'method':method,'attributes':len(P),
    'cost':int(costs[P].sum()),'audit_brier':audit_brier,
    'test_brier':float(np.mean(np.sum((truth-tpred)**2,1))),
    'test_kl':float(np.mean(np.sum(truth*np.log(np.maximum(truth,1e-300)/tpred),1))),
    'test_tv':float(np.mean(.5*np.abs(truth-tpred).sum(1))),
    'test_js':float(np.mean(.5*np.sum(truth*np.log(np.maximum(truth,1e-300)/((truth+tpred)/2)),1)+.5*np.sum(tpred*np.log(tpred/((truth+tpred)/2)),1)))})
  (job/'audit_results.json').write_text(json.dumps(rows,indent=2))
  (job/'complete.json').write_text(json.dumps({'elapsed_seconds':time.perf_counter()-started},indent=2))
  print(ds,'DONE',rep,'dev',dev_n,[(r['method'],r['attributes'],r['cost'],round(r['audit_brier'],4)) for r in rows],'%.0fs'%(time.perf_counter()-started),flush=True)
  del model,blocks2,dcache;gc.collect()

def summarize():
 allrows=[]
 for stem in NEW:
  ds=safe(stem)
  for rep in range(1,4):
   f=here/ds/f'repeat_{rep}'/'audit_results.json'
   if f.exists():allrows+=json.loads(f.read_text())
 if allrows:
  with (here/'audit_results.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
  print('wrote',here/'audit_results.csv',len(allrows),'rows')

if __name__=='__main__':
 if 'runonly' not in sys.argv:prepare()
 for stem in NEW:
  if (datadir/f'{safe(stem)}_X.npy').exists():run(stem)
  else:print(stem,'not prepared; skipped')
 summarize()

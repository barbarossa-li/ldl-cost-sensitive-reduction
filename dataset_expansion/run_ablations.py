"""Two ablations on the guard-empty expansion datasets.

Stage 'noguard' (Plan B): guard-disabled variant of the selection procedure --
classic greedy forward selection with early stopping when no attribute improves
the inner validation loss, followed by cost-descending deletion of attributes
whose removal does not increase inner loss (never emptying the set). This is
the conventional published-selector protocol; it cannot return the empty set.
We record the cost it spends and its reserved-audit loss next to the zero-cost
guard answer, quantifying the acquisition budget the guard saves on domains
where local kNN signal does not transfer.

Stage 'ridge' (Plan C): predictor-sensitivity probe -- repeat the guard
decision (full-attribute inner loss vs global-mean inner loss) with a ridge
regression predictor instead of 5-NN, under the same frozen splits. Datasets
whose ridge full-set loss beats the ridge global-mean loss are 'revived':
their geometry does carry transferable signal, just not for Euclidean kNN.
Ridge hyperparameter alpha is selected by inner-CV from a fixed grid, using
training data only.

Reproduction: python run_ablations.py noguard|ridge|all
Writes noguard_results.csv / ridge_results.csv in this directory.
"""
from pathlib import Path
import sys,json,csv,time,gc
import numpy as np

here=Path(__file__).resolve().parent
root=here.parent
sys.path.insert(0,str(root));sys.path.insert(0,str(root/'stability'))
datadir=here/'data'

GUARD_EMPTY=['Yeast_spo','Yeast_alpha','Yeast_cdc','Yeast_diau','Yeast_dtt','Yeast_elu','Yeast_heat','Yeast_spo5','Yeast_spoem',
             'FG-Net','Music','Painting','SCUT_FBP','emotion6','Movie','Flickr','Twitter','MOR-PH','Human_Gene','M2B']
INFORMATIVE=['SJAFFE','fbp5500','RAF_ML']

def safe(stem):return stem.replace('-','_')

def load(stem,rep):
 """Return the frozen nested split for one dataset repeat."""
 ds=safe(stem)
 X=np.load(datadir/f'{ds}_X.npy');L=np.load(datadir/f'{ds}_L.npy');m=X.shape[1]
 split=json.loads((here/ds/f'repeat_{rep}'/'split.json').read_text())
 costs=np.array(split['costs'])
 dev=np.array(split['dev_indices']);test=np.array(split['test_indices'])
 sp=np.array(split['search_positions']);ap=np.array(split['audit_positions'])
 Xd=X[dev];Ld=L[dev]
 return dict(X=Xd[sp],L=Ld[sp],auditX=Xd[ap],auditL=Ld[ap],Xt=X[test],Lt=L[test],
             costs=costs,m=m,dev_n=split['dev_n'],ds=ds,stem=stem,rep=rep)

def blocks_for(d):
 """Inner 3-fold squared-distance blocks (kNN-5 evaluation), fold-normalized."""
 X,L=d['X'],d['L'];n=len(X);fold_n=n//3
 folds=[np.arange(j*fold_n,(j+1)*fold_n) for j in range(3)]
 out=[]
 for v in folds:
  fit=np.setdiff1d(np.arange(n),v)
  lo=X[fit].min(0);span=X[fit].max(0)-lo;span[span==0]=1.
  z=(X[fit]-lo)/span;q=(X[v]-lo)/span
  out.append((((q[:,None,:]-z[None,:,:])**2).transpose(2,0,1).copy(),fit,v))
 return out

def nearest5(d):
 idx=np.argpartition(d,4,axis=1)[:,:5];vals=np.take_along_axis(d,idx,axis=1);th=vals.max(1)
 order=np.lexsort((idx,vals),axis=1);idx=np.take_along_axis(idx,order,axis=1)
 for i in np.flatnonzero((d==th[:,None]).sum(1)>1):
  below=np.flatnonzero(d[i]<th[i]);ties=np.flatnonzero(d[i]==th[i]);v=np.r_[below,ties[:5-len(below)]];idx[i]=v[np.lexsort((v,d[i,v]))]
 return idx

def knn_loss_factory(d):
 blocks=blocks_for(d);L=d['L'];n=len(L);cache={}
 def loss(P):
  key=tuple(sorted(P))
  if key not in cache:
   ss=0.
   for mat,fit,v in blocks:
    pred=L[fit][nearest5(mat[list(key)].sum(0))].mean(1) if key else np.tile(L[fit].mean(0),(len(v),1))
    ss+=np.sum((L[v]-pred)**2)
   cache[key]=float(ss/n)
  return cache[key]
 return loss,cache

# ---------------- Stage: noguard ----------------
def run_noguard():
 rows=[]
 for stem in GUARD_EMPTY:
  for rep in range(1,4):
   job=here/'noguard'/f'{safe(stem)}_repeat_{rep}'
   if (job/'results.json').exists():
    old=json.loads((job/'results.json').read_text())
    if all(r.get('attributes',0)>0 for r in old):rows.extend(old);continue
    # stale empty-set rows from the pre-bridge version: recompute
   d=load(stem,rep);loss,_=knn_loss_factory(d);costs=d['costs'];A=range(d['m'])
   cref=costs.mean();import math
   t0=time.perf_counter()
   # classic forward greedy, early stop when no attribute improves inner loss
   P=set();trace=[]
   while True:
    base=loss(P);gains={a:base-loss(P|{a}) for a in A if a not in P}
    positive=[a for a in gains if gains[a]>0]
    if not positive:break
    a=min(positive,key=lambda a:(-(math.log(gains[a])-math.log(costs[a]/cref)),costs[a],a))
    trace.append((a,gains[a]));P.add(a)
   # published selectors always return a non-empty set: if no attribute improved
   # the inner loss, still acquire the single best attribute (forced bridge).
   if not P:
    base=loss(P);gains={a:base-loss(P|{a}) for a in A}
    a=min(gains,key=lambda a:(-gains[a],costs[a],a))
    trace.append((a,gains[a]));P.add(a)
   # cost-descending deletion, never empty
   while True:
    removed=False
    for a in sorted(P,key=lambda a:(-costs[a],a)):
     if len(P)>1 and loss(P-{a})<=loss(P):P.remove(a);removed=True;break
    if not removed:break
   elapsed=time.perf_counter()-t0
   # audit evaluation (same frozen predictor as the main protocol)
   from scipy.spatial.distance import cdist
   X,L=d['X'],d['L'];lo=X.min(0);span=X.max(0)-lo;span[span==0]=1.;z=(X-lo)/span;qa=(d['auditX']-lo)/span
   Ps=list(P)
   pred=L[np.argsort(cdist(qa[:,Ps],z[:,Ps],'sqeuclidean'),axis=1,kind='stable')[:,:5]].mean(1)
   audit_sel=float(np.mean(np.sum((d['auditL']-pred)**2,1)))
   pred0=np.tile(L.mean(0),(len(d['auditL']),1))
   audit_empty=float(np.mean(np.sum((d['auditL']-pred0)**2,1)))
   predF=L[np.argsort(cdist(qa,z,'sqeuclidean'),axis=1,kind='stable')[:,:5]].mean(1)
   audit_full=float(np.mean(np.sum((d['auditL']-predF)**2,1)))
   r=dict(dataset=stem,repeat=rep,method='NoGuard',attributes=len(P),cost=int(costs[Ps].sum()),
          inner_loss=loss(P),inner_loss_empty=loss(()),audit_brier_sel=audit_sel,
          audit_brier_empty=audit_empty,audit_brier_full=audit_full,seconds=elapsed)
   job.mkdir(parents=True,exist_ok=True)
   (job/'results.json').write_text(json.dumps([r],indent=2));rows.append(r)
   print(stem,rep,'NoGuard attrs',len(P),'cost',r['cost'],'audit sel %.4f empty %.4f full %.4f'%(audit_sel,audit_empty,audit_full),flush=True)
   gc.collect()
 with (here/'noguard_results.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('noguard complete',len(rows),'rows')

# ---------------- Stage: ridge probe ----------------
ALPHAS=[0.01,0.1,1.,10.,100.]

def ridge_fit(X,L,alpha):
 """Closed-form ridge on centered labels; X assumed standardized."""
 n,m=X.shape
 XtX=X.T@X;XtL=X.T@L
 W=np.linalg.solve(XtX+alpha*np.eye(m),XtL)
 return W

def ridge_inner(d):
 """Inner 3-fold CV losses for full ridge (alpha by CV) and empty (global mean)."""
 X,L=d['X'],d['L'];n=len(X);fold_n=n//3
 folds=[np.arange(j*fold_n,(j+1)*fold_n) for j in range(3)]
 # alpha selection on inner folds, training data only
 best=None
 for alpha in ALPHAS:
  tot=0.
  for v in folds:
   fit=np.setdiff1d(np.arange(n),v)
   mu=X[fit].mean(0);sd=X[fit].std(0);sd[sd==0]=1.
   z=(X[fit]-mu)/sd;q=(X[v]-mu)/sd
   W=ridge_fit(z,L[fit],alpha)
   pred=q@W+(L[fit].mean(0)-mu@W)
   tot+=np.sum((L[v]-pred)**2)
  if best is None or tot<best[1]:best=(alpha,tot)
 alpha=best[0]
 # full-set inner loss under chosen alpha
 tot=0.
 for v in folds:
  fit=np.setdiff1d(np.arange(n),v)
  mu=X[fit].mean(0);sd=X[fit].std(0);sd[sd==0]=1.
  z=(X[fit]-mu)/sd;q=(X[v]-mu)/sd
  W=ridge_fit(z,L[fit],alpha)
  pred=q@W+(L[fit].mean(0)-mu@W)
  tot+=np.sum((L[v]-pred)**2)
 full_inner=tot/n
 # empty predictor (global mean) inner loss
 tot=0.
 for v in folds:
  fit=np.setdiff1d(np.arange(n),v);pred=np.tile(L[fit].mean(0),(len(v),1))
  tot+=np.sum((L[v]-pred)**2)
 empty_inner=tot/n
 return alpha,full_inner,empty_inner

def run_ridge():
 rows=[]
 for stem in GUARD_EMPTY:
  for rep in range(1,4):
   job=here/'ridge'/f'{safe(stem)}_repeat_{rep}'
   if (job/'results.json').exists():
    rows.extend(json.loads((job/'results.json').read_text()));continue
   d=load(stem,rep)
   t0=time.perf_counter()
   alpha,full_inner,empty_inner=ridge_inner(d)
   # audit evaluation of the ridge full predictor (descriptive)
   X,L=d['X'],d['L'];n=len(X)
   mu=X.mean(0);sd=X.std(0);sd[sd==0]=1.
   z=(X-mu)/sd;qa=(d['auditX']-mu)/sd
   W=ridge_fit(z,L,alpha)
   pred=qa@W+(L.mean(0)-mu@W)
   audit_ridge=float(np.mean(np.sum((d['auditL']-pred)**2,1)))
   pred0=np.tile(L.mean(0),(len(d['auditL']),1))
   audit_empty=float(np.mean(np.sum((d['auditL']-pred0)**2,1)))
   r=dict(dataset=stem,repeat=rep,alpha=alpha,ridge_inner_full=full_inner,ridge_inner_empty=empty_inner,
          inner_gain=empty_inner-full_inner,audit_brier_ridge=audit_ridge,audit_brier_empty=audit_empty,
          audit_gain=audit_empty-audit_ridge,revived=bool(full_inner<empty_inner),
          seconds=time.perf_counter()-t0)
   job.mkdir(parents=True,exist_ok=True)
   (job/'results.json').write_text(json.dumps([r],indent=2));rows.append(r)
   print(stem,rep,'ridge alpha',alpha,'inner full %.5f empty %.5f revived %s'%(full_inner,empty_inner,r['revived']),flush=True)
   gc.collect()
 with (here/'ridge_results.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('ridge complete',len(rows),'rows')

if __name__=='__main__':
 stage=sys.argv[1] if len(sys.argv)>1 else 'all'
 if stage in ('noguard','all'):run_noguard()
 if stage in ('ridge','all'):run_ridge()

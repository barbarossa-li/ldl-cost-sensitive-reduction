"""Explicit-assumption, leakage-controlled ProLSFEO paper-setting reconstruction.
Not original-author code or a verified numerical paper reproduction.
"""
from pathlib import Path
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np
import time,json,hashlib

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def groups(X,seed,nfold=10):
 _,g=np.unique(X,axis=0,return_inverse=True);ng=int(g.max())+1
 if ng<nfold:raise ValueError('Too few feature groups')
 rng=np.random.default_rng(seed);folds=[[] for _ in range(nfold)]
 for gi in rng.permutation(ng):
  j=min(range(nfold),key=lambda z:(len(folds[z]),z));folds[j].extend(np.flatnonzero(g==gi).tolist())
 return [np.array(sorted(z),int) for z in folds]
def normalize(v,p=None):
 sums=v.sum(1);bad=sums<=0
 if np.any(bad):
  if p is None:raise ValueError('Zero-total component prediction')
  v=v.copy();v[bad]=p;sums=v.sum(1)
 return v/sums[:,None]
def scale(A,Q):
 lo=A.min(0);span=A.max(0)-lo;return np.divide(A-lo,span,out=np.zeros_like(A),where=span>0),np.divide(Q-lo,span,out=np.zeros_like(Q),where=span>0),lo,span

def component_predict(A,L,Q,features,k=4):
 diff=(Q[:,None,:]-A[None,:,:])**2;D=np.einsum('hnd,qd->qhn',diff,features,optimize=True);out=np.empty((len(Q),L.shape[1]))
 for j in range(L.shape[1]):ids=np.argsort(D[j],axis=1,kind='stable')[:,:min(k,len(A))];out[:,j]=L[ids,j].mean(1)
 return out
class Fitness:
 def __init__(self,X,Y,folds):
  self.X=X;self.Y=Y;self.n,self.q=Y.shape;self.d=X.shape[1];self.parts=[];self.cache={};self.evaluations=0;self.invalid=0
  for held in folds:
   fit=np.setdiff1d(np.arange(self.n),held);A,Q,_,_=scale(X[fit],X[held]);self.parts.append((held,fit,(Q[:,None,:]-A[None,:,:])**2))
 def prediction(self,bits):
  keep=bits[:self.n];mask=bits[self.n:].reshape(self.q,self.d)
  if not keep.any() or not mask.any(1).all():raise ValueError('Empty prototypes or empty label-specific features')
  pred=np.zeros_like(self.Y)
  for held,fit,diff in self.parts:
   selected=np.flatnonzero(keep[fit])
   if not len(selected):raise ValueError('No retained prototypes in an inner training fold')
   D=np.einsum('hnd,qd->qhn',diff[:,selected,:],mask,optimize=True)
   for j in range(self.q):ids=np.argsort(D[j],axis=1,kind='stable')[:,:min(4,len(selected))];pred[held,j]=self.Y[fit[selected[ids]],j].mean(1)
  return normalize(pred)
 def __call__(self,bits):
  key=np.packbits(bits).tobytes()
  if key in self.cache:return self.cache[key]
  self.evaluations+=1
  try:
   v=self.prediction(bits);y=self.Y;value=float(np.where(y>0,y*np.log(np.where(y>0,y,1)/np.maximum(v,1e-12)),0).sum(1).mean())
   if not np.isfinite(value):raise ValueError('Nonfinite fitness')
  except ValueError:value=float('inf');self.invalid+=1
  self.cache[key]=value;return value

def hux(a,b,rng):
 diff=np.flatnonzero(a!=b);exchange=rng.choice(diff,size=len(diff)//2,replace=False);x=a.copy();y=b.copy();x[exchange]=b[exchange];y[exchange]=a[exchange];return x,y

def optimize(X,Y,seed,population=100,generations=500,threshold=10,deadline_seconds=3600,progress=None):
 rng=np.random.default_rng(seed);folds=groups(X,seed+1);F=Fitness(X,Y,folds);length=len(X)+Y.shape[1]*X.shape[1];P=rng.integers(0,2,(population,length),dtype=np.uint8).astype(bool);start=time.perf_counter();L=threshold;history=[];restarts=0
 def ranked(Z):return sorted(Z,key=lambda x:(F(x),np.packbits(x).tobytes()))
 def canonical(Z):return sorted(np.packbits(x).tobytes() for x in Z)
 completed=0
 for gen in range(generations):
  if time.perf_counter()-start>deadline_seconds:break
  P=np.array(ranked(P));perm=rng.permutation(population);children=[]
  for i in range(0,population,2):children.extend(hux(P[perm[i]],P[perm[i+1]],rng))
  new=np.array(ranked(np.concatenate([P,np.array(children)]))[:population]);unchanged=canonical(new)==canonical(P)
  if unchanged:L-=1
  if L==1:
   best=P[0].copy();P=np.concatenate([best[None,:],rng.integers(0,2,(population-1,length),dtype=np.uint8).astype(bool)]);L=threshold;restarts+=1
  else:P=new
  completed=gen+1;entry={'generation':completed,'best_inner_kl':F(ranked(P)[0]),'unique_fitness_evaluations':F.evaluations,'invalid_evaluations':F.invalid,'stagnation_counter':L,'restarts':restarts,'seconds':time.perf_counter()-start};history.append(entry)
  if progress and completed%10==0:save(progress,{'history':history,'status':'running'})
 best=ranked(P)[0];value=F(best)
 return best,folds,{'status':'complete' if completed==generations else 'time_limit_incomplete','generations_completed':completed,'generations_requested':generations,'best_inner_kl':value if np.isfinite(value) else None,'seconds':time.perf_counter()-start,'unique_fitness_evaluations':F.evaluations,'invalid_evaluations':F.invalid,'history':history}

def deploy_state(X,Y,bits):
 n,q=Y.shape;mask=bits[n:].reshape(q,X.shape[1]);P=np.flatnonzero(mask.any(0));keep=np.flatnonzero(bits[:n]);lo=X[:,P].min(0);span=X[:,P].max(0)-lo;Z=np.divide(X[keep][:,P]-lo,span,out=np.zeros((len(keep),len(P))),where=span>0)
 return {'P':P.astype(np.int64),'lo':lo,'span':span,'Z':Z,'Y':Y[keep].copy(),'label_masks':mask[:,P].copy(),'p':Y.mean(0)}
def predict(state,Q):
 Q=Q[:,state['P']];Q=np.divide(Q-state['lo'],state['span'],out=np.zeros_like(Q),where=state['span']>0);raw=component_predict(state['Z'],state['Y'],Q,state['label_masks']);return normalize(raw,state['p'])

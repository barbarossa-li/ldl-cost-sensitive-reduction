"""Training-only model selection helpers; all point predictions use explicit state."""
from pathlib import Path
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['KERAS_BACKEND']='torch';os.environ.setdefault('MPLCONFIGDIR','/tmp/ldl_matplotlib')
import sys,copy,json,gc,time,warnings
import numpy as np
PARENT=Path(os.environ['LDL_PARENT_SOURCE']);sys.path.insert(0,str(PARENT));import runner as r
from sklearn.linear_model import Ridge
from pyldl.algorithms.utils import pairwise_euclidean,kernel
from sklearn.metrics.pairwise import rbf_kernel
import torch,keras
KS=(3,5,10,20);BUDGETS=(25,50,75,100);AA_KS=(4,5,10,20,40);RIDGE_ALPHAS=(1e-4,1e-2,1.,100.);ANCHORS=((5,1e-3,1e-6),(20,1e-3,1e-6),(20,1e-2,1e-5));EPOCHS=(100,500,1000)

def group_folds(X,seed):
 _,inv=np.unique(X,axis=0,return_inverse=True);ng=int(inv.max())+1
 if ng<3:raise ValueError('Fewer than three exact-feature groups')
 order=np.argsort(inv,kind='stable');groups=np.split(order,np.flatnonzero(np.diff(inv[order]))+1);folds=[[],[],[]];sizes=[0,0,0]
 for g in np.random.default_rng(seed).permutation(ng):
  j=min(range(3),key=lambda i:(sizes[i],i));folds[j].extend(groups[int(g)].tolist());sizes[j]+=len(groups[int(g)])
 out=[np.array(sorted(x),dtype=int) for x in folds]
 assert sorted(np.concatenate(out).tolist())==list(range(len(X)))
 for i in range(3):
  for j in range(i):assert not set(inv[out[i]])&set(inv[out[j]])
 return out

def project(v):
 u=np.sort(v,axis=1)[:,::-1];cs=np.cumsum(u,axis=1)-1.;rho=np.sum(u-cs/np.arange(1,v.shape[1]+1)>0,axis=1)-1;theta=cs[np.arange(len(v)),rho]/(rho+1);return np.maximum(v-theta[:,None],0.)

def scaling(X,P):
 A=X[:,P];lo=A.min(0);span=A.max(0)-lo;Z=np.divide(A-lo,span,out=np.zeros_like(A),where=span>0);return Z,lo,span

def features(Q,state):
 A=Q[:,state['P']];return np.divide(A-state['lo'],state['span'],out=np.zeros_like(A),where=state['span']>0)

def aa_state(X,L,P,k):
 A,lo,span=scaling(X,P);m=r.AA_KNN(k=int(k));m.fit(A,L)
 return {'kind':'AA','k':int(k),'P':np.array(P,np.int64),'lo':lo,'span':span,'Z':m._X.copy(order='K'),'L':m._D.copy(order='K')}

def ridge_state(X,L,P,lam):
 A,lo,span=scaling(X,P);m=Ridge(alpha=float(lam),fit_intercept=True,solver='cholesky');m.fit(A,L)
 return {'kind':'RidgeProj','P':np.array(P,np.int64),'lo':lo,'span':span,'W':m.coef_.T.copy(),'b':m.intercept_.copy()}

def prototype_state(m,P,g,budget,k):
 D,_=r.geometry(m,P,g);K=max(1,int(budget)*m.n//100);groups=r.bisection_path(D,m.L,[K],0.)[K];obj=r.PrototypePredictor(m,P,g,groups);obj.k=int(k);st=obj.state();st['kind']='Proto';return st

def predict(state,Q,native=False):
 kind=state['kind']
 if kind=='Mean':return np.tile(state['p'],(len(Q),1))
 if kind=='Proto':
  x={k:v for k,v in state.items() if k not in ['kind','alpha']};v=r.PrototypePredictor.from_state(x).predict(Q[:,state['P']])
 elif kind=='AA':
  m=r.AA_KNN(k=state['k']);m.fit(state['Z'],state['L']);v=m.predict(features(Q,state))
 elif kind=='RidgeProj':v=project(features(Q,state)@state['W']+state['b'])
 elif kind=='RKNN':
  B=features(Q,state);m=r.AA_KNN(k=state['k']);m.fit(state['Z'],state['L']);inds=m._knn.kneighbors(B)[1];rho=state['rho'][inds].mean(1)[:,None];neighbor=state['L'][inds].mean(1);K=kernel(B,state['Z']) if native else rbf_kernel(B,state['Z'],gamma=state['gamma']);logits=torch.from_numpy(np.asarray(K,dtype=np.float32))@torch.from_numpy(state['W']);network=torch.softmax(logits,dim=1).detach().numpy();v=rho*neighbor+(1-rho)*network
 else:raise ValueError(kind)
 alpha=float(state.get('alpha',0.));return (1-alpha)*v+alpha*state['p'] if alpha else v

def arrays(state):return {k:v for k,v in state.items() if isinstance(v,np.ndarray)}
def byte_count(state):return sum(x.nbytes for x in arrays(state).values())
def with_prior(state,alpha,p):
 s=dict(state);s['alpha']=float(alpha)
 if alpha or state['kind'] in ['Proto','Mean']:s['p']=p.copy()
 return s

def proto_candidates(m,Q,P,g):
 D,_=r.geometry(m,P,g);ks={str(b):max(1,b*m.n//100) for b in BUDGETS};path=r.bisection_path(D,m.L,list(ks.values()),0.);out={};params={}
 for b,K in ks.items():
  obj=r.PrototypePredictor(m,P,g,path[K]);order=np.argsort(obj.distances(Q[:,P]),axis=1,kind='stable')
  for k in KS:
   v=obj.labels[order[:,:min(k,K)]].mean(1)
   for alpha in [0.,.5]:
    key=f'b{b}_k{k}_a{alpha}';out[key]=(1-alpha)*v+alpha*m.p;params[key]={'budget':int(b),'k':k,'alpha':alpha}
 return out,params

class Capture(keras.callbacks.Callback):
 def __init__(self,Q,P,lo,span,epochs):super().__init__();self.Q=Q;self.P=P;self.lo=lo;self.span=span;self.epochs=set(epochs);self.states={};self.pred={};self.training=[]
 def on_train_begin(self,logs=None):
  x=self.model._to_numpy(self.model._X);self.gamma=float(1/(2*np.mean(pairwise_euclidean(x,x))**2))
 def on_epoch_end(self,epoch,logs=None):
  loss=float((logs or {})['loss'])
  if not np.isfinite(loss):raise FloatingPointError('Nonfinite RKNN training loss')
  if epoch not in self.epochs:return
  m=self.model;s={'kind':'RKNN','k':int(m.k),'gamma':self.gamma,'P':np.array(self.P,np.int64),'lo':self.lo.copy(),'span':self.span.copy(),'Z':m._aaknn._X.copy(order='K'),'L':m._aaknn._D.copy(order='K'),'rho':m._to_numpy(m._rho).copy(),'W':m.get_weights()[0].copy()};v=predict(s,self.Q);r.valid(v,len(self.Q),s['L'].shape[1]);B=features(self.Q,s);i=m._aaknn._knn.kneighbors(B)[1];rho=m._to_numpy(m._rho)[i].mean(1)[:,None];expected=rho*m._aaknn._D[i].mean(1)+(1-rho)*m._to_numpy(m._call(rbf_kernel(B,m._to_numpy(m._X),gamma=self.gamma)));assert np.allclose(v,expected,atol=1e-7,rtol=0);assert np.allclose(predict(s,self.Q,native=True),m.predict(B),atol=1e-7,rtol=0)
  self.pred[int(epoch)]=v;self.states[int(epoch)]=s;self.training.append({'epoch':int(epoch),'loss':loss,'state_max_error':float(abs(v-expected).max())})

def rknn_checkpoints(X,L,Q,P,anchor,seed,epochs=EPOCHS):
 keras.backend.clear_session();A,lo,span=scaling(X,P);k,alpha,beta=anchor;cb=Capture(Q,P,lo,span,epochs);m=r.RKNN_LDL(k=k,alpha=alpha,beta=beta,random_state=int(seed%(2**32)));failure=None;start=time.perf_counter()
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always')
  try:m.fit(A,L,epochs=max(epochs),kernel=True,sparse=True,verbose=0,callbacks=[cb])
  except Exception as exc:failure={'type':type(exc).__name__,'message':str(exc)}
 details={'failure':failure,'warnings':[{'type':w.category.__name__,'message':str(w.message)} for w in caught],'training':cb.training,'seconds':time.perf_counter()-start};del m;keras.backend.clear_session();gc.collect();return cb.pred,cb.states,details

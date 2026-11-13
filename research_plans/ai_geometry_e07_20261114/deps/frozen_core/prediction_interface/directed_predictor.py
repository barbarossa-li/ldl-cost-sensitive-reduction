"""Training-source -> unseen-target directed 5NN interface.
No query labels, query-side information weights, or retraining on queries.
This candidate mapping leaves the original training geometry untouched.
"""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from granulation_reference import TrainingGeometry

class DirectedNeighborPredictor:
 def __init__(self,X,labels,kinds=None,beta=1.,zeta=.1,k=5):
  self.geometry=TrainingGeometry(X,labels,kinds=kinds,beta=beta,zeta=zeta)
  self.X=self.geometry.X;self.L=self.geometry.L.copy();self.n,self.m=self.X.shape
  if int(k)!=k or k<1 or k>self.n:raise ValueError('k must be an integer in [1,n_train]')
  self.k=int(k);self.global_mean=self.L.mean(0);self.numeric={};self.profiles={}
  for a,kind in enumerate(self.geometry.kinds):
   v=self.X[:,a]
   if kind=='numerical':
    values=np.asarray(v,float);lo=values.min();span=values.max()-lo
    self.numeric[a]=(lo,span,(values-lo)/span if span>0 else np.zeros(self.n))
   else:
    self.profiles[a]={value:self.L[v==value].mean(0) for value in dict.fromkeys(v)}
 def incoming(self,queries,P):
  Q=np.asarray(queries,dtype=object)
  if Q.ndim!=2 or Q.shape[1]!=self.m:raise ValueError('Queries must have all original feature columns')
  P=tuple(sorted(P))
  if len(set(P))!=len(P) or any(a<0 or a>=self.m for a in P):raise ValueError('Invalid attribute subset')
  D=np.zeros((len(Q),self.n));zeta=self.geometry.zeta
  for a in P:
   w=self.geometry.weights[a]
   if self.geometry.kinds[a]=='numerical':
    lo,span,z=self.numeric[a];q=np.asarray(Q[:,a],float)
    if not np.isfinite(q).all():raise ValueError('Query values must be finite')
    q=(q-lo)/span if span>0 else np.zeros(len(q))
    raw=np.abs(q[:,None]-z[None,:]);delta=raw
   else:
    values=Q[:,a]
    if any(v is None or (isinstance(v,(float,np.floating)) and not np.isfinite(v)) for v in values):raise ValueError('Missing categorical query value')
    profile=np.array([self.profiles[a].get(v,self.global_mean) for v in values],float).reshape(len(Q),self.L.shape[1])
    training=np.array([self.profiles[a][v] for v in self.X[:,a]])
    delta=.5*np.abs(profile[:,None,:]-training[None,:,:]).sum(2)
    raw=(values[:,None]!=self.X[None,:,a]).astype(float)
   D+=((1-zeta)*delta*w[None,:]+zeta*raw)/self.m
  if not np.isfinite(D).all():raise ValueError('Nonfinite extrapolated dissimilarity')
  return D
 def predict(self,queries,P):
  Q=np.asarray(queries,dtype=object);P=tuple(P)
  D=self.incoming(Q,P)
  if not P:return np.tile(self.global_mean,(len(Q),1))
  neighbors=np.argsort(D,axis=1,kind='stable')[:,:self.k]
  return self.L[neighbors].mean(1)

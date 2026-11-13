"""Numerical source-geometry retrieval and candidate granular prototype readouts.

The training partition is the frozen anchored oracle. Residual and global
readouts are distinct. Queries provide only selected feature values, no labels.
"""
from pathlib import Path
import os,operator,time
import numpy as np
CORE=Path(os.environ.get('AI_LDL_CORE','/workspace/work/asc_review/code_bundle'))
import sys
sys.path.insert(0,str(CORE))
from information_reduction.safe_extension import granulate

class NumericalPrototypeModel:
 def __init__(self,X,L,beta=1.,zeta=.1,k=5,smin=2,kappa=.1,alpha=1.,pair_budget=256):
  self.X=np.array(X,dtype=float,copy=True);self.L=np.array(L,dtype=float,copy=True)
  if self.X.ndim!=2 or self.L.ndim!=2 or not self.X.shape[0] or not self.X.shape[1] or len(self.X)!=len(self.L) or not self.L.shape[1]:raise ValueError('Nonempty matched training matrices required')
  if not np.isfinite(self.X).all() or not np.isfinite(self.L).all() or np.any(self.L<0) or not np.allclose(self.L.sum(1),1,rtol=0,atol=1e-12):raise ValueError('Finite features and probability labels required')
  self.n,self.m=self.X.shape
  if isinstance(k,bool) or operator.index(k)<1 or k>self.n:raise ValueError('Require 1 <= k <= training n')
  if not np.isfinite(beta) or beta<=0 or not np.isfinite(zeta) or not 0<=zeta<1:raise ValueError('Invalid beta/zeta')
  if isinstance(smin,bool) or operator.index(smin)<2 or not 0<=kappa<1 or not 0<=alpha<=1:raise ValueError('Invalid granulation settings')
  if isinstance(pair_budget,bool) or operator.index(pair_budget)<1:raise ValueError('Invalid pair budget')
  self.k=int(k);self.beta=float(beta);self.zeta=float(zeta);self.settings=dict(smin=int(smin),kappa=float(kappa),alpha=float(alpha),pair_budget=int(pair_budget))
  self.p=self.L.mean(0);self.lo=self.X.min(0);self.span=self.X.max(0)-self.lo
  if not np.isfinite(self.span).all():raise ValueError('Numerical feature span overflow')
  self.Z=np.zeros_like(self.X);self.W=np.empty_like(self.X);self.bandwidth=np.empty(self.m)
  active=self.p>0;norm=float(np.log2(1/self.p[active].min())) if active.sum()>1 else 0.
  if not np.isfinite(norm):raise ValueError('Unsupported extreme label normalizer')
  for a in range(self.m):
   z=(self.X[:,a]-self.lo[a])/self.span[a] if self.span[a]>0 else np.zeros(self.n);self.Z[:,a]=z
   raw=np.abs(z[:,None]-z[None,:]);positive=raw[np.triu_indices(self.n,1)];positive=positive[positive>0];tau=float(np.median(positive)) if len(positive) else 1.;self.bandwidth[a]=tau
   K=np.exp(-raw**2/(2*tau**2));support=K.sum(1);other=support-1.;u=K@self.L/support[:,None];eta=other/(other+self.beta);regular=eta[:,None]*u+(1-eta[:,None])*self.p
   w=np.sum(regular[:,active]*np.log2(regular[:,active]/self.p[active]),axis=1)/norm if norm else np.zeros(self.n);self.W[:,a]=np.clip(w,0.,1.)
  self.G=np.broadcast_to(self.W.mean(0),self.W.shape).copy();self.cache={}
  for a in [self.X,self.L,self.Z,self.W,self.G,self.p,self.lo,self.span,self.bandwidth]:a.setflags(write=False)
 def selection(self,P):
  P=tuple(P)
  if any(isinstance(a,(bool,np.bool_)) for a in P):raise ValueError('Boolean masks are not original feature indices')
  P=tuple(operator.index(a) for a in P)
  if any(a<0 or a>=self.m for a in P) or any(a>=b for a,b in zip(P,P[1:])):raise ValueError('P must contain distinct increasing original indices')
  return P
 def weights(self,mode,permutation=None):
  if mode=='U':return np.ones_like(self.W)
  if mode=='G':return self.G
  if mode=='L':return self.W
  if mode=='S':
   pi=np.asarray(permutation)
   if pi.shape!=(self.n,) or not np.issubdtype(pi.dtype,np.integer) or not np.array_equal(np.sort(pi),np.arange(self.n)):raise ValueError('S requires a full integer source-row permutation')
   return self.W[pi]
  raise ValueError('Unknown weight mode')
 def directed_training(self,P,mode='L',permutation=None):
  P=self.selection(P);W=self.weights(mode,permutation);D=np.zeros((self.n,self.n))
  for a in P:
   raw=np.abs(self.Z[:,a,None]-self.Z[None,:,a]);information=W[:,a,None]*raw/self.m;geometry=raw/self.m;D+=(1-self.zeta)*information+self.zeta*geometry
  return D
 def training_distances(self,P,mode='L',sym='max',permutation=None):
  D=self.directed_training(P,mode,permutation)
  if sym=='max':return np.maximum(D,D.T)
  if sym=='mean':return .5*(D+D.T)
  raise ValueError('Unknown reciprocal aggregation')
 def incoming_selected(self,Q,P,mode='L',permutation=None):
  P=self.selection(P);Q=np.asarray(Q,float)
  if Q.ndim!=2 or Q.shape[1]!=len(P) or not np.isfinite(Q).all():raise ValueError('Finite selected query columns must follow P')
  W=self.weights(mode,permutation);D=np.zeros((len(Q),self.n))
  for j,a in enumerate(P):
   q=(Q[:,j]-self.lo[a])/self.span[a] if self.span[a]>0 else np.zeros(len(Q));raw=np.abs(q[:,None]-self.Z[None,:,a])
   # Preserve the existing directed_predictor's operation order and orientation.
   D+=((1-self.zeta)*raw*W[None,:,a]+self.zeta*raw)/self.m
  if not np.isfinite(D).all():raise ValueError('Nonfinite query extrapolation')
  return D
 def incoming_full(self,Q,P,mode='L',permutation=None):
  P=self.selection(P);Q=np.asarray(Q,float)
  if Q.ndim!=2 or Q.shape[1]!=self.m:raise ValueError('Expected all-column compatibility input')
  return self.incoming_selected(Q[:,P],P,mode,permutation)
 def neighbors(self,Q,P,mode='L',permutation=None):
  P=self.selection(P);D=self.incoming_selected(Q,P,mode,permutation)
  return np.argsort(D,axis=1,kind='stable')[:,:self.k] if P else np.empty((len(D),0),int)
 def representation(self,P,mode='L',sym='max',permutation=None):
  P=self.selection(P);key=(P,mode,sym,None if permutation is None else tuple(map(int,permutation)))
  if key in self.cache:return self.cache[key]
  t=time.perf_counter();D=self.training_distances(P,mode,sym,permutation);keys=[tuple(row[list(P)]) for row in self.X];g=granulate(D,self.L,keys,**self.settings)
  residual=self.L.copy();legacy=np.tile(self.p,(self.n,1));covered=np.zeros(self.n,bool)
  for B in g[1]:
   mu=self.L[B].mean(0);residual[B]=mu;legacy[B]=mu;covered[B]=True
  for a in [residual,legacy,covered]:a.setflags(write=False)
  record={'raw':self.L,'ball_residual':residual,'ball_global_legacy':legacy,'leaves':tuple(tuple(B) for B in g[0]),'accepted':tuple(tuple(B) for B in g[1]),'covered':covered,'granulation_audit':g[4],
   'statistics':{'coverage':float(covered.mean()),'accepted_leaves':len(g[1]),'terminal_leaves':len(g[0]),'residual_objects':int(np.sum(~covered)),'prototype_slots_residual':len(g[1])+int(np.sum(~covered)),'prototype_slots_legacy':len(g[1])+int(np.any(~covered))},'construction_wall_seconds':time.perf_counter()-t}
  self.cache[key]=record;return record
 def predict_selected(self,Q,P,mode='L',readout='ball_residual',sym='max',permutation=None,return_neighbors=False):
  P=self.selection(P)
  if readout not in ['raw','ball_residual','ball_global_legacy']:raise ValueError('Unknown readout')
  ids=self.neighbors(Q,P,mode,permutation)
  if not P:pred=np.tile(self.p,(len(ids),1))
  else:
   R=self.L if readout=='raw' else self.representation(P,mode,sym,permutation)[readout]
   pred=R[ids].mean(1)
  return (pred,ids) if return_neighbors else pred
 def evaluate_selected(self,Q,P,mode='L',sym='max',permutation=None):
  P=self.selection(P);representation=self.representation(P,mode,sym,permutation);ids=self.neighbors(Q,P,mode,permutation)
  predictions={name:(np.tile(self.p,(len(ids),1)) if not P else representation[name][ids].mean(1)) for name in ['raw','ball_residual','ball_global_legacy']}
  return {'predictions':predictions,'neighbors':ids,'representation':representation}
 def predict_full(self,Q,P,**kwargs):
  P=self.selection(P);Q=np.asarray(Q,float)
  if Q.ndim!=2 or Q.shape[1]!=self.m:raise ValueError('Expected all-column compatibility input')
  return self.predict_selected(Q[:,P],P,**kwargs)
 def randomized_representation(self,P,seed,mode='L',sym='max'):
  original=self.representation(P,mode,sym);order=np.random.default_rng(seed).permutation(self.n);accepted=set(original['accepted']);pos=0;R=self.L.copy();groups=[];flags=[]
  for B in original['leaves']:
   C=tuple(sorted(map(int,order[pos:pos+len(B)])));pos+=len(B);flag=B in accepted;groups.append(C);flags.append(flag)
   if flag:R[list(C)]=self.L[list(C)].mean(0)
  assert pos==self.n
  return {'readout':R,'groups':tuple(groups),'accepted_flags':tuple(flags)}

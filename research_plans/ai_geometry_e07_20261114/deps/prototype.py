"""Standalone selected-column prototype retrieval with fixed training scales."""
import numpy as np

def groups_from_representation(model,rep,seed=None):
 accepted=sorted([tuple(map(int,B)) for B in rep['accepted']],key=lambda B:min(B));covered=np.flatnonzero(rep['covered']);groups=[];keys=[]
 order=None if seed is None else np.random.default_rng(seed).permutation(covered);pos=0
 for B in accepted:
  group=B if order is None else tuple(map(int,order[pos:pos+len(B)]));pos+=len(B);groups.append(group);keys.append(min(B))
 for i in np.flatnonzero(~rep['covered']):groups.append((int(i),));keys.append(int(i))
 order=np.argsort(keys,kind='stable');groups=[groups[i] for i in order];flags=np.array([keys[i] in set(map(min,accepted)) for i in order],bool)
 assert sorted(i for B in groups for i in B)==list(range(model.n));assert sum(len(B) for B,flag in zip(groups,flags) if flag)==len(covered)
 return groups,flags

class PrototypePredictor:
 def __init__(self,model,P,mode,groups=None,accepted_flags=None,covered_mean=False):
  P=model.selection(P)
  if mode not in ['U','G']:raise ValueError('Prototype geometry must use source-independent U/G weights')
  groups=groups if groups is not None else [(i,) for i in range(model.n)];flags=np.zeros(len(groups),bool) if accepted_flags is None else np.asarray(accepted_flags,bool)
  if len(flags)!=len(groups) or sorted(i for B in groups for i in B)!=list(range(model.n)):raise ValueError('Groups must partition training objects')
  self.original_m=model.m;self.k=model.k;self.zeta=model.zeta;self.scales=dict(model.scales(mode));self.P=np.array(P,dtype=np.int64);self.lo=model.lo[list(P)].copy();self.span=model.span[list(P)].copy();self.weight=model.weights(mode)[0,list(P)].copy()
  self.centers=np.array([model.Z[list(B)][:,P].mean(0) for B in groups]);self.labels=np.array([model.p if covered_mean and flag else model.L[list(B)].mean(0) for B,flag in zip(groups,flags)]);self.masses=np.array([len(B) for B in groups],dtype=np.int64);self.p=model.p.copy()
  for name in ['P','lo','span','weight','centers','labels','masses','p']:getattr(self,name).setflags(write=False)
 def distances(self,Q):
  Q=np.asarray(Q,float)
  if Q.ndim!=2 or Q.shape[1]!=len(self.P) or not np.isfinite(Q).all():raise ValueError('Finite selected query columns required')
  I=np.zeros((len(Q),len(self.labels)));G=np.zeros_like(I)
  for j in range(len(self.P)):
   q=(Q[:,j]-self.lo[j])/self.span[j] if self.span[j]>0 else np.zeros(len(Q));raw=np.abs(q[:,None]-self.centers[None,:,j]);I+=raw*self.weight[j]/self.original_m;G+=raw/self.original_m
  s=self.scales
  if s['kind']=='constant_geometry':D=np.zeros_like(G)
  elif s['kind']=='zero_information_raw_fallback':D=G/s['raw_full_mean']/s['fixed_full_max']
  else:D=((1-self.zeta)*I/s['information_full_mean']+self.zeta*G/s['raw_full_mean'])/s['fixed_full_max']
  if not np.isfinite(D).all():raise ValueError('Nonfinite query extrapolation')
  return D
 def predict(self,Q,mass_weighted=False):
  D=self.distances(Q)
  if not len(self.P):return np.tile(self.p,(len(D),1))
  ids=np.argsort(D,axis=1,kind='stable')[:,:min(self.k,len(self.labels))]
  if mass_weighted:
   mass=self.masses[ids];return np.sum(self.labels[ids]*mass[:,:,None],axis=1)/mass.sum(1)[:,None]
  return self.labels[ids].mean(1)
 def array_bytes(self):return sum(v.nbytes for v in vars(self).values() if isinstance(v,np.ndarray))
 def state(self):return {k:(v.copy() if isinstance(v,np.ndarray) else v.copy() if isinstance(v,dict) else v) for k,v in vars(self).items()}
 @classmethod
 def from_state(cls,state):
  obj=cls.__new__(cls)
  for k,v in state.items():setattr(obj,k,v.copy() if isinstance(v,(np.ndarray,dict)) else v)
  return obj

"""Training-Full component calibration; old E0 remains unchanged."""
from pathlib import Path
import os,sys,numpy as np
E0=Path(os.environ.get('AI_LDL_E0','/workspace/work/ai_e0_20261113'));sys.path.insert(0,str(E0))
from model import NumericalPrototypeModel
class CalibratedPrototypeModel(NumericalPrototypeModel):
 def __init__(self,*args,calibration='balanced',**kwargs):
  if calibration not in ['original','balanced']:raise ValueError('Unknown calibration')
  self.calibration_kind=calibration;super().__init__(*args,**kwargs);self.scale_cache={}
 def components_training(self,P,mode='L',permutation=None):
  P=self.selection(P);W=self.weights(mode,permutation);I=np.zeros((self.n,self.n));G=np.zeros_like(I)
  for a in P:
   raw=np.abs(self.Z[:,a,None]-self.Z[None,:,a]);I+=W[:,a,None]*raw/self.m;G+=raw/self.m
  return I,G
 def scales(self,mode='L',permutation=None):
  key=(mode,None if permutation is None else tuple(map(int,permutation)))
  if key in self.scale_cache:return self.scale_cache[key]
  I,G=self.components_training(tuple(range(self.m)),mode,permutation);off=~np.eye(self.n,dtype=bool)
  si=float(I[off].mean()) if self.n>1 else 0.;sg=float(G[off].mean()) if self.n>1 else 0.
  if sg==0:kind='constant_geometry';full=np.zeros_like(G)
  elif si==0:kind='zero_information_raw_fallback';full=G/sg
  else:kind='balanced';full=(1-self.zeta)*(I/si)+self.zeta*(G/sg)
  normalizer=float(full.max());normalizer=normalizer if normalizer>0 else 1.
  record={'information_full_mean':si,'raw_full_mean':sg,'fixed_full_max':normalizer,'kind':kind,'zeta':self.zeta};self.scale_cache[key]=record;return record
 def normalize_components(self,I,G,mode,permutation):
  s=self.scales(mode,permutation)
  if s['kind']=='constant_geometry':return np.zeros_like(G)
  if s['kind']=='zero_information_raw_fallback':return (G/s['raw_full_mean'])/s['fixed_full_max']
  return ((1-self.zeta)*(I/s['information_full_mean'])+self.zeta*(G/s['raw_full_mean']))/s['fixed_full_max']
 def directed_training(self,P,mode='L',permutation=None):
  if self.calibration_kind=='original':return super().directed_training(P,mode,permutation)
  I,G=self.components_training(P,mode,permutation);return self.normalize_components(I,G,mode,permutation)
 def incoming_selected(self,Q,P,mode='L',permutation=None):
  if self.calibration_kind=='original':return super().incoming_selected(Q,P,mode,permutation)
  P=self.selection(P);Q=np.asarray(Q,float)
  if Q.ndim!=2 or Q.shape[1]!=len(P) or not np.isfinite(Q).all():raise ValueError('Finite selected queries must follow P')
  W=self.weights(mode,permutation);I=np.zeros((len(Q),self.n));G=np.zeros_like(I)
  for j,a in enumerate(P):
   q=(Q[:,j]-self.lo[a])/self.span[a] if self.span[a]>0 else np.zeros(len(Q));raw=np.abs(q[:,None]-self.Z[None,:,a]);I+=raw*W[None,:,a]/self.m;G+=raw/self.m
  D=self.normalize_components(I,G,mode,permutation)
  if not np.isfinite(D).all():raise ValueError('Nonfinite calibrated query dissimilarity')
  return D

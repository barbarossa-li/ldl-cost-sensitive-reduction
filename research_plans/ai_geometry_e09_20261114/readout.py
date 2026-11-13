"""A compact, training-source-independent all-centroid inverse-Lp5 predictor."""
import numpy as np
from scipy.spatial.distance import cdist
from strong import ld_weights
class P5Predictor:
 def __init__(self,prototype,alpha):
  self.P=prototype.P.copy();self.lo=prototype.lo.copy();self.span=prototype.span.copy();self.centers=prototype.centers.copy();self.labels=prototype.labels.copy();self.alpha=float(alpha)
  if alpha!=0:self.p=prototype.p.copy()
  for v in vars(self).values():
   if isinstance(v,np.ndarray):v.setflags(write=False)
 def predict(self,Q):
  Q=np.asarray(Q,float)
  if Q.ndim!=2 or Q.shape[1]!=len(self.P) or not np.isfinite(Q).all():raise ValueError('Finite selected columns required')
  Z=np.divide(Q-self.lo,self.span,out=np.zeros_like(Q),where=self.span>0)
  v=ld_weights(cdist(Z,self.centers,'minkowski',p=5))@self.labels
  return v if self.alpha==0 else (1-self.alpha)*v+self.alpha*self.p
 def array_bytes(self):return sum(v.nbytes for v in vars(self).values() if isinstance(v,np.ndarray))

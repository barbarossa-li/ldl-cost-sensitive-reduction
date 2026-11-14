"""External library clustering constructors and an explicit LDLKM reconstruction."""
import warnings
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax
from sklearn.cluster import KMeans,BisectingKMeans
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from budget import canonical,validate

def exact_groups(labels,n,k):
 groups=canonical([tuple(np.flatnonzero(labels==j)) for j in np.unique(labels)]);initial=len(groups)
 while len(groups)<k:
  j=max(range(len(groups)),key=lambda j:(len(groups[j]),-min(groups[j])))
  B=groups.pop(j)
  if len(B)<2:raise ValueError('Cannot repair nonempty exactK')
  groups.extend([B[:-1],(B[-1],)]);groups=canonical(groups)
 return validate(groups,n,k),k-initial

def cluster(Y,k,method,seed):
 n=len(Y)
 if k==n:return [(i,) for i in range(n)],0,0
 cls=KMeans if method=='kmeanspp' else BisectingKMeans
 effective=min(k,len(np.unique(Y,axis=0)))
 kwargs=dict(n_clusters=effective,n_init=10,max_iter=300,tol=1e-4,random_state=seed%(2**32))
 kwargs.update(init='k-means++' if method=='kmeanspp' else 'random',algorithm='lloyd')
 if method=='bisecting':kwargs['bisecting_strategy']='biggest_inertia'
 with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always',ConvergenceWarning)
  labels=cls(**kwargs).fit_predict(Y)
 groups,repairs=exact_groups(labels,n,k)
 return groups,repairs,len(caught)

def ld_weights(distance):
 distance=np.asarray(distance,float)
 if distance.ndim!=2 or not np.isfinite(distance).all() or np.any(distance<0):raise ValueError('Invalid distance')
 out=np.zeros_like(distance)
 for i,d in enumerate(distance):
  zero=d==0
  if zero.any():out[i,zero]=1/int(zero.sum())
  else:
   # Shift before reciprocal evaluation: (1/d - 1/dmin) is nonpositive.
   # Longdouble prevents overflow for subnormal nonzero distances where available.
   v=d.astype(np.longdouble)
   with np.errstate(over='ignore'):
    logits=(v.min()/v-1)/v.min()
   out[i]=softmax(np.asarray(np.maximum(logits,-1e300),dtype=float))
 return out

def native_ldlkm(model,Q,P,seed):
 P=model.selection(P);Z=model.Z[:,P];k=min(5,model.n)
 with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always',ConvergenceWarning)
  labels=KMeans(n_clusters=min(k,len(np.unique(Z,axis=0))),init='random',n_init=1,max_iter=20,tol=1e-4,random_state=seed%(2**32),algorithm='lloyd').fit_predict(Z)
 groups=canonical([tuple(np.flatnonzero(labels==j)) for j in np.unique(labels)])
 centers=np.array([Z[list(B)].mean(0) for B in groups]);targets=np.array([model.L[list(B)].mean(0) for B in groups])
 span=model.span[list(P)];scaled=np.divide(Q[:,P]-model.lo[list(P)],span,out=np.zeros((len(Q),len(P))),where=span>0)
 distance=cdist(scaled,centers,'minkowski',p=5)
 pred=ld_weights(distance)@targets
 # Deployed state: column indices, scaling, centers, label means; prior only in adapted branch.
 native_bytes=sum(x.nbytes for x in [np.asarray(P,dtype=np.int64),model.lo[list(P)],span,centers,targets])
 return pred,{'K':len(groups),'requested_K':k,'native_array_bytes':native_bytes,'shrinkage_array_bytes':native_bytes+model.p.nbytes,'warnings':len(caught),'exactK_repair':False}

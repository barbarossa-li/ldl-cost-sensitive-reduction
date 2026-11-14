"""Budgeted geometric bisection and deterministic Lloyd reference partitions."""
import numpy as np
from scipy.spatial.distance import cdist

def canonical(groups):return sorted([tuple(sorted(map(int,B))) for B in groups],key=lambda B:B[0])
def validate(groups,n,k):
 assert len(groups)==k and all(len(B)>0 for B in groups) and sorted(i for B in groups for i in B)==list(range(n))
 return canonical(groups)
def distance_to_centroids(D,groups):
 return np.column_stack([np.maximum(0.,D[:,B].mean(1)-.5*D[np.ix_(B,B)].mean()) for B in groups])
def geometry(model,P,mode):
 P=model.selection(P);s=model.scales(mode)
 if s['kind']=='balanced':w=(1-model.zeta)*model.weights(mode)[0,list(P)]/s['information_full_mean']+model.zeta/s['raw_full_mean']
 else:w=np.ones(len(P))
 w=w/w.sum();Y=model.Z[:,P]*np.sqrt(w);return cdist(Y,Y,'sqeuclidean'),Y

def bisection_path(D,L,budgets,label_priority=.5):
 n=len(D);budgets=sorted(set(map(int,budgets)))
 if any(k<1 or k>n for k in budgets):raise ValueError('Require1<=K<=n')
 if not 0<=label_priority<=1:raise ValueError('Priority must be convex')
 DY=cdist(L,L,'sqeuclidean');sx=.5*D.mean();sy=.5*DY.mean();leaves=[tuple(range(n))];out={}
 def score(B):
  vx=.5*D[np.ix_(B,B)].mean()/sx if sx>0 else 0.;vy=.5*DY[np.ix_(B,B)].mean()/sy if sy>0 else 0.
  return len(B)*((1-label_priority)*vx+label_priority*vy)/n
 for k in range(1,max(budgets)+1):
  if k in budgets:out[k]=validate(leaves,n,k)
  if k==max(budgets):break
  candidates=[(score(B),len(B),-min(B),j) for j,B in enumerate(leaves) if len(B)>1];j=max(candidates)[-1];B=leaves.pop(j);sub=D[np.ix_(B,B)];ia,ib=np.unravel_index(np.argmax(sub),sub.shape)
  if sub[ia,ib]==0:
   left=B[:len(B)//2];right=B[len(B)//2:]
  else:
   labels=sub[:,ib]<sub[:,ia];left=tuple(B[i] for i in range(len(B)) if not labels[i]);right=tuple(B[i] for i in range(len(B)) if labels[i])
  assert left and right;leaves.extend([left,right])
 return out

def lloyd(D,k,max_iter=20):
 n=len(D)
 if k<1 or k>n:raise ValueError('Require1<=K<=n')
 if k==n:return [(i,) for i in range(n)]
 first=int(np.argmin(D.mean(1)));seeds=[first]
 while len(seeds)<k:
  dist=np.min(D[:,seeds],axis=1);dist[seeds]=-1;seeds.append(int(np.argmax(dist)))
 cost=D[:,seeds];previous=None
 for iteration in range(max_iter):
  labels=np.argmin(cost,axis=1);counts=np.bincount(labels,minlength=k)
  for empty in np.flatnonzero(counts==0):
   eligible=counts[labels]>1;errors=cost[np.arange(n),labels].copy();errors[~eligible]=-1;obj=int(np.argmax(errors));counts[labels[obj]]-=1;labels[obj]=empty;counts[empty]+=1
  groups=validate([tuple(np.flatnonzero(labels==j)) for j in range(k)],n,k)
  if groups==previous:return groups
  previous=groups;cost=distance_to_centroids(D,groups)
 return groups

def random_matched(groups,n,seed):
 order=np.random.default_rng(seed).permutation(n);out=[];pos=0
 for B in groups:out.append(tuple(order[pos:pos+len(B)]));pos+=len(B)
 assert pos==n;return validate(out,n,len(groups))

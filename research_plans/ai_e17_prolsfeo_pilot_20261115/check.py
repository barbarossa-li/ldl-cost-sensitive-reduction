import numpy as np,json
from reconstruction import *
rng=np.random.default_rng(72);X=rng.normal(size=(40,5));Y=rng.dirichlet(np.ones(3),40);folds=groups(X,19);bits=np.ones(40+3*5,bool);bits[40+2]=False;F=Fitness(X,Y,folds);v=F.prediction(bits);expected=np.zeros_like(Y);mask=bits[40:].reshape(3,5)
for held,fit,_ in F.parts:
 A,Q,_,_=scale(X[fit],X[held])
 for i,row in enumerate(Q):
  for j in range(3):
   distances=np.array([sum((row[d]-x[d])**2 for d in np.flatnonzero(mask[j])) for x in A]);ids=np.argsort(distances,kind='stable')[:4];expected[held[i],j]=Y[fit[ids],j].mean()
expected/=expected.sum(1)[:,None];assert np.allclose(v,expected,atol=1e-14,rtol=0)
held=folds[0];changed=Y.copy();changed[held]=[1.,0.,0.];assert np.array_equal(v[held],Fitness(X,changed,folds).prediction(bits)[held])
x=np.zeros(21,bool);y=np.ones(21,bool);a,b=hux(x,y,rng);assert np.sum(a!=x)==10 and np.sum(b!=y)==10 and np.all(a!=b)
a=optimize(X,Y,41,population=8,generations=4,threshold=3,deadline_seconds=60);b=optimize(X,Y,41,population=8,generations=4,threshold=3,deadline_seconds=60);assert np.array_equal(a[0],b[0]) and a[2]['status']==b[2]['status']=='complete'
st=deploy_state(X,Y,a[0]);z=predict(st,X[:3]);assert np.isfinite(z).all() and np.allclose(z.sum(1),1)
xx=X.copy();xx[-4:]=xx[:4];ff=groups(xx,43);_,g=np.unique(xx,axis=0,return_inverse=True)
for i in range(10):
 for j in range(i):assert not set(g[ff[i]])&set(g[ff[j]])
print(json.dumps({'passed':True,'checks':['Ten-fold label-specific fitness predictions agree with independent scalar-distance reference','Inner held-label replacement cannot affect predictions of that held fold','HUX swaps floor(half differing bits), preserving paired bit complementarity','Synthetic optimization is deterministic and reaches declared generation count','Exported label-specific numeric state gives valid simplex predictions','Exact-feature duplicate groups stay together'],'scope':'Synthetic reduced population/generation checks only; not paper-setting benchmark evidence'},indent=2))

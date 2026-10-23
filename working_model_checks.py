"""Cross-check adopted formulas, edge cases and subset-search integration."""
from pathlib import Path
import numpy as np
from granulation_reference import TrainingGeometry,granulate
from reduction_reference import heuristic,exact_backtracking
X=np.array([[0.,'A','constant'],[.5,'B','constant'],[1.,'C','constant']],dtype=object)
L=np.array([[1.,0.],[.5,.5],[0.,1.]])
model=TrainingGeometry(X,L,kinds=['numerical','categorical','categorical'])
for P,Q in [((),(0,)),((0,),(0,1)),((0,1),(0,1,2))]:
 d=model.directed(P);e=model.directed(Q)
 assert np.all(d<=e+1e-14) and np.max(e)<=len(Q)/3+1e-14
 assert np.allclose(np.diag(e),0)
assert np.all(model.directed([2])==0)
assert np.any(np.abs(model.directed([0])-model.directed([0]).T)>1e-8)
# Numerical triangle counterexample under every tested nondegenerate mixture.
for zeta in [.001,.1,.5,.99]:
 counter=TrainingGeometry(X[:,:1],L,kinds=['numerical'],zeta=zeta,bandwidths=[.5])
 d=counter.directed([0]);gap=d[0,2]-d[0,1]-d[1,2]
 assert gap>0 and np.isclose(gap,(1-zeta)*counter.weights[0][0]/2)
print('Triangle gap at zeta=.1:',(model.directed([0])[0,2]-model.directed([0])[0,1]-model.directed([0])[1,2])*3)
# Numeric and categorical signal, inactive labels, constant padding, soft scaling.
lab=np.array([[.9,.1,0],[.9,.1,0],[.1,.9,0],[.1,.9,0]])
for kind in ['numerical','categorical']:
 fitted=TrainingGeometry([[0],[0],[1],[1]],lab,kinds=[kind]);d,k=fitted.subset([0]);g=granulate(d,lab,k)[2];assert np.isclose(g,1)
 padded=TrainingGeometry([[0,7],[0,7],[1,7],[1,7]],lab,kinds=[kind,'numerical']);e,j=padded.subset([0]);assert np.allclose(e,d/2);assert np.isclose(granulate(e,lab,j)[2],g)
# XOR joint gain is now visible to both searches after a zero-gain bridge.
x=np.repeat(np.array([[0,0],[0,1],[1,0],[1,1]]),2,axis=0)
pfirst=np.array([.9 if a==b else .1 for a,b in x]);y=np.c_[pfirst,1-pfirst]
fitted=TrainingGeometry(x,y)
def dep(P):
 d,k=fitted.subset(P);return granulate(d,y,k)[2]
assert dep((0,))==dep((1,))==0 and np.isclose(dep((0,1)),1)
h=heuristic([1,2],dep);e=exact_backtracking([1,2],dep)
assert h[0]==e[0]==(0,1) and h[3][0][3]=='nonpositive bridge'
# Recompute the saved negative gain under the authoritative configuration.
data=np.loadtxt(Path(__file__).resolve().parent/'negative_gain_construction.csv',delimiter=',',skiprows=1)
fitted=TrainingGeometry(data[:,:2].astype(int),data[:,2:]);values=[]
for P in [[0],[0,1]]:
 d,k=fitted.subset(P);values.append(granulate(d,data[:,2:],k)[2])
assert values[1]<values[0]
text=('Adopted working model checks passed:\nBounds, zero diagonal, subset monotonicity, genuine direction difference.\nNumerical triangle failure at zeta=.001,.1,.5,.99.\nNumerical/categorical signal, inactive label and constant-padding checks.\nRepeated-XOR joint dependency 1; both searches reach the pair through a zero-gain bridge.\nSaved negative gain reproduced: '+str(values)+'\nSynthetic mechanism evidence only.\n')
print(text);(Path(__file__).resolve().parent/'working_model_checks.txt').write_text(text)

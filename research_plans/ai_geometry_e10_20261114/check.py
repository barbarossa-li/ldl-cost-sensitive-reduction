"""Synthetic invariance checks only; no benchmark evidence."""
from pathlib import Path
import tempfile,json,copy
import numpy as np
from runner import grid,one_case,design,splits,CalibratedPrototypeModel,PrototypePredictor

def main():
 rng=np.random.default_rng(91);X=rng.normal(size=(24,5));L=rng.dirichlet(np.ones(3),24);Q=rng.normal(size=(7,5));masks={'Full':tuple(range(5)),'cost40':(0,2)}
 pred,stats=grid(X,L,Q,masks,123);assert len(pred)==321 and len(stats)==320;checks=[]
 for s in stats:
  mask,c,method,budget,q,a=s['key'].split('__');d=len(masks[mask]);assert s['array_bytes']==8*(s['K']*(d+3+1)+4*d+3)
 checks.append('All four geometry cells have equal actual array bytes andK; state formula verified')
 for mask in masks:
  for method in ['geometric','kmeanspp','bisecting','paper20']:
   for q in ['U','G']:
    for a in ['0.0','0.5']:assert np.array_equal(pred[f'{mask}__U__{method}__100__{q}__a{a}'],pred[f'{mask}__G__{method}__100__{q}__a{a}'])
 checks.append('Construction has exactly zero effect at singleton endpoint for fixed query geometry')
 for mask in masks:
  for method in ['geometric','kmeanspp','bisecting','paper20']:
   for c in ['U','G']:
    for a in ['0.0','0.5']:assert np.max(abs(pred[f'{mask}__{c}__{method}__K5__U__a{a}']-pred[f'{mask}__{c}__{method}__K5__G__a{a}']))<=1e-15
 checks.append('K5 query contrast is zero within summation tolerance')
 m=CalibratedPrototypeModel(X,L,zeta=.5);proto=PrototypePredictor(m,(0,2),'G');state=proto.state();expected=proto.predict(Q[:,(0,2)]);del m,proto
 assert np.array_equal(PrototypePredictor.from_state(copy.deepcopy(state)).predict(Q[:,(0,2)]),expected)
 checks.append('NN5 inference does not retain training model')
 const,_=grid(np.ones((12,5)),np.tile([.2,.3,.5],(12,1)),Q,masks,123);assert all(np.isfinite(v).all() and np.allclose(v.sum(1),1) for v in const.values())
 checks.append('Constant and repeated features retain valid exactK partitions and probabilities')
 f={'family':'synthetic','N':48,'m':5,'q':3};x=rng.normal(size=(48,5));y=rng.dirichlet(np.ones(3),48);pool,_,_=design(f,0);held=pool[splits(len(pool),202611135000)[0]];changed=y.copy();changed[held]=[1.,0.,0.]
 with tempfile.TemporaryDirectory() as tmp:
  a=Path(tmp)/'a';b=Path(tmp)/'b';one_case(x,y,f,0,0,a);one_case(x,changed,f,0,0,b)
  with np.load(a/'predictions_PRIVATE.npz') as aa,np.load(b/'predictions_PRIVATE.npz') as bb:
   for key in aa.files:assert np.array_equal(aa[key],bb[key])
 checks.append('Heldout label replacement leaves all321 predictions unchanged')
 print(json.dumps({'passed':True,'scope':'Synthetic implementation invariants, not benchmark evidence','checks':checks},indent=2))
if __name__=='__main__':main()

"""Synthetic factorial and deployment checks; not benchmark evidence."""
from pathlib import Path
import json,tempfile,copy
import numpy as np
from runner import grid,one_case,design,splits
from readout import P5Predictor
from prototype import PrototypePredictor
from calibrated import CalibratedPrototypeModel

def main():
 rng=np.random.default_rng(9);X=rng.normal(size=(24,5));L=rng.dirichlet(np.ones(3),24);Q=rng.normal(size=(7,5));masks={'Full':tuple(range(5)),'cost40':(0,2)};v,stats=grid(X,L,Q,masks,123);assert len(v)==321;checks=[]
 for mask in masks:
  for g in ['U','G']:
   for readout in ['nn5','allp5']:
    for alpha in [0.,.5]:
     b=v[f'{mask}__{g}__geometric__100__{readout}__a{alpha}']
     for m in ['kmeanspp','bisecting','paper20']:assert np.array_equal(b,v[f'{mask}__{g}__{m}__100__{readout}__a{alpha}'])
 checks.append('Both readouts have identical singleton endpoints across four constructors, not necessarily across readouts')
 for s in stats:
  mask,g,method,budget,readout,alpha=s['key'].split('__');d=len(masks[mask]);a=float(alpha[1:]);k=s['K'];expected=8*(k*(d+3+1)+4*d+3) if readout=='nn5' else 8*(k*(d+3)+3*d+(3 if a else 0));assert s['array_bytes']==expected
 checks.append('Measured state bytes equal each readout formula and match constructors at equalK')
 model=CalibratedPrototypeModel(X,L,zeta=.5);proto=PrototypePredictor(model,(0,2),'G');state=P5Predictor(proto,.5);before=state.predict(Q[:,(0,2)]);del model,proto
 after=copy.deepcopy(state).predict(Q[:,(0,2)]);assert np.array_equal(before,after)
 checks.append('P5 inference state operates independently of training source and prototype object')
 # grid internally asserts U/paper20/K5/allp5 1e-12 prediction/byte equality to E08 native reconstruction.
 checks.append('LDLKM K5 prediction anchor within1e-12 and exact numeric byte identity in each U mask')
 const,_=grid(np.ones((12,5)),np.tile([.2,.3,.5],(12,1)),Q,masks,123);assert all(np.isfinite(x).all() and np.allclose(x.sum(1),1) for x in const.values())
 checks.append('Duplicate/constant-feature partitions and zero-distance prediction remain valid')
 f={'family':'synthetic','N':48,'m':5,'q':3};x=rng.normal(size=(48,5));y=rng.dirichlet(np.ones(3),48);pool,_,_=design(f,0);held=pool[splits(len(pool),202611135000)[0]];changed=y.copy();changed[held]=[1.,0.,0.]
 with tempfile.TemporaryDirectory() as tmp:
  a=Path(tmp)/'a';b=Path(tmp)/'b';one_case(x,y,f,0,0,a);one_case(x,changed,f,0,0,b)
  with np.load(a/'predictions_PRIVATE.npz') as aa,np.load(b/'predictions_PRIVATE.npz') as bb:
   for key in aa.files:assert np.array_equal(aa[key],bb[key])
 checks.append('Adversarial heldout-label replacement cannot alter any frozen prediction')
 print(json.dumps({'passed':True,'checks':checks,'scope':'Synthetic checks only; no new benchmark results'},indent=2))
if __name__=='__main__':main()

"""Technical checks only; no benchmark datasets or scores."""
import json,tempfile
from pathlib import Path
import numpy as np
from runner import select_budget,partitions,predict_grid,design,one_case

def main():
 checks=[]
 assert select_budget({25:.2,50:.1,75:.3,100:.05},50)==50
 assert select_budget({25:.1,50:.1+5e-13,75:.05,100:.01},50)==25
 assert select_budget({25:.2,50:.1,75:.3,100:.05},100)==100
 try:select_budget({25:float('nan')},25);raise AssertionError('NaN accepted')
 except ValueError:pass
 checks.append('Budget cap, deterministic tolerance tie and nonfinite rejection')
 rng=np.random.default_rng(701);X=rng.normal(size=(24,6));L=rng.dirichlet(np.ones(3),24);Q=rng.normal(size=(7,6));masks={'Full':tuple(range(6)),'cost40':(0,2)}
 v,st=predict_grid(X,L,Q,masks)
 for mask in masks:
  for g in ['U','G']:
   base=v[f'{mask}__{g}__geometric__100']
   assert np.array_equal(base,v[f'{mask}__{g}__supervised__100'])
   assert np.array_equal(base,v[f'{mask}__{g}__lloyd__100'])
 for row in st:
  key=row['key'];d=len(masks[key.split('__')[0]]);k=row['K']
  assert row['array_bytes']==8*(k*(d+3+1)+4*d+3)
 checks+=['Full endpoint equality across three constructors and two geometries','Exact capacity formula and fixed-budget bytes matching']
 # Deliberate validation-label alteration cannot enter the feature-only query API.
 assert np.array_equal(v['Full__G__geometric__25'],predict_grid(X,L,Q,masks)[0]['Full__G__geometric__25'])
 vf,_=predict_grid(np.ones((12,6)),np.tile([.2,.3,.5],(12,1)),Q,masks)
 for value in vf.values():assert np.isfinite(value).all() and np.allclose(value.sum(1),1)
 checks+=['Deterministic feature-only prediction interface','Constant-feature/zero-information fallback']
 p=json.loads(Path(__file__).with_name('PROTOCOL.json').read_text())
 for fi,f in enumerate(p['families']):
  pool,excluded,masks=design(f,fi);assert not set(pool)&set(excluded)
  blocks=partitions(len(pool),202611134000+10000*fi+1000)
  assert sorted(np.concatenate(blocks))==list(range(len(pool)))
  for fold,test in enumerate(blocks):
   fit=np.setdiff1d(np.arange(len(pool)),test)
   for iv in partitions(len(fit),202611140700+10000*fi+100*fold):
    itr=np.setdiff1d(np.arange(len(fit)),iv)
    assert not set(fit[itr])&set(fit[iv]) and not set(test)&set(fit)
 checks.append('All16 family outer/inner/excluded-original-test index separation')
 # Corrupt only outer holdout labels: no selected setting or prediction may change.
 f={'family':'synthetic','N':48,'m':6,'q':3};xs=rng.normal(size=(48,6));ls=rng.dirichlet(np.ones(3),48)
 pool,_,_=design(f,0);outer=partitions(len(pool),202611135000);held=pool[outer[0]];changed=ls.copy();changed[held]=np.tile([1.,0.,0.],(len(held),1))
 with tempfile.TemporaryDirectory() as td:
  a=Path(td)/'a';b=Path(td)/'b';one_case(xs,ls,f,0,0,a);one_case(xs,changed,f,0,0,b)
  ra=json.loads((a/'record_PRIVATE.json').read_text());rb=json.loads((b/'record_PRIVATE.json').read_text())
  assert ra['inner_risks']==rb['inner_risks'] and ra['selected_percent']==rb['selected_percent']
  with np.load(a/'predictions_PRIVATE.npz') as av,np.load(b/'predictions_PRIVATE.npz') as bv:
   for key in av.files:assert np.array_equal(av[key],bv[key])
 checks.append('Adversarial outer-holdout-label replacement leaves all inner risks, selected budgets and predictions unchanged')
 print(json.dumps({'passed':True,'scope':'Synthetic technical checks; no benchmark predictions or new result claims','checks':checks},indent=2))
if __name__=='__main__':main()

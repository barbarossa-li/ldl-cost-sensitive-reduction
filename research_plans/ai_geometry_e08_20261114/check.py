"""E08 technical checks, no benchmark score generation."""
from pathlib import Path
import sys,tempfile,json
import numpy as np
from runner import predict_grid,select_budget,partitions,design,one_case
from strong import cluster,ld_weights
from scipy.special import softmax

def main():
 rng=np.random.default_rng(8);checks=[];X=rng.normal(size=(24,5));L=rng.dirichlet(np.ones(3),24);Q=rng.normal(size=(7,5));masks={'Full':tuple(range(5)),'cost40':(0,3)}
 for method in ['kmeanspp','bisecting']:
  for Y in [X,np.ones_like(X),np.repeat(X[:2],12,axis=0)]:
   for k in [1,6,12,24]:
    groups,repair,warn=cluster(Y,k,method,8);assert len(groups)==k and sorted(i for B in groups for i in B)==list(range(24));assert cluster(Y,k,method,8)==(groups,repair,warn)
 checks.append('Standard library constructors deterministic, exactK and full coverage, with declared duplicate repair')
 dist=np.array([[.2,.3,1.],[0.,0.,1.],[1e-320,2e-320,1.]])
 w=ld_weights(dist);assert np.allclose(w[0],softmax(1/dist[0]));assert np.array_equal(w[1],[.5,.5,0]);assert np.isfinite(w).all() and np.allclose(w.sum(1),1)
 checks.append('LDLKM positive-distance inverse-softmax identity, declared zero-distance rule and extreme small distances')
 pred,stats=predict_grid(X,L,Q,masks,8);assert len(pred)==133
 for mask in masks:
  for g in ['U','G']:
   for a in [0.,.5]:
    base=pred[f'{mask}__{g}__geometric__100__a{a}']
    for m in ['lloyd','kmeanspp','bisecting']:assert np.array_equal(base,pred[f'{mask}__{g}__{m}__100__a{a}'])
 for row in stats:
  if len(row['key'].split('__'))==5:
   d=len(masks[row['key'].split('__')[0]]);assert row['array_bytes']==8*(row['K']*(d+3+1)+4*d+3)
 checks.append('All four constructors have same bytes per exactK and identical singleton endpoints under both priors')
 const,_=predict_grid(np.ones((12,5)),np.tile([.2,.3,.5],(12,1)),Q,masks,8)
 assert all(np.isfinite(v).all() and np.allclose(v.sum(1),1) for v in const.values())
 checks.append('Constant-feature and duplicate-center prototype predictions remain simplex-valued')
 assert select_budget({25:.2,50:.1,75:.1+5e-13,100:.3},100)==50
 checks.append('Inner pooled risk selection and smaller-capacity deterministic tie')
 p=json.loads(Path(__file__).with_name('PROTOCOL.json').read_text())
 for fi,f in enumerate(p['families']):
  pool,exclude,_=design(f,fi);assert not set(pool)&set(exclude)
  for fold,iv in enumerate(partitions(len(pool),202611134000+10000*fi+1000)):
   fit=np.setdiff1d(np.arange(len(pool)),iv)
   for inner in partitions(len(fit),202611140700+10000*fi+100*fold):assert set(fit[inner]).isdisjoint(iv)
 checks.append('All sixteen family fold designs exclude original outer-test indices')
 f={'family':'synthetic','N':48,'m':5,'q':3};x=rng.normal(size=(48,5));y=rng.dirichlet(np.ones(3),48);pool,_,_=design(f,0);held=pool[partitions(len(pool),202611135000)[0]];changed=y.copy();changed[held]=[1.,0.,0.]
 with tempfile.TemporaryDirectory() as tmp:
  a=Path(tmp)/'a';b=Path(tmp)/'b';one_case(x,y,f,0,0,a);one_case(x,changed,f,0,0,b);ra=json.loads((a/'record_PRIVATE.json').read_text());rb=json.loads((b/'record_PRIVATE.json').read_text());assert ra['inner_risks']==rb['inner_risks'] and ra['selected_percent']==rb['selected_percent']
  with np.load(a/'predictions_PRIVATE.npz') as aa,np.load(b/'predictions_PRIVATE.npz') as bb:
   assert len(aa.files)==165
   for key in aa.files:assert np.array_equal(aa[key],bb[key])
 checks.append('Adversarial holdout-label mutation leaves all inner selection and predictions unchanged')
 print(json.dumps({'passed':True,'checks':checks,'scope':'Synthetic checks only; no benchmark prediction or performance claim'},indent=2))
if __name__=='__main__':main()

from pathlib import Path
import os,sys,json
os.environ['LDL_PARENT_SOURCE']=os.environ.get('LDL_PARENT_SOURCE','/workspace/work/ai_external_v1_20261114/source')
import numpy as np
from scipy.optimize import minimize
import learners as l
from study import select_and_fit

def main():
 rng=np.random.default_rng(93);X=rng.uniform(size=(60,5));Y=rng.dirichlet(np.ones(4),60);masks={'Full':tuple(range(5))};checks=[]
 for v in rng.normal(size=(4,4)):
  z=l.project(v[None,:])[0];r=minimize(lambda x:np.sum((x-v)**2),np.full(4,.25),method='SLSQP',bounds=[(0,1)]*4,constraints={'type':'eq','fun':lambda x:x.sum()-1},options={'ftol':1e-12});assert r.success and np.linalg.norm(z-r.x)<1e-6 and abs(z.sum()-1)<1e-12
 checks.append('Simplex projection agrees with independent constrained quadratic optimization')
 xx=X.copy();xx[50:]=xx[:10];parts=l.group_folds(xx,7);_,groups=np.unique(xx,axis=0,return_inverse=True)
 for i in range(3):
  for j in range(i):assert not set(groups[parts[i]])&set(groups[parts[j]])
 checks.append('Exact-feature duplicate groups stay in one shared inner fold')
 a=select_and_fit(X[:48],Y[:48],X[48:],masks,0,0,71,epochs=(1,2));changed=Y.copy();changed[48:]=[1.,0.,0.,0.];b=select_and_fit(X[:48],changed[:48],X[48:],masks,0,0,71,epochs=(1,2))
 assert a[6]['failures']==b[6]['failures']=={} and len(a[0])==11
 for key in a[0]:assert np.array_equal(a[0][key],b[0][key])
 assert a[2]==b[2]
 checks.append('Full selection pipeline, including independent RKNN states, reproduces exactly and ignores outer query-label replacement')
 assert l.byte_count(a[1]['Full__G_atU'])==l.byte_count(a[1]['Full__U_min'])
 checks.append('Uniform-selected matchedG reference has identical actual state bytes')
 for key,st in a[1].items():assert np.allclose(l.predict(st,X[48:],native=('native_atSelected' in key)),a[0][key],atol=1e-7,rtol=0)
 checks.append('Every selected predictor is reconstructed from standalone numeric state')
 model=l.r.CalibratedPrototypeModel(X[:48],Y[:48],zeta=.5)
 for geom in ['G','U']:
  vv,pp=l.proto_candidates(model,X[48:],masks['Full'],geom)
  for key,params in pp.items():
   state=l.with_prior(l.prototype_state(model,masks['Full'],geom,params['budget'],params['k']),params['alpha'],model.p)
   assert np.allclose(l.predict(state,X[48:]),vv[key],atol=1e-12,rtol=0)
 checks.append('All64 geometric and uniform budget/neighborhood/prior candidate predictions agree with independent standalone reconstruction')
 print(json.dumps({'passed':True,'checks':checks,'scope':'Synthetic checks with RKNNepochs1/2 only. Real-data protocol uses100/500/1000; no real-data outcomes used here.'},indent=2))
if __name__=='__main__':main()

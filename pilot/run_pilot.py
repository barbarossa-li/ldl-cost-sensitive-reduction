"""Small real-LDL pilot; fixed protocol, no selection from test results.
Natural Scene authors' files. Dense reference, not final benchmark runner.
"""
from pathlib import Path
import sys,time,json,csv
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from granulation_reference import TrainingGeometry,granulate
from reduction_reference import heuristic,exact_backtracking
from scipy.spatial.distance import cdist
p=Path(__file__).resolve().parent
X=np.load(p/'data/feature.npy',allow_pickle=False);L=np.load(p/'data/label.npy',allow_pickle=False)
assert X.shape==(2000,294) and L.shape==(2000,9)
assert np.isfinite(X).all() and np.isfinite(L).all() and np.all(L>=0) and np.allclose(L.sum(1),1,atol=1e-12)
seed=20260930;order=np.random.default_rng(seed).permutation(len(X));train=order[:128];test=order[128:384]
x=X[train];lab=L[train];xt=X[test];truth=L[test];m=X.shape[1]
costs=np.random.default_rng(seed+1).integers(1,11,m)
lo=x.min(0);span=x.max(0)-lo;span[span==0]=1.;scaled=(x-lo)/span;scaled_test=(xt-lo)/span
v0=float(np.mean(np.sum((lab-lab.mean(0))**2,axis=1)))
protocol={'dataset':'Natural Scene','seed':seed,'cost_seed':seed+1,'source_shape':list(X.shape),'labels':9,'train_indices':train.tolist(),'test_indices':test.tolist(),'unused_count':len(X)-len(train)-len(test),'costs':costs.tolist(),'label_global_variation':v0,'settings':{'beta':1,'smin':2,'alpha':1,'kappa':.1,'zeta':.1,'lambda':-1,'epsilon':0,'knn_k':5},'note':'Simulated acquisition costs; single subsampled split; no tuning using test labels.'}
(p/'protocol.json').write_text(json.dumps(protocol,indent=2))
start=time.perf_counter();model=TrainingGeometry(x,lab,kinds=['numerical']*m);fitsec=time.perf_counter()-start
print('Fitted training geometry',round(fitsec,3),'seconds; V0',v0,flush=True)

def evaluate(P,zeta=.1,kappa=.1):
 D,keys=model.subset(P,zeta=zeta);leaves,eligible,gamma,history=granulate(D,lab,keys,kappa=kappa)
 return {'gamma':gamma,'coverage':sum(len(B) for B in eligible)/len(train),'leaves':len(leaves),'eligible_balls':len(eligible),'unresolved_balls':len(leaves)-len(eligible),'singleton_leaves':sum(len(B)==1 for B in leaves),'accepted_splits':len(history)}

def prediction(P):
 P=list(P)
 if not P:return np.tile(lab.mean(0),(len(test),1))
 distances=cdist(scaled_test[:,P],scaled[:,P],metric='sqeuclidean');neighbors=np.argsort(distances,axis=1,kind='stable')[:,:5]
 return lab[neighbors].mean(1)

def losses(pred):
 pred=np.maximum(pred,1e-12);pred=pred/pred.sum(1,keepdims=True)
 mid=(truth+pred)/2
 return {'brier':float(np.mean(np.sum((truth-pred)**2,axis=1))),
         'kl':float(np.mean(np.sum(np.where(truth>0,truth*np.log(np.maximum(truth,1e-300)/pred),0),axis=1))),
         'tv':float(np.mean(.5*np.abs(truth-pred).sum(1))),
         'js':float(np.mean(.5*np.sum(truth*np.log(np.maximum(truth,1e-300)/mid),axis=1)+.5*np.sum(pred*np.log(pred/mid),axis=1)))}
rows=[];traces={}
def addrow(name,P,zeta=.1,searchsec=0.,evals=0):
 info=evaluate(P,zeta=zeta);row={'method':name,'attributes':len(P),'cost':int(sum(costs[a] for a in P)),'selected':json.dumps(list(map(int,P))),'search_seconds':searchsec,'distinct_search_evaluations':evals,**info,**losses(prediction(P))};rows.append(row);print(name,row,flush=True)
addrow('All attributes',tuple(range(m)))
addrow('Global mean',())
for zeta,name in [(.1,'Adopted heuristic'),(0.,'Information-only ablation')]:
 start=time.perf_counter();cache={}
 def dep(P):
  key=tuple(P)
  if key not in cache:cache[key]=evaluate(P,zeta=zeta)
  return cache[key]['gamma']
 h=heuristic(costs,dep,lam=-1.);seconds=time.perf_counter()-start
 traces[name]={'target':h[2],'steps':h[3],'cache_size':len(cache)}
 addrow(name,h[0],zeta=zeta,searchsec=seconds,evals=h[4])
 if name=='Adopted heuristic':selected=h[0]
k=len(selected)
addrow('Cheapest same cardinality',tuple(sorted(np.argsort(costs,kind='stable')[:k])))
random_order=np.random.default_rng(seed+2).permutation(m)
addrow('Random same cardinality',tuple(sorted(random_order[:k])))
with (p/'pilot_results.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(p/'heuristic_traces.json').write_text(json.dumps(traces,indent=2))
# Prespecified training-only sensitivity checks. No held-out losses used here.
sensitivity=[]
for zeta in [0.,.1,.25]:
 for kappa in [.1,.25,.5]:
  info=evaluate(tuple(range(m)),zeta=zeta,kappa=kappa);sensitivity.append({'zeta':zeta,'kappa':kappa,**info})
with (p/'training_sensitivity.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(sensitivity[0]));w.writeheader();w.writerows(sensitivity)
# Exactness refers only to a screened eight-feature subproblem, NOT all 294.
meanw=np.mean(np.stack(model.weights),axis=1);candidates=tuple(sorted(sorted(range(m),key=lambda a:(-meanw[a],a))[:8]))
def restricted(P):return evaluate(tuple(candidates[a] for a in P))['gamma']
cc=[int(costs[a]) for a in candidates]
start=time.perf_counter();hh=heuristic(cc,restricted);hs=time.perf_counter()-start
start=time.perf_counter();ee=exact_backtracking(cc,restricted);es=time.perf_counter()-start
exact={'scope':'8 features screened by training-only mean information weight; target is Gamma(candidate set), not Gamma(all 294)','candidates':list(map(int,candidates)),'costs':cc,'target':ee[2],'heuristic_selected_original':[int(candidates[a]) for a in hh[0]],'heuristic_cost':hh[1],'heuristic_seconds':hs,'optimal_selected_original':[int(candidates[a]) for a in ee[0]],'optimal_cost':ee[1],'exact_seconds':es,'exact_nodes':ee[3],'exact_distinct_evaluations':ee[4],'completed':True,'degenerate':ee[2]==0,'excess_cost_ratio':((hh[1]-ee[1])/ee[1] if ee[1]>0 else None)}
(p/'restricted_exact.json').write_text(json.dumps(exact,indent=2));print('Restricted exact comparison',exact,flush=True)
(p/'runtime.json').write_text(json.dumps({'geometry_fit_seconds':fitsec,'note':'Single-process timings for this workspace; not hardware-independent benchmarks.'},indent=2))

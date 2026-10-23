"""Fresh real-LDL pilot with three-fold TRAINING-INTERNAL Brier guard.
Prior pilot's 384 objects are excluded; outer test labels never select features.
"""
from pathlib import Path
import sys,time,json,csv
import numpy as np
from scipy.spatial.distance import cdist
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from granulation_reference import TrainingGeometry,granulate
from reduction_reference import heuristic,exact_backtracking
p=Path(__file__).resolve().parent;root=p.parent
X=np.load(root/'pilot/data/feature.npy');L=np.load(root/'pilot/data/label.npy')
prior=json.loads((root/'pilot/protocol.json').read_text());used=set(prior['train_indices']+prior['test_indices'])
unused=np.array([i for i in range(len(X)) if i not in used]);seed=20261030
order=np.random.default_rng(seed).permutation(unused);train=order[:192];test=order[192:448]
assert set(train).isdisjoint(test) and used.isdisjoint(set(train)|set(test))
x=X[train];lab=L[train];xt=X[test];truth=L[test];m=X.shape[1];costs=np.array(prior['costs'])
fold_seed=20261031;folds=np.array_split(np.random.default_rng(fold_seed).permutation(len(train)),3)
protocol={'dataset':'Natural Scene','seed':seed,'inner_fold_seed':fold_seed,'train_indices':train.tolist(),'test_indices':test.tolist(),'excluded_previous_count':len(used),'train_count':192,'test_count':256,'inner_validation_positions':[v.tolist() for v in folds],'costs':costs.tolist(),'settings':{'zeta':.1,'kappa':.1,'beta':1,'smin':2,'alpha':1,'lambda':-1,'dependency_tolerance':0.,'prediction_tolerance':0.,'inner_folds':3,'knn_k':5},'note':'Fresh objects; simulated costs; fixed protocol; no test-selected tolerances.'}
(p/'protocol.json').write_text(json.dumps(protocol,indent=2))
start=time.perf_counter();model=TrainingGeometry(x,lab,kinds=['numerical']*m);fitsec=time.perf_counter()-start
print('Outer training geometry fitted',fitsec,flush=True)
class InnerLoss:
 def __init__(self):
  self.blocks=[];self.cache={};self.fold_scores={}
  for validation in folds:
   fit=np.setdiff1d(np.arange(len(train)),validation);a=x[fit];b=x[validation]
   lo=a.min(0);span=a.max(0)-lo;span[span==0]=1.
   a=(a-lo)/span;b=(b-lo)/span
   matrices=((b[:,None,:]-a[None,:,:])**2).transpose(2,0,1).copy()
   self.blocks.append((matrices,fit,validation))
 def __call__(self,P):
  key=tuple(sorted(P))
  if key not in self.cache:
   scores=[];total=0.;count=0
   for matrices,fit,validation in self.blocks:
    if key:
     distances=matrices[list(key)].sum(0);neighbors=np.argsort(distances,axis=1,kind='stable')[:,:5];pred=lab[fit][neighbors].mean(1)
    else:pred=np.tile(lab[fit].mean(0),(len(validation),1))
    losses=np.sum((lab[validation]-pred)**2,axis=1);total+=losses.sum();count+=len(validation);scores.append(float(losses.mean()))
   self.cache[key]=float(total/count);self.fold_scores[key]=scores
  return self.cache[key]
inner=InnerLoss();full=tuple(range(m));reference=inner(full)
print('Inner full-feature Brier reference',reference,flush=True)
train_cache={}
def stats(P):
 key=tuple(sorted(P))
 if key not in train_cache:
  D,keys=model.subset(key);leaves,eligible,gamma,hist=granulate(D,lab,keys)
  train_cache[key]={'gamma':gamma,'coverage':sum(len(B) for B in eligible)/len(train),'eligible_balls':len(eligible),'leaves':len(leaves)}
 return train_cache[key]
def dep(P):return stats(P)['gamma']
lo=x.min(0);span=x.max(0)-lo;span[span==0]=1.;scaled=(x-lo)/span;scaled_test=(xt-lo)/span
def predict(P):
 if not P:return np.tile(lab.mean(0),(len(test),1))
 dist=cdist(scaled_test[:,P],scaled[:,P],metric='sqeuclidean');neighbors=np.argsort(dist,axis=1,kind='stable')[:,:5]
 return lab[neighbors].mean(1)
def metrics(P):
 pred=np.maximum(predict(tuple(P)),1e-12);pred/=pred.sum(1,keepdims=True);mid=(truth+pred)/2
 return {'test_brier':float(np.mean(np.sum((truth-pred)**2,axis=1))),'test_kl':float(np.mean(np.sum(truth*np.log(np.maximum(truth,1e-300)/pred),axis=1))),'test_tv':float(np.mean(.5*np.abs(truth-pred).sum(1))),'test_js':float(np.mean(.5*np.sum(truth*np.log(np.maximum(truth,1e-300)/mid),axis=1)+.5*np.sum(pred*np.log(pred/mid),axis=1)))}
# Select EVERY subset before computing any outer test loss.
selected={'All attributes':full,'Global mean':()};searchinfo={}
for name,guard in [('Dependency-only',None),('Guarded heuristic',inner)]:
 start=time.perf_counter();before_dep=len(train_cache);before_risk=len(inner.cache)
 h=heuristic(costs,dep,prediction_loss=guard,prediction_tolerance=0.)
 selected[name]=h[0];searchinfo[name]={'seconds':time.perf_counter()-start,'trace':h[3],'dependency_target':h[2],'distinct_dependency_evaluations_this_search':h[4],'new_shared_dependency_evaluations':len(train_cache)-before_dep,'new_inner_loss_evaluations':len(inner.cache)-before_risk}
 print('Selected',name,'features',len(h[0]),'cost',h[1],'Gamma',dep(h[0]),'inner loss',inner(h[0]),'search seconds',searchinfo[name]['seconds'],flush=True)
 if guard is not None:assert inner(h[0])<=reference and dep(h[0])>=dep(full)
# Component ablation: predictive wrapper without a dependency requirement.
start=time.perf_counter();before_risk=len(inner.cache)
h=heuristic(costs,lambda P:0.,prediction_loss=inner,prediction_tolerance=0.)
selected['Prediction-only ablation']=h[0]
searchinfo['Prediction-only ablation']={'seconds':time.perf_counter()-start,'trace':h[3],'dependency_target':0.,'new_inner_loss_evaluations':len(inner.cache)-before_risk}
print('Selected prediction-only ablation',len(h[0]),h[1],inner(h[0]),flush=True)
k=len(selected['Guarded heuristic']);selected['Cheapest same cardinality']=tuple(sorted(np.argsort(costs,kind='stable')[:k]));selected['Random same cardinality']=tuple(sorted(np.random.default_rng(seed+2).choice(m,size=k,replace=False)))
(p/'selections_before_test.json').write_text(json.dumps({name:list(map(int,P)) for name,P in selected.items()},indent=2))
# Restricted exact comparison has its OWN eight-feature reference (Gamma and risk).
meanw=np.mean(np.stack(model.weights),axis=1);candidates=tuple(sorted(sorted(range(m),key=lambda a:(-meanw[a],a))[:8]));cc=[int(costs[a]) for a in candidates]
def restricted_dep(P):return dep(tuple(candidates[a] for a in P))
def restricted_loss(P):return inner(tuple(candidates[a] for a in P))
start=time.perf_counter();hh=heuristic(cc,restricted_dep,prediction_loss=restricted_loss);hs=time.perf_counter()-start
start=time.perf_counter();ee=exact_backtracking(cc,restricted_dep,prediction_loss=restricted_loss);es=time.perf_counter()-start
exact={'scope':'8 training-screened candidates; own Gamma(candidate set) and inner-risk(candidate set) references','candidates':list(map(int,candidates)),'dependency_target':ee[2],'inner_reference':restricted_loss(tuple(range(8))),'heuristic_selected_original':[int(candidates[a]) for a in hh[0]],'heuristic_cost':hh[1],'heuristic_seconds':hs,'optimal_selected_original':[int(candidates[a]) for a in ee[0]],'optimal_cost':ee[1],'exact_seconds':es,'exact_nodes':ee[3],'exact_distinct_dependency_evaluations':ee[4],'completed':True}
(p/'restricted_exact.json').write_text(json.dumps(exact,indent=2));print('Restricted exact',exact,flush=True)
rows=[]
for name,P in selected.items():
 row={'method':name,'attributes':len(P),'cost':int(sum(costs[a] for a in P)),'selected':json.dumps(list(map(int,P))),'inner_brier':inner(P),'inner_guard_pass':int(inner(P)<=reference),'search_seconds':searchinfo.get(name,{}).get('seconds',0.),**stats(P),**metrics(P)};rows.append(row);print('Outer evaluation',row,flush=True)
with (p/'results.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(p/'search_traces.json').write_text(json.dumps(searchinfo,indent=2))
cvrows=[{'selected':json.dumps(list(map(int,P))),'inner_brier':v,'fold_scores':json.dumps(inner.fold_scores[P])} for P,v in inner.cache.items()]
with (p/'inner_loss_cache.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['selected','inner_brier','fold_scores']);w.writeheader();w.writerows(cvrows)
(p/'runtime.json').write_text(json.dumps({'geometry_fit_seconds':fitsec,'inner_cache_size':len(inner.cache),'dependency_cache_size':len(train_cache),'note':'Timing shares caches across multiple searches; not an isolated runtime competition.'},indent=2))

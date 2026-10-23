"""Five paired repeated splits, three nested training sizes. No test-driven tuning."""
from pathlib import Path
import sys,json,csv,time,gc
import numpy as np
from scipy.spatial.distance import cdist
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from granulation_reference import TrainingGeometry
from fast_granulate import granulate
from reduction_reference import heuristic
out=Path(__file__).resolve().parent
X=np.load(root/'pilot/data/feature.npy');L=np.load(root/'pilot/data/label.npy')
old=json.loads((root/'pilot/protocol.json').read_text());costs=np.array(old['costs']);m=X.shape[1]
# Exclude all previously used pilot objects. This is a repeated diagnostic on remaining objects.
recent=json.loads((root/'guarded_pilot/protocol.json').read_text())
excluded=set(old['train_indices']+old['test_indices']+recent['train_indices']+recent['test_indices'])
pool=np.array([i for i in range(len(X)) if i not in excluded]);assert len(pool)>=832
sizes=[192,384,576];seeds=[20261100+i for i in range(5)]
protocol={'dataset':'Natural Scene','sizes':sizes,'repetitions':5,'seeds':seeds,'test_count':256,'excluded_previous_objects':sorted(excluded),'available_objects':len(pool),'costs':costs.tolist(),'settings':{'inner_folds':3,'knn_k':5,'dependency_tolerance':0,'prediction_tolerance':0,'zeta':.1,'kappa':.1,'beta':1,'smin':2,'alpha':1,'lambda':-1},'note':'Fixed settings before evaluation. Nested train sizes and common test within each repeat; overlap across repeats, descriptive SD only.'}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2));rows=[]
for rep,seed in enumerate(seeds):
 order=np.random.default_rng(seed).permutation(pool);test=order[:256]
 for n in sizes:
  job=out/f'repeat_{rep+1}_n{n}';job.mkdir(exist_ok=True)
  if (job/'results.json').exists():
   rows.extend(json.loads((job/'results.json').read_text()));continue
  train=order[256:256+n];assert set(train).isdisjoint(test)
  x=X[train];lab=L[train];folds=np.array_split(np.random.default_rng(seed+10000+n).permutation(n),3)
  (job/'split.json').write_text(json.dumps({'train_indices':train.tolist(),'test_indices':test.tolist(),'fold_validation_positions':[v.tolist() for v in folds]},indent=2))
  began=time.perf_counter();model=TrainingGeometry(x,lab,kinds=['numerical']*m);dcache={};rcache={};blocks=[]
  for val in folds:
   fit=np.setdiff1d(np.arange(n),val);a=x[fit];b=x[val];lo=a.min(0);span=a.max(0)-lo;span[span==0]=1
   a=(a-lo)/span;b=(b-lo)/span;mat=((b[:,None,:]-a[None,:,:])**2).transpose(2,0,1).copy();blocks.append((mat,fit,val))
  def risk(P):
   key=tuple(sorted(P))
   if key not in rcache:
    total=0
    for mat,fit,val in blocks:
     if key:
      dist=mat[list(key)].sum(0);nn=np.argsort(dist,axis=1,kind='stable')[:,:5];pred=lab[fit][nn].mean(1)
     else:pred=np.tile(lab[fit].mean(0),(len(val),1))
     total+=np.sum((lab[val]-pred)**2)
    rcache[key]=float(total/n)
   return rcache[key]
  def dep(P):
   key=tuple(sorted(P))
   if key not in dcache:
    d,k=model.subset(key);leaves,eligible,g,h=granulate(d,lab,k);dcache[key]=(g,sum(map(len,eligible))/n)
   return dcache[key][0]
  full=tuple(range(m));gref=dep(full);rref=risk(full);selected={'All attributes':full};traces={}
  for name,guard in [('Dependency-only',None),('Joint guard',risk)]:
   t=time.perf_counter();h=heuristic(costs,dep,prediction_loss=guard);selected[name]=h[0];traces[name]={'trace':h[3],'seconds':time.perf_counter()-t}
   if guard:assert dep(h[0])>=gref and risk(h[0])<=rref
   print('SELECT',rep+1,n,name,len(h[0]),h[1],risk(h[0]),flush=True)
  # Selection checkpoint precedes all test predictions.
  (job/'selection_before_test.json').write_text(json.dumps({k:list(map(int,v)) for k,v in selected.items()},indent=2))
  (job/'search_traces.json').write_text(json.dumps(traces,indent=2))
  lo=x.min(0);span=x.max(0)-lo;span[span==0]=1;a=(x-lo)/span;b=(X[test]-lo)/span;truth=L[test];local=[]
  for name,P in selected.items():
   if P:
    d=cdist(b[:,P],a[:,P],'sqeuclidean');nn=np.argsort(d,axis=1,kind='stable')[:,:5];pred=lab[nn].mean(1)
   else:pred=np.tile(lab.mean(0),(len(test),1))
   pred=np.maximum(pred,1e-12);pred/=pred.sum(1,keepdims=True);mid=(truth+pred)/2
   row={'repeat':rep+1,'seed':seed,'train_n':n,'method':name,'attributes':len(P),'cost':int(sum(costs[i] for i in P)),'selected':list(map(int,P)),'inner_brier':risk(P),'inner_reference':rref,'gamma':dep(P),'gamma_reference':gref,'coverage':dcache[tuple(sorted(P))][1],'joint_pass':int(dep(P)>=gref and risk(P)<=rref),'test_brier':float(np.mean(np.sum((truth-pred)**2,axis=1))),'test_kl':float(np.mean(np.sum(truth*np.log(np.maximum(truth,1e-300)/pred),axis=1))),'test_tv':float(np.mean(.5*np.abs(truth-pred).sum(1))),'test_js':float(np.mean(.5*np.sum(truth*np.log(np.maximum(truth,1e-300)/mid),axis=1)+.5*np.sum(pred*np.log(pred/mid),axis=1)))};local.append(row)
  (job/'results.json').write_text(json.dumps(local,indent=2));rows.extend(local)
  with (out/'results.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  print('DONE',rep+1,n,'seconds',time.perf_counter()-began,'test',[(r['method'],r['test_brier']) for r in local],flush=True)
  del model,blocks,dcache,rcache;gc.collect()

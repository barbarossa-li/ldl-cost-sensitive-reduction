"""Frozen new-dataset replication: joint, predictive ablation, budget bridge and constraint audit."""
from pathlib import Path
import sys,json,csv,gc,time
import numpy as np
from scipy.spatial.distance import cdist
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root));sys.path.insert(0,str(root/'stability'))
from granulation_reference import TrainingGeometry
from fast_granulate import granulate
from reduction_reference import heuristic
out=Path(__file__).resolve().parent
def nearest5(d):
 idx=np.argpartition(d,4,axis=1)[:,:5];vals=np.take_along_axis(d,idx,axis=1);th=vals.max(1)
 order=np.lexsort((idx,vals),axis=1);idx=np.take_along_axis(idx,order,axis=1)
 for i in np.flatnonzero((d==th[:,None]).sum(1)>1):
  below=np.flatnonzero(d[i]<th[i]);ties=np.flatnonzero(d[i]==th[i]);v=np.r_[below,ties[:5-len(below)]];idx[i]=v[np.lexsort((v,d[i,v]))]
 return idx
protocol={'datasets':['SBU_3DFE','Yeast_cold'],'sizes':[192,384,576],'repeats':5,'test_n':256,'cost_seed':20260931,'parameters':{'beta':1,'zeta':.1,'kappa':.1,'smin':2,'alpha':1,'lambda':-1,'dependency_tolerance':0,'prediction_tolerance':0,'inner_folds':3,'knn_k':5},'scope':'All frozen before new-dataset test evaluation. Same original joint algorithm and prediction-only ablation. Strong bridge baseline under joint training-selected cost budget. Shared-prefix constraint-only audit; shared search failures retained. Original gamma-only deletion candidates diagnosed. Repeats overlap, descriptive results only.'}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2));allrows=[];auditrows=[];diagrows=[]
for di,name in enumerate(protocol['datasets']):
 X=np.load(out/'data'/f'{name}_X.npy');L=np.load(out/'data'/f'{name}_L.npy');m=X.shape[1];costs=np.random.default_rng(protocol['cost_seed']).integers(1,11,m)
 for rep in range(1,6):
  seed=20261200+di*100+rep;order=np.random.default_rng(seed).permutation(len(X));test=order[:256]
  for n in protocol['sizes']:
   job=out/name/f'repeat_{rep}_n{n}';job.mkdir(parents=True,exist_ok=True)
   if (job/'complete.json').exists():
    allrows.extend(json.loads((job/'results.json').read_text()));auditrows.extend(json.loads((job/'audit_results.json').read_text()));diagrows.extend(json.loads((job/'blocked_test.json').read_text()));continue
   started=time.perf_counter();train=order[256:256+n];x=X[train];lab=L[train];folds=np.array_split(np.random.default_rng(seed+10000+n).permutation(n),3);cache={};dcache={};blocks=[];model=TrainingGeometry(x,lab,kinds=['numerical']*m)
   (job/'split.json').write_text(json.dumps({'seed':seed,'train_indices':train.tolist(),'test_indices':test.tolist(),'fold_validation_positions':[v.tolist() for v in folds],'costs':costs.tolist()},indent=2))
   assert set(train).isdisjoint(test)
   for val in folds:
    fit=np.setdiff1d(np.arange(n),val);a=x[fit];b=x[val];lo=a.min(0);span=a.max(0)-lo;span[span==0]=1.;a=(a-lo)/span;b=(b-lo)/span;blocks.append((((b[:,None,:]-a[None,:,:])**2).transpose(2,0,1).copy(),fit,val))
   def risk(P):
    key=tuple(sorted(P))
    if key not in cache:
     total=0.
     for mat,fit,val in blocks:
      pred=lab[fit][nearest5(mat[list(key)].sum(0))].mean(1) if key else np.tile(lab[fit].mean(0),(len(val),1));total+=np.sum((lab[val]-pred)**2)
     cache[key]=float(total/n)
    return cache[key]
   def stats(P):
    key=tuple(sorted(P))
    if key not in dcache:
     d,k=model.subset(key);leaves,eligible,g,h=granulate(d,lab,k);dcache[key]=(g,sum(map(len,eligible))/n)
    return dcache[key]
   def dep(P):return stats(P)[0]
   full=tuple(range(m));rref=risk(full);gref=dep(full);selected={'All attributes':full};traces={}
   for mode,d in [('Joint guard',dep),('Prediction-only',lambda P:0.)]:
    h=heuristic(costs,d,prediction_loss=risk);selected[mode]=h[0];traces[mode]=h[3];print('SELECT',name,rep,n,mode,len(h[0]),h[1],flush=True)
   joint=selected['Joint guard'];budget=int(sum(costs[list(joint)]));assert dep(joint)>=gref and risk(joint)<=rref
   P=set();prefixes=[()];best=();bestloss=risk(());bridge=[]
   while True:
    remain=budget-sum(costs[list(P)]);cand=[a for a in range(m) if a not in P and costs[a]<=remain]
    if not cand:break
    base=risk(P);gains={a:base-risk(P|{a}) for a in cand};positive=[a for a in cand if gains[a]>0]
    aa=min(positive,key=lambda a:(-gains[a]/costs[a],costs[a],a)) if positive else min(cand,key=lambda a:(-gains[a],costs[a],a))
    P.add(aa);key=tuple(sorted(P));prefixes.append(key);v=risk(P);bridge.append([aa,gains[aa],v])
    if v<bestloss or (v==bestloss and sum(costs[list(P)])<sum(costs[list(best)])):best=key;bestloss=v
   selected['Budget-forward bridge']=best;events=[{'phase':'prefix','selected':list(P),'risk_pass':risk(P)<=rref,'gamma_pass':dep(P)>=gref,'inner_brier':risk(P),'gamma':dep(P)} for P in prefixes];audits={}
   for mode,on in [('Constraint off',False),('Constraint on',True)]:
    feasible=[P for P in prefixes if risk(P)<=rref and (not on or dep(P)>=gref)]
    if not feasible:audits[mode]=None;continue
    P=set(min(feasible,key=lambda P:(sum(costs[list(P)]),len(P),P)))
    while True:
     removed=False
     for aa in sorted(P,key=lambda a:(-costs[a],a)):
      C=P-{aa};rp=risk(C)<=rref;gp=dep(C)>=gref;accept=rp and (not on or gp);events.append({'phase':'deletion','mode':mode,'removed':int(aa),'selected':sorted(C),'risk_pass':rp,'gamma_pass':gp,'accepted':accept})
      if accept:P=C;removed=True;break
     if not removed:break
    audits[mode]=tuple(sorted(P))
   drops=[]
   for aa in joint:
    C=tuple(sorted(set(joint)-{aa}));rp=risk(C)<=rref;gp=dep(C)>=gref;drops.append({'removed':int(aa),'selected':list(C),'risk_pass':rp,'gamma_pass':gp,'gamma_only_block':rp and not gp,'inner_brier':risk(C),'gamma':dep(C)})
   # All selections and all diagnostic training decisions checkpointed before test metrics.
   (job/'selection_before_test.json').write_text(json.dumps({'selected':{k:list(map(int,P)) for k,P in selected.items()},'audit':{k:list(map(int,P)) if P is not None else None for k,P in audits.items()},'budget':budget,'risk_reference':rref,'gamma_reference':gref,'bridge_path':bridge},indent=2));(job/'audit_events.json').write_text(json.dumps(events,indent=2));(job/'original_deletions.json').write_text(json.dumps(drops,indent=2));(job/'search_traces.json').write_text(json.dumps(traces,indent=2))
   lo=x.min(0);span=x.max(0)-lo;span[span==0]=1.;a=(x-lo)/span;b=(X[test]-lo)/span;truth=L[test]
   def evaluate(P):
    pred=lab[nearest5(cdist(b[:,P],a[:,P],'sqeuclidean'))].mean(1) if P else np.tile(lab.mean(0),(len(test),1));pred=np.maximum(pred,1e-12);pred/=pred.sum(1,keepdims=True);mid=(truth+pred)/2
    # x log x conventions permit genuine zero description degrees.
    kl=float(np.mean(np.sum(truth*np.log(np.maximum(truth,1e-300)/pred),axis=1)))
    js=float(np.mean(.5*np.sum(truth*np.log(np.maximum(truth,1e-300)/mid),axis=1)+.5*np.sum(pred*np.log(pred/mid),axis=1)))
    return {'test_brier':float(np.mean(np.sum((truth-pred)**2,axis=1))),'test_kl':kl,'test_tv':float(np.mean(.5*np.abs(truth-pred).sum(1))),'test_js':js}
   local=[]
   for mode,P in selected.items():local.append({'dataset':name,'repeat':rep,'train_n':n,'method':mode,'attributes':len(P),'cost':int(sum(costs[list(P)])),'inner_brier':risk(P),'gamma':dep(P),'coverage':stats(P)[1],**evaluate(P)})
   ars=[]
   for mode,P in audits.items():
    ars.append({'dataset':name,'repeat':rep,'train_n':n,'mode':mode,'success':P is not None,'selected':list(P) if P is not None else None,'attributes':len(P) if P is not None else None,'cost':int(sum(costs[list(P)])) if P is not None else None,'test_brier':evaluate(P)['test_brier'] if P is not None else None})
   orig=next(r for r in local if r['method']=='Joint guard');ds=[]
   for e in drops:
    if e['gamma_only_block']:
     C=tuple(e['selected']);v=evaluate(C)['test_brier'];ds.append({'dataset':name,'repeat':rep,'train_n':n,'removed':e['removed'],'original_cost':orig['cost'],'candidate_cost':int(sum(costs[list(C)])),'original_test_brier':orig['test_brier'],'candidate_test_brier':v,'candidate_minus_original':v-orig['test_brier']})
   (job/'results.json').write_text(json.dumps(local,indent=2));(job/'audit_results.json').write_text(json.dumps(ars,indent=2));(job/'blocked_test.json').write_text(json.dumps(ds,indent=2));(job/'complete.json').write_text(json.dumps({'seconds':time.perf_counter()-started,'dependency_evaluations':len(dcache),'risk_evaluations':len(cache)},indent=2));allrows.extend(local);auditrows.extend(ars);diagrows.extend(ds)
   print('DONE',name,rep,n,[(r['method'],round(r['test_brier'],5)) for r in local], 'audit',[(r['mode'],r['success']) for r in ars],'gamma-blocks',len(ds),flush=True)
   for fname,rr in [('results',allrows),('audit_results',auditrows),('blocked_test',diagrows)]:
    if rr:
     with (out/(fname+'.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
   del model,blocks,cache,dcache;gc.collect()

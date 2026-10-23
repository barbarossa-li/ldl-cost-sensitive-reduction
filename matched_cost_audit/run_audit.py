"""Frozen subsets vs label-free cost-profile/cardinality-matched controls."""
from pathlib import Path
import json,csv,hashlib,math,itertools
import numpy as np
from scipy.spatial.distance import cdist
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent
names=['Natural Scene','SBU_3DFE','Yeast_cold'];methods=['Prediction only','Guided'];manifest=[]
protocol=dict(scope='576 development size, three saved repeats,432 search/144 reserved audit; no outer test; original refined subsets frozen',controls=20,matching='Exact cardinality, total cost and number of attributes at each cost value; stricter than cost/cardinality alone',sampling='Uniform independent draws within each cost stratum, retain unique subsets excluding selected subset; fixed seeds; not uniform over all subsets with only matched total cost',seed='20260930+10000*dataset_index+100*repeat+method_index',empty='One identical empty reference, no substantive comparison',metrics='Raw Brier as in search/audit; all controls evaluated and distribution summarized; no best-control selection',interpretation='Descriptive conditional controls, not an exchangeable permutation test, p-value or significance claim')
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
for di,name in enumerate(names):
 for rep in [1,2,3]:
  parent=root/'holdout_size_audit' if rep==1 else root/'large_holdout_replication';job=parent/name.replace(' ','_')/f'repeat_{rep}_n576';part=json.loads((parent/(name.replace(' ','_')+f'_repeat_{rep}_n576_partition.json')).read_text())
  costs=np.array(json.loads((root/'stability/protocol.json').read_text())['costs']) if name=='Natural Scene' else np.array(json.loads((root/'cross_dataset'/name/f'repeat_{rep}_n576/split.json').read_text())['costs'])
  for mi,method in enumerate(methods):
   source=job/(method.replace(' ','_')+'.json');frozen=json.loads(source.read_text());P=tuple(frozen['after']);profile={int(c):int(sum(costs[list(P)]==c)) for c in np.unique(costs) if sum(costs[list(P)]==c)};seed=20260930+10000*di+100*rep+mi;rng=np.random.default_rng(seed);draws=[];capacity=math.prod(math.comb(int(sum(costs==c)),k) for c,k in profile.items())
   if not P:draws=[()]
   elif capacity-1<=20:
    choices=[list(itertools.combinations(np.flatnonzero(costs==c).tolist(),k)) for c,k in profile.items()];draws=[tuple(sorted(itertools.chain.from_iterable(v))) for v in itertools.product(*choices)];draws=[v for v in draws if v!=P]
   else:
    seen={P}
    while len(draws)<20:
     q=tuple(sorted(itertools.chain.from_iterable(rng.choice(np.flatnonzero(costs==c),k,replace=False).tolist() for c,k in profile.items())))
     if q not in seen:draws.append(q);seen.add(q)
   for Q in draws:assert len(Q)==len(P) and sum(costs[list(Q)])==sum(costs[list(P)]) and all(sum(costs[list(Q)]==c)==k for c,k in profile.items())
   manifest.append(dict(dataset=name,repeat=rep,method=method,selected=list(P),cost=int(costs[list(P)].sum()),profile=profile,seed=seed,controls=[list(v) for v in draws],search_ids=part['search_ids'],audit_ids=part['audit_ids'],folds=part['search_fold_validation_positions'],source=str(source.relative_to(root)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),saved_inner_loss=frozen['results']['after_loss'],inner_reference=frozen['results']['full_reference']))
# Freeze all label-free controls before any audit or internal loss evaluation.
(out/'controls_checkpoint_before_evaluation.json').write_text(json.dumps(manifest,indent=2));rows=[];summary=[]
for entry in manifest:
 name=entry['dataset'];rep=entry['repeat'];method=entry['method'];fitids=entry['search_ids'];auditids=entry['audit_ids'];assert set(fitids).isdisjoint(auditids)
 if name=='Natural Scene':X=np.load(root/'pilot/data/feature.npy');L=np.load(root/'pilot/data/label.npy')
 else:X=np.load(root/'cross_dataset/data'/f'{name}_X.npy');L=np.load(root/'cross_dataset/data'/f'{name}_L.npy')
 x=X[fitids];lab=L[fitids];n=len(x);blocks=[]
 for v in entry['folds']:
  v=np.array(v);fit=np.setdiff1d(np.arange(n),v);lo=x[fit].min(0);span=x[fit].max(0)-lo;span[span==0]=1;blocks.append(((x[fit]-lo)/span,(x[v]-lo)/span,fit,v))
 lo=x.min(0);span=x.max(0)-lo;span[span==0]=1;z=(x-lo)/span;query=(X[auditids]-lo)/span
 def evaluate(P):
  ss=0.
  for a,b,fit,v in blocks:
   pred=lab[fit][np.argsort(cdist(b[:,P],a[:,P],'sqeuclidean'),axis=1,kind='stable')[:,:5]].mean(1) if P else np.tile(lab[fit].mean(0),(len(v),1));ss+=np.sum((lab[v]-pred)**2)
  pred=lab[np.argsort(cdist(query[:,P],z[:,P],'sqeuclidean'),axis=1,kind='stable')[:,:5]].mean(1) if P else np.tile(lab.mean(0),(len(auditids),1));return float(ss/n),float(np.mean(np.sum((L[auditids]-pred)**2,axis=1)))
 inn,audit=evaluate(entry['selected']);assert abs(inn-entry['saved_inner_loss'])<1e-12
 oldparent=root/'holdout_size_audit' if rep==1 else root/'large_holdout_replication';old=json.loads((oldparent/'audit_results.json').read_text());expected=next(v['audit_brier'] for v in old if v['dataset']==name and v['repeat']==rep and v['development_size']==576 and v['method']==method+' after');assert abs(audit-expected)<1e-12
 results=[]
 for j,P in enumerate(entry['controls']):
  a,b=evaluate(P);r=dict(dataset=name,repeat=rep,method=method,draw=j,attributes=len(P),cost=entry['cost'],inner_brier=a,audit_brier=b,inner_pass=a<=entry['inner_reference'],selected_inner=inn,selected_audit=audit,subset=json.dumps(P));rows.append(r);results.append(r)
 ci=np.array([v['inner_brier'] for v in results]);ca=np.array([v['audit_brier'] for v in results]);s=dict(dataset=name,repeat=rep,method=method,attributes=len(entry['selected']),cost=entry['cost'],control_count=len(results),degenerate_empty=not bool(entry['selected']),selected_inner=inn,selected_audit=audit,control_inner_mean=float(ci.mean()),control_audit_mean=float(ca.mean()),audit_q25=float(np.quantile(ca,.25)),audit_q75=float(np.quantile(ca,.75)),selected_minus_control_audit_mean=float(audit-ca.mean()),selected_minus_control_inner_mean=float(inn-ci.mean()),controls_worse_audit=int(sum(ca>audit+1e-12)),controls_better_audit=int(sum(ca<audit-1e-12)),controls_equal_audit=int(sum(np.abs(ca-audit)<=1e-12)),controls_inner_feasible=sum(v['inner_pass'] for v in results));summary.append(s)
 assert hashlib.sha256((root/entry['source']).read_bytes()).hexdigest()==entry['source_sha256'];print('DONE',s,flush=True)
with (out/'controls.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'validation.json').write_text(json.dumps(dict(contexts=18,control_records=len(rows),selected_inner_and_audit_reproduced=18,matching_cardinality_total_cost_profile=True,controls_frozen_before_loss_evaluation=True,selection_sources_unchanged=True,controls_use_no_labels=True,outer_test_used=False,new_selection=False,retuning=False),indent=2))

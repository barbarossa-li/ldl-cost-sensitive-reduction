"""Frozen training-only comparisons; reuse 15 existing splits. Not published-method replicas."""
from pathlib import Path
import sys,json,csv,time,gc
import numpy as np
from scipy.spatial.distance import cdist
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from reduction_reference import heuristic
out=Path(__file__).resolve().parent;X=np.load(root/'pilot/data/feature.npy');L=np.load(root/'pilot/data/label.npy');costs=np.array(json.loads((root/'stability/protocol.json').read_text())['costs']);m=X.shape[1]
def nearest5(d):
 # Exact stable-argsort nearest five, using partial selection and explicit boundary ties.
 idx=np.argpartition(d,4,axis=1)[:,:5];vals=np.take_along_axis(d,idx,axis=1);threshold=vals.max(1)
 for i in range(len(d)):
  below=np.flatnonzero(d[i]<threshold[i]);ties=np.flatnonzero(d[i]==threshold[i]);v=np.r_[below,ties[:5-len(below)]]
  idx[i]=v[np.lexsort((v,d[i,v]))]
 return idx
rng=np.random.default_rng(20260930)
for shape in [(64,128),(192,384)]:
 for tied in [False,True]:
  d=rng.integers(0,10,shape).astype(float) if tied else rng.random(shape)
  assert np.array_equal(nearest5(d),np.argsort(d,axis=1,kind='stable')[:,:5])
(out/'nearest_checks.txt').write_text('PASS: exact equality to stable argsort nearest5 on random continuous and tied distance matrices (64x128 and 192x384).\n')
protocol={'scope':'15 saved Natural Scene splits; no retuning joint method','methods':{'Cheapest k':'k cheapest, matched cardinality','Random k':'20 fixed random k subsets; average test metrics, not best chosen','Correlation k':'sum squared feature-label correlations, top k','Forward k':'training-inner Brier sequential forward selection to exactly k; no cost ranking','Cost-correlation budget':'correlation score/cost, greedy fit within joint cost budget','Cost-forward budget':'positive inner Brier decrease/cost, greedy within joint cost budget; stop if no positive gain','Prediction-only':'same original guarded heuristic, dependency identically zero; no matched quota'},'random_draws':20,'fairness':'k and cost budget are joint outputs determined by training only. Same predictor, folds, normalization and costs. Test data already examined in prior diagnostics; exploratory paired comparison, not untouched confirmation. Generic controls and wrapper baselines, no claimed replicas of named published algorithms.'}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2));rows=[]
for rep in range(1,6):
 for n in [192,384,576]:
  job=out/f'repeat_{rep}_n{n}';job.mkdir(exist_ok=True)
  if not (job/'results.json').exists():raise RuntimeError('Run main baselines first')
  if (job/'bridge_result.json').exists():continue
  saved=root/'stability'/job.name;split=json.loads((saved/'split.json').read_text());old=json.loads((saved/'results.json').read_text());joint=next(r for r in old if r['method']=='Joint guard');k=joint['attributes'];budget=joint['cost'];train=np.array(split['train_indices']);test=np.array(split['test_indices']);x=X[train];lab=L[train];blocks=[];cache={}
  for val in map(np.array,split['fold_validation_positions']):
   fit=np.setdiff1d(np.arange(n),val);a=x[fit];b=x[val];lo=a.min(0);span=a.max(0)-lo;span[span==0]=1.;a=(a-lo)/span;b=(b-lo)/span;blocks.append((((b[:,None,:]-a[None,:,:])**2).transpose(2,0,1).copy(),fit,val))
  def risk(P):
   key=tuple(sorted(P))
   if key not in cache:
    total=0.
    for mat,fit,val in blocks:
     pred=lab[fit][nearest5(mat[list(key)].sum(0))].mean(1) if key else np.tile(lab[fit].mean(0),(len(val),1))
     total+=np.sum((lab[val]-pred)**2)
    cache[key]=float(total/n)
   return cache[key]
  assert abs(risk(tuple(joint['selected']))-joint['inner_brier'])<1e-12
  P=set();best=();bestloss=risk(());trace=[]
  while True:
   remaining=budget-sum(costs[list(P)]);candidates=[a for a in range(m) if a not in P and costs[a]<=remaining]
   if not candidates:break
   base=risk(P);gains={a:base-risk(P|{a}) for a in candidates};positive=[a for a in candidates if gains[a]>0]
   if positive:a=min(positive,key=lambda a:(-gains[a]/costs[a],costs[a],a))
   else:a=min(candidates,key=lambda a:(-gains[a],costs[a],a))
   P.add(a);v=risk(P);trace.append([a,gains[a],v])
   if v<bestloss or (v==bestloss and sum(costs[list(P)])<sum(costs[list(best)])):best=tuple(sorted(P));bestloss=v
  # Best inner-validation prefix, including empty; not a test-selected subset.
  (job/'bridge_selection_before_test.json').write_text(json.dumps({'selected':list(map(int,best)),'budget':budget,'trace':trace,'inner_loss':bestloss},indent=2))
  lo=x.min(0);span=x.max(0)-lo;span[span==0]=1.;aa=(x-lo)/span;bb=(X[test]-lo)/span;truth=L[test]
  pred=lab[nearest5(cdist(bb[:,best],aa[:,best],'sqeuclidean'))].mean(1) if best else np.tile(lab.mean(0),(len(test),1));pred=np.maximum(pred,1e-12);pred/=pred.sum(1,keepdims=True);mid=(truth+pred)/2
  row={'repeat':rep,'train_n':n,'method':'Budget-forward bridge','attributes':len(best),'cost':int(sum(costs[list(best)])),'inner_brier':risk(best),'test_brier':float(np.mean(np.sum((truth-pred)**2,axis=1))),'test_kl':float(np.mean(np.sum(truth*np.log(np.maximum(truth,1e-300)/pred),axis=1))),'test_tv':float(np.mean(.5*np.abs(truth-pred).sum(1))),'test_js':float(np.mean(.5*np.sum(truth*np.log(np.maximum(truth,1e-300)/mid),axis=1)+.5*np.sum(pred*np.log(pred/mid),axis=1)))}
  assert row['cost']<=budget
  (job/'bridge_result.json').write_text(json.dumps(row,indent=2));print('BRIDGE DONE',rep,n,row['attributes'],row['cost'],row['test_brier'],flush=True)
  del blocks,cache;gc.collect()
rows=[]
for job in sorted(out.glob('repeat_*_n*')):
 rows.extend(json.loads((job/'results.json').read_text()));rows.append(json.loads((job/'bridge_result.json').read_text()))
with (out/'results.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
protocol['methods']['Budget-forward bridge']='cost-constrained forward search with nonpositive bridge; best inner-risk prefix incl empty; preserve positive-only baseline'
protocol['note']='Added to address established zero-singleton interaction issue; descriptive extension after early baseline diagnostics, not new untouched evaluation.'
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))

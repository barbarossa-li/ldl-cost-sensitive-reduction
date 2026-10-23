from pathlib import Path
import json,csv
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent;root=p.parent;X=np.load(root/'pilot/data/feature.npy');L=np.load(root/'pilot/data/label.npy');paired=[];counts=[];blocked=[]
for n in [192,384,576]:
 count={'train_n':n,'off_feasible':0,'on_feasible':0,'different_outputs':0,'prefix_gamma_only_blocks':0,'original_blocked_deletions':0,'original_affected_splits':0}
 for rep in range(1,6):
  job=p/f'repeat_{rep}_n{n}';rows=json.loads((job/'results.json').read_text());off,on=rows;events=json.loads((job/'events.json').read_text());saved=json.loads((job/'selection_before_test.json').read_text())
  count['off_feasible']+=off['success'];count['on_feasible']+=on['success'];count['different_outputs']+=off['success'] and on['success'] and off['selected']!=on['selected'];count['prefix_gamma_only_blocks']+=sum(e['phase']=='common prefix' and e['risk_pass'] and not e['gamma_pass'] for e in events)
  assert not on['success'] or off['success']
  for r in rows:
   assert r['selected']==saved['selected'][r['mode']]
   if r['success']:
    assert r['cost']<=saved['budget'] and r['inner_brier']<=saved['inner_reference']
    if r['mode']=='Constraint on':assert r['gamma']>=saved['gamma_reference']
  if off['success'] and on['success']:paired.append({'repeat':rep,'train_n':n,'changed':off['selected']!=on['selected'],'off_attributes':off['attributes'],'on_attributes':on['attributes'],'off_cost':off['cost'],'on_cost':on['cost'],'off_brier':off['test_brier'],'on_brier':on['test_brier'],'on_minus_off_brier':on['test_brier']-off['test_brier']})
  es=json.loads((job/'original_deletion_events.json').read_text());rejected=[e for e in es if e['gamma_only_block']];count['original_blocked_deletions']+=len(rejected);count['original_affected_splits']+=bool(rejected)
  split=json.loads((root/'stability'/job.name/'split.json').read_text());train=split['train_indices'];test=split['test_indices'];x=X[train];lab=L[train];lo=x.min(0);span=x.max(0)-lo;span[span==0]=1;a=(x-lo)/span;b=(X[test]-lo)/span;truth=L[test]
  original=next(r for r in json.loads((root/'stability'/job.name/'results.json').read_text()) if r['method']=='Joint guard')
  for e in rejected:
   P=tuple(e['selected']);pred=lab[np.argsort(cdist(b[:,P],a[:,P],'sqeuclidean'),axis=1,kind='stable')[:,:5]].mean(1) if P else np.tile(lab.mean(0),(len(test),1));pred=np.maximum(pred,1e-12);pred/=pred.sum(1,keepdims=True);v=float(np.mean(np.sum((truth-pred)**2,axis=1)))
   blocked.append({'repeat':rep,'train_n':n,'removed':e['removed'],'original_cost':original['cost'],'candidate_cost':e['cost'],'original_brier':original['test_brier'],'candidate_brier':v,'candidate_minus_original':v-original['test_brier'],'candidate_inner_brier':e['inner_brier'],'inner_reference':saved['inner_reference'],'candidate_gamma':e['gamma'],'gamma_reference':saved['gamma_reference']})
 counts.append(count)
assert sum(c['off_feasible'] for c in counts)==len(paired)
for name,rr in [('counts',counts),('paired',paired),('blocked_deletions',blocked)]:
 with (p/f'{name}.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
(p/'validation.txt').write_text('PASS: 15 shared frozen paths; selection checkpoints, common budget, identical risk reference, on implies off feasibility, exact selected sets, no invalid output reported feasible. Original deletion diagnostics use saved training decisions before test evaluation.\n')
plt.rcParams.update({'font.size':10,'pdf.fonttype':42});fig,axes=plt.subplots(2,1,figsize=(6.5,5.8));axes[0].bar([str(c['train_n']) for c in counts],[c['original_blocked_deletions'] for c in counts]);axes[0].set_ylabel('Gamma-only blocked deletions');axes[0].set_xlabel('Training objects');axes[0].set_ylim(0,max(c['original_blocked_deletions'] for c in counts)+1)
labels=[f"R{r['repeat']}, n={r['train_n']}" for r in paired];axes[1].bar(labels,[r['on_minus_off_brier'] for r in paired]);axes[1].axhline(0,color='gray');axes[1].set_ylabel('On minus off: test Brier');axes[1].tick_params(axis='x',labelsize=8);axes[1].set_title('Only the five jointly available pairs')
fig.tight_layout();fig.savefig(p/'constraint_effect.pdf');fig.savefig(p/'constraint_effect.png',dpi=180)
print(json.dumps({'counts':counts,'paired':paired,'blocked_deletions':blocked},indent=2))

"""Compatibility audit of frozen saved candidates; no search or tuning."""
from pathlib import Path
import csv,json,sys,gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent
sys.path.insert(0,str(root));sys.path.insert(0,str(root/'stability'))
from granulation_reference import TrainingGeometry
from fast_granulate import granulate
rows=[]
for dataset,source in [('Natural Scene','constraint_audit/blocked_deletions.csv'),('SBU_3DFE','cross_dataset/blocked_test.csv'),('Yeast_cold','cross_dataset/blocked_test.csv')]:
 for r in csv.DictReader(open(root/source)):
  if 'dataset' in r and r['dataset']!=dataset:continue
  rep=int(r['repeat']);n=int(r['train_n']);removed=int(r['removed']);job=f'repeat_{rep}_n{n}'
  folder=root/'stability'/job if dataset=='Natural Scene' else root/'cross_dataset'/dataset/job
  parent=next(x for x in json.loads((folder/'results.json').read_text()) if x['method']=='Joint guard')
  ref=next(x for x in json.loads((folder/'results.json').read_text()) if x['method']=='All attributes')
  events=root/'constraint_audit'/job/'original_deletion_events.json' if dataset=='Natural Scene' else folder/'original_deletions.json'
  e=next(x for x in json.loads(events.read_text()) if x['removed']==removed)
  assert e['risk_pass'] and not e['gamma_pass'] and e['gamma_only_block']
  candtest=float(r['candidate_brier'] if dataset=='Natural Scene' else r['candidate_test_brier'])
  cost=int(r['candidate_cost']);assert cost<parent['cost']
  rows.append({'dataset':dataset,'repeat':rep,'train_n':n,'removed':removed,'parent_cost':parent['cost'],'candidate_cost':cost,'parent_gamma':parent['gamma'],'candidate_gamma':e['gamma'],'gamma_reference':ref['gamma'],'parent_inner_brier':parent['inner_brier'],'candidate_inner_brier':e['inner_brier'],'inner_reference':ref['inner_brier'],'parent_test_brier':parent['test_brier'],'candidate_test_brier':candtest,'delta_gamma':e['gamma']-parent['gamma'],'delta_inner':e['inner_brier']-parent['inner_brier'],'delta_test':candtest-parent['test_brier'],'selected':e['selected']})
assert len(rows)==47
with (out/'candidates.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
counts=[]
for name in ['Natural Scene','SBU_3DFE','Yeast_cold']:
 rr=[r for r in rows if r['dataset']==name]
 counts.append({'dataset':name,'blocked_candidates':len(rr),'affected_splits':len({(r['repeat'],r['train_n']) for r in rr}),'inner_better_than_parent':sum(r['delta_inner']<0 for r in rr),'inner_equal_to_parent':sum(r['delta_inner']==0 for r in rr),'test_better_than_parent':sum(r['delta_test']<0 for r in rr),'both_losses_better':sum(r['delta_inner']<0 and r['delta_test']<0 for r in rr)})
(out/'counts.json').write_text(json.dumps(counts,indent=2));print('COUNTS',counts,flush=True)
# Witness selection uses only training quantities, before consulting saved test loss.
# First dataset/repeat/size/removed in deterministic order among CV-dominating drops.
witnesses=[]
for name in ['Natural Scene','SBU_3DFE','Yeast_cold']:
 eligible=sorted([r for r in rows if r['dataset']==name and r['delta_inner']<0],key=lambda r:(r['repeat'],r['train_n'],r['removed']))
 if eligible:witnesses.append(eligible[0])
(out/'witnesses.json').write_text(json.dumps(witnesses,indent=2));print('WITNESSES',witnesses,flush=True)
# Independent direct residual-sum identity check: full, parent and one rejected drop,
# on the first affected 192-object split per dataset. This is not a new experiment.
identities=[]
for name in ['Natural Scene','SBU_3DFE','Yeast_cold']:
 r=min([r for r in rows if r['dataset']==name and r['train_n']==192],key=lambda r:(r['repeat'],r['removed']))
 job=f"repeat_{r['repeat']}_n192";folder=root/'stability'/job if name=='Natural Scene' else root/'cross_dataset'/name/job
 split=json.loads((folder/'split.json').read_text());ids=split['train_indices']
 if name=='Natural Scene':X=np.load(root/'pilot/data/feature.npy');L=np.load(root/'pilot/data/label.npy');parent=next(x for x in json.loads((folder/'results.json').read_text()) if x['method']=='Joint guard')['selected']
 else:
  X=np.load(root/'cross_dataset/data'/f'{name}_X.npy');L=np.load(root/'cross_dataset/data'/f'{name}_L.npy');parent=json.loads((folder/'selection_before_test.json').read_text())['selected']['Joint guard']
 x=X[ids];lab=L[ids];p=lab.mean(0);v0=float(np.mean(np.sum((lab-p)**2,axis=1)));model=TrainingGeometry(x,lab,kinds=['numerical']*x.shape[1])
 for mode,P in [('Full',list(range(x.shape[1]))),('Parent',parent),('Rejected drop',r['selected'])]:
  d,k=model.subset(P);leaves,accepted,g,_=granulate(d,lab,k);pred=np.tile(p,(len(lab),1))
  for B in accepted:pred[B]=lab[B].mean(0)
  direct=float(np.mean(np.sum((lab-pred)**2,axis=1)));rhs=v0*(1-g);err=abs(direct-rhs);assert err<1e-12
  identities.append({'dataset':name,'repeat':r['repeat'],'train_n':192,'subset':mode,'v0':v0,'gamma':g,'direct_training_distortion':direct,'identity_rhs':rhs,'absolute_error':err})
 del model;gc.collect()
(out/'identity_checks.json').write_text(json.dumps(identities,indent=2));print('IDENTITY_MAX_ERROR',max(r['absolute_error'] for r in identities),flush=True)
fig,axes=plt.subplots(1,3,figsize=(10,3.5))
for ax,name in zip(axes,['Natural Scene','SBU_3DFE','Yeast_cold']):
 rr=[r for r in rows if r['dataset']==name]
 for better,color,marker,label in [(True,'#187b59','o','Test loss lower'),(False,'#b04d38','x','Test loss higher')]:
  points=[r for r in rr if (r['delta_test']<0)==better]
  ax.scatter([r['delta_gamma'] for r in points],[r['delta_inner'] for r in points],s=28,c=color,marker=marker,label=label)
 ax.axhline(0,color='grey',lw=.8);ax.axvline(0,color='grey',lw=.8);ax.set_title(name.replace('_',' '));ax.set_xlabel(r'$\Gamma(C)-\Gamma(P)$');ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.grid(alpha=.15)
axes[0].set_ylabel('Inner Brier(C) - Inner Brier(P)');axes[0].legend(fontsize=7)
fig.tight_layout();fig.savefig(out/'compatibility.pdf');fig.savefig(out/'compatibility.png',dpi=180);plt.close(fig)
print('DONE: frozen candidates only; no changed selection, parameters or constraints.')

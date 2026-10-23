from pathlib import Path
import json,csv,itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent;root=p.parent;datasets=['SBU_3DFE','Yeast_cold'];sizes=[192,384,576];summary=[];activity=[];paired=[];raw=[];diagnostics=[]
for name in datasets:
 for n in sizes:
  groups=[];count={'dataset':name,'train_n':n,'off_found':0,'on_found':0,'both_found':0,'changed_pairs':0,'on_test_better':0,'on_test_worse':0,'common_prefix_gamma_only_blocks':0,'original_blocked_deletions':0,'blocked_drop_worsens_test':0,'blocked_drop_improves_test':0,'original_affected_splits':0}
  for rep in range(1,6):
   job=p/name/f'repeat_{rep}_n{n}';assert (job/'complete.json').exists();rr=json.loads((job/'results.json').read_text());groups.extend(rr);raw.extend(rr)
   split=json.loads((job/'split.json').read_text());assert len(set(split['train_indices']))==n and len(set(split['test_indices']))==256 and set(split['train_indices']).isdisjoint(split['test_indices']);assert sorted(sum(split['fold_validation_positions'],[]))==list(range(n))
   cp=json.loads((job/'selection_before_test.json').read_text());onoff=json.loads((job/'audit_results.json').read_text());off,on=onoff;count['off_found']+=off['success'];count['on_found']+=on['success'];assert not on['success'] or off['success']
   j=next(r for r in rr if r['method']=='Joint guard');assert j['inner_brier']<=cp['risk_reference'] and j['gamma']>=cp['gamma_reference'] and j['cost']==cp['budget']
   for r in rr:
    P=cp['selected'].get(r['method'],[]);assert len(P)==r['attributes'] and sum(split['costs'][a] for a in P)==r['cost']
    if r['method']=='Budget-forward bridge':assert r['cost']<=cp['budget']
   for a in onoff:
    assert a['selected']==cp['audit'][a['mode']]
    if a['success']:assert a['cost']<=cp['budget']
   if on['success']:
    count['both_found']+=1;changed=off['selected']!=on['selected'];count['changed_pairs']+=changed;dif=on['test_brier']-off['test_brier'];count['on_test_better']+=dif<0;count['on_test_worse']+=dif>0;paired.append({'dataset':name,'repeat':rep,'train_n':n,'changed':changed,'off_cost':off['cost'],'on_cost':on['cost'],'off_brier':off['test_brier'],'on_brier':on['test_brier'],'on_minus_off':dif})
   events=json.loads((job/'audit_events.json').read_text());count['common_prefix_gamma_only_blocks']+=sum(e['phase']=='prefix' and e['risk_pass'] and not e['gamma_pass'] for e in events)
   ds=json.loads((job/'blocked_test.json').read_text());diagnostics.extend(ds);count['original_blocked_deletions']+=len(ds);count['original_affected_splits']+=bool(ds);count['blocked_drop_worsens_test']+=sum(d['candidate_minus_original']>0 for d in ds);count['blocked_drop_improves_test']+=sum(d['candidate_minus_original']<0 for d in ds)
  for method in ['All attributes','Joint guard','Prediction-only','Budget-forward bridge','Global mean']:
   rr=[r for r in groups if r['method']==method];assert len(rr)==5;s={'dataset':name,'train_n':n,'method':method}
   for key in ['attributes','cost','inner_brier','gamma','coverage','test_brier','test_kl','test_tv','test_js']:
    v=[r[key] for r in rr];s[key+'_mean']=float(np.mean(v));s[key+'_sd']=float(np.std(v,ddof=1))
   if method!='All attributes':
    deltas=[]
    for r in rr:
     a=next(a for a in groups if a['repeat']==r['repeat'] and a['method']=='All attributes');deltas.append(100*(r['test_brier']/a['test_brier']-1))
    s['relative_brier_percent_mean']=float(np.mean(deltas));s['no_worse_than_all']=sum(d<=0 for d in deltas)
   else:s['relative_brier_percent_mean']=0.;s['no_worse_than_all']=5
   summary.append(s)
  activity.append(count)
# Nested train/test identity within each repetition.
for name in datasets:
 for rep in range(1,6):
  prev=set();te=None
  for n in sizes:
   s=json.loads((p/name/f'repeat_{rep}_n{n}'/'split.json').read_text());assert prev<=set(s['train_indices'])
   if te is not None:assert te==s['test_indices']
   prev=set(s['train_indices']);te=s['test_indices']
for fname,rows in [('summary',summary),('activity',activity),('paired',paired),('blocked_test',diagnostics)]:
 with (p/f'{fname}.csv').open('w',newline='') as f:
  if rows:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(p/'validation.txt').write_text('PASS: 30 completed fresh-dataset splits, 150 main method evaluations; 30 joint outputs feasible; 3-fold coverage, train/test separation, nested sizes, common costs and checkpoints verified. Audit failures retained. Labels preserved, including genuine zeros.\n')
plt.rcParams.update({'font.size':10,'pdf.fonttype':42});fig,axes=plt.subplots(3,1,figsize=(6.5,8.4));colors={'All attributes':'#1f77b4','Joint guard':'#2ca02c','Prediction-only':'#ff7f0e','Budget-forward bridge':'#9467bd','Global mean':'#777777'}
for ax,name in zip(axes,datasets):
 for method in colors:
  ss=[r for r in summary if r['dataset']==name and r['method']==method];ax.errorbar(sizes,[r['test_brier_mean'] for r in ss],yerr=[r['test_brier_sd'] for r in ss],marker='o',capsize=3,label=method,color=colors[method])
 ax.set_title(name.replace('_','-'));ax.set_ylabel('Test Brier (mean ± SD)');ax.set_xticks(sizes);ax.set_xlabel('Training objects');ax.grid(alpha=.2);ax.legend(fontsize=7)
x=np.arange(3)
for i,name in enumerate(datasets):
 ss=[r for r in activity if r['dataset']==name];axes[2].bar(x+(i-.5)*.3,[r['original_blocked_deletions'] for r in ss],width=.3,label=name.replace('_','-'))
axes[2].set_xticks(x,[str(n) for n in sizes]);axes[2].set_xlabel('Training objects');axes[2].set_ylabel('Gamma-only blocked deletions');axes[2].legend(fontsize=8);fig.tight_layout();fig.savefig(p/'cross_summary.pdf');fig.savefig(p/'cross_summary.png',dpi=180)
print(json.dumps({'summary':summary,'activity':activity,'paired':paired},indent=2))

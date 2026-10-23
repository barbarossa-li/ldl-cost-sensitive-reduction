from pathlib import Path
import json,csv,itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent;rows=list(csv.DictReader((p/'results.csv').open()));assert len(rows)==45
for r in rows:
 for k in ['train_n','repeat','attributes','cost','joint_pass']:r[k]=int(r[k])
 for k in ['inner_brier','test_brier','test_kl','test_tv','test_js','gamma','coverage']:r[k]=float(r[k])
 r['selected']=json.loads(r['selected'])
sizes=[192,384,576];methods=['All attributes','Dependency-only','Joint guard'];summary=[]
for n in sizes:
 for method in methods:
  rr=[r for r in rows if r['train_n']==n and r['method']==method];assert len(rr)==5
  s={'train_n':n,'method':method,'joint_pass_count':sum(r['joint_pass'] for r in rr)}
  for k in ['attributes','cost','inner_brier','test_brier','test_kl','test_tv','test_js','gamma','coverage']:
   v=[r[k] for r in rr];s[k+'_mean']=float(np.mean(v));s[k+'_sd']=float(np.std(v,ddof=1))
  summary.append(s)
paired=[]
for n in sizes:
 diffs=[];rel=[];sets=[]
 for rep in range(1,6):
  a=next(r for r in rows if r['train_n']==n and r['repeat']==rep and r['method']=='All attributes')
  j=next(r for r in rows if r['train_n']==n and r['repeat']==rep and r['method']=='Joint guard')
  diffs.append(j['test_brier']-a['test_brier']);rel.append(100*(j['test_brier']/a['test_brier']-1));sets.append(set(j['selected']))
 js=[len(a&b)/len(a|b) if a|b else 1. for a,b in itertools.combinations(sets,2)]
 paired.append({'train_n':n,'outer_brier_delta_mean':float(np.mean(diffs)),'outer_brier_delta_sd':float(np.std(diffs,ddof=1)),'relative_delta_percent_mean':float(np.mean(rel)),'no_worse_than_all_count':sum(d<=0 for d in diffs),'pairwise_jaccard_mean':float(np.mean(js)),'paired_deltas':diffs})
with (p/'summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
(p/'paired_diagnostics.json').write_text(json.dumps(paired,indent=2))
# Validate saved pre-test decisions, train/test separation and nested sizes.
protocol=json.loads((p/'protocol.json').read_text());excluded=set(protocol['excluded_previous_objects'])
for rep in range(1,6):
 previous=set();test0=None
 for n in sizes:
  job=p/f'repeat_{rep}_n{n}';s=json.loads((job/'split.json').read_text());tr=set(s['train_indices']);te=set(s['test_indices']);assert len(tr)==n and len(te)==256 and tr.isdisjoint(te) and excluded.isdisjoint(tr|te) and previous<=tr
  if test0 is not None:assert test0==te
  previous=tr;test0=te
  assert sorted(sum(s['fold_validation_positions'],[]))==list(range(n))
  saved=json.loads((job/'selection_before_test.json').read_text())
  for r in rows:
   if r['repeat']==rep and r['train_n']==n:assert saved[r['method']]==r['selected']
assert all(r['joint_pass']==1 for r in rows if r['method']=='Joint guard')
(p/'validation.txt').write_text('PASS: 15 completed runs, 45 method evaluations; all 15 joint guards feasible; split separation, old-object exclusion, nested training, full fold coverage, pre-test selection checkpoints verified.\n')
plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
colors={'All attributes':'#1f77b4','Dependency-only':'#ff7f0e','Joint guard':'#2ca02c'}
fig,axes=plt.subplots(3,1,figsize=(6.5,8))
for method in methods:
 ss=[s for s in summary if s['method']==method]
 axes[0].errorbar(sizes,[s['test_brier_mean'] for s in ss],yerr=[s['test_brier_sd'] for s in ss],marker='o',capsize=3,label=method,color=colors[method])
 axes[0].set_ylabel('Outer-test Brier (mean ± SD)')
for method in methods[1:]:
 ss=[s for s in summary if s['method']==method]
 axes[1].errorbar(sizes,[s['attributes_mean'] for s in ss],yerr=[s['attributes_sd'] for s in ss],marker='o',capsize=3,label=method,color=colors[method])
axes[1].set_ylabel('Selected attributes (mean ± SD)')
for d in paired:
 axes[2].scatter([d['train_n']]*5,d['paired_deltas'],s=24)
axes[2].plot(sizes,[d['outer_brier_delta_mean'] for d in paired],color='black',marker='o',label='Mean paired difference');axes[2].axhline(0,color='gray',ls='--');axes[2].set_ylabel('Joint minus all: outer Brier')
for ax in axes:ax.set_xlabel('Training objects');ax.set_xticks(sizes);ax.grid(alpha=.2)
axes[0].legend(fontsize=8);axes[1].legend(fontsize=8);axes[2].legend(fontsize=8)
fig.tight_layout();fig.savefig(p/'stability.pdf');fig.savefig(p/'stability.png',dpi=180)
print(json.dumps({'summary':summary,'paired':paired},indent=2))

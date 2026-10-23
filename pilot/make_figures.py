from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
with (p/'pilot_results.csv').open() as f:rows=list(csv.DictReader(f))
labels=['All','Mean','New','Info','Cheap','Random']
colors=['#3976ae','#a5a5a5','#df7b28','#efb36b','#9e90bd','#76aaa0']
fig,axes=plt.subplots(2,2,figsize=(9,6))
for ax,key,title in zip(axes.ravel(),['gamma','coverage','brier','kl'],['Training dependency (higher)','Eligible training coverage','Held-out Brier loss (lower)','Held-out KL loss (lower)']):
 vals=np.array([float(r[key]) for r in rows]);ax.bar(range(len(rows)),vals,color=colors)
 ax.axhline(vals[0],color=colors[0],linestyle='--',linewidth=1,alpha=.8)
 ax.set_xticks(range(len(rows)),labels,fontsize=10);ax.set_title(title,fontsize=11);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 for i,v in enumerate(vals):ax.text(i,v+max(vals)*.015,f'{v:.3f}',ha='center',va='bottom',fontsize=10)
 ax.set_ylim(0,max(vals)*1.15)
fig.suptitle('Natural Scene pilot: 128 training / 256 test objects; one fixed split',fontsize=12);fig.tight_layout(rect=(0,0,1,.95))
for ext in ['pdf','png']:fig.savefig(p/f'pilot_summary.{ext}',dpi=180)
with (p/'single_attribute_diagnostic.csv').open() as f:single=list(csv.DictReader(f))
with (p/'cardinality_profile.csv').open() as f:profile=list(csv.DictReader(f))
diag=json.loads((p/'training_diagnosis.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(8.5,3.4))
g=np.array([float(r['gamma']) for r in single]);axes[0].hist(g,bins=24,color='#8cacc9',edgecolor='white');axes[0].axvline(diag['training_full_gamma'],color='#c65537',linestyle='--',label='Full-feature reference');axes[0].legend(frameon=False,fontsize=8);axes[0].set(xlabel='Single-feature dependency',ylabel='Number of features',title='15 of 294 single features meet the target')
sizes=sorted(set(int(r['attributes']) for r in profile))
means=[];lo=[];hi=[]
for size in sizes:
 vals=[float(r['gamma']) for r in profile if int(r['attributes'])==size];means.append(np.mean(vals));lo.append(np.min(vals));hi.append(np.max(vals))
axes[1].plot(sizes,means,'o-',color='#3976ae');axes[1].fill_between(sizes,lo,hi,color='#3976ae',alpha=.18);axes[1].set_xscale('log',base=2);axes[1].set(xlabel='Random subset feature count',ylabel='Training dependency',title='Mean and range of 5 subsets per count');axes[1].axhline(diag['training_full_gamma'],color='#c65537',linestyle='--');axes[1].grid(alpha=.2)
fig.tight_layout()
for ext in ['pdf','png']:fig.savefig(p/f'training_diagnostic.{ext}',dpi=180)

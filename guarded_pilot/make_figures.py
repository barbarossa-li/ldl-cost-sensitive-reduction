from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
with (p/'results.csv').open() as f:by={r['method']:r for r in csv.DictReader(f)}
methods=['All attributes','Dependency-only','Guarded heuristic','Prediction-only ablation','Cheapest same cardinality','Random same cardinality'];labels=['All','Dep','Joint','Pred','Cheap','Random'];colors=['#3976ae','#bbbbbb','#df7b28','#8c90bd','#77a99c','#c0a581']
rows=[by[name] for name in methods];reference=float(by['All attributes']['inner_brier'])
fig,axes=plt.subplots(1,2,figsize=(8.3,3.5))
for ax,key,title,ref in zip(axes,['inner_brier','test_brier'],['Training-internal 3-fold Brier','Fresh external-test Brier'],[reference,float(by['All attributes']['test_brier'])]):
 vals=[float(r[key]) for r in rows];ax.bar(range(len(rows)),vals,color=colors);ax.axhline(ref,color='#3976ae',linestyle='--',linewidth=1);ax.set_xticks(range(len(rows)),labels,fontsize=10);ax.set_title(title,fontsize=11);ax.set_ylim(0,max(vals)*1.16);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 for i,v in enumerate(vals):ax.text(i,v+.004,f'{v:.3f}',ha='center',fontsize=10)
fig.suptitle('Natural Scene: 192 training / 256 fresh test objects',fontsize=12);fig.tight_layout(rect=(0,0,1,.92))
for ext in ['pdf','png']:fig.savefig(p/f'guarded_summary.{ext}',dpi=180)
selected=json.loads(by['Guarded heuristic']['selected']);all_scores=joint_scores=None
for r in csv.DictReader((p/'inner_loss_cache.csv').open()):
 P=json.loads(r['selected'])
 if len(P)==294:all_scores=json.loads(r['fold_scores'])
 if P==selected:joint_scores=json.loads(r['fold_scores'])
fig,ax=plt.subplots(figsize=(5.5,3.3));xx=np.arange(3);ax.bar(xx-.17,all_scores,.34,label='All features',color='#3976ae');ax.bar(xx+.17,joint_scores,.34,label='Joint reduction',color='#df7b28');ax.set_xticks(xx,['Fold 1','Fold 2','Fold 3']);ax.set_ylabel('Validation Brier loss');ax.set_title('Mean guard passes; not every fold improves');ax.legend(frameon=False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);fig.tight_layout()
for ext in ['pdf','png']:fig.savefig(p/f'fold_diagnostic.{ext}',dpi=180)

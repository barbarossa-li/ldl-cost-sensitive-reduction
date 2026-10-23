from pathlib import Path
import json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
p=Path(__file__).resolve().parent;out=p/'paper_figures';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.titlesize':12,'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10})
blue='#226c93';orange='#bb6138'
paired=json.loads((p/'large_holdout_replication/summary.json').read_text())['paired']
fig,axs=plt.subplots(3,1,figsize=(5.6,7.6),layout='constrained')
for ax,name in zip(axs,['Natural Scene','SBU_3DFE','Yeast_cold']):
 for method,color,offset,marker in [('Prediction only',blue,-.04,'o'),('Guided',orange,.04,'s')]:
  q=sorted([v for v in paired if v['dataset']==name and v['method']==method],key=lambda v:v['repeat']);ax.plot([v['repeat']+offset for v in q],[v['audit_gap'] for v in q],color=color,marker=marker,ms=6,label='P+' if method=='Prediction only' else 'G+')
 ax.axhline(0,color='gray',ls='--',lw=1);ax.set_title(name.replace('_',' '));ax.set_xticks([1,2,3]);ax.set_ylabel('Audit Brier minus full');ax.set_xlabel('Saved development repeat');ax.yaxis.set_major_locator(MaxNLocator(4));ax.grid(alpha=.2)
axs[0].legend(loc='best',ncol=2);fig.savefig(out/'transfer_large.pdf');fig.savefig(out/'transfer_large.png',dpi=160);plt.close(fig)
controls=list(csv.DictReader(open(p/'matched_cost_audit/controls.csv')));summary=json.loads((p/'matched_cost_audit/summary.json').read_text())
for name,stem in [('Natural Scene','natural'),('SBU_3DFE','sbu')]:
 fig,axs=plt.subplots(2,1,figsize=(5.6,6.8),layout='constrained')
 for ax,method in zip(axs,['Prediction only','Guided']):
  for rep in [1,2,3]:
   q=next(v for v in summary if v['dataset']==name and v['method']==method and v['repeat']==rep);values=[float(r['audit_brier']) for r in controls if r['dataset']==name and r['method']==method and int(r['repeat'])==rep]
   ax.boxplot([values],positions=[rep],widths=.35,patch_artist=True,boxprops=dict(facecolor='#bbd7e3',edgecolor=blue),medianprops=dict(color=blue),flierprops=dict(markersize=4));ax.scatter(rep,q['selected_audit'],marker='D',color=orange,s=52,zorder=5)
  ax.set_title(name.replace('_',' ')+' | '+('P+' if method=='Prediction only' else 'G+'));ax.set_xticks([1,2,3]);ax.set_ylabel('Reserved-audit Brier');ax.set_xlabel('Saved development repeat');ax.grid(axis='y',alpha=.2)
 fig.legend(handles=[Line2D([0],[0],color=blue,lw=5,label='20 matched controls'),Line2D([0],[0],color=orange,marker='D',ls='',label='Frozen refined subset')],loc='outside upper center',ncol=2,fontsize=10)
 fig.savefig(out/f'matched_{stem}.pdf');fig.savefig(out/f'matched_{stem}.png',dpi=160);plt.close(fig)
points=json.loads((p/'pareto_audit/fronts.json').read_text())
for name,stem in [('Natural Scene','natural'),('SBU_3DFE','sbu')]:
 fig,axs=plt.subplots(3,1,figsize=(5.6,7.8),layout='constrained')
 for ax,rep in zip(axs,[1,2,3]):
  q=[v for v in points if v['dataset']==name and v['repeat']==rep]
  for method,color,marker in [('Prediction ranking',blue,'o'),('Dependency guidance',orange,'s')]:
   v=[r for r in q if r['method']==method];ax.scatter([r['spent'] for r in v],[r['audit_brier'] for r in v],color=color,marker=marker,s=55,zorder=4)
  empty=next(v for v in q if v['method']=='Empty');full=next(v for v in q if v['method']=='Full');ax.scatter(0,empty['audit_brier'],color='#666666',marker='D',s=42,zorder=4);ax.axhline(full['audit_brier'],color='#777777',ls='--',lw=1)
  unique=sorted({(v['spent'],v['audit_brier']) for v in q if v['nondominated'] and v['method']!='Full'});ax.plot([v[0] for v in unique],[v[1] for v in unique],color='#777777',lw=1,alpha=.8)
  for budget in [5,10,20]:
   pr=next(v for v in q if v['method']=='Prediction ranking' and v['budget']==budget);g=next(v for v in q if v['method']=='Dependency guidance' and v['budget']==budget);ax.plot([pr['spent'],g['spent']],[pr['audit_brier'],g['audit_brier']],color='#aaaaaa',ls=':',lw=1,zorder=1)
  ax.set_title(name.replace('_',' ')+f' | repeat {rep}');ax.set_xlim(-.6,21);ax.set_xticks([0,5,10,15,20]);ax.set_xlabel('Actual acquisition cost');ax.set_ylabel('Reserved-audit Brier');ax.grid(alpha=.15);ax.yaxis.set_major_locator(MaxNLocator(4))
 fig.legend(handles=[Line2D([0],[0],color=blue,marker='o',ls='',label='Prediction ranking'),Line2D([0],[0],color=orange,marker='s',ls='',label='Dependency guidance'),Line2D([0],[0],color='#666666',marker='D',ls='',label='Empty'),Line2D([0],[0],color='#777777',ls='--',label='Full loss (cost outside view)')],loc='outside upper center',ncol=2,fontsize=9.5)
 fig.savefig(out/f'pareto_{stem}.pdf');fig.savefig(out/f'pareto_{stem}.png',dpi=160);plt.close(fig)
# Unscaled, separated budget tables; preserve actual spending and add feasibility counts.
budget=json.loads((p/'common_budget_audit/summary.json').read_text())['means'];tables=[]
for name,stem in [('Natural Scene','natural'),('SBU_3DFE','sbu')]:
 t=r'''\begin{table}[htbp]
\centering\small\renewcommand{\arraystretch}{1.35}\setlength{\tabcolsep}{7pt}
'''+f'\\caption{{Common-budget ranking ablation on {name.replace("_"," ")}: three-repeat means. Actual spending may be below the cap; passes refer to the inner guard. No deletion/exchange phase is used.}}\n'+f'\\label{{tab:budget-{stem}}}\n'+r'''\begin{tabular}{rrrrrr}\hline
Cap & Rank & Spent & Attributes & Audit Brier & Inner passes\\\hline
'''
 for cap in [5,10,20]:
  for method,short in [('Prediction ranking','P'),('Dependency guidance','G')]:
   v=next(v for v in budget if v['dataset']==name and v['budget']==cap and v['method']==method)
   t+=f"{cap} & {short} & {v['mean_spent']:.2f} & {v['mean_attributes']:.2f} & {v['mean_audit_brier']:.6f} & {v['inner_pass']}/3\\\\\n"
 t+=r'\hline\end{tabular}\end{table}'+'\n';tables.append(t)
(p/'paper_budget_tables.tex').write_text('\n'.join(tables)+r'For Yeast-cold, both rankings return zero attributes with cost zero, audit Brier 0.006046, and three inner passes at every cap; the coincident outcomes provide no ranking comparison.'+'\n')

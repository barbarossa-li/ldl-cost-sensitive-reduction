"""Budget-sweep figure and ridge-protocol table for the paper.

  budget_sweep.pdf     audit Brier vs actual spending across caps {1,2,5,10,20,50}
                      on the three informative datasets (both ranking rules)
  paper_ridge_table.tex  per-dataset ridge-protocol results table
Reproduction: python generate_final_figures.py
"""
from pathlib import Path
import csv,json,collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here=Path(__file__).resolve().parent
figdir=here.parent/'paper_figures'
plt.rcParams.update({'font.size':11,'axes.titlesize':12,'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10})
blue='#226c93';orange='#bb6138';gray='#666666'

def texname(ds):return ds.replace('_','\\_').replace('-','\\_')

# ---------------- budget sweep figure ----------------
rows=list(csv.DictReader(open(here/'budget_sweep_results.csv')))
for r in rows:r['cap']=int(r['cap']);r['spent']=int(r['spent']);r['audit_brier']=float(r['audit_brier'])
INF=['SJAFFE','fbp5500','RAF_ML']
fig,axs=plt.subplots(1,3,figsize=(7.9,3.4),layout='constrained')
for ax,ds in zip(axs,INF):
 rs=[r for r in rows if r['dataset']==ds]
 for method,color,mk in [('Prediction ranking',blue,'o'),('Dependency guidance',orange,'s')]:
  xs=[];ys=[]
  for cap in [1,2,5,10,20,50]:
   sel=[r for r in rs if r['method']==method and r['cap']==cap]
   if sel:xs.append(np.mean([r['spent'] for r in sel]));ys.append(np.mean([r['audit_brier'] for r in sel]))
  ax.plot(xs,ys,color=color,marker=mk,ms=5,lw=1.3,label='$P$ ranking' if method=='Prediction ranking' else '$G$ ranking')
 # references
 full=np.mean([float(r['audit_brier']) for r in rows if r['dataset']==ds][:1]) # placeholder
 ax.axhline(np.mean([r['audit_brier'] for r in rs if r['cap']==50 and r['method']=='Prediction ranking']),color='#999999',ls=':',lw=1)
 ax.set_xscale('log');ax.set_title(texname(ds));ax.set_xlabel('Actual spending (log)')
 ax.yaxis.set_major_locator(plt.MaxNLocator(4));ax.grid(alpha=.2)
axs[0].set_ylabel('Reserved-audit Brier');axs[0].legend(fontsize=9)
fig.savefig(figdir/'budget_sweep.pdf');fig.savefig(figdir/'budget_sweep.png',dpi=160);plt.close(fig)

# ---------------- ridge protocol table ----------------
rr=list(csv.DictReader(open(here/'ridge_protocol_results.csv')))
for r in rr:r['audit_brier']=float(r['audit_brier']);r['cost']=float(r['cost'])
by=collections.defaultdict(lambda: collections.defaultdict(list))
for r in rr:by[r['dataset']][r['method']].append(r)
ORDER=['Yeast_diau','Yeast_dtt','Yeast_heat','Yeast_spoem','FG-Net','Painting','SCUT_FBP','Movie','MOR-PH']
lines=[r'''\begin{table}[htbp]
\centering\small\renewcommand{\arraystretch}{1.2}\setlength{\tabcolsep}{4pt}
\caption{Complete protocol under the ridge predictor on the nine revived datasets (three-repeat means). $C(A)$ is the full-set cost; Empty/Full are zero-cost and full-cost audit references under the same ridge predictor. Bold marks selections whose audit Brier is no worse than both references.}
\label{tab:ridge-protocol}
\begin{tabular}{lrrrrr}\hline
Dataset & $C(A)$ & $C(G+)$ & Empty & Full & $G+$\\\hline
''']
for ds in ORDER:
 b=by[ds]
 emp=np.mean([r['audit_brier'] for r in b['Empty']])
 ful=np.mean([r['audit_brier'] for r in b['Full']])
 g=np.mean([r['audit_brier'] for r in b['Guided after']])
 c=np.mean([r['cost'] for r in b['Guided after']])
 ca=b['Full'][0]['cost'] if 'cost' in b['Full'][0] else None
 # full-set cost from the kNN protocol table (same costs)
 fullcost={'Yeast_diau':127,'Yeast_dtt':127,'Yeast_heat':127,'Yeast_spoem':127,'FG-Net':1412,
           'Painting':831,'SCUT_FBP':1690,'Movie':10168,'MOR-PH':1162}[ds]
 mark=r'\textbf' if g<=emp+1e-9 and g<=ful+1e-9 else ''
 gtxt=('%.4f'%g) if not mark else (r'\textbf{%.4f}'%g)
 lines.append(f"{texname(ds)} & {fullcost} & {c:.1f} & {emp:.4f} & {ful:.4f} & {gtxt}\\\\\n")
lines.append(r'\hline\end{tabular}\end{table}')
(here.parent/'paper_ridge_table.tex').write_text(''.join(lines))
print('budget_sweep figure + ridge table written')

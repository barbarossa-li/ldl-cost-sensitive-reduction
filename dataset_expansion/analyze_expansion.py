"""Summaries, figures, and LaTeX tables for the dataset-expansion experiments.

Reads dataset_expansion/audit_results.csv (run_expansion.py) and
dataset_expansion/published_baselines/all_results.csv (run_expansion_published.py)
and produces:
  - expansion_summary.csv                     per dataset x method aggregates
  - paper_figures/expansion_cost.pdf/.png     full vs selected cost per dataset (log)
  - paper_figures/expansion_gap.pdf/.png      audit-loss gap vs full reference
  - paper_figures/expansion_pareto.pdf/.png   normalized cost-loss plane, all repeats
  - paper_figures/expansion_cd.pdf/.png       Friedman critical-difference diagram
  - paper_expansion_table.tex                 main per-dataset results table
  - paper_expansion_published_table.tex       published-core comparison table
Reproduction: python analyze_expansion.py
"""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

here=Path(__file__).resolve().parent
root=here.parent
figdir=root/'paper_figures';figdir.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.titlesize':12,'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10})
blue='#226c93';orange='#bb6138';gray='#666666';green='#4a7c3f';red='#a13434'

rows=list(csv.DictReader(open(here/'audit_results.csv')))
for r in rows:
 r['repeat']=int(r['repeat']);r['attributes']=int(r['attributes']);r['cost']=int(r['cost'])
 r['audit_brier']=float(r['audit_brier']);r['test_brier']=float(r['test_brier'])
 r['dev_n']=int(r['dev_n'])

DATASETS=[]
for r in rows:
 if r['dataset'] not in DATASETS:DATASETS.append(r['dataset'])

def get(ds,method,field='audit_brier'):
 return [r[field] for r in rows if r['dataset']==ds and r['method']==method]

def mean(ds,method,field='audit_brier'):
 v=get(ds,method,field);return float(np.mean(v)) if v else None

def passes(ds,method):
 full=get(ds,'Full','audit_brier');v=get(ds,method,'audit_brier')
 return sum(1 for a,b in zip(v,full) if a<=b) if v else 0

# ---------- per-dataset summary ----------
summary=[]
for ds in DATASETS:
 rec={'dataset':ds,'n_repeats':len(get(ds,'Full','audit_brier')),
      'full_cost':mean(ds,'Full','cost'),'empty_audit':mean(ds,'Empty'),
      'full_audit':mean(ds,'Full')}
 for name,method in [('P+','Prediction only after'),('G+','Guided after')]:
  rec[name+'_cost']=mean(ds,method,'cost');rec[name+'_attrs']=mean(ds,method,'attributes')
  rec[name+'_audit']=mean(ds,method);rec[name+'_passes']=passes(ds,method)
 summary.append(rec)
with (here/'expansion_summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)

def texname(ds):return ds.replace('_','\\_').replace('-','\\_')

# ---------- Figure 1: cost reduction ----------
order=sorted(DATASETS,key=lambda ds:mean(ds,'Full','cost'))
fig,ax=plt.subplots(figsize=(6.3,0.42*len(order)+1.3),layout='constrained')
y=np.arange(len(order))
full=[mean(ds,'Full','cost') for ds in order]
gp=[max(mean(ds,'Guided after','cost'),0.5) for ds in order]
ax.barh(y+.2,full,height=.4,color=blue,label='Full acquisition cost')
ax.barh(y-.2,gp,height=.4,color=orange,label='$G+$ selected cost')
ax.set_xscale('log');ax.set_yticks(y);ax.set_yticklabels([texname(ds) for ds in order],fontsize=10.5)
ax.set_xlabel('Attribute acquisition cost (log scale)');ax.invert_yaxis()
ax.legend(loc='lower right',fontsize=11);ax.grid(axis='x',alpha=.2);ax.tick_params(axis='x',labelsize=10)
fig.savefig(figdir/'expansion_cost.pdf');fig.savefig(figdir/'expansion_cost.png',dpi=200);plt.close(fig)

# ---------- Figure 2: audit-loss gap ----------
fig,ax=plt.subplots(figsize=(6.3,0.42*len(order)+1.3),layout='constrained')
gaps=[];errs=[]
for ds in order:
 d=np.array(get(ds,'Guided after'))-np.array(get(ds,'Full'))
 gaps.append(float(np.mean(d)));errs.append(float(np.std(d)))
colors=[green if g<=0 else red for g in gaps]
ax.barh(y,gaps,xerr=errs,height=.62,color=colors,error_kw=dict(lw=1.2))
ax.axvline(0,color='gray',ls='--',lw=1)
ax.set_yticks(y);ax.set_yticklabels([texname(ds) for ds in order],fontsize=10.5);ax.invert_yaxis()
ax.set_xlabel('Reserved-audit Brier of $G+$ minus full-attribute reference',fontsize=11)
ax.grid(axis='x',alpha=.2);ax.tick_params(axis='x',labelsize=10)
fig.savefig(figdir/'expansion_gap.pdf');fig.savefig(figdir/'expansion_gap.png',dpi=200);plt.close(fig)

# ---------- Figure 3: normalized cost-loss plane ----------
fig,ax=plt.subplots(figsize=(5.6,4.2),layout='constrained')
for method,color,marker,label in [('Prediction only after',blue,'o','$P+$'),('Guided after',orange,'s','$G+$')]:
 xs=[];ys=[]
 for ds in DATASETS:
  c_full=mean(ds,'Full','cost');a_full=mean(ds,'Full')
  for c,a in zip(get(ds,method,'cost'),get(ds,method)):
   xs.append(max(c,0.45)/max(c_full,1));ys.append(a/max(a_full,1e-12))
 ax.scatter(xs,ys,color=color,marker=marker,s=30,alpha=.75,label=label,edgecolors='none')
ax.axhline(1,color=gray,ls='--',lw=1)
ax.set_xscale('log');ax.set_xlabel('Selected cost relative to full set');ax.set_ylabel('Audit Brier relative to full set')
ax.legend(loc='upper left');ax.grid(alpha=.2)
fig.savefig(figdir/'expansion_pareto.pdf');fig.savefig(figdir/'expansion_pareto.png',dpi=160);plt.close(fig)

# ---------- Figure 4 + table: published cores ----------
pub_path=here/'published_baselines'/'all_results.csv'
nc_path=here/'newcore_results.csv'
if pub_path.exists():
 pub=list(csv.DictReader(open(pub_path)))
 if nc_path.exists():
  pub=pub+[r for r in csv.DictReader(open(nc_path)) if r['method'].startswith(('NRS','FMI'))]
 for r in pub:
  r['audit_brier']=float(r['audit_brier']);r['cost']=int(r['cost']);r['attributes']=int(r['attributes'])
 methods=['G+','GBFRS k','GBFRS budget','DSIM k','DSIM budget','FSLC k','FSLC budget']
 extra=[m for m in ['NRS k','NRS budget','FMI k','FMI budget'] if any(r['method']==m for r in pub)]
 methods+=extra
 units=sorted({(r['dataset'],r['repeat']) for r in pub})
 mat=np.array([[next(r['audit_brier'] for r in pub if r['dataset']==d and r['repeat']==rep and r['method']==m) for m in methods] for d,rep in units])
 from scipy.stats import friedmanchisquare,rankdata
 stat,p=friedmanchisquare(*[mat[:,i] for i in range(len(methods))])
 ranks=rankdata(mat,axis=1).mean(0)
 # Nemenyi critical difference (q at alpha=0.05; Studentized-range table values)
 N,k=len(units),len(methods)
 QTAB={2:1.960,3:2.343,4:2.569,5:2.728,6:2.850,7:2.949,8:3.031,9:3.102,10:3.164,11:3.219,12:3.268}
 q_alpha=QTAB.get(k,3.031)
 cd=q_alpha*np.sqrt(k*(k+1)/(6.*N))
 # Standard Demsar critical-difference diagram: rank axis 1..k (reversed),
 # methods at their mean ranks, thick bars joining groups that are NOT
 # significantly different (|rank diff| < CD).
 order_idx=np.argsort(ranks)
 srt=ranks[order_idx]
 labels=[methods[i] for i in order_idx]
 lo=max(0.5,np.floor(min(srt)-cd));hi=min(len(methods)+0.5,np.ceil(max(srt)+cd))
 fig,ax=plt.subplots(figsize=(6.3,3.4),layout='constrained')
 ax.set_xlim(hi,lo)  # reversed: best (low rank) on the right
 ax.set_ylim(-1.5,1.25);ax.set_yticks([])
 # rank axis at top, method labels at bottom
 for x in range(1,len(methods)+1):ax.plot([x,x],[-0.55,-0.45],color='gray',lw=1)
 ax.plot(np.arange(1,len(methods)+1),np.zeros(len(methods)),color='gray',lw=0.8)
 for i,(r,lab) in enumerate(zip(srt,labels)):
  ax.plot(r,0,'o',color=blue,ms=6,zorder=3)
  ax.annotate('',xy=(r,-0.42),xytext=(r,0),arrowprops=dict(arrowstyle='-',color='#999999',lw=0.8))
  ax.text(r,-0.62,lab,rotation=32,ha='right',va='top',fontsize=9.5)
  ax.text(r,0.14,str(round(r,2)),ha='center',fontsize=9,color='#444444')
 # CD bar
 ax.plot([srt[0],srt[0]+cd],[0.62,0.62],color=orange,lw=1.6)
 for x in (srt[0],srt[0]+cd):ax.plot([x,x],[0.56,0.68],color=orange,lw=1.6)
 ax.text(srt[0]+cd/2,0.72,'CD=%.2f'%cd,ha='center',fontsize=9.5,color=orange)
 # groups not significantly different from the best method
 for i in range(1,len(methods)):
  if srt[i]-srt[0]<cd:
   ax.plot([srt[0],srt[i]],[0.42-0.0*i,0.42-0.0*i],color='#aaaaaa',lw=3,alpha=.8,solid_capstyle='butt')
  else:break
 ax.set_xlabel('Mean Friedman rank (lower is better)')
 ax.set_title('Friedman over %d dataset-repeat units: $\\chi^2$=%.2f, p=%.3g'%(N,stat,p),fontsize=11)
 fig.savefig(figdir/'expansion_cd.pdf');fig.savefig(figdir/'expansion_cd.png',dpi=160);plt.close(fig)
 stats={'friedman_chi2':float(stat),'friedman_p':float(p),'n_units':N,'cd':float(cd),
        'mean_ranks':{m:float(r) for m,r in zip(methods,ranks)},
        'mean_cost':{m:float(np.mean([r['cost'] for r in pub if r['method']==m])) for m in methods},
        'mean_audit':{m:float(np.mean([r['audit_brier'] for r in pub if r['method']==m])) for m in methods}}
 # informative subset: units where G+ selected a non-empty set
 informative=[(d,rep) for d,rep in units if any(r['dataset']==d and r['method']=='G+' and r['attributes']>0 for r in pub)]
 if informative:
  gcol=methods.index('G+')
  imat=np.array([[next(r['audit_brier'] for r in pub if r['dataset']==d and r['repeat']==rep and r['method']==m) for m in methods] for d,rep in informative])
  stats['informative_units']=len(informative)
  stats['informative_mean_audit']={m:float(np.mean(imat[:,i])) for i,m in enumerate(methods)}
  stats['informative_mean_cost']={m:float(np.mean([r['cost'] for r in pub if r['method']==m and (r['dataset'],r['repeat']) in informative])) for m in methods}
  stats['informative_paired_delta']={m:float(np.mean(imat[:,gcol]-imat[:,i])) for i,m in enumerate(methods) if m!='G+'}
  stats['informative_wins']={m:int(np.sum(imat[:,gcol]<imat[:,i])) for i,m in enumerate(methods) if m!='G+'}
 (here/'expansion_stats.json').write_text(json.dumps(stats,indent=2))
 print('Friedman chi2=%.3f p=%.3g ranks=%s'%(stat,p,dict(zip(methods,np.round(ranks,2)))))
 # published-core table: informative datasets per row + pooled aggregate
 def esc(m):return m.replace('+','\\ensuremath{+}')
 inf_ds=[d for d in ['SJAFFE','fbp5500','RAF_ML'] if any(r['dataset']==d and r['method']=='G+' and r['attributes']>0 for r in pub)]
 lines=[r'''\begin{table}[htbp]
\centering\small\renewcommand{\arraystretch}{1.25}\setlength{\tabcolsep}{4pt}
\caption{Published scoring cores on the informative expansion datasets (three-repeat means, reserved-audit Brier). Baselines marked $k$ share cardinality with $G+$; budget variants share its cost. On the twenty guard-empty datasets every matched protocol returns the empty set and ties exactly, so those units enter only the pooled statistics reported in the text.}
\label{tab:expansion-published}
\begin{tabular}{llrrr}\hline
Dataset & Method & Attributes & Cost & Audit Brier\\\hline
''']
 for ds in inf_ds:
  first=True
  for m in methods:
   rs=[r for r in pub if r['dataset']==ds and r['method']==m]
   dsx=ds.replace('_','\\_').replace('-','\\_')
   nm=esc(m) if m!='G+' else r'\textbf{$G+$ (proposed)}'
   lines.append(('%s & %s & %.1f & %.1f & %.4f\\\\\n'%(dsx if first else '',nm,np.mean([r['attributes'] for r in rs]),np.mean([r['cost'] for r in rs]),np.mean([r['audit_brier'] for r in rs]))))
   first=False
  lines.append('\\hline\n')
 lines.append(r'\end{tabular}\end{table}')
 (root/'paper_expansion_published_table.tex').write_text(''.join(lines))

# ---------- main expansion table ----------
lines=[r'''\begin{table}[htbp]
\centering\scriptsize\renewcommand{\arraystretch}{1.15}\setlength{\tabcolsep}{3.5pt}
\caption{Dataset expansion under the frozen nested audit protocol (three repeats). Costs and audit Brier scores are means; passes compare the audit loss of each selection with the full-attribute reference. Datasets with fewer than 832 objects use the single-192-block development variant.}
\label{tab:expansion-main}
\begin{tabular}{lrrrrrrr}\hline
Dataset & $C(A)$ & $C(G+)$ & Attrs & Full & $G+$ & Pass & Test\\\hline
''']
for ds in DATASETS:
 rec=next(r for r in summary if r['dataset']==ds)
 ratio=rec['G+_cost']/rec['full_cost'] if rec['full_cost'] else 0
 lines.append(f"{texname(ds)} & {rec['full_cost']:.0f} & {rec['G+_cost']:.1f} & {rec['G+_attrs']:.1f} & {rec['full_audit']:.4f} & {rec['G+_audit']:.4f} & {rec['G+_passes']}/3 & {mean(ds,'Guided after','test_brier'):.4f}\\\\\n")
lines.append(r'\hline\end{tabular}\end{table}')
(root/'paper_expansion_table.tex').write_text(''.join(lines))
print('figures + tables written;',len(DATASETS),'datasets')

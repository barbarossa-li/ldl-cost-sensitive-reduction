"""Figures for the ablation and probe experiments (Plans B and C) plus supporting
visuals from the frozen expansion results.

  noguard_guard.pdf      guard-disabled ablation: wasted cost + audit gap vs free empty
  ridge_probe.pdf        predictor-sensitivity probe: ridge vs empty audit scatter
  refine_convergence.pdf refinement cost trajectories on informative datasets
  method_heatmap.pdf     mean audit Brier heatmap, methods x informative datasets
  audit_test.pdf         reserved-audit vs held-out test Brier (audit-proxy validity)
  cost_strata.pdf        cost distribution of selected vs all attributes
  full_empty.pdf         full vs empty audit Brier across all 25 datasets (prevalence)

Reproduction: python generate_ablation_figures.py
Writes to ../paper_figures/.
"""
from pathlib import Path
import csv,json,glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

here=Path(__file__).resolve().parent
figdir=here.parent/'paper_figures'
plt.rcParams.update({'font.size':11,'axes.titlesize':12,'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10})
blue='#226c93';orange='#bb6138';gray='#666666';green='#4a7c3f';red='#a13434'

def texname(ds):return ds.replace('_','\\_')

def read(p):return list(csv.DictReader(open(p)))

# ---------------- Fig: noguard ablation ----------------
ng=read(here/'noguard_results.csv')
import collections
by=collections.defaultdict(list)
for r in ng:by[r['dataset']].append(r)
order=sorted(by,key=lambda ds:-np.mean([float(r['cost']) for r in by[ds]]))
fig,axs=plt.subplots(1,2,figsize=(7.6,7.4),layout='constrained',sharey=True)
y=np.arange(len(order))
costs=[np.mean([float(r['cost']) for r in by[ds]]) for ds in order]
gaps=[np.mean([float(r['audit_brier_sel'])-float(r['audit_brier_empty']) for r in by[ds]]) for ds in order]
gerr=[np.std([float(r['audit_brier_sel'])-float(r['audit_brier_empty']) for r in by[ds]]) for ds in order]
axs[0].barh(y,costs,color=blue,height=.62)
axs[0].set_xlabel('Acquisition cost spent by guard-disabled selection',fontsize=11)
axs[0].set_yticks(y);axs[0].set_yticklabels([texname(ds) for ds in order],fontsize=10)
axs[0].invert_yaxis();axs[0].grid(axis='x',alpha=.2);axs[0].tick_params(axis='x',labelsize=9.5)
axs[1].barh(y,gaps,xerr=gerr,height=.62,color=[red if g>0 else green for g in gaps],error_kw=dict(lw=1.2))
axs[1].axvline(0,color='gray',ls='--',lw=1)
axs[1].set_xlabel('Audit Brier: spent selection minus free empty',fontsize=11)
axs[1].grid(axis='x',alpha=.2);axs[1].tick_params(axis='x',labelsize=9.5)
axs[1].tick_params(labelleft=False)
fig.savefig(figdir/'noguard_guard.pdf');fig.savefig(figdir/'noguard_guard.png',dpi=160);plt.close(fig)

# ---------------- Fig: ridge probe ----------------
rr=read(here/'ridge_results.csv')
fig,ax=plt.subplots(figsize=(5.4,4.6),layout='constrained')
for ds in sorted({r['dataset'] for r in rr}):
 rs=[r for r in rr if r['dataset']==ds]
 rev=sum(1 for r in rs if r['revived']=='True')>=2
 c=orange if rev else gray
 ex=[float(r['audit_brier_empty']) for r in rs];ri=[float(r['audit_brier_ridge']) for r in rs]
 ax.scatter(ex,ri,s=26,color=c,alpha=.8,edgecolors='none')
lims=[.9*min(min(float(r['audit_brier_empty']),float(r['audit_brier_ridge'])) for r in rr),
      1.08*max(max(float(r['audit_brier_empty']),float(r['audit_brier_ridge'])) for r in rr)]
ax.plot(lims,lims,color='gray',ls='--',lw=1)
ax.set_xlim(lims);ax.set_ylim(lims)
ax.set_xlabel('Reserved-audit Brier: global mean (free)')
ax.set_ylabel('Reserved-audit Brier: full-set ridge')
ax.legend(handles=[Line2D([0],[0],color=orange,marker='o',ls='',label='Revived under ridge (9 datasets)'),
                   Line2D([0],[0],color=gray,marker='o',ls='',label='Not revived (11 datasets)')],loc='upper left',fontsize=9)
ax.grid(alpha=.2)
fig.savefig(figdir/'ridge_probe.pdf');fig.savefig(figdir/'ridge_probe.png',dpi=160);plt.close(fig)

# ---------------- Fig: refinement convergence ----------------
INF=['SJAFFE','fbp5500','RAF_ML']
fig,axs=plt.subplots(1,3,figsize=(7.9,3.4),layout='constrained',sharey=False)
for ax,ds in zip(axs,INF):
 for rep in [1,2,3]:
  for method,color,lab in [('Guided',orange,'$G+$'),('Prediction only',blue,'$P+$')]:
   f=here/ds.replace('-','_')/f'repeat_{rep}'/(method.replace(' ','_')+'.json')
   if not f.exists():continue
   g=json.loads(f.read_text())
   acc=g['refinement']['accepted']
   xs=[0]+[i+1 for i,mv in enumerate(acc)]
   cs=[g['results']['before_cost']]+[mv['cost'] for mv in acc]
   ax.step(xs,cs,where='post',color=color,alpha=.75,lw=1.4,
           label=lab if rep==1 else None)
 ax.set_title(texname(ds));ax.set_xlabel('Accepted refinement move')
 ax.yaxis.set_major_locator(plt.MaxNLocator(4));ax.grid(alpha=.2)
axs[0].set_ylabel('Acquisition cost')
axs[0].legend(fontsize=9)
fig.savefig(figdir/'refine_convergence.pdf');fig.savefig(figdir/'refine_convergence.png',dpi=160);plt.close(fig)

# ---------------- Fig: method heatmap (informative datasets) ----------------
pub=read(here/'published_baselines'/'all_results.csv')
methods=['G+','GBFRS k','GBFRS budget','DSIM k','DSIM budget','FSLC k','FSLC budget']
# include new cores if merged
extra=[m for m in ['NRS k','NRS budget','FMI k','FMI budget'] if any(r['method']==m for r in pub)]
methods+=extra
mat=np.zeros((len(methods),len(INF)))
for i,mth in enumerate(methods):
 for j,ds in enumerate(INF):
  rs=[float(r['audit_brier']) for r in pub if r['dataset']==ds and r['method']==mth]
  mat[i,j]=np.mean(rs) if rs else np.nan
fig,ax=plt.subplots(figsize=(5.6,0.52*len(methods)+1.6),layout='constrained')
vmax=np.nanmax(mat);vmin=np.nanmin(mat)
im=ax.imshow(mat,aspect='auto',cmap='Blues',vmin=vmin,vmax=vmax)
ax.set_xticks(range(len(INF)),[texname(d) for d in INF],fontsize=10)
ax.set_yticks(range(len(methods)),[m.replace(' ','\\ ') for m in methods],fontsize=10)
for i in range(len(methods)):
 for j in range(len(INF)):
  if not np.isnan(mat[i,j]):
   ax.text(j,i,'%.4f'%mat[i,j],ha='center',va='center',fontsize=9,
           color='white' if mat[i,j]>(vmin+vmax)/2 else 'black')
ax.set_title('Reserved-audit Brier on informative datasets',fontsize=11)
fig.savefig(figdir/'method_heatmap.pdf');fig.savefig(figdir/'method_heatmap.png',dpi=160);plt.close(fig)

# ---------------- Fig: audit vs test ----------------
ex=read(here/'audit_results.csv')
fig,ax=plt.subplots(figsize=(5.0,4.4),layout='constrained')
for method,color,mk in [('Guided after',orange,'s'),('Prediction only after',blue,'o'),('Full',gray,'D')]:
 xs=[float(r['audit_brier']) for r in ex if r['method']==method]
 ys=[float(r['test_brier']) for r in ex if r['method']==method]
 ax.scatter(xs,ys,s=22,color=color,marker=mk,alpha=.7,edgecolors='none',label=method.replace(' after','+'))
r=np.corrcoef([float(r['audit_brier']) for r in ex],[float(r['test_brier']) for r in ex])[0,1]
lim=[0,max(max(float(r['audit_brier']) for r in ex),max(float(r['test_brier']) for r in ex))*1.05]
ax.plot(lim,lim,color='gray',ls='--',lw=1)
ax.set_xlabel('Reserved-audit Brier');ax.set_ylabel('Held-out test Brier (descriptive)')
ax.set_title('Audit-test agreement: r = %.3f (n = %d)'%(r,len(ex)),fontsize=11)
ax.legend(fontsize=9);ax.grid(alpha=.2)
fig.savefig(figdir/'audit_test.pdf');fig.savefig(figdir/'audit_test.png',dpi=160);plt.close(fig)

# ---------------- Fig: cost strata of selected attributes ----------------
fig,axs=plt.subplots(1,3,figsize=(7.9,3.4),layout='constrained')
for ax,ds in zip(axs,INF):
 for rep in [1,2,3]:
  sp=json.loads((here/ds/f'repeat_{rep}'/'split.json').read_text())
  costs=np.array(sp['costs'])
  g=json.loads((here/ds/f'repeat_{rep}'/'Guided.json').read_text())
  sel=costs[g['after']] if g['after'] else np.array([])
  ax.hist([costs,sel],bins=np.arange(.5,11.5,1),color=['#c8dbe4',orange],label=['All attributes','Selected by $G+$'] if rep==1 else None,density=True)
 ax.set_title(texname(ds));ax.set_xlabel('Attribute cost');ax.set_xticks(range(1,11,2))
 ax.yaxis.set_major_locator(plt.MaxNLocator(3));ax.grid(alpha=.2)
axs[0].set_ylabel('Density');axs[0].legend(fontsize=9)
fig.savefig(figdir/'cost_strata.pdf');fig.savefig(figdir/'cost_strata.png',dpi=160);plt.close(fig)

# ---------------- Fig: full vs empty across all datasets ----------------
summ=read(here/'expansion_summary.csv')
# add the four anchor datasets from the paper table (frozen three-repeat records)
anchors={'Natural Scene':(0.232987,0.276018),'SBU-3DFE':(0.028781,0.032800),'Yeast_cold':(0.007124,0.006046),'Yeast_spo':(None,None)}
rows=[]
for r in summ:
 if r['dataset']=='Yeast_spo':
  au=json.loads((here/'Yeast_spo'/'repeat_1'/'audit_results.json').read_text())
  full=np.mean([x['audit_brier'] for x in au if x['method']=='Full'])
  emp=np.mean([x['audit_brier'] for x in au if x['method']=='Empty'])
  rows.append((r['dataset'],full,emp))
 else:rows.append((r['dataset'],float(r['full_audit']),float(r['empty_audit'])))
for k,(f,e) in anchors.items():
 if k!='Yeast_spo':rows.append((k,f,e))
rows.sort(key=lambda v:-(v[1]-v[2]))
fig,ax=plt.subplots(figsize=(6.3,8.6),layout='constrained')
y=np.arange(len(rows))
ax.barh(y-.2,[v[1] for v in rows],height=.4,color=blue,label='Full attribute set')
ax.barh(y+.2,[v[2] for v in rows],height=.4,color='#c9c9c9',label='Global mean (free)')
ax.set_yticks(y);ax.set_yticklabels([texname(v[0]) for v in rows],fontsize=10.5)
ax.invert_yaxis();ax.set_xlabel('Reserved-audit Brier (three-repeat mean)',fontsize=11)
ax.legend(loc='lower right',fontsize=11);ax.grid(axis='x',alpha=.2);ax.tick_params(axis='x',labelsize=10)
ax.set_title('Acquiring everything vs. predicting the mean',fontsize=12)
fig.savefig(figdir/'full_empty.pdf');fig.savefig(figdir/'full_empty.png',dpi=160);plt.close(fig)

print('ablation figures written')

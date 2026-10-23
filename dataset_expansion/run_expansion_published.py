"""Published-LDL-selector scoring cores (GBFRS / DSIM / FSLC) on the expansion datasets.

Same scoring-core implementations and matched-k / matched-budget protocol as
baselines_published/run_cross_published.py, evaluated on the frozen
dataset_expansion splits. k and budget are matched to the Guided-after (G+)
outputs recorded in dataset_expansion/<ds>/repeat_*/Guided.json. Identical
predictor (5-NN), inner folds, normalization, and per-dataset cost vector as
the expansion runs. Selections are frozen before any audit or test evaluation.

Reproduction: python run_expansion_published.py   (checkpoints per split; resumable)
Requires run_expansion.py to have completed the target datasets.
"""
from pathlib import Path
import sys,json,csv,time,gc
import numpy as np
from scipy.spatial.distance import cdist

here=Path(__file__).resolve().parent
root=here.parent
sys.path.insert(0,str(root));sys.path.insert(0,str(root/'baselines_published'))
out=here/'published_baselines'
out.mkdir(exist_ok=True)
from run_published import nearest5,normed,attr_slice,gbfrs_dependency,dsim_scores,fslc_selection

DATASETS=['Yeast_spo','Yeast_alpha','Yeast_cdc','Yeast_diau','Yeast_dtt','Yeast_elu','Yeast_heat','Yeast_spo5','Yeast_spoem',
          'FG-Net','SJAFFE','Music','Painting','SCUT_FBP','fbp5500','emotion6','RAF_ML','Movie','Flickr','Twitter','MOR-PH',
          'Human_Gene','M2B']

def safe(stem):return stem.replace('-','_')

def main():
 datadir=here/'data'
 allrows=[]
 for stem in DATASETS:
  ds=safe(stem)
  if not (datadir/f'{ds}_X.npy').exists():print(stem,'no data, skip');continue
  X=np.load(datadir/f'{ds}_X.npy');L=np.load(datadir/f'{ds}_L.npy');m=X.shape[1]
  rows=[]
  for rep in range(1,4):
   job=out/f'{ds}_repeat_{rep}'
   if (job/'results.json').exists():rows.extend(json.loads((job/'results.json').read_text()));continue
   job.mkdir(parents=True,exist_ok=True)
   src=here/ds/f'repeat_{rep}'
   if not (src/'complete.json').exists():print(ds,'repeat',rep,'main run incomplete, skip');continue
   split=json.loads((src/'split.json').read_text())
   guided_full=json.loads((src/'Guided.json').read_text())
   guided=guided_full['results']
   k=guided['after_attributes'];budget=guided['after_cost']
   costs=np.array(split['costs']);assert len(costs)==m
   dev=np.array(split['dev_indices']);test=np.array(split['test_indices'])
   sp=np.array(split['search_positions']);ap=np.array(split['audit_positions'])
   Xd=X[dev];Ld=L[dev];x=Xd[sp];lab=Ld[sp];auditX=Xd[ap];auditL=Ld[ap];n=len(x)
   blocks=[];cache={}
   for v in map(np.array,split['fold_validation_positions']):
    fit=np.setdiff1d(np.arange(n),v);a=x[fit];b=x[val:=v];lo=a.min(0);span=a.max(0)-lo;span[span==0]=1.;a=(a-lo)/span;b=(b-lo)/span
    blocks.append((((b[:,None,:]-a[None,:,:])**2).transpose(2,0,1).copy(),fit,v))
   def risk(P):
    key=tuple(sorted(P))
    if key not in cache:
     total=0.
     for mat,fit,v in blocks:
      pred=lab[fit][nearest5(mat[list(key)].sum(0))].mean(1) if key else np.tile(lab[fit].mean(0),(len(v),1))
      total+=np.sum((lab[v]-pred)**2)
     cache[key]=float(total/n)
    return cache[key]
   t0=time.perf_counter()
   # ---------- training-only statistics ----------
   z,sig=normed(x)
   dsim=dsim_scores(z,sig,lab)
   xc=x-x.mean(0);yc=lab-lab.mean(0);den=np.sum(xc**2,0)[:,None]*np.sum(yc**2,0)[None,:];cross=xc.T@yc
   rel=np.sum(np.divide(cross**2,den,out=np.zeros_like(cross),where=den>0),axis=1)
   xz=(x-x.mean(0))/np.where(x.std(0)>0,x.std(0),1.);fden=np.outer(np.sum(xz**2,0),np.sum(xz**2,0));fcorr=np.zeros((m,m));np.divide(xz.T@xz,fden,out=fcorr,where=fden>0);np.fill_diagonal(fcorr,1.)
   # ---------- selections ----------
   selected={'DSIM k':tuple(sorted(sorted(range(m),key=lambda a:(-dsim[a],a))[:k]))}
   P=[];remaining=budget
   for a in sorted(range(m),key=lambda a:(-dsim[a]/costs[a],costs[a],a)):
    if costs[a]<=remaining:P.append(a);remaining-=costs[a]
   selected['DSIM budget']=tuple(sorted(P))
   selected['FSLC k']=fslc_selection(rel,fcorr,k=k)
   selected['FSLC budget']=fslc_selection(rel,fcorr,budget=budget,costs=costs)
   for variant in ['k','budget']:
    P=[];Rp=np.ones((n,n));gamma_P=gbfrs_dependency(Rp,lab);trace=[]
    while True:
     if variant=='k' and len(P)>=k:break
     cands=[a for a in range(m) if a not in P]
     if variant=='budget':
      cands=[a for a in cands if sum(costs[P])+costs[a]<=budget]
      if not cands:break
     if not cands:break
     best=None;best_val=None
     for a in cands:
      Rn=np.minimum(Rp,attr_slice(z,sig,a));g=gbfrs_dependency(Rn,lab)
      if best is None or g>best_val+1e-15 or (abs(g-best_val)<=1e-15 and (costs[a],a)<(costs[best],best)):
       best,best_val=a,g
     if variant=='budget' and best_val<=gamma_P+1e-15:break
     trace.append([best,best_val-gamma_P]);P.append(best);Rp=np.minimum(Rp,attr_slice(z,sig,best));gamma_P=best_val
    selected[f'GBFRS {variant}']=tuple(sorted(P))
    print(ds,'GBFRS',variant,rep,'k_sel',len(P),'gamma',round(gamma_P,4),flush=True)
   (job/'selection_before_test.json').write_text(json.dumps({name:list(map(int,pp)) for name,pp in selected.items()},indent=2))
   # ---------- evaluation (identical predictor; audit first, test descriptive) ----------
   lo=x.min(0);span=x.max(0)-lo;span[span==0]=1.;a=(x-lo)/span
   qa=(auditX-lo)/span;b=(X[test]-lo)/span;truth=L[test]
   def evaluate(Pp):
    Pp=list(Pp)
    ap_=lab[nearest5(cdist(qa[:,Pp],a[:,Pp],'sqeuclidean'))].mean(1) if Pp else np.tile(lab.mean(0),(len(auditL),1))
    tp=lab[nearest5(cdist(b[:,Pp],a[:,Pp],'sqeuclidean'))].mean(1) if Pp else np.tile(lab.mean(0),(len(truth),1))
    tp=np.maximum(tp,1e-12);tp/=tp.sum(1,keepdims=True);mid=(truth+tp)/2
    return {'inner_brier':risk(tuple(Pp)),'audit_brier':float(np.mean(np.sum((auditL-ap_)**2,1))),
     'test_brier':float(np.mean(np.sum((truth-tp)**2,1))),
     'test_kl':float(np.mean(np.sum(truth*np.log(np.maximum(truth,1e-300)/tp),1))),
     'test_tv':float(np.mean(.5*np.abs(truth-tp).sum(1))),
     'test_js':float(np.mean(.5*np.sum(truth*np.log(np.maximum(truth,1e-300)/mid),1)+.5*np.sum(tp*np.log(tp/mid),1)))}
   local=[]
   for name,pp in selected.items():
    if name.endswith(' k'):assert len(pp)==k,(ds,name,len(pp),k)
    if name.endswith('budget'):assert sum(costs[list(pp)])<=budget,(ds,name,sum(costs[list(pp)]),budget)
    local.append({'dataset':stem,'repeat':rep,'method':name,'attributes':len(pp),'cost':int(sum(costs[list(pp)])),**evaluate(pp)})
   # G+ reference rows for paired comparison on identical splits
   for nm in ['Guided after']:
    pp=guided_full['after']
    local.append({'dataset':stem,'repeat':rep,'method':'G+','attributes':len(pp),'cost':int(sum(costs[list(pp)])),**evaluate(pp)})
   (job/'results.json').write_text(json.dumps(local,indent=2));rows.extend(local)
   print(ds,'DONE',rep,[(r['method'],round(r['audit_brier'],4)) for r in local],'%.0fs'%(time.perf_counter()-t0),flush=True)
   del blocks,cache,z,sig;gc.collect()
  with (out/f'{ds}_results.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  allrows+=rows
 if allrows:
  with (out/'all_results.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
  print('COMPLETE',len(allrows),'rows')

if __name__=='__main__':main()

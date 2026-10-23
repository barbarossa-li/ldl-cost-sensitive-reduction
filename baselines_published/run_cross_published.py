"""Published-LDL-selector scoring cores on SBU_3DFE and Yeast_cold.

Reuses the GBFRS/DSIM/FSLC scoring cores from run_published.py and the frozen
cross_dataset/ splits (5 repeats x 192/384/576 per dataset). k and budget are
matched to the Joint guard outputs recorded in cross_dataset/<ds>/repeat_*/results.json.
Same predictor, folds, normalization and per-dataset cost vector. Selections saved
before test evaluation. FSLC is included despite its Natural Scene redundancy finding;
its behaviour on these datasets is an open empirical question.

Reproduction: python run_cross_published.py  (checkpoints per split; safe to restart).
"""
from pathlib import Path
import sys,json,csv,time,gc
import numpy as np
from scipy.spatial.distance import cdist

root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
out=Path(__file__).resolve().parent
sys.path.insert(0,str(out))
from run_published import nearest5, normed, attr_slice, gbfrs_dependency, dsim_scores, fslc_selection

def main():
 for ds in ['SBU_3DFE','Yeast_cold','Human_Gene','M2B']:
  dsdir=root/'cross_dataset'/ds
  X=np.load(root/'cross_dataset'/'data'/f'{ds}_X.npy');L=np.load(root/'cross_dataset'/'data'/f'{ds}_L.npy');m=X.shape[1]
  rows=[]
  for rep in range(1,6):
   for n in [192,384,576]:
    job=out/f'{ds}_repeat_{rep}_n{n}';job.mkdir(exist_ok=True)
    if (job/'results.json').exists():rows.extend(json.loads((job/'results.json').read_text()));continue
    saved=dsdir/f'repeat_{rep}_n{n}';split=json.loads((saved/'split.json').read_text());old=json.loads((saved/'results.json').read_text())
    joint=next(r for r in old if r['method']=='Joint guard');k=joint['attributes'];budget=joint['cost']
    costs=np.array(split['costs']);assert len(costs)==m
    train=np.array(split['train_indices']);test=np.array(split['test_indices']);x=X[train];lab=L[train]
    blocks=[];cache={}
    for val in map(np.array,split['fold_validation_positions']):
     fit=np.setdiff1d(np.arange(n),val);a=x[fit];b=x[val];lo=a.min(0);span=a.max(0)-lo;span[span==0]=1.;a=(a-lo)/span;b=(b-lo)/span;blocks.append((((b[:,None,:]-a[None,:,:])**2).transpose(2,0,1).copy(),fit,val))
    def risk(P):
     key=tuple(sorted(P))
     if key not in cache:
      total=0.
      for mat,fit,val in blocks:
       pred=lab[fit][nearest5(mat[list(key)].sum(0))].mean(1) if key else np.tile(lab[fit].mean(0),(len(val),1))
       total+=np.sum((lab[val]-pred)**2)
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
     print(ds,'GBFRS',variant,rep,n,'k_sel',len(P),'gamma',round(gamma_P,4),flush=True)
    (job/'selection_before_test.json').write_text(json.dumps({name:list(map(int,P)) for name,P in selected.items()},indent=2))
    # ---------- evaluation (identical predictor and metrics) ----------
    lo=x.min(0);span=x.max(0)-lo;span[span==0]=1.;a=(x-lo)/span;b=(X[test]-lo)/span;truth=L[test]
    def evaluate(P):
     pred=lab[nearest5(cdist(b[:,P],a[:,P],'sqeuclidean'))].mean(1) if P else np.tile(lab.mean(0),(len(test),1));pred=np.maximum(pred,1e-12);pred/=pred.sum(1,keepdims=True);mid=(truth+pred)/2
     return {'inner_brier':risk(P),'test_brier':float(np.mean(np.sum((truth-pred)**2,axis=1))),'test_kl':float(np.mean(np.sum(truth*np.log(np.maximum(truth,1e-300)/pred),axis=1))),'test_tv':float(np.mean(.5*np.abs(truth-pred).sum(1))),'test_js':float(np.mean(.5*np.sum(truth*np.log(np.maximum(truth,1e-300)/mid),axis=1)+.5*np.sum(pred*np.log(pred/mid),axis=1)))}
    local=[]
    for name,P in selected.items():
     if name.endswith(' k'):assert len(P)==k,(ds,name,len(P),k)
     if name.endswith('budget'):assert sum(costs[list(P)])<=budget,(ds,name,sum(costs[list(P)]),budget)
     local.append({'dataset':ds,'repeat':rep,'train_n':n,'method':name,'attributes':len(P),'cost':int(sum(costs[list(P)])),**evaluate(P)})
    (job/'results.json').write_text(json.dumps(local,indent=2));rows.extend(local)
    print(ds,'DONE',rep,n,[(r['method'],round(r['test_brier'],4)) for r in local],'%.0fs'%(time.perf_counter()-t0),flush=True)
    del blocks,cache,z,sig;gc.collect()
  with (out/f'{ds}_results.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  print(ds,'COMPLETE',len(rows),'rows',flush=True)

if __name__ == '__main__':
 main()

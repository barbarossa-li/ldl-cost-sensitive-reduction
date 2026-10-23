"""Published-LDL-selector scoring cores under the frozen cost-feasibility protocol.

Implements the scoring formulations of three published LDL feature selectors:
  GBFRS  fuzzy-rough LDL dependency (following Qian et al. 2023, KBS, granular-ball
         fuzzy rough sets); we compute exact fuzzy lower approximations rather than
         ball-level approximations, which is a strengthening of the same criterion.
  DSIM   dual-similarity dependence (following Deng et al. 2022, Inf. Sci.); score is
         the normalized covariance between the attribute-induced Gaussian affinity
         matrix and the label-distribution affinity matrix (HSIC-style statistic).
  FSLC   feature-similarity / label-correlation greedy selection (following Qian et al.
         2022, Inf. Sci.); relevance by feature-label distribution correlation with an
         MIFS-style feature-feature redundancy penalty.

All statistics use training data only. Protocol: 15 saved Natural Scene splits, same
five-neighbour Euclidean predictor, same folds, normalization and cost vector as the
proposed method. k and budget are matched to the Joint guard outputs, exactly as the
generic reference selectors in baselines/. Selections are saved before any test
evaluation. These are implementations of the cited scoring cores under a unified
protocol, not verbatim replicas of the published pipelines.

Reproduction order: python run_published.py  (checkpoints per split; safe to restart).
"""
from pathlib import Path
import sys,json,csv,time,gc
import numpy as np
from scipy.spatial.distance import cdist

root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
out=Path(__file__).resolve().parent
X=np.load(root/'pilot/data/feature.npy');L=np.load(root/'pilot/data/label.npy')
costs=np.array(json.loads((root/'stability/protocol.json').read_text())['costs']);m=X.shape[1]

def nearest5(d):
 # Exact stable-argsort nearest five, identical to baselines/run_baselines.py.
 idx=np.argpartition(d,4,axis=1)[:,:5];vals=np.take_along_axis(d,idx,axis=1);threshold=vals.max(1)
 for i in range(len(d)):
  below=np.flatnonzero(d[i]<threshold[i]);ties=np.flatnonzero(d[i]==threshold[i]);v=np.r_[below,ties[:5-len(below)]]
  idx[i]=v[np.lexsort((v,d[i,v]))]
 return idx

def normed(x):
 # Min-max normalize and record per-attribute bandwidth (training standard deviation).
 lo=x.min(0);span=x.max(0)-lo;span[span==0]=1.;z=(x-lo)/span
 sig=z.std(0);sig[sig<1e-12]=1e-12
 return z,sig

def attr_slice(z,sig,a):
 # Gaussian affinity R_a for one attribute: exp(-(za(x)-za(y))^2 / (2 sigma_a^2)).
 return np.exp(-((z[:,None,a]-z[None,:,a])**2)/(2*sig[a]**2))

def gbfrs_dependency(R,lab):
 # Fuzzy-rough LDL dependency of attribute-induced similarity R (n x n):
 # (R down D_k)(x) = min_y max(1-R(x,y), l^k_y); gamma = (1/n) sum_i sum_k l^k_i (R down D_k)(x_i).
 A=1.-R
 low=np.maximum(A[:,:,None],lab[None,:,:]).min(axis=1)  # n x q
 return float((lab*low).sum()/len(lab))

def dsim_scores(z,sig,lab):
 # Dual-similarity dependence per attribute: centered covariance of the attribute
 # affinity matrix with the label-distribution affinity matrix (HSIC-style statistic).
 tl=lab@lab.T;n=len(lab)
 T=tl-tl.mean(0,keepdims=True)-tl.mean(1,keepdims=True)+tl.mean()
 s=np.zeros(z.shape[1])
 for a in range(z.shape[1]):
  s[a]=float((attr_slice(z,sig,a)*T).sum()/((n-1)**2))
 return s

def fslc_selection(rel,ffcorr,k=None,budget=None,costs=None):
 # MIFS-style greedy: maximize relevance minus max correlation with selected features.
 # Exactly one of k (matched cardinality) or budget (matched budget, score/cost) is used.
 P=[];pen=np.zeros(len(rel))
 while True:
  if k is not None and len(P)>=k:break
  cands=[a for a in range(len(rel)) if a not in P]
  if budget is not None:
   cands=[a for a in cands if sum(costs[P])+costs[a]<=budget]
   if not cands:break
   key=lambda a:(-(rel[a]-pen[a])/costs[a],costs[a],a)
  else:
   if not cands:break
   key=lambda a:(-(rel[a]-pen[a]),a)
  a=min(cands,key=key);P.append(a)
  pen=np.maximum(pen,ffcorr[a])
 return tuple(sorted(P))

protocol={'scope':'15 saved Natural Scene splits; published LDL scoring cores under the frozen cost-feasibility protocol','methods':{
 'GBFRS k':'exact fuzzy-rough LDL dependency (Qian et al. 2023 core), forward greedy to matched k',
 'GBFRS budget':'same criterion, positive-gain/cost greedy within the Joint guard budget',
 'DSIM k':'dual-similarity dependence (Deng et al. 2022 core), top-k by score',
 'DSIM budget':'score/cost greedy within budget',
 'FSLC k':'feature-similarity/label-correlation greedy (Qian et al. 2022 core), matched k',
 'FSLC budget':'relevance-per-cost with redundancy penalty, within budget'},
 'fairness':'k and cost budget are Joint guard outputs determined by training only. Same predictor, folds, normalization and costs as baselines/. Training-only statistics; selections saved before test evaluation. Implementations of cited scoring cores under a unified protocol, not verbatim replicas of published pipelines.','note':'GBFRS uses exact fuzzy lower approximations (strengthening of the ball-level approximation); bandwidths are per-attribute training standard deviations of min-max normalized features.'}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))

def main():
 rows=[]
 for rep in range(1,6):
  for n in [192,384,576]:
   job=out/f'repeat_{rep}_n{n}';job.mkdir(exist_ok=True)
   if (job/'results.json').exists():rows.extend(json.loads((job/'results.json').read_text()));continue
   saved=root/'stability'/job.name;split=json.loads((saved/'split.json').read_text());old=json.loads((saved/'results.json').read_text())
   joint=next(r for r in old if r['method']=='Joint guard');k=joint['attributes'];budget=joint['cost']
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
   dsim=dsim_scores(z,sig,lab)                         # m
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
   # GBFRS forward search on exact fuzzy-rough dependency
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
      # maximize dependency; ties broken by smaller cost then smaller index
      if best is None or g>best_val+1e-15 or (abs(g-best_val)<=1e-15 and (costs[a],a)<(costs[best],best)):
       best,best_val=a,g
     if variant=='budget' and best_val<=gamma_P+1e-15:break
     trace.append([best,best_val-gamma_P]);P.append(best);Rp=np.minimum(Rp,attr_slice(z,sig,best));gamma_P=best_val
    selected[f'GBFRS {variant}']=tuple(sorted(P))
    print('GBFRS',variant,rep,n,'k_sel',len(P),'gamma',round(gamma_P,4),flush=True)
   (job/'selection_before_test.json').write_text(json.dumps({name:list(map(int,P)) for name,P in selected.items()},indent=2))
   # ---------- evaluation (identical predictor and metrics) ----------
   lo=x.min(0);span=x.max(0)-lo;span[span==0]=1.;a=(x-lo)/span;b=(X[test]-lo)/span;truth=L[test]
   def evaluate(P):
    pred=lab[nearest5(cdist(b[:,P],a[:,P],'sqeuclidean'))].mean(1) if P else np.tile(lab.mean(0),(len(test),1));pred=np.maximum(pred,1e-12);pred/=pred.sum(1,keepdims=True);mid=(truth+pred)/2
    return {'inner_brier':risk(P),'test_brier':float(np.mean(np.sum((truth-pred)**2,axis=1))),'test_kl':float(np.mean(np.sum(truth*np.log(np.maximum(truth,1e-300)/pred),axis=1))),'test_tv':float(np.mean(.5*np.abs(truth-pred).sum(1))),'test_js':float(np.mean(.5*np.sum(truth*np.log(np.maximum(truth,1e-300)/mid),axis=1)+.5*np.sum(pred*np.log(pred/mid),axis=1)))}
   local=[]
   for name,P in selected.items():
    if name.endswith(' k'):assert len(P)==k,(name,len(P),k)
    if name.endswith('budget'):assert sum(costs[list(P)])<=budget,(name,sum(costs[list(P)]),budget)
    local.append({'repeat':rep,'train_n':n,'method':name,'attributes':len(P),'cost':int(sum(costs[list(P)])),**evaluate(P)})
   (job/'results.json').write_text(json.dumps(local,indent=2));rows.extend(local)
   with (out/'results.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
   print('DONE',rep,n,[(r['method'],round(r['test_brier'],4)) for r in local],'%.0fs'%(time.perf_counter()-t0),flush=True)
   del blocks,cache,z,sig;gc.collect()
 print('ALL COMPLETE',len(rows),'rows',flush=True)

if __name__ == '__main__':
 main()

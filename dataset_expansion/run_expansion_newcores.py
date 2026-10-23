"""Two additional published-LDL-selector scoring cores (2024-2025) on the expansion datasets.

  NRS   neighborhood rough set LDL dependency (following the neighborhood-rough-set
        label-distribution feature selection core, CPE 2025). For subset P the
        Chebyshev delta-neighborhood relation on min-max-normalized attributes
        (delta = 0.1, fixed) induces a hard neighborhood N_P(x); the label-wise
        lower approximation is (N_P | D_k)(x) = min_{y in N_P(x)} l^k_y, and the
        dependency is gamma(P) = (1/n) sum_i sum_k l^k_i (N_P | D_k)(x_i).
        Forward greedy maximizing gamma, positive-gain/cost stopping under the
        budget variant -- structurally identical to the GBFRS core protocol.
  FMI   fuzzy mutual information core (following the fuzzy-MI label-distribution
        feature selection, Inf. Sci. 2024). Each attribute (min-max normalized)
        and each label dimension are fuzzified into three triangular fuzzy sets;
        relevance(a) = sum_k I_fuzzy(a; Y_k) via membership-mass joint entropies;
        greedy selection maximizes relevance minus the maximum fuzzy MI with the
        already-selected attributes (MIFS-style), as in the FSLC core protocol.

Both cores are evaluated under the frozen expansion protocol: matched-k and
matched-budget to the frozen G+ selections, identical folds, predictor, and
cost vectors; selections frozen before evaluation. These are implementations of
the cited scoring cores under a unified protocol, not verbatim replicas.

Reproduction: python run_expansion_newcores.py   (checkpoints per split; resumable)
"""
from pathlib import Path
import sys,json,csv,time,gc
import numpy as np
from scipy.spatial.distance import cdist

here=Path(__file__).resolve().parent
root=here.parent
sys.path.insert(0,str(root));sys.path.insert(0,str(root/'baselines_published'))
out=here/'published_baselines'   # same output tree; results.json per job keeps cores distinct by method name
from run_published import nearest5

DATASETS=['Yeast_spo','Yeast_alpha','Yeast_cdc','Yeast_diau','Yeast_dtt','Yeast_elu','Yeast_heat','Yeast_spo5','Yeast_spoem',
          'FG-Net','SJAFFE','Music','Painting','SCUT_FBP','fbp5500','emotion6','RAF_ML','Movie','Flickr','Twitter','MOR-PH',
          'Human_Gene','M2B']
DELTA=0.1

def safe(stem):return stem.replace('-','_')

# ---------------- NRS core ----------------
def nrs_lower(B,lab):
 """(N|D_k)(x) = min over y with B[x,y] of lab[y,k]; B is 0/1 float (n x n)."""
 n,q=lab.shape
 BIG=2.0
 # masked values: lab[y,k] where B=1 else BIG; take min over y
 vals=lab[None,:,:]*(B[:,:,None])+(1.-B[:,:,None])*BIG   # n x n x q
 low=vals.min(axis=1)                                     # n x q
 low=np.minimum(low,1.0)                                  # diagonal guarantees lab[x] bounds
 return low

def nrs_dependency(B,lab):
 low=nrs_lower(B,lab)
 return float((lab*low).sum()/len(lab))

# ---------------- FMI core ----------------
def tri3(v):
 """Three triangular fuzzy memberships on [0,1]: low / mid / high. v is (n,)."""
 low=np.clip(1.-2.*v,0.,1.);mid=np.clip(2.-np.abs(2.*v-1.),0.,1.);high=np.clip(2.*v-1.,0.,1.)
 return np.vstack([low,mid,high])   # 3 x n

def fmi_pair(mu,nu):
 """Fuzzy MI between two fuzzified variables given memberships (3 x n) each."""
 n=mu.shape[1]
 pa=mu.sum(1)/n;pb=nu.sum(1)/n
 joint=mu@nu.T/n
 with np.errstate(divide='ignore',invalid='ignore'):
  t=np.where((joint>1e-15)&(pa[:,None]>1e-15)&(pb[None,:]>1e-15),
             joint*np.log(joint/(pa[:,None]*pb[None,:])),0.)
 return float(t.sum())

def fmi_relevance(z,lab):
 """relevance(a) = sum_k fuzzy MI(a; label dim k). z: n x m in [0,1]; lab: n x q in [0,1]."""
 n,m=z.shape;q=lab.shape[1]
 labmem=[tri3(lab[:,k]) for k in range(q)]
 rel=np.zeros(m)
 for a in range(m):
  am=tri3(z[:,a])
  rel[a]=sum(fmi_pair(am,lm) for lm in labmem)
 return rel

def fmi_feature_mi(z):
 """Pairwise fuzzy MI between attributes (m x m)."""
 n,m=z.shape
 mems=[tri3(z[:,a]) for a in range(m)]
 M=np.zeros((m,m))
 for a in range(m):
  for b in range(a+1,m):
   v=fmi_pair(mems[a],mems[b]);M[a,b]=v;M[b,a]=v
 return M

def fmi_selection(rel,fmi,k=None,budget=None,costs=None):
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
  pen=np.maximum(pen,fmi[a])
 return tuple(sorted(P))

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
   # merge new-core rows into the same job results file as the other cores
   existing=[]
   if (job/'results.json').exists():existing=json.loads((job/'results.json').read_text())
   if any(r['method'].startswith('NRS') or r['method'].startswith('FMI') for r in existing):
    rows.extend([r for r in existing if r['method'].startswith(('NRS','FMI'))]);continue
   src=here/ds/f'repeat_{rep}'
   if not (src/'complete.json').exists():print(ds,'repeat',rep,'main run incomplete, skip');continue
   split=json.loads((src/'split.json').read_text())
   gf=json.loads((src/'Guided.json').read_text())
   k=gf['results']['after_attributes'];budget=gf['results']['after_cost']
   costs=np.array(split['costs']);assert len(costs)==m
   dev=np.array(split['dev_indices']);test=np.array(split['test_indices'])
   sp=np.array(split['search_positions']);ap=np.array(split['audit_positions'])
   Xd=X[dev];Ld=L[dev];x=Xd[sp];lab=Ld[sp];auditX=Xd[ap];auditL=Ld[ap];n=len(x)
   t0=time.perf_counter()
   lo=x.min(0);span=x.max(0)-lo;span[span==0]=1.;z=(x-lo)/span
   # ---------- NRS ----------
   # per-attribute delta-neighborhood indicator matrices accumulated by min (Chebyshev)
   B_attrs=[(np.abs(z[:,None,a]-z[None,:,a])<=DELTA).astype(float) for a in range(m)]
   selected={}
   for variant in ['k','budget']:
    P=[];Bp=np.ones((n,n));gamma_P=nrs_dependency(Bp,lab);trace=[]
    while True:
     if variant=='k' and len(P)>=k:break
     cands=[a for a in range(m) if a not in P]
     if variant=='budget':
      cands=[a for a in cands if sum(costs[P])+costs[a]<=budget]
      if not cands:break
     if not cands:break
     best=None;best_val=None
     for a in cands:
      Bn=np.minimum(Bp,B_attrs[a]);g=nrs_dependency(Bn,lab)
      if best is None or g>best_val+1e-15 or (abs(g-best_val)<=1e-15 and (costs[a],a)<(costs[best],best)):
       best,best_val=a,g
     if variant=='budget' and best_val<=gamma_P+1e-15:break
     trace.append([best,best_val-gamma_P]);P.append(best);Bp=np.minimum(Bp,B_attrs[best]);gamma_P=best_val
    selected[f'NRS {variant}']=tuple(sorted(P))
    print(ds,'NRS',variant,rep,'k_sel',len(P),'gamma',round(gamma_P,4),flush=True)
   # ---------- FMI ----------
   rel=fmi_relevance(z,lab)
   fmi=fmi_feature_mi(z)
   selected['FMI k']=fmi_selection(rel,fmi,k=k)
   selected['FMI budget']=fmi_selection(rel,fmi,budget=budget,costs=costs)
   print(ds,'FMI',rep,'done',flush=True)
   # ---------- evaluation ----------
   za=z;qa=(auditX-lo)/span;b=(X[test]-lo)/span;truth=L[test]
   def evaluate(Pp):
    Pp=list(Pp)
    ap_=lab[nearest5(cdist(qa[:,Pp],za[:,Pp],'sqeuclidean'))].mean(1) if Pp else np.tile(lab.mean(0),(len(auditL),1))
    tp=lab[nearest5(cdist(b[:,Pp],za[:,Pp],'sqeuclidean'))].mean(1) if Pp else np.tile(lab.mean(0),(len(truth),1))
    tp=np.maximum(tp,1e-12);tp/=tp.sum(1,keepdims=True);mid=(truth+tp)/2
    return {'audit_brier':float(np.mean(np.sum((auditL-ap_)**2,1))),
     'test_brier':float(np.mean(np.sum((truth-tp)**2,1))),
     'test_kl':float(np.mean(np.sum(truth*np.log(np.maximum(truth,1e-300)/tp),1))),
     'test_tv':float(np.mean(.5*np.abs(truth-tp).sum(1))),
     'test_js':float(np.mean(.5*np.sum(truth*np.log(np.maximum(truth,1e-300)/mid),1)+.5*np.sum(tp*np.log(tp/mid),1)))}
   local=[]
   for name,pp in selected.items():
    if name.endswith(' k'):assert len(pp)==k,(ds,name,len(pp),k)
    if name.endswith('budget'):assert sum(costs[list(pp)])<=budget,(ds,name,sum(costs[list(pp)]),budget)
    local.append({'dataset':stem,'repeat':rep,'method':name,'attributes':len(pp),'cost':int(sum(costs[list(pp)])),**evaluate(pp)})
   merged=existing+local
   (job/'results.json').write_text(json.dumps(merged,indent=2))
   (job/'newcore_selection_before_test.json').write_text(json.dumps({name:list(map(int,pp)) for name,pp in selected.items()},indent=2))
   rows.extend(local);print(ds,'DONE',rep,[(r['method'],r['cost'],round(r['audit_brier'],4)) for r in local],'%.0fs'%(time.perf_counter()-t0),flush=True)
   del B_attrs;gc.collect()
  allrows+=rows
 if allrows:
  with (here/'newcore_results.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
  print('COMPLETE',len(allrows),'rows')

if __name__=='__main__':main()

"""E10 independent construction/query weighting factorial development; original outer tests never scored."""
from pathlib import Path
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,hashlib,time,argparse,warnings
import numpy as np
ROOT=Path(__file__).resolve().parent;os.environ['AI_LDL_E0']=str(ROOT/'deps/e0');os.environ['AI_LDL_CORE']=str(ROOT/'deps/frozen_core');sys.path[:0]=[str(ROOT/'deps'),str(ROOT/'deps/e02')]
from calibrated import CalibratedPrototypeModel
from prototype import PrototypePredictor
from budget import geometry,bisection_path
from strong import cluster,exact_groups,native_ldlkm
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
METHODS=('geometric','kmeanspp','bisecting','paper20');BUDGETS=('K5','25','50','75','100')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(path)
def verify():
 for n,h in json.loads((ROOT/'MANIFEST.json').read_text()).items():
  if sha(ROOT/n)!=h:raise ValueError('Changed source '+n)
 return json.loads((ROOT/'PROTOCOL.json').read_text())
def design(f,fi):
 N=f['N'];nt=min(384,7*N//10);nv=min(512,N-nt);seed=2026111300+100*(fi+1)+1;order=np.random.default_rng(seed).permutation(N);excluded=order[:nv];pool=order[nv:nv+nt][:192]
 costs=np.random.default_rng(202611130000+fi+1).integers(1,11,f['m']);orderp=np.random.default_rng(seed+100000).permutation(f['m']);cum=np.cumsum(costs[orderp]);P=tuple(sorted(map(int,orderp[cum<=40*int(costs.sum())//100])))
 if not P:raise ValueError('Empty P')
 return pool,excluded,{'Full':tuple(range(f['m'])),'cost40':P}
def splits(n,seed):return np.array_split(np.random.default_rng(seed).permutation(n),3)
def paper20(Y,k,seed):
 if k==len(Y):return [(i,) for i in range(k)],0,0
 effective=min(k,len(np.unique(Y,axis=0)))
 with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as ww:
  warnings.simplefilter('always');labels=KMeans(n_clusters=effective,init='random',n_init=1,max_iter=20,tol=1e-4,random_state=seed%(2**32),algorithm='lloyd').fit_predict(Y)
 groups,repair=exact_groups(labels,len(Y),k);return groups,repair,len(ww)
def grid(X,L,Q,masks,seed):
 m=CalibratedPrototypeModel(X,L,zeta=.5);ks={'K5':min(5,m.n),**{str(p):max(1,p*m.n//100) for p in [25,50,75,100]}};pred={};stats=[];alignment={}
 for mi,(mask,P) in enumerate(masks.items()):
  raw={q:PrototypePredictor(m,P,q) for q in ['U','G']}
  raw_prediction={q:raw[q].predict(Q[:,P]) for q in raw}
  for gi,c in enumerate(['U','G']):
   D,Y=geometry(m,P,c);path=bisection_path(D,L,list(ks.values()),0.)
   for method in METHODS:
    for budget,k in ks.items():
     repair=warn=0
     if method=='geometric':groups=path[k]
     elif method=='paper20':groups,repair,warn=paper20(m.Z[:,P] if c=='U' else Y,k,seed+700000+mi)
     else:groups,repair,warn=cluster(Y,k,method,seed+1000*mi+(5 if budget=='K5' else int(budget)))
     protos={q:PrototypePredictor(m,P,q,groups) for q in ['U','G']}
     assert np.array_equal(protos['U'].centers,protos['G'].centers) and np.array_equal(protos['U'].labels,protos['G'].labels)
     nn={q:protos[q].predict(Q[:,P]) for q in protos}
     if k==m.n:
      for q in protos:assert np.array_equal(nn[q],raw_prediction[q])
     if k<=5:assert np.allclose(nn['U'],nn['G'],rtol=0,atol=1e-15)
     for q,proto in protos.items():
      for alpha in [0.,.5]:
       v=(1-alpha)*nn[q]+alpha*m.p
       if not np.isfinite(v).all() or np.any(v<0) or not np.allclose(v.sum(1),1,rtol=0,atol=1e-12):raise ValueError('Bad simplex')
       key=f'{mask}__{c}__{method}__{budget}__{q}__a{alpha}';pred[key]=v
       byte=proto.array_bytes();basebyte=raw[q].array_bytes();slot=(mask,method,budget,alpha)
       if slot in alignment:assert alignment[slot]==(k,byte,basebyte)
       alignment[slot]=(k,byte,basebyte)
       stats.append({'key':key,'K':k,'nfit':m.n,'array_bytes':byte,'singleton_bytes_same_readout':basebyte,'array_fraction':byte/basebyte,'repair_count':repair,'warning_count':warn})
 pred['plain_mean']=np.tile(m.p,(len(Q),1));return pred,stats

def one_case(X,L,f,fi,fold,out):
 pool,exclude,masks=design(f,fi);blocks=splits(len(pool),202611134000+10000*fi+1000);held=pool[blocks[fold]];fit=pool[np.setdiff1d(np.arange(len(pool)),blocks[fold])]
 if set(fit)&set(held) or (set(fit)|set(held))&set(exclude):raise ValueError('Leakage')
 pred,stats=grid(X[fit],L[fit],X[held],masks,202611140800+10000*fi+100*fold+900000)
 out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'predictions_PRIVATE.npz',**pred)
 save(out/'record_PRIVATE.json',{'family':f['family'],'fold':fold,'fit_indices':fit.tolist(),'validation_indices':held.tolist(),'excluded_original_test_indices':exclude.tolist(),'predictions_sha256':sha(out/'predictions_PRIVATE.npz'),'protocol_sha256':sha(ROOT/'PROTOCOL.json'),'statistics':stats})
def run(args):
 p=verify();out=args.output;out.mkdir(parents=True,exist_ok=True)
 if (out/'ALL_PREDICTIONS_FROZEN.json').exists():raise ValueError('Already frozen')
 records={};start=time.perf_counter()
 for fi,f in enumerate(p['families']):
  arrays=[]
  for t in ['X','L']:
   path=args.data/f"{f['family']}_{t}.npy"
   if sha(path)!=f[t+'_sha256']:raise ValueError('Input hash')
   arrays.append(np.load(path,mmap_mode='r'))
  X,L=arrays
  if X.shape!=(f['N'],f['m']) or L.shape!=(f['N'],f['q']):raise ValueError('Shape')
  for fold in range(3):
   case=out/f['family']/f'fold_{fold}';rec=case/'record_PRIVATE.json'
   if rec.exists():
    old=json.loads(rec.read_text());assert old['protocol_sha256']==sha(ROOT/'PROTOCOL.json') and old['predictions_sha256']==sha(case/'predictions_PRIVATE.npz')
   else:one_case(X,L,f,fi,fold,case)
   records[str(rec.relative_to(out))]=sha(rec);print('FROZEN',f['family'],fold,flush=True)
 save(out/'ALL_PREDICTIONS_FROZEN.json',{'records':records,'protocol_sha256':sha(ROOT/'PROTOCOL.json'),'source_manifest_sha256':sha(ROOT/'MANIFEST.json'),'wall_seconds':time.perf_counter()-start,'original_outer_test_scored':False});print('ALL48 FROZEN',flush=True)
def score(args):
 p=verify();out=args.output;fr=json.loads((out/'ALL_PREDICTIONS_FROZEN.json').read_text());assert len(fr['records'])==48 and fr['protocol_sha256']==sha(ROOT/'PROTOCOL.json') and fr['source_manifest_sha256']==sha(ROOT/'MANIFEST.json');rows=[]
 for name,h in fr['records'].items():
  path=out/name;assert sha(path)==h;rec=json.loads(path.read_text());f=next(f for f in p['families'] if f['family']==rec['family']);yp=args.data/f"{rec['family']}_L.npy";assert sha(yp)==f['L_sha256'] and sha(path.parent/'predictions_PRIVATE.npz')==rec['predictions_sha256'];y=np.load(yp,mmap_mode='r')[rec['validation_indices']];stats={s['key']:s for s in rec['statistics']}
  with np.load(path.parent/'predictions_PRIVATE.npz') as arr:
   for key in arr.files:rows.append({'family':rec['family'],'fold':rec['fold'],'key':key,'n_validation':len(y),'brier':float(np.square(arr[key]-y).sum(1).mean()),**({k:v for k,v in stats[key].items() if k!='key'} if key!='plain_mean' else {})})
 summary=[]
 for f,key in sorted({(r['family'],r['key']) for r in rows}):
  rr=[r for r in rows if r['family']==f and r['key']==key];n=sum(r['n_validation'] for r in rr);item={'family':f,'key':key,'brier':sum(r['brier']*r['n_validation'] for r in rr)/n}
  if key!='plain_mean':
   for field in ['array_bytes','array_fraction','K']:item[field]=sum(r[field]*r['n_validation'] for r in rr)/n
  summary.append(item)
 save(out/'RESULTS_PUBLIC.json',{'scope':p['scope'],'rows':rows,'family_summary':summary,'protocol_sha256':sha(ROOT/'PROTOCOL.json'),'original_outer_test_scored':False});print('SCORED',len(rows),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['run','score']);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();globals()[args.action](args)

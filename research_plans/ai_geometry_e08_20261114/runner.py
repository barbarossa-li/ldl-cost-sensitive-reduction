"""E0.8 strong-baseline training-pool development. Does not score original outer tests."""
from pathlib import Path
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import sys,json,hashlib,argparse,time
import numpy as np
ROOT=Path(__file__).resolve().parent
os.environ['AI_LDL_E0']=str(ROOT/'deps/e0');os.environ['AI_LDL_CORE']=str(ROOT/'deps/frozen_core')
sys.path[:0]=[str(ROOT/'deps'),str(ROOT/'deps/e02')]
from calibrated import CalibratedPrototypeModel
from prototype import PrototypePredictor
from budget import geometry,bisection_path,lloyd
from strong import cluster,native_ldlkm
PCTS=(25,50,75,100);METHODS=('geometric','lloyd','kmeanspp','bisecting');GEOMS=('U','G')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
def verify():
 manifest=json.loads((ROOT/'MANIFEST.json').read_text())
 for name,h in manifest.items():
  if sha(ROOT/name)!=h:raise ValueError('Code/protocol changed: '+name)
 return json.loads((ROOT/'PROTOCOL.json').read_text())
def select_budget(losses,cap):
 candidates=[p for p in PCTS if p<=cap]
 if not candidates or any(not np.isfinite(losses[p]) for p in candidates):raise ValueError('Missing/nonfinite CV loss')
 best=min(losses[p] for p in candidates)
 return min(p for p in candidates if losses[p]<=best+1e-12)
def partitions(n,seed):return np.array_split(np.random.default_rng(seed).permutation(n),3)
def design(f,fi):
 n=f['N'];nt=min(384,7*n//10);nv=min(512,n-nt);seed=2026111300+100*(fi+1)+1
 order=np.random.default_rng(seed).permutation(n);excluded=order[:nv];pool=order[nv:nv+nt][:192]
 costs=np.random.default_rng(202611130000+fi+1).integers(1,11,f['m'])
 orderp=np.random.default_rng(seed+100000).permutation(f['m']);cum=np.cumsum(costs[orderp]);P=tuple(sorted(map(int,orderp[cum<=40*int(costs.sum())//100])))
 if not P:raise ValueError('Empty fixed cost40 mask')
 return pool,excluded,{'Full':tuple(range(f['m'])),'cost40':P}
def predict_grid(X,L,Q,masks,seed=0,include_native=True):
 model=CalibratedPrototypeModel(X,L,zeta=.5);n=len(X);ks=[max(1,p*n//100) for p in PCTS];values={};stats=[]
 for mi,(mask,P) in enumerate(masks.items()):
  for gi,g in enumerate(GEOMS):
   D,Y=geometry(model,P,g);path=bisection_path(D,L,ks,0.);raw=PrototypePredictor(model,P,g);base=raw.predict(Q[:,P])
   for method in METHODS:
    for pct,k in zip(PCTS,ks):
     repairs=warn=0
     if method=='geometric':groups=path[k]
     elif method=='lloyd':groups=lloyd(D,k)
     else:groups,repairs,warn=cluster(Y,k,method,seed+1000*mi+100*gi+pct)
     pr=PrototypePredictor(model,P,g,groups);v=pr.predict(Q[:,P])
     if k==n and not np.array_equal(v,base):raise ValueError('Singleton endpoint changed')
     for alpha in [0.,.5]:
      key=f'{mask}__{g}__{method}__{pct}__a{alpha}';values[key]=(1-alpha)*v+alpha*model.p
      stats.append({'key':key,'K':k,'nfit':n,'array_bytes':pr.array_bytes(),'raw_array_bytes':raw.array_bytes(),'array_fraction':pr.array_bytes()/raw.array_bytes(),'repair_count':repairs,'warning_count':warn})
  if include_native:
   native,st=native_ldlkm(model,Q,P,seed+700000+mi)
   values[f'{mask}__native_LDLKM']=native;values[f'{mask}__native_LDLKM_half']=.5*native+.5*model.p
   stats.append({'key':f'{mask}__native_LDLKM','nfit':n,**st})
 values['plain_mean']=np.tile(model.p,(len(Q),1))
 for v in values.values():
  if not np.isfinite(v).all() or np.any(v<0) or not np.allclose(v.sum(1),1,rtol=0,atol=1e-12):raise ValueError('Invalid prediction')
 return values,stats

def one_case(X,L,f,fi,fold,out):
 pool,excluded,masks=design(f,fi);outer=partitions(len(pool),202611134000+10000*fi+1000)
 test=pool[outer[fold]];fit=pool[np.setdiff1d(np.arange(len(pool)),outer[fold])]
 if (set(fit)|set(test))&set(excluded) or set(fit)&set(test):raise ValueError('Split leakage')
 inner=partitions(len(fit),202611140700+10000*fi+100*fold);sums={};counts={};splits=[]
 for j,valpos in enumerate(inner):
  trpos=np.setdiff1d(np.arange(len(fit)),valpos);tr=fit[trpos];va=fit[valpos]
  if set(tr)&set(va) or (set(tr)|set(va))&set(test):raise ValueError('Nested leakage')
  pred,_=predict_grid(X[tr],L[tr],X[va],masks,seed=202611140800+10000*fi+100*fold+10*j,include_native=False)
  for key,v in pred.items():
   loss=float(np.square(v-L[va]).sum());sums[key]=sums.get(key,0.)+loss;counts[key]=counts.get(key,0)+len(va)
  splits.append({'inner':j,'fit_indices':tr.tolist(),'validation_indices':va.tolist()})
 losses={k:sums[k]/counts[k] for k in sums};selection={}
 for mask in masks:
  for g in GEOMS:
   for method in METHODS:
    for alpha in [0.,.5]:
     loss={p:losses[f'{mask}__{g}__{method}__{p}__a{alpha}'] for p in PCTS}
     selection[f'{mask}__{g}__{method}__cap100__a{alpha}']=select_budget(loss,100)
 pred,stats=predict_grid(X[fit],L[fit],X[test],masks,seed=202611140800+10000*fi+100*fold+900000)
 for key,p in selection.items():pred[key]=pred[key.replace('cap100',str(p))]
 t0=time.time();out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'predictions_PRIVATE.npz',**pred)
 record={'family':f['family'],'fold':fold,'fit_indices':fit.tolist(),'validation_indices':test.tolist(),'excluded_original_test_indices':excluded.tolist(),'inner_splits':splits,'inner_risks':losses,'selected_percent':selection,'statistics':stats,'prediction_sha256':sha(out/'predictions_PRIVATE.npz'),'protocol_sha256':sha(ROOT/'PROTOCOL.json'),'note':'Consulted training-pool nested development; not independent confirmation','saved_at_unix':t0}
 save(out/'record_PRIVATE.json',record)

def run(args):
 p=verify();root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
 if (root/'ALL_PREDICTIONS_FROZEN.json').exists():raise ValueError('Already frozen; use score or a new directory')
 records={};start=time.perf_counter()
 for fi,f in enumerate(p['families']):
  arrays=[]
  for a in ['X','L']:
   file=args.data/f"{f['family']}_{a}.npy"
   if sha(file)!=f[a+'_sha256']:raise ValueError('Input mismatch '+str(file))
   arrays.append(np.load(file,mmap_mode='r'))
  X,L=arrays
  if X.shape!=(f['N'],f['m']) or L.shape!=(f['N'],f['q']):raise ValueError('Input shape')
  for fold in range(3):
   out=root/f['family']/f'fold_{fold}';record=out/'record_PRIVATE.json'
   if record.exists():
    r=json.loads(record.read_text())
    if r['protocol_sha256']!=sha(ROOT/'PROTOCOL.json') or r['prediction_sha256']!=sha(out/'predictions_PRIVATE.npz'):raise ValueError('Invalid resume')
   else:one_case(X,L,f,fi,fold,out)
   records[str(record.relative_to(root))]=sha(record)
   print('FROZEN',f['family'],fold,flush=True)
 save(root/'ALL_PREDICTIONS_FROZEN.json',{'records':records,'protocol_sha256':sha(ROOT/'PROTOCOL.json'),'code_manifest_sha256':sha(ROOT/'MANIFEST.json'),'wall_seconds':time.perf_counter()-start,'original_outer_test_scored':False})
 print('ALL48 FROZEN',flush=True)

def score(args):
 p=verify();root=args.output;fr=json.loads((root/'ALL_PREDICTIONS_FROZEN.json').read_text());rows=[]
 if len(fr['records'])!=48 or fr['protocol_sha256']!=sha(ROOT/'PROTOCOL.json') or fr['code_manifest_sha256']!=sha(ROOT/'MANIFEST.json'):raise ValueError('Incomplete/changed freeze')
 for name,h in fr['records'].items():
  file=root/name
  if sha(file)!=h:raise ValueError('Changed record')
  r=json.loads(file.read_text());f=next(f for f in p['families'] if f['family']==r['family']);yp=args.data/f"{r['family']}_L.npy"
  if sha(yp)!=f['L_sha256'] or sha(file.parent/'predictions_PRIVATE.npz')!=r['prediction_sha256']:raise ValueError('Changed scoring input')
  y=np.load(yp,mmap_mode='r')[r['validation_indices']];stats={a['key']:a for a in r['statistics']}
  with np.load(file.parent/'predictions_PRIVATE.npz') as bundle:
   for key in bundle.files:
    v=bundle[key];loss=float(np.square(v-y).sum(1).mean());parts=key.split('__');entry={'family':r['family'],'fold':r['fold'],'key':key,'n_validation':len(y),'brier':loss}
    if len(parts)==5:
     pct=r['selected_percent'][key] if parts[3].startswith('cap') else int(parts[3]);st=stats['__'.join(parts[:3])+f'__{pct}__'+parts[4]];entry.update(array_fraction=st['array_fraction'],selected_percent=pct,K=st['K'],nfit=st['nfit'],repair_count=st['repair_count'],warning_count=st['warning_count'])
    elif len(parts)==2:
     st=stats[parts[0]+'__native_LDLKM'];entry.update(K=st['K'],nfit=st['nfit'],native_array_bytes=st['shrinkage_array_bytes'] if parts[1].endswith('_half') else st['native_array_bytes'])
    rows.append(entry)
 summary=[]
 for family,key in sorted({(r['family'],r['key']) for r in rows}):
  rr=[r for r in rows if r['family']==family and r['key']==key];n=sum(r['n_validation'] for r in rr);item={'family':family,'key':key,'brier':sum(r['brier']*r['n_validation'] for r in rr)/n}
  if 'array_fraction' in rr[0]:item.update(array_fraction=sum(r['array_fraction']*r['n_validation'] for r in rr)/n,selected_percent=sum(r['selected_percent']*r['n_validation'] for r in rr)/n)
  if 'native_array_bytes' in rr[0]:item['native_array_bytes']=sum(r['native_array_bytes']*r['n_validation'] for r in rr)/n
  summary.append(item)
 save(root/'RESULTS_PUBLIC.json',{'scope':p['scope'],'rows':rows,'family_summary':summary,'original_outer_test_scored':False,'protocol_sha256':sha(ROOT/'PROTOCOL.json')})
 print('SCORED',len(rows),'rows',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['run','score']);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();globals()[args.action](args)

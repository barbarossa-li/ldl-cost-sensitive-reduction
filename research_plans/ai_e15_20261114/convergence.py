"""Post-outcome numerical SA-BFGS convergence check, with explicit inference state."""
from pathlib import Path
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,argparse,hashlib,time,concurrent.futures,multiprocessing
import numpy as np
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def worker(job):
 source,base,out,known,novel,stage,name=job;sys.path.insert(0,str(Path(source)/'vendor'));from pyldl.algorithms import SA_BFGS;from pyldl.algorithms.utils import softmax;import pyldl.algorithms._specialized_algorithms as spec
 parent=json.loads((Path(source)/'PROTOCOL.json').read_text());oldfile=Path(base)/stage/name;old=json.loads(oldfile.read_text());folder=Path(known if stage=='e11' else novel);family=old['family'];f=next(f for f in parent['stages']['e11']['families'] if f['family']==family) if stage=='e11' else parent['stages']['e12']['family'];xp=folder/(family+'_X.npy');yp=folder/(family+'_L.npy');assert sha(xp)==f['X_sha256'] and sha(yp)==f['L_sha256'];X=np.load(xp,mmap_mode='r');L=np.load(yp,mmap_mode='r');fit=np.array(old['fit_indices']);held=np.array(old['validation_indices']);fitX=X[fit];fitL=L[fit];Q=X[held];prior=fitL.mean(0)
 if stage=='e11':
  fi=next(i for i,f in enumerate(parent['stages']['e11']['families']) if f['family']==family);N=f['N'];nt=min(384,7*N//10);seed=2026111300+100*(fi+1)+1;costs=np.random.default_rng(202611130000+fi+1).integers(1,11,f['m']);order=np.random.default_rng(seed+100000).permutation(f['m'])
 else:
  fi=16;costs=np.random.default_rng(202611150013).integers(1,11,f['m']);order=np.random.default_rng(202611150014).permutation(f['m'])
 P=tuple(sorted(map(int,order[np.cumsum(costs[order])<=40*int(costs.sum())//100])));masks={'Full':tuple(range(f['m'])),'cost40':P};case=Path(out)/stage/family/f"fold_{old['fold']}";record=case/'record_PRIVATE.json'
 if record.exists():
  rec=json.loads(record.read_text());assert rec['predictions_sha256']==sha(case/'predictions_PRIVATE.npz') and rec['protocol_sha256']==sha(ROOT/'PROTOCOL.json');return str(record.relative_to(out)),sha(record)
 values={};states={};details={};start=time.perf_counter()
 for mi,(mask,P) in enumerate(masks.items()):
  lo=fitX[:,P].min(0);span=fitX[:,P].max(0)-lo;A=np.divide(fitX[:,P]-lo,span,out=np.zeros_like(fitX[:,P]),where=span>0);B=np.divide(Q[:,P]-lo,span,out=np.zeros_like(Q[:,P]),where=span>0);seed=int((202611150000+1000*fi+100*old['fold']+10*mi+2)%(2**32));assert seed==old['model_diagnostics'][mask+'__SABFGS']['seed'];original=spec.minimize;info=[]
  def observed(*args,**kwargs):
   rr=original(*args,**kwargs);info.append({'success':bool(rr.success),'status':int(rr.status),'iterations':int(rr.nit),'message':str(rr.message),'objective':float(rr.fun),'gradient_max':float(abs(rr.jac).max())});return rr
  spec.minimize=observed
  try:model=SA_BFGS(random_state=seed);model.fit(A,fitL,max_iterations=20000,convergence_criterion=1e-7)
  finally:spec.minimize=original
  W=model.get_weights();v=model.predict(B);plain=softmax(B@W);assert np.array_equal(v,plain);half=.5*v+.5*prior
  for label,pred in [('native',v),('half_prior',half)]:assert np.isfinite(pred).all() and (pred>=0).all() and abs(pred.sum(1)-1).max()<1e-12;values[mask+'__SABFGS20000__'+label]=pred
  state={'W':W,'P':np.array(P,dtype=np.int64),'lo':lo,'span':span};rawbytes=sum(z.nbytes for z in state.values());states.update({mask+'__'+k:z for k,z in state.items()});states[mask+'__prior']=prior;details[mask]={'seed':seed,'optimizer':info[0],'old_optimizer':old['model_diagnostics'][mask+'__SABFGS']['optimizer'][0],'native_array_bytes':rawbytes,'half_prior_array_bytes':rawbytes+prior.nbytes,'standalone_prediction_bit_exact':True}
 case.mkdir(parents=True,exist_ok=True);np.savez_compressed(case/'predictions_PRIVATE.npz',**values);np.savez_compressed(case/'inference_state_PRIVATE.npz',**states);save(record,{k:old[k] for k in ['family','fold','fit_indices','validation_indices','excluded_original_test_indices']}|{'stage':stage,'diagnostics':details,'predictions_sha256':sha(case/'predictions_PRIVATE.npz'),'inference_state_sha256':sha(case/'inference_state_PRIVATE.npz'),'protocol_sha256':sha(ROOT/'PROTOCOL.json'),'source_manifest_sha256':sha(ROOT/'MANIFEST.json'),'wall_seconds':time.perf_counter()-start});return str(record.relative_to(out)),sha(record)

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['run','check']);p.add_argument('--external-source',type=Path,required=True);p.add_argument('--external-run',type=Path);p.add_argument('--known-data',type=Path);p.add_argument('--novel-data',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();sys.path.insert(0,str(a.external_source/'vendor'))
 if a.action=='check':
  from pyldl.algorithms import SA_BFGS
  from pyldl.algorithms.utils import softmax
  rng=np.random.default_rng(91);X=rng.uniform(size=(48,7));L=rng.dirichlet(np.ones(4),48);Q=rng.uniform(size=(9,7));m=SA_BFGS(random_state=91);m.fit(X,L,max_iterations=20000,convergence_criterion=1e-7);W=m.get_weights();assert np.array_equal(m.predict(Q),softmax(Q@W));print(json.dumps({'passed':True,'scope':'Synthetic long-budget SA-BFGS fit and independent weight-only inference check'}));return
 for n,h in json.loads((ROOT/'MANIFEST.json').read_text()).items():assert sha(ROOT/n)==h
 for n,h in json.loads((a.external_source/'MANIFEST.json').read_text()).items():assert sha(a.external_source/n)==h
 plan=json.loads((ROOT/'PROTOCOL.json').read_text());assert sha(a.external_source/'PROTOCOL.json')==plan['parent_protocol_sha256'] and sha(a.external_source/'MANIFEST.json')==plan['parent_manifest_sha256'];assert not (a.output/'ALL_PREDICTIONS_FROZEN.json').exists();jobs=[]
 for stage,n in [('e11',48),('e12',5)]:
  fr=json.loads((a.external_run/stage/'ALL_PREDICTIONS_FROZEN.json').read_text());assert len(fr['records'])==n
  for name,h in fr['records'].items():assert sha(a.external_run/stage/name)==h;jobs.append((str(a.external_source),str(a.external_run),str(a.output),str(a.known_data),str(a.novel_data),stage,name))
 records={};start=time.perf_counter()
 with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
  for name,h in pool.map(worker,jobs):records[name]=h;print('FROZEN',name,flush=True)
 assert len(records)==53;save(a.output/'ALL_PREDICTIONS_FROZEN.json',{'records':records,'protocol_sha256':sha(ROOT/'PROTOCOL.json'),'source_manifest_sha256':sha(ROOT/'MANIFEST.json'),'wall_seconds':time.perf_counter()-start})
if __name__=='__main__':main()

"""Label-associated weights versus permutation controls, with unchanged targets."""
from pathlib import Path
import sys,argparse,json,copy,time
import numpy as np
p=argparse.ArgumentParser();p.add_argument('action',choices=['check','run','score']);p.add_argument('--external-source',type=Path,required=True);p.add_argument('--known-data',type=Path);p.add_argument('--novel-data',type=Path);p.add_argument('--external-run',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();sys.path.insert(0,str(a.external_source));import runner as r
ROOT=Path(__file__).resolve().parent

def replace_weights(base,w):
 m=copy.copy(base);m.G=np.broadcast_to(w,base.G.shape).copy();m.G.setflags(write=False);m.scale_cache={};m.cache={};assert m.L is base.L and m.p is base.p;return m

def prediction(m,Q,masks):
 out={};st={};k=m.n//2
 for mask,P in masks.items():
  D,_=r.geometry(m,P,'G');groups=r.bisection_path(D,m.L,[k],0.)[k];proto=r.PrototypePredictor(m,P,'G',groups);v=.5*proto.predict(Q[:,P])+.5*m.p;r.valid(v,len(Q),m.L.shape[1],1e-12);out[mask]=v;st[mask]={'K':k,'array_bytes':proto.array_bytes()}
 return out,st

def grid(X,L,Q,masks,fi,fold,repeats=5):
 base=r.CalibratedPrototypeModel(X,L,zeta=.5);original=base.G[0].copy();out={};stats={};permutations={}
 def add(name,model):
  v,s=prediction(model,Q,masks)
  for mask in masks:out[mask+'__'+name]=v[mask];stats[mask+'__'+name]=s[mask]
 add('GG50',base);add('UU50',replace_weights(base,np.ones(base.m)))
 for j in range(repeats):
  seed=202611150100+1000*fi+100*fold+j;pi=np.random.default_rng(seed).permutation(len(L));null=r.CalibratedPrototypeModel(X,L[pi],zeta=.5);assert np.allclose(base.p,null.p,atol=1e-14,rtol=0);axis=np.random.default_rng(seed+500000).permutation(base.m);permutations[str(j)]={'label_permutation':pi.tolist(),'axis_permutation':axis.tolist(),'seed':seed};add('LP'+str(j),replace_weights(base,null.G[0]));add('AX'+str(j),replace_weights(base,original[axis]))
 for mask in masks:assert len({(s['K'],s['array_bytes']) for k,s in stats.items() if k.startswith(mask+'__')})==1
 return out,stats,permutations

def verify():
 for n,h in json.loads((ROOT/'MANIFEST.json').read_text()).items():assert r.sha(ROOT/n)==h
 parent=r.verify();plan=json.loads((ROOT/'PROTOCOL.json').read_text());assert r.sha(a.external_source/'PROTOCOL.json')==plan['parent_protocol_sha256'] and r.sha(a.external_source/'MANIFEST.json')==plan['parent_manifest_sha256'];return parent,plan

def run():
 parent,plan=verify();out=a.output;out.mkdir(parents=True,exist_ok=True);assert not (out/'ALL_PREDICTIONS_FROZEN.json').exists();jobs=[];records={};start=time.perf_counter()
 for fi,f in enumerate(parent['stages']['e11']['families']):
  pool,excluded,masks=r.known_design(f,fi);parts=r.blocks(len(pool),202611134000+10000*fi+1000)
  for fold in range(3):jobs.append(('e11',f,fi,fold,pool[np.setdiff1d(np.arange(len(pool)),parts[fold])],pool[parts[fold]],excluded,masks))
 f=parent['stages']['e12']['family'];X,L=r.checked_data(a.novel_data,f);cases,masks=r.novel_design(X)
 for fold,(fit,held) in enumerate(cases):jobs.append(('e12',f,16,fold,fit,held,np.array([],int),masks))
 current=None
 for stage,f,fi,fold,fit,held,excluded,masks in jobs:
  if current!=f['family']:X,L=r.checked_data(a.known_data if stage=='e11' else a.novel_data,f);current=f['family']
  assert not set(fit)&set(held) and not (set(fit)|set(held))&set(excluded);case=out/stage/f['family']/f'fold_{fold}';record=case/'record_PRIVATE.json'
  if not record.exists():
   values,stats,perms=grid(X[fit],L[fit],X[held],masks,fi,fold,plan['repeats']);assert len(values)==24;case.mkdir(parents=True,exist_ok=True);np.savez_compressed(case/'predictions_PRIVATE.npz',**values);r.save(record,{'stage':stage,'family':f['family'],'fold':fold,'fit_indices':fit.tolist(),'validation_indices':held.tolist(),'excluded_original_test_indices':excluded.tolist(),'statistics':stats,'permutations':perms,'predictions_sha256':r.sha(case/'predictions_PRIVATE.npz'),'protocol_sha256':r.sha(ROOT/'PROTOCOL.json'),'source_manifest_sha256':r.sha(ROOT/'MANIFEST.json')})
  rec=json.loads(record.read_text());assert rec['protocol_sha256']==r.sha(ROOT/'PROTOCOL.json') and rec['source_manifest_sha256']==r.sha(ROOT/'MANIFEST.json') and rec['predictions_sha256']==r.sha(case/'predictions_PRIVATE.npz');records[str(record.relative_to(out))]=r.sha(record);print('FROZEN',stage,f['family'],fold,flush=True)
 assert len(records)==53;r.save(out/'ALL_PREDICTIONS_FROZEN.json',{'records':records,'protocol_sha256':r.sha(ROOT/'PROTOCOL.json'),'source_manifest_sha256':r.sha(ROOT/'MANIFEST.json'),'wall_seconds':time.perf_counter()-start})

def score():
 parent,plan=verify();out=a.output;fr=json.loads((out/'ALL_PREDICTIONS_FROZEN.json').read_text());assert len(fr['records'])==53
 for stage,n in [('e11',48),('e12',5)]:assert len(json.loads((a.external_run/stage/'ALL_PREDICTIONS_FROZEN.json').read_text())['records'])==n
 assert not (out/'RESULTS_PUBLIC.json').exists();rows=[]
 for name,h in fr['records'].items():
  file=out/name;assert r.sha(file)==h;rec=json.loads(file.read_text());stage=rec['stage'];folder=a.known_data if stage=='e11' else a.novel_data;f=next(f for f in parent['stages']['e11']['families'] if f['family']==rec['family']) if stage=='e11' else parent['stages']['e12']['family'];path=folder/f"{rec['family']}_L.npy";assert r.sha(path)==f['L_sha256'];y=np.load(path,mmap_mode='r')[rec['validation_indices']];assert r.sha(file.parent/'predictions_PRIVATE.npz')==rec['predictions_sha256']
  with np.load(file.parent/'predictions_PRIVATE.npz') as values:
   for key in values.files:rows.append({'stage':stage,'family':rec['family'],'fold':rec['fold'],'key':key,'n_validation':len(y),**r.metrics(y,values[key]),**rec['statistics'][key]})
 r.save(out/'RESULTS_PUBLIC.json',{'scope':plan['scope'],'rows':rows,'repeats':5,'seed_aggregation':'Average individual predictor risks; never risk of averaged predictions'});print('SCORED',len(rows),flush=True)

def check():
 rng=np.random.default_rng(93);X=rng.normal(size=(24,5));L=rng.dirichlet(np.ones(3),24);Q=rng.normal(size=(7,5));masks={'Full':tuple(range(5)),'cost40':(0,2)};v,st,_=grid(X,L,Q,masks,0,0,1);assert len(v)==8;reference=r.own(X,L,Q,masks)[0]
 for mask in masks:
  for g in ['UU','GG']:assert np.array_equal(v[mask+'__'+g+'50'],reference[mask+'__'+g+'50'])
 vv,_,_=grid(X,L,Q,masks,0,0,1)
 for key in v:assert np.array_equal(v[key],vv[key])
 print(json.dumps({'passed':True,'checks':['GG50 andUU50 exactly reproduce parent references','All controls matchK and actual array bytes','Weight-only replacement preserves original training labels andprior byobjectidentity','Permutation seeds reproduce all arrays'],'scope':'Synthetic checks only'},indent=2))
globals()[a.action]()

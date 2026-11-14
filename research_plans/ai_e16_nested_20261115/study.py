"""Restricted-grid fair training-only selection on53 already-consulted cases."""
from pathlib import Path
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,argparse,time,concurrent.futures,multiprocessing,traceback
import numpy as np
ROOT=Path(__file__).resolve().parent

def setup(source):
 os.environ['LDL_PARENT_SOURCE']=str(source);sys.path.insert(0,str(ROOT));import learners as l;return l

def resource(method,param,n,d,q):
 if param.get('kind')=='Mean':return 8*q
 alpha=param['alpha']
 if method in ['G','U']:return 8*(max(1,param['budget']*n//100)*(d+q+1)+4*d+q)
 if method=='AA':return 8*(n*(d+q)+3*d+(q if alpha else 0))
 if method=='Ridge':return 8*(d*q+3*d+q+(q if alpha else 0))
 return 8*n*(d+q)+4*n*q+4*n+24*d+(8*q if alpha else 0)

def choose(rows,method,n,d,q,one_se=False):
 available=[r for r in rows if r['available']];best=min(r['risk'] for r in available);top=min([r for r in available if r['risk']<=best+1e-12],key=lambda r:(resource(method,r['params'],n,d,q),r['id']));threshold=top['risk']+(top['se'] if one_se else 1e-12);eligible=[r for r in available if r['risk']<=threshold]
 if one_se:return min(eligible,key=lambda r:(resource(method,r['params'],n,d,q),r['risk'],r['id']))
 return top

def select_and_fit(X,L,Q,masks,fi,fold,seed,epochs=None):
 l=setup(os.environ['LDL_PARENT_SOURCE']);r=l.r;epochs=tuple(epochs or l.EPOCHS);n,q=L.shape;parts=l.group_folds(X,seed);methods=['G','U','AA','Ridge','RKNN'];cv={};parameters={};diag=[];mean=np.zeros((n,q))
 for mask in masks:
  for method in methods:cv[mask,method]={};parameters[mask,method]={}
 for inner,held in enumerate(parts):
  fit=np.setdiff1d(np.arange(n),held);A=X[fit];D=L[fit];B=X[held];p=D.mean(0);mean[held]=p;model=r.CalibratedPrototypeModel(A,D,zeta=.5)
  for mi,(mask,P) in enumerate(masks.items()):
   def add(method,key,v,param):
    r.valid(v,len(held),q);target=cv[mask,method].setdefault(key,np.full((n,q),np.nan));target[held]=v;parameters[mask,method][key]=param
   for g in ['G','U']:
    vv,pp=l.proto_candidates(model,B,P,g)
    for key,v in vv.items():add(g,key,v,pp[key])
   for k in l.AA_KS:
    state=l.aa_state(A,D,P,min(k,len(A)));v=l.predict(state,B)
    for alpha in [0.,.5]:add('AA',f'k{k}_a{alpha}',(1-alpha)*v+alpha*p,{'k':k,'alpha':alpha})
   for lam in l.RIDGE_ALPHAS:
    state=l.ridge_state(A,D,P,lam);v=l.predict(state,B)
    for alpha in [0.,.5]:add('Ridge',f'lam{lam}_a{alpha}',(1-alpha)*v+alpha*p,{'lambda':lam,'alpha':alpha})
   for ai,anchor in enumerate(l.ANCHORS):
    fitseed=202611160000+10000*fi+1000*fold+100*inner+10*mi+ai;vv,states,detail=l.rknn_checkpoints(A,D,B,P,anchor,fitseed,epochs);diag.append({'inner':inner,'mask':mask,'anchor':ai,**detail})
    for ep in epochs:
     for alpha in [0.,.5]:
      key=f'c{ai}_e{ep}_a{alpha}';param={'anchor':ai,'epoch':ep,'alpha':alpha};parameters[mask,'RKNN'][key]=param
      if ep in vv:add('RKNN',key,(1-alpha)*vv[ep]+alpha*p,param)
      else:cv[mask,'RKNN'].setdefault(key,np.full((n,q),np.nan))
 for mask in masks:
  for method in methods:cv[mask,method]['mean']=mean;parameters[mask,method]['mean']={'kind':'Mean'}
 rows={};choices={};publiccv=[]
 for (mask,method),values in cv.items():
  rr=[]
  for key,v in values.items():
   available=bool(np.isfinite(v).all());loss=np.square(v-L).sum(1) if available else None;foldrisk=[float(loss[h].mean()) for h in parts] if available else None;row={'id':key,'params':parameters[mask,method][key],'available':available,'risk':float(loss.mean()) if available else None,'fold_risks':foldrisk,'se':float(np.std(foldrisk,ddof=1)/np.sqrt(3)) if available else None};rr.append(row);publiccv.append({'mask':mask,'method':method,**row})
  rows[mask,method]=rr;choices[mask,method+'_min']=choose(rr,method,n,len(masks[mask]),q)
  if method=='G':choices[mask,'G_oneSE']=choose(rr,method,n,len(masks[mask]),q,True)
 for mask in masks:choices[mask,'G_atU']={**choices[mask,'U_min'],'selected_on':'U'}
 output={};states={};failures={};outerdiag=[];base=r.CalibratedPrototypeModel(X,L,zeta=.5);p=base.p
 for mi,(mask,P) in enumerate(masks.items()):
  fitted={}
  for policy in ['G_min','U_min','G_atU','G_oneSE','AA_min','Ridge_min','RKNN_min']:
   choice=choices[mask,policy];param=choice['params'];key=mask+'__'+policy
   try:
    if param.get('kind')=='Mean':state={'kind':'Mean','p':p.copy()}
    else:
     if policy.startswith(('G_','U_')):state=l.prototype_state(base,P,'U' if policy=='U_min' else 'G',param['budget'],param['k'])
     elif policy=='AA_min':state=l.aa_state(X,L,P,min(param['k'],n))
     elif policy=='Ridge_min':state=l.ridge_state(X,L,P,param['lambda'])
     else:
      ai=param['anchor'];vv,ss,details=l.rknn_checkpoints(X,L,Q,P,l.ANCHORS[ai],202611169000+10000*fi+1000*fold+10*mi+ai,(param['epoch'],));outerdiag.append({'mask':mask,'policy':policy,**details})
      if param['epoch'] not in ss:raise FloatingPointError('Selected RKNN outer fit failed')
      state=ss[param['epoch']]
     state=l.with_prior(state,param['alpha'],p)
    value=l.predict(state,Q);r.valid(value,len(Q),q);output[key]=value;states[key]=state
    if l.byte_count(state)!=resource('U' if policy=='U_min' else policy.split('_')[0],param,n,len(P),q):raise SystemExit('State byte formula mismatch')
    if policy=='RKNN_min':output[mask+'__RKNN_native_atSelected']=l.predict(state,Q,native=True);states[mask+'__RKNN_native_atSelected']=state
   except Exception as exc:
    output.pop(key,None);states.pop(key,None);failures[key]={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
    if policy=='RKNN_min':failures[mask+'__RKNN_native_atSelected']=failures[key]
  for g in ['G','U']:
   state=l.with_prior(l.prototype_state(base,P,g,50,5),.5,p);key=mask+'__'+g+g+'50_fixed';output[key]=l.predict(state,Q);states[key]=state
 output['plain_mean']=np.tile(p,(len(Q),1));states['plain_mean']={'kind':'Mean','p':p.copy()}
 assert len(output)+len(failures)==10*len(masks)+1
 return output,states,choices,publiccv,cv,parts,{'inner_models':diag,'outer_models':outerdiag,'failures':failures}

def worker(job):
 source,parentrun,out,known,novel,stage,name=job;l=setup(source);r=l.r;parent=r.verify();file=Path(parentrun)/stage/name;old=json.loads(file.read_text());f=next(f for f in parent['stages']['e11']['families'] if f['family']==old['family']) if stage=='e11' else parent['stages']['e12']['family'];fi=next(i for i,f in enumerate(parent['stages']['e11']['families']) if f['family']==old['family']) if stage=='e11' else 16;X,L=r.checked_data(Path(known if stage=='e11' else novel),f);fit=np.array(old['fit_indices']);held=np.array(old['validation_indices']);assert not set(fit)&set(held)
 if stage=='e11':_,excluded,masks=r.known_design(f,fi);assert excluded.tolist()==old['excluded_original_test_indices']
 else:_,masks=r.novel_design(X)
 case=Path(out)/stage/f['family']/f"fold_{old['fold']}";recfile=case/'record_PRIVATE.json'
 if recfile.exists():
  rec=json.loads(recfile.read_text());assert rec['protocol_sha256']==r.sha(ROOT/'PROTOCOL.json') and rec['predictions_sha256']==r.sha(case/'predictions_PRIVATE.npz');return str(recfile.relative_to(out)),r.sha(recfile)
 t=time.perf_counter();values,states,choices,scores,cv,parts,details=select_and_fit(X[fit],L[fit],X[held],masks,fi,old['fold'],202611160500+1000*fi+100*old['fold']);case.mkdir(parents=True,exist_ok=True);np.savez_compressed(case/'predictions_PRIVATE.npz',**values);flat={};meta={}
 for key,state in states.items():
  meta[key]={k:v for k,v in state.items() if not isinstance(v,np.ndarray)};meta[key]['array_bytes']=l.byte_count(state)
  for k,v in l.arrays(state).items():flat[key+'___'+k]=v
 np.savez_compressed(case/'inference_state_PRIVATE.npz',**flat);rawcv={mask+'___'+method+'___'+key:v for (mask,method),vv in cv.items() for key,v in vv.items()};np.savez_compressed(case/'inner_predictions_PRIVATE.npz',**rawcv)
 r.save(recfile,{k:old[k] for k in ['family','fold','fit_indices','validation_indices','excluded_original_test_indices']}|{'stage':stage,'inner_folds':[h.tolist() for h in parts],'choices':{mask+'__'+method:c for (mask,method),c in choices.items()},'cv_scores':scores,'state_metadata':meta,'details':details,'predictions_sha256':r.sha(case/'predictions_PRIVATE.npz'),'states_sha256':r.sha(case/'inference_state_PRIVATE.npz'),'inner_predictions_sha256':r.sha(case/'inner_predictions_PRIVATE.npz'),'protocol_sha256':r.sha(ROOT/'PROTOCOL.json'),'source_manifest_sha256':r.sha(ROOT/'MANIFEST.json'),'seconds':time.perf_counter()-t});return str(recfile.relative_to(out)),r.sha(recfile)

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['run']);p.add_argument('--external-source',type=Path,required=True);p.add_argument('--external-run',type=Path,required=True);p.add_argument('--known-data',type=Path,required=True);p.add_argument('--novel-data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();l=setup(a.external_source);r=l.r
 for n,h in json.loads((ROOT/'MANIFEST.json').read_text()).items():assert r.sha(ROOT/n)==h
 plan=json.loads((ROOT/'PROTOCOL.json').read_text());assert r.sha(a.external_source/'PROTOCOL.json')==plan['parent_protocol_sha256'] and r.sha(a.external_source/'MANIFEST.json')==plan['parent_manifest_sha256'];assert not (a.output/'ALL_PREDICTIONS_FROZEN.json').exists();jobs=[]
 for stage,n in [('e11',48),('e12',5)]:
  fr=json.loads((a.external_run/stage/'ALL_PREDICTIONS_FROZEN.json').read_text());assert len(fr['records'])==n
  for name,h in fr['records'].items():assert r.sha(a.external_run/stage/name)==h;jobs.append((str(a.external_source),str(a.external_run),str(a.output),str(a.known_data),str(a.novel_data),stage,name))
 records={};start=time.perf_counter()
 with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
  for name,h in pool.map(worker,jobs):records[name]=h;print('FROZEN',name,flush=True)
 assert len(records)==53;r.save(a.output/'ALL_PREDICTIONS_FROZEN.json',{'records':records,'protocol_sha256':r.sha(ROOT/'PROTOCOL.json'),'source_manifest_sha256':r.sha(ROOT/'MANIFEST.json'),'seconds':time.perf_counter()-start})
if __name__=='__main__':main()

"""Pre-outcome common-prior overlay; retain all unchanged parent baseline outputs."""
from pathlib import Path
import sys,argparse,json
import numpy as np
p=argparse.ArgumentParser();p.add_argument('action',choices=['run','score','check']);p.add_argument('--external-source',type=Path,required=True);p.add_argument('--external-run',type=Path);p.add_argument('--e13-run',type=Path);p.add_argument('--known-data',type=Path);p.add_argument('--novel-data',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();sys.path.insert(0,str(a.external_source));import runner as r
ROOT=Path(__file__).resolve().parent

def verify():
 for n,h in json.loads((ROOT/'MANIFEST.json').read_text()).items():assert r.sha(ROOT/n)==h
 plan=json.loads((ROOT/'PROTOCOL.json').read_text());parent=r.verify();assert r.sha(a.external_source/'PROTOCOL.json')==plan['parent_protocol_sha256'];return parent,plan

def run():
 parent,plan=verify();out=a.output;out.mkdir(parents=True,exist_ok=True);assert not (out/'ALL_PREDICTIONS_FROZEN.json').exists();records={}
 for stage,n in [('e11',48),('e12',5)]:
  base=a.external_run/stage;fr=json.loads((base/'ALL_PREDICTIONS_FROZEN.json').read_text());assert len(fr['records'])==n
  for name,h in fr['records'].items():
   oldfile=base/name;assert r.sha(oldfile)==h;old=json.loads(oldfile.read_text());assert r.sha(oldfile.parent/'predictions_PRIVATE.npz')==old['predictions_sha256'];case=out/stage/old['family']/f"fold_{old['fold']}";record=case/'record_PRIVATE.json'
   if not record.exists():
    values={}
    with np.load(oldfile.parent/'predictions_PRIVATE.npz') as arr:
     for mask in ['Full','cost40']:
      for method in r.EXTERNAL:
       key=mask+'__'+method
       if key in arr.files:
        v=.5*arr[key]+.5*arr['plain_mean'];r.valid(v,v.shape[0],v.shape[1]);values[key+'__half_prior']=v
    case.mkdir(parents=True,exist_ok=True);np.savez_compressed(case/'predictions_PRIVATE.npz',**values);r.save(record,{k:old[k] for k in ['family','fold','fit_indices','validation_indices','excluded_original_test_indices']}|{'stage':stage,'protocol_sha256':r.sha(ROOT/'PROTOCOL.json'),'source_manifest_sha256':r.sha(ROOT/'MANIFEST.json'),'parent_predictions_sha256':old['predictions_sha256'],'predictions_sha256':r.sha(case/'predictions_PRIVATE.npz'),'missing_parent_slots':list(old['failures'])})
   rec=json.loads(record.read_text());assert rec['protocol_sha256']==r.sha(ROOT/'PROTOCOL.json') and rec['source_manifest_sha256']==r.sha(ROOT/'MANIFEST.json') and rec['predictions_sha256']==r.sha(case/'predictions_PRIVATE.npz');records[str(record.relative_to(out))]=r.sha(record)
 assert len(records)==53;r.save(out/'ALL_PREDICTIONS_FROZEN.json',{'records':records,'protocol_sha256':r.sha(ROOT/'PROTOCOL.json'),'source_manifest_sha256':r.sha(ROOT/'MANIFEST.json')});print('ALL53 DERIVED PREDICTIONS FROZEN',flush=True)

def score():
 parent,plan=verify();out=a.output;fr=json.loads((out/'ALL_PREDICTIONS_FROZEN.json').read_text());assert len(fr['records'])==53 and len(json.loads((a.e13_run/'ALL_PREDICTIONS_FROZEN.json').read_text())['records'])==53;assert not (out/'RESULTS_PUBLIC.json').exists();rows=[]
 for name,h in fr['records'].items():
  file=out/name;assert r.sha(file)==h;rec=json.loads(file.read_text());stage=rec['stage'];f=next(f for f in parent['stages']['e11']['families'] if f['family']==rec['family']) if stage=='e11' else parent['stages']['e12']['family'];folder=a.known_data if stage=='e11' else a.novel_data;yp=folder/f"{rec['family']}_L.npy";assert r.sha(yp)==f['L_sha256'];y=np.load(yp,mmap_mode='r')[rec['validation_indices']];assert r.sha(file.parent/'predictions_PRIVATE.npz')==rec['predictions_sha256']
  with np.load(file.parent/'predictions_PRIVATE.npz') as arr:
   for key in arr.files:rows.append({'stage':stage,'family':rec['family'],'fold':rec['fold'],'key':key,'n_validation':len(y),**r.metrics(y,arr[key])})
 r.save(out/'RESULTS_PUBLIC.json',{'scope':plan['scope'],'rows':rows,'missing_slots':sum(len(json.loads((out/n).read_text())['missing_parent_slots']) for n in fr['records'])});print('SCORED',len(rows),flush=True)

def check():
 y=np.array([[.1,.9],[.8,.2]]);v=np.array([[.7,.3],[.4,.6]]);prior=np.tile([.45,.55],(2,1));mix=.5*v+.5*prior;r.valid(mix,2,2);lhs=np.square(mix-y).sum(1);rhs=.5*np.square(v-y).sum(1)+.5*np.square(prior-y).sum(1)-.25*np.square(v-prior).sum(1);assert np.allclose(lhs,rhs,atol=1e-15,rtol=0);print(json.dumps({'passed':True,'scope':'Synthetic half-prior arithmetic identity only'}))
globals()[a.action]()

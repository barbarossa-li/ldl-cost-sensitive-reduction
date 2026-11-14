"""Execution-only helper: same frozen E11 cases/settings, no scoring or tuning."""
from pathlib import Path
import sys,argparse,json,time,shutil
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--known-data',type=Path,required=True);p.add_argument('--e10',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.source));import runner as r
protocol=r.verify();out=a.output/'e11'
# Fixed lastfour families; original process advances from the front. No outcome-based scheduling.
for fi in range(12,16):
 f=protocol['stages']['e11']['families'][fi];X,L=r.checked_data(a.known_data,f);pool,excluded,masks=r.known_design(f,fi);parts=r.blocks(len(pool),202611134000+10000*fi+1000)
 for fold in range(3):
  case=out/f['family']/f'fold_{fold}';record=case/'record_PRIVATE.json'
  if record.exists():
   rec=json.loads(record.read_text());assert rec['protocol_sha256']==r.sha(a.source/'PROTOCOL.json') and rec['predictions_sha256']==r.sha(case/'predictions_PRIVATE.npz');continue
  held=pool[parts[fold]];fit=pool[r.np.setdiff1d(r.np.arange(len(pool)),parts[fold])];tmp=out/'.parallel_staging'/f['family']/f'fold_{fold}'
  r.one_case(X,L,f,fi,fold,fit,held,excluded,masks,tmp,a.e10)
  if case.exists():
   for attempt in range(100):
    if record.exists():break
    time.sleep(.1)
   assert record.exists()
   with r.np.load(case/'predictions_PRIVATE.npz') as old,r.np.load(tmp/'predictions_PRIVATE.npz') as new:
    assert set(old.files)==set(new.files)
    for key in old.files:assert r.np.array_equal(old[key],new[key])
   shutil.rmtree(tmp)
  else:
   assert not (out/'ALL_PREDICTIONS_FROZEN.json').exists();case.parent.mkdir(parents=True,exist_ok=True);tmp.rename(case)
  print('CHECKPOINT',f['family'],fold,flush=True)
print('HELPER COMPLETE; original runner alone writes the global frozen manifest',flush=True)

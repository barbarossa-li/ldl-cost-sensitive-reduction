"""Explicit numerical amendment for the frozen LDL-25 GBFRS mean-radius boundary.
Preserves original files. New final freezes include per-case amendment provenance.
"""
from pathlib import Path
import sys,os,json,hashlib,inspect,importlib.util,subprocess,threading,queue,time,datetime
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['PYTHONUTF8']='1';os.environ['PYTHONIOENCODING']='utf-8'
SELF=Path(__file__).resolve();BASE=SELF.parent
PATCH_ID='GBFRS-mean-radius-roundoff-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):
 p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.pending');temp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');temp.replace(p)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def apply_patch():
 import numpy as np
 from baseline_revision import gbfrs_reconstruction as impl
 original=impl.local_consistency
 old="radius=float(distance[b].mean());neighbors=np.flatnonzero(distance<=radius)"
 source=inspect.getsource(original)
 assert source.count(old)==1,'Unknown implementation: refusing patch'
 replacement="radius=float(distance[b].mean());neighbors=np.flatnonzero(distance<=radius)\n        if not len(neighbors):\n            minimum=float(distance[b].min())\n            if not np.isfinite(distance).all() or not np.isfinite(radius) or not radius<minimum:\n                raise AssertionError('Empty radius neighborhood is not the recognized roundoff boundary')\n            radius=minimum;neighbors=np.flatnonzero(distance<=radius)\n            _radius_repairs[0]+=1"
 namespace=original.__globals__.copy();repairs=[0];namespace['_radius_repairs']=repairs
 exec(compile(source.replace(old,replacement),'<'+PATCH_ID+'>','exec'),namespace)
 impl.local_consistency=namespace['local_consistency']
 return original,impl.local_consistency,repairs

def provenance():
 return dict(amendment=PATCH_ID,patch_sha256=sha(SELF),original_gbfrs_sha256=sha(BASE/'baseline_revision/gbfrs_reconstruction.py'),original_protocol_sha256=sha(BASE/'expansion25/protocol.json'),change='Only when radius neighborhood is empty and finite computed mean radius is below minimum member distance, raise radius to that minimum. All nonempty original neighborhoods unchanged; no general epsilon or feature removal.',compatibility='Existing successful checkpoints unchanged: old implementation returned no output at the amended boundary. New selections explicitly record this runtime amendment.')

def seal(folder,worker):
 freeze_path=folder/'selection_frozen.json';freeze=read(freeze_path)
 freeze['files']['runtime_amendment.json']=sha(folder/'runtime_amendment.json')
 worker.save(freeze_path,freeze)

def work(args):
 sys.path[:0]=[str(BASE),str(BASE/'stability')]
 worker=module('ldl_amended_case',BASE/'expansion25/run_case.py')
 stage,name,rep=args;rep=int(rep);folder=BASE/'expansion25/results'/name/f'repeat_{rep}'
 done=folder/'external_evaluation.json';frozen=folder/'selection_frozen.json';marker=folder/'runtime_amendment.json'
 if done.exists():
  worker.verify(folder);assert read(done)['freeze_sha256']==sha(frozen)
  print('SKIP completed',name,rep,stage,flush=True);return
 if frozen.exists() and not marker.exists():
  # Existing already-frozen selection remains unamended even if test is pending.
  worker.verify(folder)
  if stage=='evaluate':worker.evaluate(name,rep)
  else:print('SKIP original frozen',name,rep,flush=True)
  return
 info=provenance()
 if marker.exists():assert read(marker)==info,'Existing runtime amendment differs'
 else:save(marker,info)
 _,_,repairs=apply_patch()
 print('ACTIVE AMENDMENT',PATCH_ID,name,rep,stage,flush=True)
 if stage=='train':
  old_candidate=worker.candidate
  def progress(p,token,x,l,split,kind,a,b=None):
   if kind=='GBFRS':print('GBFRS candidate start',a,b,'(three folds; checkpoint after all folds)',flush=True)
   return old_candidate(p,token,x,l,split,kind,a,b)
  worker.candidate=progress
  # Recover a marker seal if a prior run stopped between original freezing and amendment sealing.
  if frozen.exists():seal(folder,worker)
  worker.train(name,rep);seal(folder,worker)
 else:
  seal(folder,worker);worker.evaluate(name,rep)
 save(folder/('runtime_'+stage+'.json'),dict(amendment=PATCH_ID,repair_events_in_this_process=repairs[0]))
 print('AMENDMENT REPAIR EVENTS',repairs[0],flush=True)

def main():
 assert (BASE/'run_all.py').exists() and (BASE/'.venv').exists(),'Place this script inside the existing LDL_25_Expanded_Run folder'
 sys.path[:0]=[str(BASE),str(BASE/'stability')]
 controller=module('ldl_original_controller',BASE/'run_all.py')
 controller.LOGS.mkdir(parents=True,exist_ok=True)
 save(controller.LOGS/'runtime_amendment.json',provenance())
 def command(args,label):
  logfile=controller.LOGS/(label+'.log');start=time.monotonic()
  with logfile.open('a',encoding='utf-8') as log:
   p=subprocess.Popen([sys.executable,'-u',str(SELF),'--worker',*args],cwd=BASE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
   messages=queue.Queue()
   def drain():
    try:
     for line in p.stdout:messages.put(line)
    finally:messages.put(None)
   threading.Thread(target=drain,daemon=True).start()
   try:
    while True:
     try:line=messages.get(timeout=60)
     except queue.Empty:
      line=f'STILL RUNNING {label}: {int(time.monotonic()-start)} seconds in this stage (no percentage estimate)\n'
     if line is None:break
     print(line,end='',flush=True);log.write(line);log.flush()
    code=p.wait()
    if code:raise RuntimeError(f'{label} stopped with exit code {code}; keep return archive and report error')
   except KeyboardInterrupt:
    if os.name=='nt':subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    else:p.terminate()
    p.wait();raise
 controller.command=command
 print('RESUMING ORIGINAL RESULTS WITH EXPLICIT NUMERICAL AMENDMENT',PATCH_ID,flush=True)
 controller.main()
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='--worker':work(sys.argv[2:])
 else:main()

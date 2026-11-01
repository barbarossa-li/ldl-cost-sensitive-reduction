"""Checkpoint-preserving optimizer layered over the explicit radius amendment."""
from pathlib import Path
import sys,os,json,hashlib,importlib.util,subprocess,queue,threading,time,socket
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['PYTHONUTF8']='1';os.environ['PYTHONIOENCODING']='utf-8'
SELF=Path(__file__).resolve();BASE=SELF.parent;VERSION='GBFRS-exact-cache-v1'
REPAIR_SHA='5a69d164367c350daf55e3330e2bd3591162916943345a736f2747210bd0d716'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.pending');q.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');q.replace(p)
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def validate_original():
 assert sha(BASE/'RESUME_25_MOVIE.py')==REPAIR_SHA,'Unexpected numerical-amendment version; keep files and report this message'
 assert (BASE/'GBFRS_FAST_V1.py').exists()
 sys.path[:0]=[str(BASE),str(BASE/'stability')]
 return module('ldl_radius_amendment',BASE/'RESUME_25_MOVIE.py')
def provenance():
 return dict(optimization=VERSION,optimizer_sha256=sha(SELF),implementation_sha256=sha(BASE/'GBFRS_FAST_V1.py'),numerical_amendment_sha256=REPAIR_SHA,
  changes='Share exact-byte-identical standardized-column scores within each round; cache unchanged label agreements and prefix geometry; group equal-shaped lower-approximation operations while retaining value order and stable ties.',
  preserved='All attributes, indices, 200 GBFRS combinations, all folds, paper quota, selections, original protocol, and existing successful checkpoints retained. Radius amendment remains explicit.',
  reference_scope='Exact randomized local and full-ranking trace tests, Movie first12-round full traces, and the two already reported Movie CV scores. Not a proof that all future parameters have already run.')
def work(args):
 repair=validate_original();stage,name,rep=args;rep=int(rep)
 folder=BASE/'expansion25/results'/name/f'repeat_{rep}';freeze=folder/'selection_frozen.json';done=folder/'external_evaluation.json';marker=folder/'runtime_optimization.json'
 # Completed or previously frozen outputs retain their original version and hashes.
 active=marker.exists() or (not freeze.exists() and not done.exists())
 if active:
  info=provenance()
  if marker.exists():assert read(marker)==info,'Optimizer version changed during resume'
  else:save(marker,info)
  if freeze.exists() and not done.exists():
   value=read(freeze);value['files']['runtime_optimization.json']=sha(marker);save(freeze,value)
 original_module=repair.module;original_apply=repair.apply_patch;holder={}
 def capture(name,path):
  value=original_module(name,path)
  if name=='ldl_amended_case':holder['worker']=value
  return value
 def apply_optimized():
  result=original_apply()
  fast=module('ldl_gbfrs_fast',BASE/'GBFRS_FAST_V1.py');adapter=fast.install(repair_counter=result[2]);holder['adapter']=adapter
  worker=holder['worker'];worker.rank=adapter['full_rank'];worker.rank_prefix=adapter['prefix_rank']
  original_candidate=worker.candidate
  def reuse_notice(p,token,x,l,split,kind,a,b=None):
   if kind=='GBFRS' and (p/'grid'/f'GBFRS_{a:.2f}_{b}.json').exists():print('REUSE completed GBFRS parameter',a,b,flush=True)
   return original_candidate(p,token,x,l,split,kind,a,b)
  worker.candidate=reuse_notice
  print('ACTIVE OPTIMIZATION',VERSION,'(same complete grid and attributes)',flush=True)
  return result
 repair.module=capture
 if active:repair.apply_patch=apply_optimized
 repair.work([stage,name,str(rep)])
 if active and freeze.exists() and not done.exists():
  value=read(freeze);value['files']['runtime_optimization.json']=sha(marker);save(freeze,value)
 if 'adapter' in holder:
  state=holder['adapter']['state'];stats={k:state[k] for k in ['lower_calls','lower_preparations','local_calls','local_preparations','roundoff_repairs','duplicate_score_reuses']}
  save(folder/('runtime_optimization_'+stage+'.json'),dict(version=VERSION,**stats));print('OPTIMIZATION COUNTERS',stats,flush=True)
def command(args,label,controller):
 start=time.monotonic()
 with (controller.LOGS/(label+'.log')).open('a',encoding='utf-8') as log:
  p=subprocess.Popen([sys.executable,'-u',str(SELF),'--worker',*args],cwd=BASE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace');messages=queue.Queue()
  def drain():
   try:
    for line in p.stdout:messages.put(line)
   finally:messages.put(None)
  threading.Thread(target=drain,daemon=True).start()
  try:
   while True:
    try:line=messages.get(timeout=60)
    except queue.Empty:line=f'STILL RUNNING {label}: {int(time.monotonic()-start)} seconds (optimized; no percentage estimate)\n'
    if line is None:break
    print(line,end='',flush=True);log.write(line);log.flush()
   code=p.wait()
   if code:raise RuntimeError(f'{label} stopped with exit code {code}; keep return archive and report error')
  except KeyboardInterrupt:
   if os.name=='nt':subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
   else:p.terminate()
   p.wait();raise
def self_test(repair):
 import numpy as np
 repair.apply_patch()
 from baseline_revision import gbfrs_reconstruction as impl
 original_lower,original_local=impl.lower,impl.local_consistency
 rng=np.random.default_rng(901);x=rng.normal(size=(18,12));x[:,5:9]=x[:,0,None];x[:,9:]=0;l=rng.dirichlet([1,1,1],18)
 expected=impl.rank(x,l,.45,7)
 fast=module('ldl_fast_selftest',BASE/'GBFRS_FAST_V1.py');adapter=fast.install()
 try:assert adapter['full_rank'](x,l,.45,7)==expected,'Optimizer local exactness test failed'
 finally:impl.lower=original_lower;impl.local_consistency=original_local
 save(BASE/'expansion25/local_logs/optimization_selftest.json',dict(version=VERSION,exact_full_ranking_and_candidate_scores=True,includes_duplicate_and_constant_columns=True))
 print('OPTIMIZATION EXACTNESS SELF-TEST PASSED',flush=True)
def main():
 assert (BASE/'run_all.py').exists(),'Place this file inside the EXISTING LDL_25_Expanded_Run folder'
 # Avoid touching output provenance while another controller holds the experiment lock.
 with socket.socket() as check:
  check.settimeout(1)
  if check.connect_ex(('127.0.0.1',49784))==0:raise RuntimeError('Existing experiment is running. Stop its console with Ctrl+C and wait for it to finish before starting this optimizer.')
 repair=validate_original();controller=module('ldl_original_controller',BASE/'run_all.py')
 controller.command=lambda args,label:command(args,label,controller)
 original_preflight=controller.preflight
 def checked_preflight():
  protocol=original_preflight();self_test(repair);return protocol
 controller.preflight=checked_preflight
 save(controller.LOGS/'runtime_optimization.json',provenance())
 print('RESUMING WITH EXACT GBFRS CACHING; completed checkpoints are retained.',flush=True)
 controller.main()
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='--worker':work(sys.argv[2:])
 else:main()

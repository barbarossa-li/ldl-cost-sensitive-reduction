"""Fetch recorded public source matrices and verify every cache byte hash."""
from pathlib import Path
import argparse,json,urllib.request,hashlib,concurrent.futures
import numpy as np
from scipy.io import loadmat
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fetch(f,out):
 if all((out/f"{f['family']}_{t}.npy").exists() and sha(out/f"{f['family']}_{t}.npy")==f[t+'_sha256'] for t in ['X','L']):return f['family']+' cached'
 mat=out/(f['family']+'.mat')
 if not mat.exists() or sha(mat)!=f['mat_sha256']:
  tmp=mat.with_suffix('.download');urllib.request.urlretrieve(f['url'],tmp)
  if sha(tmp)!=f['mat_sha256']:raise ValueError('Source MAT hash mismatch '+f['family'])
  tmp.replace(mat)
 data=loadmat(mat);X=np.asarray(data['features'],dtype=float);L=np.asarray(data['labels'],dtype=float)
 if not np.isfinite(X).all() or not np.isfinite(L).all() or np.any(L<0) or np.any(L.sum(1)<=0):raise ValueError('Invalid arrays')
 L=L/L.sum(1,keepdims=True)
 for t,a in [('X',X),('L',L)]:
  file=out/f"{f['family']}_{t}.npy";np.save(file,a)
  if sha(file)!=f[t+'_sha256']:raise ValueError('NPY byte mismatch '+str(file))
 return f['family']+' verified'
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);p=json.loads((ROOT/'PROTOCOL.json').read_text())
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for msg in pool.map(lambda f:fetch(f,a.output),p['families']):print(msg,flush=True)

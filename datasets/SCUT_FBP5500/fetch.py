"""Verify/reuse local or private-Drive cache; official download requires an explicit flag."""
from pathlib import Path
import argparse,hashlib,json
EXPECTED='b8a756d0634a514d3f2f956c3a9363bb8f5a727c7652cbf9d4a87d80bb55a23c'
SIZE=179937702
FILENAME='SCUT-FBP5500_v2.1.zip'
PARTS=[('SCUT-FBP5500_v2.1.zip.part01',90000000,'ce297fe3376caa7bcbe6c17ea5b6a9509b5fcf519a3edfc0a9dbb2c0c74805f4'),('SCUT-FBP5500_v2.1.zip.part02',89937702,'ff4003fde841a9edde3a0443e034a7fb50e8b47431d1d73b4f117a87af28e08d')]
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def verify(path,size,expected):
 if path.stat().st_size!=size or digest(path)!=expected:raise ValueError('Data identity mismatch: '+str(path))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache-dir',type=Path,default=Path('data/cache'));p.add_argument('--download-official',action='store_true');a=p.parse_args();target=a.cache_dir/FILENAME
 if target.exists():verify(target,SIZE,EXPECTED);print('Verified existing cache; no download:',target);return
 if all((a.cache_dir/name).exists() for name,_,_ in PARTS):
  tmp=target.with_suffix('.pending')
  with tmp.open('wb') as output:
   for name,size,h in PARTS:
    part=a.cache_dir/name;verify(part,size,h)
    with part.open('rb') as f:
     for block in iter(lambda:f.read(1024*1024),b''):output.write(block)
  verify(tmp,SIZE,EXPECTED);tmp.replace(target);print('Restored private cache; no download:',target);return
 if not a.download_official:raise SystemExit('No local cache. First locate the author-private Drive folder ASC_LDL_实验数据缓存（私有） and reuse its two parts. Use --download-official only if no valid private cache is accessible.')
 import gdown
 a.cache_dir.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix('.pending')
 result=gdown.download('https://drive.google.com/uc?id=1w0TorBfTIqbquQVd6k3h_77ypnrvfGwf',str(tmp),quiet=False)
 if result is None:raise RuntimeError('Official download failed')
 verify(tmp,SIZE,EXPECTED);tmp.replace(target);print('Verified official download:',target)
if __name__=='__main__':main()

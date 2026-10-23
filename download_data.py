"""Fetch the public LDL benchmark datasets used in the paper and place them where
the reproduction scripts expect them.

All datasets come from the public PyLDL repository (github.com/SpriteMisaka/PyLDL,
datasets/) and the LDL-LDM repository (github.com/wangjing4research/LDL-LDM).
Natural Scene is converted to the pilot/data/{feature,label}.npy convention used
by the frozen stability splits; the cross_dataset and dataset_expansion loaders
read {ds}_X.npy/{ds}_L.npy (produced by the prepare() stages of the respective
run scripts) or the raw .mat files directly.

Usage:  python download_data.py          (requires: requests, scipy, numpy)
"""
import urllib.request
from pathlib import Path
import numpy as np
import scipy.io as sio

PYLDL_RAW="https://raw.githubusercontent.com/SpriteMisaka/PyLDL/main/datasets"

# .mat files consumed by cross_dataset/ and dataset_expansion/
MAT_DATASETS={
    'cross_dataset/data':['SBU_3DFE','Yeast_cold','Human_Gene','M2B'],
    'dataset_expansion/data':['Yeast_spo','Yeast_alpha','Yeast_cdc','Yeast_diau','Yeast_dtt','Yeast_elu',
        'Yeast_heat','Yeast_spo5','Yeast_spoem','FG-Net','SJAFFE','Music','Painting','SCUT_FBP',
        'fbp5500','emotion6','RAF_ML','Movie','Flickr','Twitter','MOR-PH'],
}

def fetch(url,dst):
    if dst.exists():
        print('exists, skip:',dst);return
    print('downloading',url)
    dst.parent.mkdir(parents=True,exist_ok=True)
    urllib.request.urlretrieve(url,dst)

def main():
    root=Path(__file__).resolve().parent
    for folder,names in MAT_DATASETS.items():
        for name in names:
            fetch(f'{PYLDL_RAW}/{name}.mat',root/folder/f'{name}.mat')
    # Natural Scene -> pilot/data/{feature,label}.npy (stability convention)
    ns=root/'pilot/data'
    if not (ns/'feature.npy').exists():
        tmp=root/'Natural_Scene.mat'
        fetch(f'{PYLDL_RAW}/Natural_Scene.mat',tmp)
        m=sio.loadmat(tmp)
        X=np.asarray(m['features'],dtype=float);L=np.asarray(m['labels'],dtype=float)
        s=L.sum(1,keepdims=True);s[s==0]=1.;L=L/s
        ns.mkdir(parents=True,exist_ok=True)
        np.save(ns/'feature.npy',X);np.save(ns/'label.npy',L)
        tmp.unlink()
        print('Natural Scene converted:',X.shape,L.shape)
    print('all datasets ready')

if __name__=='__main__':main()

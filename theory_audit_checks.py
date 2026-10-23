"""Reproducible analytical audit, not a benchmark study.
Working model uses the adopted mixture and relative threshold.
"""
from pathlib import Path
import numpy as np
from granulation_reference import TrainingGeometry,granulate
from archive.granulation_v1 import categorical_geometry as old_geometry,granulate as old_granulate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
X=np.repeat(np.array([[0,0],[0,1],[1,0],[1,1]]),2,axis=0)
first=np.array([.9 if a==b else .1 for a,b in X]);L=np.c_[first,1-first]
D,k=old_geometry(X,L,[0,1]);assert np.max(D)<1e-14
old=old_granulate(D,L,k,delta=.032)[2];assert old==0
G=np.mean(X[:,None,:]!=X[None,:,:],axis=2);zeta=.1
mixed,k=TrainingGeometry(X,L).subset([0,1])
assert np.allclose(mixed,(1-zeta)*D+zeta*G)
new=granulate(mixed,L,k,kappa=.1)[2];assert np.isclose(new,1)
print('XOR: original max distance',D.max(),'original dependency',old,'mixed candidate dependency',new)
fig,axes=plt.subplots(1,2,figsize=(7.5,3.2))
for ax,arr,title in zip(axes,[D,mixed],['Archived information-only distance','Adopted geometry mixture (zeta=0.1)']):
 im=ax.imshow(arr,vmin=0,vmax=.1,cmap='Blues');ax.set_title(title,fontsize=10);ax.set_xticks(range(8),range(1,9));ax.set_yticks(range(8),range(1,9));ax.set_xlabel('Object');ax.set_ylabel('Object')
fig.colorbar(im,ax=axes.ravel().tolist(),label='Symmetric descriptor',fraction=.035,pad=.03);fig.subplots_adjust(left=.06,right=.87,bottom=.15,top=.86,wspace=.32)
for ext in ['pdf','png']:fig.savefig(p/f'theory_audit_xor.{ext}',dpi=180)
# Same separated signal, progressively contracted toward the global profile.
X2=np.array([[0],[0],[1],[1]])
rows=[]
for scale in [1.,.25,.1]:
 first=np.array([.5+.4*scale,.5+.4*scale,.5-.4*scale,.5-.4*scale]);lab=np.c_[first,1-first]
 v0=np.mean(np.sum((lab-lab.mean(0))**2,axis=1));d,keys=TrainingGeometry(X2,lab).subset([0])
 fixed=old_granulate(d,lab,keys,delta=.01)[2];relative=granulate(d,lab,keys,kappa=.1)[2]
 assert np.isclose(relative,1);rows.append([scale,v0,fixed,relative]);print('scale,V0,fixed,relative',rows[-1])
np.savetxt(p/'theory_audit_threshold.csv',rows,delimiter=',',header='scale,V0,fixed_delta_dependency,relative_delta_dependency',comments='')
(p/'theory_audit_checks.txt').write_text('Original XOR joint subset has zero distances and dependency 0.\nAdopted geometry mixture zeta=.1 yields dependency 1 on the repeated-XOR construction.\nFixed delta=.01 stops the contracted perfect signal at the root when V0=.0032.\nRelative delta=.1*V0 yields dependency 1 at all three tested scales.\nSynthetic mechanism checks, not benchmark evidence; mixture and relative threshold adopted in Sections 2--3.\n')

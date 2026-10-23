from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from granulation_reference import categorical_geometry,granulate
p=Path(__file__).resolve().parent;data=np.loadtxt(p/'negative_gain_construction.csv',delimiter=',',skiprows=1);X=data[:,:2].astype(int);L=data[:,2:]
fig,axes=plt.subplots(1,2,figsize=(8,3.3),sharey=True)
for ax,P,title in zip(axes,[[0],[0,1]],['Subset {a}','Subset {a,b}']):
 D,keys=categorical_geometry(X,L,P);leaves,accepted,g,h=granulate(D,L,keys,kappa=.1,alpha=1.)
 matrix=np.zeros((8,len(leaves)))
 for j,B in enumerate(leaves):matrix[B,j]=2 if B in accepted else 1
 ax.imshow(matrix,cmap=ListedColormap(['white','#bcbcbc','#3976ae']),vmin=0,vmax=2,aspect='auto',interpolation='nearest')
 ax.set_xticks(range(len(leaves)),[f'Ball {j+1}' for j in range(len(leaves))]);ax.set_yticks(range(8),[f'Object {i+1}' for i in range(8)]);ax.set_title(f'{title}: dependency = {g:.5f}')
 ax.set_xticks(np.arange(-.5,len(leaves),1),minor=True);ax.set_yticks(np.arange(-.5,8,1),minor=True);ax.grid(which='minor',color='#e0e0e0');ax.tick_params(which='minor',bottom=False,left=False)
fig.legend(handles=[Patch(facecolor='#3976ae',label='Eligible membership'),Patch(facecolor='#bcbcbc',label='Ineligible membership')],loc='lower center',ncol=2,frameon=False)
fig.tight_layout(rect=(0,.1,1,1))
for ext in ['pdf','png']:fig.savefig(p/f'granulation_negative_gain.{ext}',dpi=180)

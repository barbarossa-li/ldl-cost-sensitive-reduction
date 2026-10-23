import os
os.environ['MPLCONFIGDIR']='/tmp/ldl-mpl-cache'
from pathlib import Path
import numpy as np, math, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
def h2(e):
 if e in (0.,1.):return 0.
 return -e*math.log2(e)-(1-e)*math.log2(1-e)
with (p/'entropy_diagnostic.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['epsilon','label_profile_TV','categorical_entropy_weight','current_directed_contribution_m2','reversed_weight_contribution_m2','unweighted_contribution_m2'])
 for e in [0.,.01,.05,.1,.25,.5]:
  h=h2(e);d=1-2*e;row=[e,d,h,h*d/2,(1-h)*d/2,d/2];w.writerow(row);print(', '.join(f'{v:.6f}' for v in row))
x=np.linspace(0,.5,501);h=np.array([h2(float(e)) for e in x]);d=1-2*x
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
fig,ax=plt.subplots(figsize=(7.2,4.1),constrained_layout=True)
ax.plot(x,d/2,label='Attribute discrepancy / m',color='#516779',lw=2)
ax.plot(x,h*d/2,label='Current entropy weight',color='#b34d3b',lw=2.3)
ax.plot(x,(1-h)*d/2,label='Reversed weight (diagnostic only)',color='#278477',lw=2,ls='--')
ax.set(xlabel=r'Label mixing $\varepsilon$ (smaller = stronger separation)',ylabel='Contribution to dissimilarity (m = 2)',xlim=(0,.5),ylim=(0,.52))
ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False)
ax.set_title('Constructed categorical example: entropy weighting suppresses separation',fontsize=10)
fig.savefig(p/'entropy_diagnostic.pdf');fig.savefig(p/'entropy_diagnostic.png',dpi=180)

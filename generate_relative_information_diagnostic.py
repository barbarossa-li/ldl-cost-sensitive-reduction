from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
def h(x):
 x=np.asarray(x); return -(x*np.log2(np.maximum(x,1e-300))+(1-x)*np.log2(np.maximum(1-x,1e-300)))
e=np.linspace(0,.5,501); old=h(e)*(1-2*e)/2; new=(1-h(.25+.5*e))*(1-2*e)/2; mixed=.9*new+.05
np.savetxt(p/'relative_information_diagnostic.csv',np.c_[e,old,new,mixed],delimiter=',',header='epsilon,previous_entropy_contribution,information_only_contribution,adopted_mixed_contribution',comments='')
fig,ax=plt.subplots(figsize=(6.6,3.8)); ax.plot(e,old,label='Previous entropy weight'); ax.plot(e,new,label='Information-only predecessor'); ax.plot(e,mixed,label='Adopted mixture (zeta = 0.1)'); ax.set(xlabel=r'Label mixing $\varepsilon$ (smaller = clearer signal)',ylabel='Directed distance contribution',title='Constructed example: fixed m = 2, beta = 1'); ax.legend(frameon=False); ax.grid(alpha=.2);fig.tight_layout()
for ext in ['pdf','png']:fig.savefig(p/f'relative_information_diagnostic.{ext}',dpi=180)
for x in [0,.01,.1,.25,.5]: print(x,float(h(x)*(1-2*x)/2),float((1-h(.25+.5*x))*(1-2*x)/2),float(.9*(1-h(.25+.5*x))*(1-2*x)/2+.05))

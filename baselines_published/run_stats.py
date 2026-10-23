"""Friedman + Wilcoxon (Holm) tests for the published scoring-core comparison.
Analysis units: 60 splits (4 datasets x 5 repeats x 3 fitting sizes; M2B excluded as
a predictor-suitability boundary case). Repeats within a dataset share test partitions;
the tests quantify descriptive patterns.
Reproduction: python run_stats.py (after run_published.py and run_cross_published.py).
"""
import csv, json
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
from pathlib import Path
root=Path(__file__).resolve().parents[1]
here=Path(__file__).resolve().parent

data={}
for rep in range(1,6):
    for n in [192,384,576]:
        old=json.loads((root/'stability'/f'repeat_{rep}_n{n}'/'results.json').read_text())
        data[('NS',str(rep),str(n),'G')]=float(next(r for r in old if r['method']=='Joint guard')['test_brier'])
for r in csv.DictReader(open(here/'results.csv')):
    data[('NS',r['repeat'],r['train_n'],r['method'])]=float(r['test_brier'])
for ds in ['SBU_3DFE','Yeast_cold','Human_Gene']:
    for rep in range(1,6):
        for n in [192,384,576]:
            old=json.loads((root/'cross_dataset'/ds/f'repeat_{rep}_n{n}'/'results.json').read_text())
            data[(ds,str(rep),str(n),'G')]=float(next(r for r in old if r['method']=='Joint guard')['test_brier'])
    for r in csv.DictReader(open(here/f'{ds}_results.csv')):
        data[(ds,r['repeat'],r['train_n'],r['method'])]=float(r['test_brier'])

methods=['G','GBFRS k','GBFRS budget','DSIM k','DSIM budget']
splits=[(d,rep,n) for d in ['NS','SBU_3DFE','Yeast_cold','Human_Gene'] for rep in ['1','2','3','4','5'] for n in ['192','384','576']]
mat=np.array([[data[(d,rep,n,m)] for m in methods] for d,rep,n in splits])
stat,p=friedmanchisquare(*[mat[:,i] for i in range(5)])
print('Friedman chi2=%.3f p=%.6f (60 splits)'%(stat,p))
pvals={}
for m in methods[1:]:
    if np.allclose(mat[:,0]-mat[:,methods.index(m)],0): pvals[m]=1.0
    else: pvals[m]=wilcoxon(mat[:,0],mat[:,methods.index(m)],zero_method='wilcox').pvalue
ms=sorted(pvals.items(),key=lambda kv:kv[1]); M=len(ms)
for i,(m,p) in enumerate(ms):
    print('  %-13s p=%.4f holm=%.4f %s'%(m,p,min(1.0,p*(M-i)),'**' if min(1.0,p*(M-i))<0.05 else ''))

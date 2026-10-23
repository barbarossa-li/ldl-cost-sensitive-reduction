from pathlib import Path
import math
import numpy as np
from reduction_reference import heuristic,exact_backtracking,powerset
rng=np.random.default_rng(20260930)
for trial in range(80):
 m=6;c=rng.integers(1,11,m);D={P:float(rng.random()) for P in powerset(m)};D[()]=0.
 R={P:float(rng.random()) for P in powerset(m)};dep=lambda P:D[P];loss=lambda P:R[P]
 h=heuristic(c,dep,tolerance=.03,prediction_loss=loss,prediction_tolerance=.02)
 e=exact_backtracking(c,dep,tolerance=.03,prediction_loss=loss,prediction_tolerance=.02)
 full=tuple(range(m));feasible=lambda P:D[P]>=e[2] and R[P]<=R[full]+.02
 truth=min(sum(c[a] for a in P) for P in powerset(m) if feasible(P))
 assert math.isclose(e[1],truth) and feasible(h[0]) and all(not feasible(tuple(a for a in h[0] if a!=b)) for b in h[0])
# High dependency alone no longer makes the cheap singleton feasible.
D={():0.,(0,):1.,(1,):.4,(0,1):.8};R={():.5,(0,):.6,(1,):.4,(0,1):.2}
h=heuristic([1,2],lambda P:D[P],prediction_loss=lambda P:R[P]);e=exact_backtracking([1,2],lambda P:D[P],prediction_loss=lambda P:R[P])
assert h[0]==e[0]==(0,1) and any('prediction loss' in step[3] for step in h[3])
text='80 nonmonotone joint dependency/loss tables: exact search agrees with exhaustive minimum feasible cost.\nHeuristic outputs jointly feasible single-deletion-irreducible sets.\nHigh-dependency but high-loss singleton rejected; forward search switches to predictive improvement.\nNo dependency or validation-loss monotonicity assumed.\n'
print(text);Path(__file__).with_name('guarded_search_checks.txt').write_text(text)

import itertools,json
from pathlib import Path
import numpy as np
from revised_search import heuristic,exact
rng=np.random.default_rng(20260930);checks=[]
for trial in range(80):
 m=6;subsets=[P for k in range(m+1) for P in itertools.combinations(range(m),k)];c=rng.integers(1,11,m)
 D={P:float(rng.random()) for P in subsets};R={P:float(rng.random()) for P in subsets};D[()]=0.
 h=heuristic(c,lambda P:D[P],lambda P:R[P]);e=exact(c,lambda P:R[P])
 truth=min(sum(c[a] for a in P) for P in subsets if R[P]<=R[tuple(range(m))])
 assert e[1]==truth and h[1]>=e[1] and R[h[0]]<=h[2]
 assert all(R[tuple(a for a in h[0] if a!=b)]>h[2] for b in h[0])
checks.append('80 arbitrary nonmonotone loss/dependency tables: exact agrees with enumeration; heuristic feasible and single-deletion irreducible')
# Improved prediction is feasible despite dependency below the full-set reference.
D={():0.,(0,):.05,(1,):.5,(0,1):.8};R={():.4,(0,):.1,(1,):.5,(0,1):.2}
h=heuristic([1,2],lambda P:D[P],lambda P:R[P]);assert h[0]==(0,)
checks.append('Lower-dependency predictive singleton is accepted')
# Complementary features with no useful singleton step must remain reachable.
R={():.5,(0,):.6,(1,):.6,(0,1):.2};h=heuristic([1,2],lambda P:D[P],lambda P:R[P]);assert h[0]==(0,1) and h[3][0]['rule']=='nonpositive prediction bridge'
checks.append('Nonpositive prediction bridge reaches complementary pair')
R={():.1,(0,):.4,(1,):.5,(0,1):.2};h=heuristic([1,2],lambda P:D[P],lambda P:R[P]);e=exact([1,2],lambda P:R[P]);assert h[0]==e[0]==() and h[1]==e[1]==0
checks.append('Feasible empty subset is retained as zero-cost optimum')
Path(__file__).with_name('search_checks.json').write_text(json.dumps(checks,indent=2));print('\n'.join(checks))

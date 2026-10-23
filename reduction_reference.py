"""Reference searches for a deterministic, potentially nonmonotone dependency.
All costs are strictly positive. With prediction_loss supplied, feasibility includes the inner-loss guard.
Without it they retain the dependency-only ablation for historical checks.
"""
import math
from itertools import combinations

def _setup(costs, dependency, tolerance):
    c=tuple(float(x) for x in costs)
    if not c or any(not math.isfinite(x) or x<=0 for x in c):
        raise ValueError('Nonempty attribute list with finite positive costs required')
    if not math.isfinite(tolerance) or tolerance<0: raise ValueError('Invalid tolerance')
    cache={}
    def dep(P):
        key=tuple(sorted(P))
        if key not in cache:
            value=float(dependency(key))
            if not math.isfinite(value) or not -1e-12<=value<=1+1e-12:
                raise ValueError('Dependency must lie in [0,1]')
            cache[key]=min(1.,max(0.,value))
        return cache[key]
    A=tuple(range(len(c)));full=dep(A);target=max(0.,full-tolerance)
    return c,dep,A,target,cache

def _loss_guard(prediction_loss, A, prediction_tolerance):
    if not math.isfinite(prediction_tolerance) or prediction_tolerance<0:
        raise ValueError('Prediction tolerance must be nonnegative')
    cache={}
    def loss(P):
        key=tuple(sorted(P))
        if key not in cache:
            value=float(prediction_loss(key))
            if not math.isfinite(value) or value<0:raise ValueError('Invalid predictive loss')
            cache[key]=value
        return cache[key]
    limit=loss(A)+prediction_tolerance if prediction_loss is not None else None
    return loss,limit,cache

def heuristic(costs, dependency, tolerance=0., lam=-1., prediction_loss=None, prediction_tolerance=0.):
    if not math.isfinite(lam) or lam>0: raise ValueError('Use finite lambda <= 0')
    c,dep,A,target,cache=_setup(costs,dependency,tolerance)
    loss,limit,loss_cache=_loss_guard(prediction_loss,A,prediction_tolerance)
    def feasible(P):return dep(P)>=target and (limit is None or loss(P)<=limit)
    cref=sum(c)/len(c);P=set();trace=[]
    while not feasible(P):
        if dep(P)<target:
            base=dep(P);gains={a:dep(P|{a})-base for a in A if a not in P};criterion='dependency'
        else:
            base=loss(P);gains={a:base-loss(P|{a}) for a in A if a not in P};criterion='prediction loss' 
        positive=[a for a,g in gains.items() if g>0]
        if positive:
            # Logarithms preserve ordering and avoid power overflow.
            a=min(positive,key=lambda a:(-(math.log(gains[a])+lam*math.log(c[a]/cref)),c[a],a))
            mode='positive cost-weighted gain'
        else:
            a=min(gains,key=lambda a:(-gains[a],c[a],a))
            mode='nonpositive bridge'
        trace.append((tuple(sorted(P)),a,gains[a],mode if limit is None else criterion+': '+mode));P.add(a)
    # Revisit every deletion after each successful removal.
    while True:
        removed=False
        for a in sorted(P,key=lambda a:(-c[a],a)):
            if feasible(P-{a}):
                trace.append((tuple(sorted(P)),a,dep(P-{a})-dep(P),'delete'))
                P.remove(a);removed=True;break
        if not removed:break
    assert feasible(P)
    assert all(not feasible(P-{a}) for a in P)
    return tuple(sorted(P)),sum(c[a] for a in P),target,trace,len(cache)

def exact_backtracking(costs, dependency, tolerance=0., prediction_loss=None, prediction_tolerance=0.):
    c,dep,A,target,cache=_setup(costs,dependency,tolerance)
    loss,limit,loss_cache=_loss_guard(prediction_loss,A,prediction_tolerance)
    def feasible(P):return dep(P)>=target and (limit is None or loss(P)<=limit)
    best=A;bestcost=sum(c);nodes=0
    order=tuple(sorted(A,key=lambda a:(c[a],a)))
    def visit(P,k,current):
        nonlocal best,bestcost,nodes
        nodes+=1
        # Every descendant retains P and can only add strictly positive cost.
        if current>=bestcost:return
        if feasible(P):
            best=tuple(sorted(P));bestcost=current;return
        if k==len(order):return
        a=order[k]
        visit(P,k+1,current)
        visit(P+(a,),k+1,current+c[a])
    visit((),0,0.)
    return best,bestcost,target,nodes,len(cache)

def powerset(m):
    for size in range(m+1):yield from combinations(range(m),size)

if __name__=='__main__':
    import numpy as np
    from pathlib import Path
    from granulation_reference import TrainingGeometry,granulate
    rng=np.random.default_rng(20260930)
    for trial in range(80):
        m=6;c=rng.integers(1,11,m);table={P:float(rng.random()) for P in powerset(m)}
        table[()]=0.;dep=lambda P:table[tuple(P)]
        h=heuristic(c,dep,tolerance=.03);e=exact_backtracking(c,dep,tolerance=.03)
        truth=min(sum(c[a] for a in P) for P in powerset(m) if table[P]>=e[2])
        assert math.isclose(e[1],truth)
        assert h[1]>=e[1] and table[h[0]]>=h[2]
    # Interaction with zero singleton gains requires a bridge, not early stopping.
    t={():0.,(0,):0.,(1,):0.,(0,1):1.}
    h=heuristic([1,2],lambda P:t[P]);assert h[0]==(0,1) and h[3][0][3]=='nonpositive bridge'
    X=np.array([[0,0],[0,1],[1,0],[1,1]])
    L=np.array([[.9,.1],[.9,.1],[.1,.9],[.1,.9]])
    fitted=TrainingGeometry(X,L)
    def actual(P):
        D,k=fitted.subset(P)
        return granulate(D,L,k)[2]
    h=heuristic([5,1],actual);e=exact_backtracking([5,1],actual)
    assert h[0]==e[0]==(0,) and h[1]==e[1]==5
    text='80 nonmonotone subset tables: exact search agrees with exhaustive enumeration.\nZero-singleton interaction: bridge reaches feasible pair.\nActual four-object LDL construction: heuristic and exact return attribute a, cost 5.\nAll are mechanism checks, not LDL benchmark comparisons.\n'
    print(text)
    (Path(__file__).resolve().parent/'reduction_checks.txt').write_text(text)

"""Equivalent vectorized farthest-pair selection; core definitions unchanged."""
import numpy as np
def granulate(D, labels, keys, smin=2, kappa=.1, alpha=1., eps_v=0., eps_r=0.):
    D=np.asarray(D,float);L=np.asarray(labels,float);n=len(L)
    if D.shape!=(n,n) or len(keys)!=n or n==0:raise ValueError('Shape mismatch')
    if not np.isfinite(D).all() or np.any(D<0) or not np.allclose(D,D.T,atol=1e-12) or not np.allclose(np.diag(D),0,atol=1e-12):
        raise ValueError('D must be finite, nonnegative, symmetric, with zero diagonal')
    if smin<2 or int(smin)!=smin or not 0<=kappa<1 or not 0<=alpha<=1 or min(eps_v,eps_r)<0:raise ValueError('Invalid parameters')
    p=L.mean(0);v0=float(np.mean(np.sum((L-p)**2,axis=1)));delta=kappa*v0
    def stats(B):
        x=L[B];mu=x.mean(0);v=np.mean(np.sum((x-mu)**2,axis=1))
        local=D[np.ix_(B,B)];rep=B[int(np.argmin(local.sum(1)))];r=float(D[rep,B].max())
        return mu,float(v),r
    leaves=[];history=[];pending=[list(range(n))]
    while pending:
        B=pending.pop(0);mu,v,r=stats(B)
        eligible=len(B)>=smin and v<=delta and r<=alpha
        if eligible or len(B)<2*smin:leaves.append(B);continue
        atoms={}
        for i in B:atoms.setdefault(keys[i],[]).append(i)
        reps=[a[0] for a in atoms.values()]
        # Same farthest pair and lexicographic original-index ties, without Python tuple list.
        rr=np.asarray(reps);ia,ja=np.triu_indices(len(rr),1)
        vv=D[rr[ia],rr[ja]]
        if not len(vv) or vv.max()<=0:leaves.append(B);continue
        pos=np.flatnonzero(vv==vv.max())
        chosen=min(pos,key=lambda k:(int(rr[ia[k]]),int(rr[ja[k]])))
        u=int(rr[ia[chosen]]);w=int(rr[ja[chosen]]);C1=[];C2=[]
        for atom in atoms.values():
            i=atom[0];(C1 if D[i,u]<=D[i,w] else C2).extend(atom)
        C1.sort();C2.sort()
        if min(len(C1),len(C2))<smin:leaves.append(B);continue
        _,v1,r1=stats(C1);_,v2,r2=stats(C2)
        gain=v-(len(C1)*v1+len(C2)*v2)/len(B)
        assert gain>=-1e-12
        if not(gain>eps_v or max(r1,r2)<r-eps_r):leaves.append(B);continue
        history.append((B,C1,C2,gain,r,max(r1,r2)));pending.extend([C1,C2])
    numerator=0.;accepted=[]
    for B in leaves:
        mu,v,r=stats(B)
        if len(B)>=smin and v<=delta and r<=alpha:
            accepted.append(B);numerator+=len(B)*np.sum((mu-p)**2)
    dep=numerator/(n*v0) if v0>0 else 0.
    assert sorted(i for B in leaves for i in B)==list(range(n))
    assert 0<=dep<=1+1e-12
    return leaves,accepted,float(dep),history

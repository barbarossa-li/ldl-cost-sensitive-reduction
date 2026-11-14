"""Adopted working model: mixed directed geometry and delta=kappa*V0.
TrainingGeometry fits once on training objects. Synthetic reference only.
"""
import numpy as np

class TrainingGeometry:
    def __init__(self,X,labels,kinds=None,beta=1.,zeta=.1,bandwidths=None):
        self.X=np.asarray(X,dtype=object);self.L=np.asarray(labels,float)
        if self.X.ndim!=2 or self.L.ndim!=2:raise ValueError('Require two-dimensional arrays')
        n,m=self.X.shape
        if not n or not m or len(self.L)!=n or self.L.shape[1]==0:raise ValueError('Invalid shapes')
        if not np.isfinite(self.L).all() or np.any(self.L<0) or not np.allclose(self.L.sum(1),1,rtol=0,atol=1e-12):
            raise ValueError('Labels must be probability distributions')
        if not np.isfinite(beta) or beta<=0 or not np.isfinite(zeta) or not 0<=zeta<1:
            raise ValueError('Require beta > 0 and 0 <= zeta < 1; zeta=0 is an ablation')
        self.kinds=tuple(kinds if kinds is not None else ['categorical']*m)
        if len(self.kinds)!=m or any(k not in ('categorical','numerical') for k in self.kinds):raise ValueError('Invalid kinds')
        if bandwidths is not None and len(bandwidths)!=m:raise ValueError('Invalid bandwidth list')
        self.beta=beta;self.zeta=zeta;self.m=m;self.n=n
        p=self.L.mean(0);active=p>0;norm=np.log2(1/p[active].min()) if active.sum()>1 else 0.
        self.information=[];self.geometry=[];self.weights=[];self.bandwidths=[]
        for a,kind in enumerate(self.kinds):
            values=self.X[:,a]
            if kind=='numerical':
                v=np.asarray(values,float)
                if not np.isfinite(v).all():raise ValueError('Numerical attributes must be finite')
                z=(v-v.min())/(v.max()-v.min()) if v.max()>v.min() else np.zeros(n)
                raw=np.abs(z[:,None]-z[None,:]);positive=raw[np.triu_indices(n,1)];positive=positive[positive>0]
                tau=bandwidths[a] if bandwidths is not None else None
                tau=float(tau) if tau is not None else (float(np.median(positive)) if len(positive) else 1.)
                if not np.isfinite(tau) or tau<=0:raise ValueError('Bandwidth must be positive')
                K=np.exp(-raw**2/(2*tau**2));discrepancy=raw
            else:
                if any(v is None or (isinstance(v,(float,np.floating)) and not np.isfinite(v)) for v in values):
                    raise ValueError('Missing categorical values require preprocessing')
                equal=values[:,None]==values[None,:];K=equal.astype(float);raw=(~equal).astype(float);tau=None
            support=K.sum(1);other=support-1.;u=K@self.L/support[:,None];eta=other/(other+beta)
            regular=eta[:,None]*u+(1-eta[:,None])*p
            w=np.sum(regular[:,active]*np.log2(regular[:,active]/p[active]),axis=1)/norm if norm else np.zeros(n)
            w=np.clip(w,0.,1.) # Roundoff guard for exact nonnegative KL.
            if kind=='categorical':
                discrepancy=np.zeros((n,n))
                for k in range(self.L.shape[1]):discrepancy+=.5*np.abs(u[:,k,None]-u[None,:,k])
            self.information.append(w[:,None]*discrepancy/m);self.geometry.append(raw/m)
            self.weights.append(w);self.bandwidths.append(tau)
    def directed(self,P,zeta=None):
        P=tuple(sorted(P))
        if len(set(P))!=len(P) or any(a<0 or a>=self.m for a in P):raise ValueError('Invalid subset')
        z=self.zeta if zeta is None else float(zeta)
        if not 0<=z<1:raise ValueError('Invalid mixture')
        d=np.zeros((self.n,self.n))
        for a in P:d+=(1-z)*self.information[a]+z*self.geometry[a]
        return d
    def subset(self,P,zeta=None):
        P=tuple(sorted(P));forward=self.directed(P,zeta)
        return np.maximum(forward,forward.T),[tuple(row[list(P)]) for row in self.X]

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
        pairs=[(float(D[i,j]),i,j) for a,i in enumerate(reps) for j in reps[a+1:] if D[i,j]>0]
        if not pairs:leaves.append(B);continue
        _,u,w=min(pairs,key=lambda x:(-x[0],x[1],x[2]));C1=[];C2=[]
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

def categorical_geometry(X,L,P,beta=1.,zeta=.1):
    """Convenience wrapper. For repeated subset searches fit TrainingGeometry once."""
    return TrainingGeometry(X,L,beta=beta,zeta=zeta).subset(P)

if __name__=='__main__':
    from pathlib import Path
    dest=Path(__file__).resolve().parent
    L=np.array([[.9,.1],[.9,.1],[.1,.9],[.1,.9]]);X=np.array([[0],[0],[1],[1]])
    model=TrainingGeometry(X,L);D,k=model.subset([0]);leaves,accepted,dep,h=granulate(D,L,k)
    assert np.isclose(dep,1) and len(leaves)==2
    D,k=model.subset([]);assert granulate(D,L,k)[2]==0
    same=np.tile([.5,.5],(4,1));D,k=categorical_geometry(X,same,[0]);assert granulate(D,same,k)[2]==0
    assert granulate(np.ones((3,3))-np.eye(3),L[:3],[(i,) for i in range(3)])[2]==0
    # Reproducible negative-gain construction for the ADOPTED parameter configuration.
    rng=np.random.default_rng(20260930)
    for trial in range(20000):
        x=rng.integers(0,4,size=(8,2));lab=rng.uniform(.05,.95,8);y=np.c_[lab,1-lab]
        model=TrainingGeometry(x,y);d1,k1=model.subset([0]);d2,k2=model.subset([0,1])
        g1=granulate(d1,y,k1)[2];g2=granulate(d2,y,k2)[2]
        if g2<g1-1e-5:
            np.savetxt(dest/'negative_gain_construction.csv',np.c_[x,y],delimiter=',',header='a,b,label_1,label_2',comments='')
            text=f'Adopted model: zeta=.1, kappa=.1, smin=2, alpha=1, beta=1, m=2\nGamma(a)={g1:.12g}\nGamma(a,b)={g2:.12g}\nGain={g2-g1:.12g}\nSeed 20260930; trial {trial}; synthetic mechanism check.\n'
            (dest/'granulation_checks.txt').write_text(text);print(text);break
    else:raise AssertionError('No negative-gain construction found')

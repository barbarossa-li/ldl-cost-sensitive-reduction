"""Anchored training-representation extension, not an unanchored new granulator.

Keep every originally eligible leaf. Admit an alternative-seed-only leaf iff
its mean is no worse than the global fallback for EVERY member. Unsafe new
leaves fall back as whole leaves. No per-object mixing of reconstruction maps.
"""
import numpy as np
from stability.fast_granulate import granulate as original
from information_reduction.alternative_seeds import granulate as alternative


def granulate(D, labels, keys, smin=2, kappa=.1, alpha=1., eps_v=0., eps_r=0.,
             pair_budget=256, atol=1e-12):
    L=np.asarray(labels,float)
    if L.ndim!=2 or not L.shape[0] or not L.shape[1] or not np.isfinite(L).all():
        raise ValueError('Finite nonempty label matrix required')
    if np.any(L<0) or not np.allclose(L.sum(1),1,atol=1e-10,rtol=0):
        raise ValueError('Labels must be probability distributions')
    if not np.isfinite(atol) or atol<0:raise ValueError('Invalid numerical tolerance')
    args=dict(smin=smin,kappa=kappa,alpha=alpha,eps_v=eps_v,eps_r=eps_r)
    base=original(D,L,keys,**args)
    revised=alternative(D,L,keys,pair_budget=pair_budget,**args)
    anchor={tuple(B) for B in base[1]}
    proposed={tuple(B) for B in revised[1]}
    if not anchor.issubset(proposed):
        raise AssertionError('Alternative construction must preserve eligible anchor leaves')
    p=L.mean(0); fallback=np.sum((L-p)**2,axis=1)
    accepted=[]; rejected=[];new_accepted=[]
    for B in revised[1]:
        key=tuple(B)
        local=np.sum((L[B]-L[B].mean(0))**2,axis=1)
        if key in anchor or np.all(local<=fallback[B]+atol):
            accepted.append(B)
            if key not in anchor:new_accepted.append(B)
        else:rejected.append(B)
    v0=float(fallback.mean()); reconstructed=np.tile(p,(len(L),1))
    baseline=np.tile(p,(len(L),1))
    for B in accepted:reconstructed[B]=L[B].mean(0)
    for B in base[1]:baseline[B]=L[B].mean(0)
    errors=np.sum((L-reconstructed)**2,axis=1)
    base_errors=np.sum((L-baseline)**2,axis=1)
    assert np.all(errors<=base_errors+atol)
    gamma=float(1-errors.mean()/v0) if v0>0 else 0.
    audit=dict(**revised[4],anchor_gamma=base[2],numerical_tolerance=atol,
               retained_anchor_leaves=len(anchor),safe_new_leaves=len(new_accepted),
               safe_new_objects=sum(map(len,new_accepted)),rejected_new_leaves=len(rejected),
               rejected_new_objects=sum(map(len,rejected)),
               maximum_objectwise_error_increase=float(np.max(errors-base_errors)))
    return revised[0],accepted,gamma,revised[3],audit

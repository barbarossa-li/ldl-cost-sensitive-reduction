"""Candidate revision: prediction feasibility with dependency-guided positive steps.
Original reduction_reference.py is retained unchanged.
"""
import sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reduction_reference import _setup,_loss_guard

def heuristic(costs,dependency,prediction_loss,lam=-1.,prediction_tolerance=0.):
 if prediction_loss is None:raise ValueError('Prediction loss is required')
 if not math.isfinite(lam) or lam>0:raise ValueError('Use finite lambda <= 0')
 c,dep,A,_,cache=_setup(costs,dependency,0.)
 loss,limit,lc=_loss_guard(prediction_loss,A,prediction_tolerance)
 cref=sum(c)/len(c);P=set();trace=[]
 def weighted(g,a):return math.log(g[a])+lam*math.log(c[a]/cref)
 while loss(P)>limit:
  before=tuple(sorted(P));base=loss(P);g={a:base-loss(P|{a}) for a in A if a not in P}
  positive=[a for a in g if g[a]>0]
  if positive:
   d0=dep(P);dg={a:dep(P|{a})-d0 for a in positive};both=[a for a in positive if dg[a]>0]
   if both:a=min(both,key=lambda a:(-weighted(dg,a),c[a],a));mode='prediction-improving dependency gain'
   else:a=min(positive,key=lambda a:(-weighted(g,a),c[a],a));mode='prediction gain fallback'
  else:
   a=min(g,key=lambda a:(-g[a],c[a],a));mode='nonpositive prediction bridge'
  P.add(a);trace.append({'phase':'add','before':before,'attribute':a,'prediction_gain':g[a],'rule':mode})
 while True:
  removed=False
  for a in sorted(P,key=lambda a:(-c[a],a)):
   if loss(P-{a})<=limit:
    trace.append({'phase':'delete','before':tuple(sorted(P)),'attribute':a});P.remove(a);removed=True;break
  if not removed:break
 assert loss(P)<=limit and all(loss(P-{a})>limit for a in P)
 return tuple(sorted(P)),sum(c[a] for a in P),limit,trace

def exact(costs,prediction_loss,prediction_tolerance=0.):
 c=tuple(float(v) for v in costs)
 if not c or any(not math.isfinite(v) or v<=0 for v in c):raise ValueError('Positive finite costs required')
 if prediction_loss is None:raise ValueError('Prediction loss is required')
 A=tuple(range(len(c)));loss,limit,_=_loss_guard(prediction_loss,A,prediction_tolerance)
 best=A;bestcost=sum(c);order=tuple(sorted(A,key=lambda a:(c[a],a)));nodes=0
 def visit(P,k,cost):
  nonlocal best,bestcost,nodes
  nodes+=1
  if cost>=bestcost:return
  if loss(P)<=limit:best=tuple(sorted(P));bestcost=cost;return
  if k==len(order):return
  a=order[k];visit(P,k+1,cost);visit(P+(a,),k+1,cost+c[a])
 visit((),0,0.)
 return best,bestcost,limit,nodes

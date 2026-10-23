"""Budgeted deterministic lower-cost deletion/exchange refinement."""
def refine(P,costs,risk,limit,budget=200):
 P=tuple(sorted(P));used=0;trace=[];evaluated=[];scans=0
 assert risk(P)<=limit
 while used<budget:
  scans+=1;before=P;cost=sum(costs[a] for a in P);moves=[]
  for a in P:
   moves.append((costs[a],a,-1,tuple(v for v in P if v!=a)))
   for b in range(len(costs)):
    if b not in P and costs[b]<costs[a]:moves.append((costs[a]-costs[b],a,b,tuple(sorted((set(P)-{a})|{b}))))
  moves.sort(key=lambda v:(-v[0],v[1],v[2]));best=None;exhausted=True
  for saving,a,b,Q in moves:
   if used>=budget:exhausted=False;break
   loss=risk(Q);used+=1;r=dict(scan=scans,before=list(P),removed=a,added=b,cost=cost-saving,loss=loss,feasible=loss<=limit,subset=list(Q));evaluated.append(r)
   if loss<=limit:
    key=(cost-saving,loss,a,b)
    if best is None or key<best[0]:best=(key,Q,r)
  if best is None:reason='neighbourhood exhausted without feasible move' if exhausted else 'budget exhausted';break
  P=best[1];assert sum(costs[a] for a in P)<cost and risk(P)<=limit;trace.append(best[2])
  if used>=budget:reason='budget exhausted after accepted move';break
 else:reason='budget exhausted'
 return P,dict(evaluations=used,scans=scans,accepted=trace,candidates=evaluated,stop_reason=reason)

"""Mechanism checks: budgets, acceptance, restart and zero-cost endpoint."""
from refine import refine
costs=[5,3,1];table={(0,):.2,(1,):.15,(2,):.1,():.9}
def risk(P):return table.get(tuple(sorted(P)),.8)
P,log=refine([0],costs,risk,.2,budget=1)
assert P==(0,) and log['evaluations']==1 and len(log['accepted'])==0
P,log=refine([0],costs,risk,.2,budget=2)
assert P==(2,) and log['evaluations']==2 and len(log['accepted'])==1
P,log=refine([0],costs,risk,.2,budget=10)
assert P==(2,) and log['stop_reason']=='neighbourhood exhausted without feasible move'
assert all(r['loss']<=.2 for r in log['accepted'])
P,log=refine([],costs,lambda P:0.,0.,10)
assert P==() and log['evaluations']==0
# Best-loss tie at identical cost; restart permits a deletion only after exchange.
costs=[5,2,2,1];table={(0,3):.2,(1,3):.2,(2,3):.15,(2,):.1}
P,log=refine([0,3],costs,lambda P:table.get(tuple(sorted(P)),.9),.2,30)
assert P==(2,) and len(log['accepted'])==2
assert log['accepted'][0]['added']==2 and log['accepted'][1]['added']==-1
print('PASS: budget cap, deterministic best move, strict cost descent, restart deletion, empty endpoint')

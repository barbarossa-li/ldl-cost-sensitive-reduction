"""Descriptive actual-cost/audit-loss dominance; no new selection/evaluation."""
from pathlib import Path
import json,csv,hashlib
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent;source=root/'common_budget_audit/results.json';source_hash=hashlib.sha256(source.read_bytes()).hexdigest();s=json.loads(source.read_text());tol=1e-12;paired=[];cross=[];fronts=[]
def dominates(x,y):
 return x['spent']<=y['spent'] and x['audit_brier']<=y['audit_brier']+tol and (x['spent']<y['spent'] or x['audit_brier']<y['audit_brier']-tol)
def classify(p,g):
 if p['spent']==g['spent'] and abs(p['audit_brier']-g['audit_brier'])<=tol:return 'coincident'
 if dominates(g,p):return 'guidance dominates'
 if dominates(p,g):return 'prediction dominates'
 return 'tradeoff'
for name in ['Natural Scene','SBU_3DFE','Yeast_cold']:
 costs=json.loads((root/'stability/protocol.json').read_text())['costs'] if name=='Natural Scene' else json.loads((root/'cross_dataset'/name/'repeat_1_n576/split.json').read_text())['costs']
 for rep in [1,2,3]:
  q=[dict(v) for v in s if v['dataset']==name and v['repeat']==rep]
  job=root/'common_budget_audit'/name.replace(' ','_')/f'repeat_{rep}';checkpoint=json.loads((job/'selection_before_audit.json').read_text())
  for v in q:v['selected']=next(r['selected'] for r in checkpoint if r['budget']==v['budget'] and r['method']==v['method'])
  for b in [5,10,20]:
   p=next(v for v in q if v['budget']==b and v['method']=='Prediction ranking');g=next(v for v in q if v['budget']==b and v['method']=='Dependency guidance');paired.append(dict(dataset=name,repeat=rep,budget=b,prediction_cost=p['spent'],guidance_cost=g['spent'],prediction_loss=p['audit_brier'],guidance_loss=g['audit_brier'],guidance_minus_prediction_cost=g['spent']-p['spent'],guidance_minus_prediction_loss=g['audit_brier']-p['audit_brier'],classification=classify(p,g),guidance_strict_both=g['spent']<p['spent'] and g['audit_brier']<p['audit_brier']-tol,prediction_strict_both=p['spent']<g['spent'] and p['audit_brier']<g['audit_brier']-tol,identical_selected_subset=p['selected']==g['selected']))
  for p in [v for v in q if v['method']=='Prediction ranking']:
   for g in [v for v in q if v['method']=='Dependency guidance']:cross.append(dict(dataset=name,repeat=rep,prediction_cap=p['budget'],guidance_cap=g['budget'],prediction_cost=p['spent'],guidance_cost=g['spent'],prediction_loss=p['audit_brier'],guidance_loss=g['audit_brier'],classification=classify(p,g),guidance_strict_both=g['spent']<p['spent'] and g['audit_brier']<p['audit_brier']-tol))
  refs=[dict(dataset=name,repeat=rep,budget=None,method='Full',spent=sum(costs),audit_brier=q[0]['audit_full'],selected=list(range(len(costs)))),dict(dataset=name,repeat=rep,budget=None,method='Empty',spent=0,audit_brier=q[0]['audit_empty'],selected=[])]
  points=q+refs
  for v in points:
   dominators=[dict(method=w['method'],budget=w['budget'],spent=w['spent'],audit_brier=w['audit_brier']) for w in points if dominates(w,v)];fronts.append(dict(dataset=name,repeat=rep,method=v['method'],budget=v['budget'],spent=v['spent'],audit_brier=v['audit_brier'],nondominated=not bool(dominators),dominators=dominators,selected=v['selected']))
for name,data in [('paired',paired),('cross_budget',cross)]:
 with (out/(name+'.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
(out/'fronts.json').write_text(json.dumps(fronts,indent=2));summary=[]
for name in ['Natural Scene','SBU_3DFE','Yeast_cold']:
 q=[v for v in paired if v['dataset']==name];fq=[v for v in fronts if v['dataset']==name and v['method'] in ['Prediction ranking','Dependency guidance']];cq=[v for v in cross if v['dataset']==name];summary.append(dict(dataset=name,paired_comparisons=len(q),counts={key:sum(v['classification']==key for v in q) for key in ['guidance dominates','prediction dominates','tradeoff','coincident']},guidance_strict_both=sum(v['guidance_strict_both'] for v in q),prediction_strict_both=sum(v['prediction_strict_both'] for v in q),front_semantic_prediction=sum(v['nondominated'] and v['method']=='Prediction ranking' for v in fq),front_semantic_guidance=sum(v['nondominated'] and v['method']=='Dependency guidance' for v in fq),cross_budget_guidance_strict_both=sum(v['guidance_strict_both'] for v in cq),cross_budget_contexts_with_guidance_strict_both=sum(any(v['guidance_strict_both'] for v in cq if v['repeat']==rep) for rep in [1,2,3])))
(out/'summary.json').write_text(json.dumps(summary,indent=2));assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
# Meaningful logical checks of dominance, not statistical tests.
a=dict(spent=1,audit_brier=.1);b=dict(spent=2,audit_brier=.2);assert dominates(a,b) and not dominates(b,a) and not dominates(a,a);assert classify(a,dict(spent=2,audit_brier=.05))=='tradeoff'
(out/'validation.json').write_text(json.dumps(dict(paired_comparisons=27,cross_budget_comparisons=81,semantic_front_records=72,source_sha256=source_hash,source_unchanged=True,new_search=False,new_prediction=False,no_budget_selected_by_audit=True,cost_comparison='exact integer',loss_tolerance=tol,dominance_irreflexivity_asymmetry_tradeoff_checks=True,front_scope='Finite existing candidate set plus full/empty per same split; no global optimality',duplicates='Repeated budgets/identical subsets retained and must not count as independent evidence'),indent=2));print(json.dumps(summary,indent=2))

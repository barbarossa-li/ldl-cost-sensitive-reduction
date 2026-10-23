"""Training-only follow-up diagnostics prompted by the prespecified pilot failure."""
from pathlib import Path
import sys,json,csv
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from granulation_reference import TrainingGeometry,granulate
p=Path(__file__).resolve().parent;cfg=json.loads((p/'protocol.json').read_text())
X=np.load(p/'data/feature.npy');L=np.load(p/'data/label.npy');idx=cfg['train_indices'];x=X[idx];lab=L[idx];m=x.shape[1]
model=TrainingGeometry(x,lab,kinds=['numerical']*m)
def evaluate(P,smin=2,kappa=.1):
 d,k=model.subset(P);leaves,elig,g,h=granulate(d,lab,k,smin=smin,kappa=kappa)
 return g,sum(len(B) for B in elig)/len(lab),len(elig)
full=evaluate(range(m))[0];rows=[]
for a in range(m):
 g,c,e=evaluate([a]);rows.append({'attribute':a,'cost':cfg['costs'][a],'gamma':g,'coverage':c,'eligible_balls':e,'feasible_against_full':int(g>=full)})
with (p/'single_attribute_diagnostic.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={'training_full_gamma':full,'single_attribute_feasible_count':sum(r['feasible_against_full'] for r in rows),'total_attributes':m,'minimum_feasible_single_cost':min(r['cost'] for r in rows if r['feasible_against_full']),'global_optimal_cost_certificate':'Empty subset is infeasible; every nonempty subset costs at least 1; the adopted heuristic finds a feasible cost-1 subset. Therefore cost 1 is globally optimal for the CURRENT empirical criterion over all 294 features; this certifies cost, not prediction quality.'}
(p/'training_diagnosis.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
rng=np.random.default_rng(cfg['seed']+3);profiles=[]
for size in [1,2,4,8,16,32,64,128,294]:
 for repeat in range(5 if size<294 else 1):
  P=tuple(sorted(rng.choice(m,size=size,replace=False)));g,c,e=evaluate(P)
  profiles.append({'attributes':size,'repeat':repeat,'gamma':g,'coverage':c,'eligible_balls':e})
with (p/'cardinality_profile.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(profiles[0]));w.writeheader();w.writerows(profiles)
support=[]
for smin in [2,4,8]:
 for kappa in [.1,.25,.5]:
  g,c,e=evaluate(range(m),smin=smin,kappa=kappa);support.append({'smin':smin,'kappa':kappa,'gamma':g,'coverage':c,'eligible_balls':e})
with (p/'support_diagnostic.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(support[0]));w.writeheader();w.writerows(support)

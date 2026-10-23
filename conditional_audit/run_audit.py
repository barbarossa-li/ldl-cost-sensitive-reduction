"""Fixed saved prediction-only/guided subsets; all one-attribute toggles.
Conditional diagnostics, no new reduct or test evaluation.
"""
from pathlib import Path
import json,csv,sys
import numpy as np
from scipy.stats import spearmanr
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent;sys.path.insert(0,str(root/'prediction_interface'))
from directed_predictor import DirectedNeighborPredictor
names=['Natural Scene','SBU_3DFE','Yeast_cold'];methods=['Euclidean','Manhattan','Incoming directed'];bases=['Prediction only','Guided'];old=list(csv.DictReader(open(root/'neighbour_audit/results.csv')));assoc=list(csv.DictReader(open(root/'attribute_prediction_audit/attributes.csv')));rows=[];summary=[]
protocol={'scope':'One addition/deletion per attribute on two saved fixed bases per dataset; same three development splits and folds. Identical empty Yeast contexts duplicate semantically, not independent.','bases':bases,'learners':methods,'gain':'Add: base_loss-candidate_loss; delete: candidate_loss-base_loss, positive means deletion harms prediction','association':'Saved observed-minus-permutation mean W; descriptive correlation with additions','no_changes':['subsets','parameters','test usage','search rules'],'comparison_tolerance':1e-12,'note':'Positive addition gain increases acquisition cost and is not by itself a better reduct'}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
for name in names:
 folder=root/'stability/repeat_1_n192' if name=='Natural Scene' else root/'cross_dataset'/name/'repeat_1_n192';split=json.loads((folder/'split.json').read_text());ids=np.array(split['train_indices']);saved=json.loads((root/'neighbour_audit'/name.replace(' ','_')/'fixed_subsets.json').read_text());subsets={b:saved['selected'][b] for b in bases};costs=np.array(saved['costs'])
 if name=='Natural Scene':X=np.load(root/'pilot/data/feature.npy');L=np.load(root/'pilot/data/label.npy')
 else:X=np.load(root/'cross_dataset/data'/f'{name}_X.npy');L=np.load(root/'cross_dataset/data'/f'{name}_L.npy')
 n=len(ids);m=X.shape[1];ss={b:np.zeros((m,3)) for b in bases};basess={b:np.zeros(3) for b in bases};job=out/name.replace(' ','_');job.mkdir(exist_ok=True);provenance=[]
 for f,v in enumerate(split['fold_validation_positions']):
  val=np.array(v);fit=np.setdiff1d(np.arange(n),val);fi=ids[fit];vi=ids[val];assert set(fi).isdisjoint(vi) and set(fi).isdisjoint(split['test_indices']) and set(vi).isdisjoint(split['test_indices']);model=DirectedNeighborPredictor(X[fi],L[fi],kinds=['numerical']*m);truth=L[vi];p=L[fi].mean(0)
  provenance.append({'fold':f,'fit_ids':fi.tolist(),'validation_ids':vi.tolist(),'geometry_fit_n':model.geometry.n})
  z=np.array([model.numeric[a][2] for a in range(m)]).T;query=np.array([(X[vi,a]-model.numeric[a][0])/model.numeric[a][1] if model.numeric[a][1]>0 else np.zeros(len(val)) for a in range(m)]).T
  raw=np.abs(query[:,None,:]-z[None,:,:]).transpose(2,0,1);weight=np.array(model.geometry.weights);mat={'Euclidean':raw**2,'Manhattan':raw,'Incoming directed':raw*(.1+.9*weight[:,None,:])/m}
  def evalsubset(P,method):
   pred=L[fi][np.argsort(mat[method][list(P)].sum(0),axis=1,kind='stable')[:,:5]].mean(1) if P else np.tile(p,(len(val),1));return float(np.sum((truth-pred)**2))
  for b,P in subsets.items():
   assert np.max(np.abs(mat['Incoming directed'][P].sum(0)-model.incoming(X[vi],P)))<1e-12
   for j,method in enumerate(methods):basess[b][j]+=evalsubset(P,method)
   for a in range(m):
    C=sorted(set(P)^set([a]))
    for j,method in enumerate(methods):ss[b][a,j]+=evalsubset(C,method)
 (job/'fold_provenance.json').write_text(json.dumps(provenance,indent=2))
 for b,P in subsets.items():
  aa={int(r['attribute']):r for r in assoc if r['dataset']==name}
  for j,method in enumerate(methods):
   base=basess[b][j]/n;prev=next(r for r in old if r['dataset']==name and r['subset']==b and r['learner']==method);assert abs(base-float(prev['inner_brier']))<1e-12;ref=float(next(r['inner_brier'] for r in old if r['dataset']==name and r['subset']=='Full' and r['learner']==method));local=[]
   for a in range(m):
    deletion=a in P;candidate=ss[b][a,j]/n;gain=candidate-base if deletion else base-candidate;r={'dataset':name,'base':b,'learner':method,'base_attributes':len(P),'base_cost':int(sum(costs[P])),'base_brier':base,'base_passes_reference':base<=ref,'attribute':a,'operation':'delete' if deletion else 'add','attribute_cost':int(costs[a]),'candidate_cost':int(sum(costs[P])+(-costs[a] if deletion else costs[a])),'candidate_brier':candidate,'gain':gain,'full_reference':ref,'candidate_passes_reference':candidate<=ref,'excess_W':float(aa[a]['excess_W'])};rows.append(r);local.append(r)
   add=[r for r in local if r['operation']=='add'];delete=[r for r in local if r['operation']=='delete'];corr=float(spearmanr([r['excess_W'] for r in add],[r['gain'] for r in add]).statistic) if len(add)>1 else None
   q={'dataset':name,'base':b,'learner':method,'base_brier':base,'base_passes_reference':base<=ref,'add_candidates':len(add),'positive_add_gains':sum(r['gain']>1e-12 for r in add),'spearman_excess_vs_conditional_add_gain':corr,'mean_add_gain':float(np.mean([r['gain'] for r in add])),'deletions':len(delete),'deletion_preserves_reference':sum(r['candidate_passes_reference'] for r in delete),'deletion_improves_loss':sum(r['gain'] < -1e-12 for r in delete)};summary.append(q)
   if method=='Euclidean':assert q['deletion_preserves_reference']==0
  print('DONE',name,b,[q for q in summary if q['dataset']==name and q['base']==b],flush=True)
with (out/'candidates.csv').open('w',newline='') as file:w=csv.DictWriter(file,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=[{k:(v.item() if isinstance(v,np.generic) else v) for k,v in r.items()} for r in summary];(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'validation.json').write_text(json.dumps({'semantic_candidate_records':len(rows),'contexts':18,'base_loss_reproduction':'All 18 saved baselines within 1e-12','original_single_deletion_irreducibility':'Both saved Euclidean bases per dataset have no feasible single deletion','no_test_no_new_reduct':True,'parameters_changed':False,'fold_fitting':'Nine fitting-only geometry models'},indent=2))

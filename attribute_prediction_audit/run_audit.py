"""All single attributes on fixed folds: association versus validation gain.
No selection rerun, test evaluation or parameter changes.
"""
from pathlib import Path
import json,csv,sys
import numpy as np
from scipy.stats import spearmanr
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent;sys.path.insert(0,str(root/'prediction_interface'))
from directed_predictor import DirectedNeighborPredictor
names=['Natural Scene','SBU_3DFE','Yeast_cold'];perm=list(csv.DictReader(open(root/'permutation_audit/attributes.csv')));old=list(csv.DictReader(open(root/'neighbour_audit/results.csv')));rows=[];foldrows=[];summary=[]
protocol={'scope':'All 561 single numerical attributes, same three development splits, three fold-refitted learners. No reduction or outer testing.','association_score':'Mean observed W minus mean permutation W across the three fitting folds; primary descriptive rank score','predictive_gain':'Inner validation Brier of fold-training global mean minus single-attribute Brier','learners':['Euclidean','Manhattan','Incoming directed'],'constants':'Original scaling, k=5, beta=1,zeta=.1 and bandwidth rules','comparison':'Spearman coefficient only, no independent-attribute or significance claim; stable permutation group used for description only','limitations':'Marginal diagnosis cannot exclude feature interactions; previously used development folds are not fresh test evidence'}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2));tiechecks=0
for name in names:
 folder=root/'stability/repeat_1_n192' if name=='Natural Scene' else root/'cross_dataset'/name/'repeat_1_n192';split=json.loads((folder/'split.json').read_text());ids=np.array(split['train_indices']);job=out/name.replace(' ','_');job.mkdir(exist_ok=True)
 if name=='Natural Scene':X=np.load(root/'pilot/data/feature.npy');L=np.load(root/'pilot/data/label.npy')
 else:X=np.load(root/'cross_dataset/data'/f'{name}_X.npy');L=np.load(root/'cross_dataset/data'/f'{name}_L.npy')
 n=len(ids);m=X.shape[1];ssE=np.zeros(m);ssD=np.zeros(m);mean_ss=0.;weightdelta=np.zeros(m);provenance=[]
 for f,v in enumerate(split['fold_validation_positions']):
  val=np.array(v);fit=np.setdiff1d(np.arange(n),val);fi=ids[fit];vi=ids[val];assert set(fi).isdisjoint(vi) and set(fi).isdisjoint(split['test_indices']) and set(vi).isdisjoint(split['test_indices'])
  model=DirectedNeighborPredictor(X[fi],L[fi],kinds=['numerical']*m);truth=L[vi];meanpred=np.tile(L[fi].mean(0),(len(val),1));base=float(np.sum((truth-meanpred)**2));mean_ss+=base
  provenance.append({'fold':f,'fitting_ids':fi.tolist(),'validation_ids':vi.tolist(),'geometry_fit_n':model.geometry.n})
  for a in range(m):
   lo,span,z=model.numeric[a];q=(X[vi,a]-lo)/span if span>0 else np.zeros(len(val));distance=np.abs(q[:,None]-z[None,:]);e=np.argsort(distance**2,axis=1,kind='stable')[:,:5];mnn=np.argsort(distance,axis=1,kind='stable')[:,:5];assert np.array_equal(e,mnn);tiechecks+=1
   d=np.argsort(model.incoming(X[vi],[a]),axis=1,kind='stable')[:,:5];pe=L[fi][e].mean(1);pd=L[fi][d].mean(1);lossE=float(np.sum((truth-pe)**2));lossD=float(np.sum((truth-pd)**2));ssE[a]+=lossE;ssD[a]+=lossD;weightdelta[a]+=float(np.sum(model.geometry.weights[a][d].mean(1)-model.geometry.weights[a][e].mean(1)))
   foldrows.append({'dataset':name,'fold':f,'attribute':a,'mean_brier':base/len(val),'euclidean_brier':lossE/len(val),'manhattan_brier':lossE/len(val),'directed_brier':lossD/len(val),'validation_n':len(val)})
 (job/'fold_provenance.json').write_text(json.dumps(provenance,indent=2));baseline=mean_ss/n;fullE=float(next(r['inner_brier'] for r in old if r['dataset']==name and r['subset']=='Full' and r['learner']=='Euclidean'));fullD=float(next(r['inner_brier'] for r in old if r['dataset']==name and r['subset']=='Full' and r['learner']=='Incoming directed'));local=[]
 assert abs(baseline-float(next(r['inner_brier'] for r in old if r['dataset']==name and r['subset']=='Empty' and r['learner']=='Euclidean')))<1e-12
 for a in range(m):
  pp=[r for r in perm if r['dataset']==name and int(r['attribute'])==a];assert len(pp)==3
  r={'dataset':name,'attribute':a,'observed_mean_W':float(np.mean([float(r['observed_mean_W']) for r in pp])),'permutation_mean_W':float(np.mean([float(r['permutation_mean_W']) for r in pp])),'excess_W':float(np.mean([float(r['observed_minus_null_mean']) for r in pp])),'above_own_q95_all_folds':all(r['above_own_q95']=='True' for r in pp),'global_mean_brier':baseline,'euclidean_brier':ssE[a]/n,'manhattan_brier':ssE[a]/n,'directed_brier':ssD[a]/n,'euclidean_gain_over_mean':baseline-ssE[a]/n,'directed_gain_over_mean':baseline-ssD[a]/n,'directed_minus_euclidean_brier':(ssD[a]-ssE[a])/n,'euclidean_passes_full':ssE[a]/n<=fullE,'directed_passes_full':ssD[a]/n<=fullD,'directed_minus_euclidean_selected_source_weight':weightdelta[a]/n};rows.append(r);local.append(r)
 ex=np.array([r['excess_W'] for r in local]);ge=np.array([r['euclidean_gain_over_mean'] for r in local]);gd=np.array([r['directed_gain_over_mean'] for r in local]);strong=[r for r in local if r['above_own_q95_all_folds']];rest=[r for r in local if not r['above_own_q95_all_folds']]
 r={'dataset':name,'attributes':m,'mean_brier':baseline,'spearman_excess_vs_euclidean_gain':float(spearmanr(ex,ge).statistic),'spearman_excess_vs_directed_gain':float(spearmanr(ex,gd).statistic),'euclidean_improves_mean':int(np.sum(ge>0)),'directed_improves_mean':int(np.sum(gd>0)),'directed_better_than_euclidean':sum(r['directed_minus_euclidean_brier'] < -1e-12 for r in local),'directed_equal_euclidean':sum(abs(r['directed_minus_euclidean_brier']) <= 1e-12 for r in local),'euclidean_passes_full_count':sum(r['euclidean_passes_full'] for r in local),'directed_passes_full_count':sum(r['directed_passes_full'] for r in local),'stable_q95_count':len(strong),'stable_q95_euclidean_improves_mean':sum(r['euclidean_gain_over_mean']>0 for r in strong),'stable_q95_euclidean_gain_mean':float(np.mean([r['euclidean_gain_over_mean'] for r in strong])) if strong else None,'other_euclidean_gain_mean':float(np.mean([r['euclidean_gain_over_mean'] for r in rest])) if rest else None,'mean_directed_minus_euclidean_brier':float(np.mean([r['directed_minus_euclidean_brier'] for r in local])),'source_weight_difference_mean':float(np.mean([r['directed_minus_euclidean_selected_source_weight'] for r in local]))};r={k:(v.item() if isinstance(v,np.generic) else v) for k,v in r.items()};summary.append(r);print('DONE',json.dumps(r),flush=True)
 for f in ['Original joint']:
  prev=next(r for r in old if r['dataset']==name and r['subset']==f and r['learner']=='Euclidean')
  if int(prev['attributes'])==1:
   subset=json.loads((root/'neighbour_audit'/name.replace(' ','_')/'fixed_subsets.json').read_text())['selected'][f];assert abs(local[subset[0]]['euclidean_brier']-float(prev['inner_brier']))<1e-12
for fname,rr in [('attributes',rows),('folds',foldrows)]:
 with (out/(fname+'.csv')).open('w',newline='') as file:w=csv.DictWriter(file,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'validation.json').write_text(json.dumps({'attributes':len(rows),'fold_attribute_checks':tiechecks,'euclidean_manhattan_neighbour_equality':'All 1683 single-attribute fold comparisons exact','mean_reference_reproduction':'All three saved global-mean losses within 1e-12','previous_singleton_reproduction':'Yeast original single-attribute result within 1e-12','parameters_changed':False,'test_labels_used':False,'selection_rerun':False},indent=2))

"""Exact information-weight decomposition on nine saved fitting folds.
No changes to beta, bandwidth, normalizer, zeta, subsets or predictions.
"""
from pathlib import Path
import sys,json,csv
import numpy as np
from scipy.special import xlogy
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent;sys.path.insert(0,str(root))
from granulation_reference import TrainingGeometry
rows=[];folds=[];support_groups=[];max_identity=0.;max_jensen_excess=0.;max_shrink_excess=0.
protocol={'scope':'Decompose fitted weights in the same nine inner fitting folds, each 128 training objects; no validation/test labels used','parameters':'All original: beta=1, Gaussian bandwidth median positive pairwise scaled differences, original common KL normalizer, zeta=.1','measures':'Raw KL, support-shrunk KL, normalised W; kernel effective size, support, eta, Jensen-averaging baseline','interpretation':'Averaging retention is a descriptive convexity ratio, not a causal bandwidth effect or predictive accuracy metric','no_change':['subset selection','bandwidths','beta','zeta','normalizer','prediction mapping'],'unit':'KL in bits'}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
for name in ['Natural Scene','SBU_3DFE','Yeast_cold']:
 folder=root/'stability/repeat_1_n192' if name=='Natural Scene' else root/'cross_dataset'/name/'repeat_1_n192';split=json.loads((folder/'split.json').read_text());ids=np.array(split['train_indices'])
 if name=='Natural Scene':X=np.load(root/'pilot/data/feature.npy');L=np.load(root/'pilot/data/label.npy')
 else:X=np.load(root/'cross_dataset/data'/f'{name}_X.npy');L=np.load(root/'cross_dataset/data'/f'{name}_L.npy')
 for f,v in enumerate(split['fold_validation_positions']):
  fit=np.setdiff1d(np.arange(len(ids)),v);source=ids[fit];assert set(source).isdisjoint(ids[np.array(v)]) and set(source).isdisjoint(split['test_indices']);x=X[source];lab=L[source];n,m=x.shape;model=TrainingGeometry(x,lab,kinds=['numerical']*m)
  p=lab.mean(0);active=p>0;B=float(np.log2(1/p[active].min())) if active.sum()>1 else 0.
  def kl(r):return np.maximum(0.,np.sum(xlogy(r[:,active],r[:,active]/p[active]),axis=1)/np.log(2.))
  objectkl=kl(lab);v0=float(np.mean(np.sum((lab-p)**2,axis=1)));localrows=[]
  for a in range(m):
   z=(x[:,a]-x[:,a].min())/(x[:,a].max()-x[:,a].min()) if x[:,a].max()>x[:,a].min() else np.zeros(n)
   diff=np.abs(z[:,None]-z[None,:]);tau=model.bandwidths[a];K=np.exp(-diff**2/(2*tau**2));support=K.sum(1);t=support-1.;eta=t/(t+1.);raw=K@lab/support[:,None];regular=eta[:,None]*raw+(1-eta[:,None])*p
   rawkl=kl(raw);regkl=kl(regular);weighted_objectkl=K@objectkl/support;neff=support**2/(K**2).sum(1);W=regkl/B if B else np.zeros(n)
   for lo,hi,label in [(0,1,'other support <1'),(1,5,'1<=other support<5'),(5,20,'5<=other support<20'),(20,float('inf'),'other support>=20')]:
    mask=(t>=lo)&(t<hi);support_groups.append({'dataset':name,'fold':f,'attribute':a,'support_bin':label,'source_count':int(mask.sum()),'raw_kl_sum':float(rawkl[mask].sum()),'regularized_kl_sum':float(regkl[mask].sum()),'eta_rawkl_sum':float((eta[mask]*rawkl[mask]).sum())})
   err=float(np.max(np.abs(W-model.weights[a])));max_identity=max(max_identity,err);assert err<1e-12
   j=float(np.max(rawkl-weighted_objectkl));s=float(np.max(regkl-eta*rawkl));max_jensen_excess=max(max_jensen_excess,j);max_shrink_excess=max(max_shrink_excess,s);assert j<1e-12 and s<1e-12
   r={'dataset':name,'fold':f,'attribute':a,'fit_n':n,'constant':bool(x[:,a].max()==x[:,a].min()),'bandwidth':tau,'normalizer_bits':B,'object_kl_bits':float(objectkl.mean()),'v0':v0,'weighted_object_kl_bits':float(weighted_objectkl.mean()),'raw_profile_kl_bits':float(rawkl.mean()),'regularized_profile_kl_bits':float(regkl.mean()),'mean_information_weight':float(W.mean()),'mean_eta':float(eta.mean()),'mean_support':float(support.mean()),'mean_other_support_fraction':float(np.mean(t/(n-1))),'mean_effective_n':float(neff.mean()),'min_effective_n':float(neff.min()),'raw_profile_sq_global_difference':float(np.mean(np.sum((raw-p)**2,axis=1))),'regularized_sq_global_difference':float(np.mean(np.sum((regular-p)**2,axis=1))), 'shrink_retention':float(regkl.sum()/rawkl.sum()) if rawkl.sum()>1e-20 else None,'averaging_retention':float(rawkl.sum()/weighted_objectkl.sum()) if weighted_objectkl.sum()>1e-20 else None}
   rows.append(r);localrows.append(r)
  folds.append({'dataset':name,'fold':f,'fit_ids':source.tolist(),'fit_n':n,'m':m,'normalizer_bits':B,'global_label_profile':p.tolist(),'object_mean_kl_bits':float(objectkl.mean()),'v0':v0})
  print('DONE',name,f,'raw',np.mean([r['raw_profile_kl_bits'] for r in localrows]),'W',np.mean([r['mean_information_weight'] for r in localrows]),flush=True)
with (out/'support_bins.csv').open('w',newline='') as file:
 w=csv.DictWriter(file,fieldnames=list(support_groups[0]));w.writeheader();w.writerows(support_groups)
with (out/'attributes.csv').open('w',newline='') as file:w=csv.DictWriter(file,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'fold_provenance.json').write_text(json.dumps(folds,indent=2));summary=[]
for name in ['Natural Scene','SBU_3DFE','Yeast_cold']:
 rr=[r for r in rows if r['dataset']==name];ff=[f for f in folds if f['dataset']==name];sumraw=sum(r['raw_profile_kl_bits'] for r in rr);sumreg=sum(r['regularized_profile_kl_bits'] for r in rr);sumobj=sum(r['weighted_object_kl_bits'] for r in rr)
 sg=[r for r in support_groups if r['dataset']==name];rawtotal=sum(r['raw_kl_sum'] for r in sg);regtotal=sum(r['regularized_kl_sum'] for r in sg);low=[r for r in sg if r['support_bin']=='other support <1']
 q={'source_attribute_records':sum(r['source_count'] for r in sg),'raw_kl_weighted_eta':sum(r['eta_rawkl_sum'] for r in sg)/rawtotal if rawtotal else None,'raw_kl_mass_other_support_lt1':sum(r['raw_kl_sum'] for r in low)/rawtotal if rawtotal else None,'regularized_kl_mass_other_support_lt1':sum(r['regularized_kl_sum'] for r in low)/regtotal if regtotal else None,'source_fraction_other_support_lt1':sum(r['source_count'] for r in low)/sum(r['source_count'] for r in sg),'dataset':name,'fitting_folds':len(ff),'attributes_per_fold':ff[0]['m'],'attribute_fold_records':len(rr),'raw_kl_bits':float(np.mean([r['raw_profile_kl_bits'] for r in rr])),'regularized_kl_bits':float(np.mean([r['regularized_profile_kl_bits'] for r in rr])),'mean_weight':float(np.mean([r['mean_information_weight'] for r in rr])),'normalizer_bits_mean':float(np.mean([f['normalizer_bits'] for f in ff])),'object_kl_bits_mean':float(np.mean([f['object_mean_kl_bits'] for f in ff])),'weighted_object_kl_bits':float(np.mean([r['weighted_object_kl_bits'] for r in rr])),'averaging_retention':sumraw/sumobj if sumobj else None,'shrink_retention':sumreg/sumraw if sumraw else None,'eta_mean':float(np.mean([r['mean_eta'] for r in rr])),'effective_n_mean':float(np.mean([r['mean_effective_n'] for r in rr])),'other_support_fraction_mean':float(np.mean([r['mean_other_support_fraction'] for r in rr])),'weight_attribute_mean_q50':float(np.median([r['mean_information_weight'] for r in rr])),'weight_attribute_mean_q95':float(np.quantile([r['mean_information_weight'] for r in rr],.95)),'constant_attribute_fold_records':sum(r['constant'] for r in rr)}
 summary.append(q)
(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'validation.json').write_text(json.dumps({'attribute_fold_records':len(rows),'fitting_folds':len(folds),'mean_fit_n':128,'original_weight_max_error':max_identity,'jensen_max_positive_excess':max_jensen_excess,'shrink_convexity_max_positive_excess':max_shrink_excess,'validation_or_test_labels_used':False,'parameter_changes':False,'no_prediction_or_selection_rerun':True},indent=2));print('SUMMARY',json.dumps(summary,indent=2),flush=True)

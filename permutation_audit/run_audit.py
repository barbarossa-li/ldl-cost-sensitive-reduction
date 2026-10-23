"""Whole-row label permutations in nine fitting folds; frozen geometry and parameters.
This is conditional association diagnosis, not prediction or feature selection.
"""
from pathlib import Path
import sys,json,csv,time
import numpy as np
from scipy.special import xlogy
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent;sys.path.insert(0,str(root))
from granulation_reference import TrainingGeometry
BPERM=199;names=['Natural Scene','SBU_3DFE','Yeast_cold'];arows=[];frows=[];max_identity=0.;started=time.perf_counter()
protocol={'scope':'Whole training-label-vector permutations, only within each of nine saved 128-object fitting folds','permutations_per_fold':BPERM,'seed_rule':'20260930+1000*dataset_index+10*fold_index','joint_attributes':'Same permutation index used across all attributes within a fold','preserved':'Features, bandwidths, kernel/support/shrinkage, global label mean, active labels, KL normalizer, label-vector rows and row-internal probabilities','statistics':'Primary descriptive fold score: average W over sources and all attributes. Secondary: each attribute average W against its own permutation background','q95_scope':'Above own permutation 95th percentile is descriptive; not a multiplicity-adjusted significant-attribute count','p_scope':'One-sided finite-permutation rank (1+exceedances)/200 for each fold, conditional on label-row exchangeability under the null. No independence between overlapping folds or combined dataset p-value claimed.','unchanged':['model definitions','parameters','subsets','prediction learner'],'no_validation_test_or_selection':True}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
for di,name in enumerate(names):
 folder=root/'stability/repeat_1_n192' if name=='Natural Scene' else root/'cross_dataset'/name/'repeat_1_n192';split=json.loads((folder/'split.json').read_text());ids=np.array(split['train_indices']);job=out/name.replace(' ','_');job.mkdir(exist_ok=True)
 if name=='Natural Scene':X=np.load(root/'pilot/data/feature.npy');L=np.load(root/'pilot/data/label.npy')
 else:X=np.load(root/'cross_dataset/data'/f'{name}_X.npy');L=np.load(root/'cross_dataset/data'/f'{name}_L.npy')
 for f,v in enumerate(split['fold_validation_positions']):
  fit=np.setdiff1d(np.arange(len(ids)),v);source=ids[fit];assert set(source).isdisjoint(ids[np.array(v)]) and set(source).isdisjoint(split['test_indices']);x=X[source];lab=L[source];n,m=x.shape;model=TrainingGeometry(x,lab,kinds=['numerical']*m)
  p=lab.mean(0);active=p>0;normalizer=float(np.log2(1/p[active].min())) if active.sum()>1 else 0.;seed=20260930+1000*di+10*f;rng=np.random.default_rng(seed);perms=np.array([rng.permutation(n) for _ in range(BPERM)])
  assert all(np.array_equal(np.sort(perm),np.arange(n)) for perm in perms)
  labels=np.concatenate((lab[None,:,:],lab[perms]),axis=0);assert np.allclose(labels.mean(1),p,atol=1e-12);flat=labels.transpose(1,0,2).reshape(n,-1);scores=np.zeros((m,BPERM+1));rawscores=np.zeros_like(scores)
  for a in range(m):
   z=(x[:,a]-x[:,a].min())/(x[:,a].max()-x[:,a].min()) if x[:,a].max()>x[:,a].min() else np.zeros(n);diff=np.abs(z[:,None]-z[None,:]);K=np.exp(-diff**2/(2*model.bandwidths[a]**2));support=K.sum(1);eta=(support-1)/support
   raw=(K@flat).reshape(n,BPERM+1,lab.shape[1])/support[:,None,None];regular=eta[:,None,None]*raw+(1-eta[:,None,None])*p
   rawkl=np.maximum(0.,np.sum(xlogy(raw[:,:,active],raw[:,:,active]/p[active]),axis=2)/np.log(2));regkl=np.maximum(0.,np.sum(xlogy(regular[:,:,active],regular[:,:,active]/p[active]),axis=2)/np.log(2));W=regkl/normalizer if normalizer else np.zeros_like(regkl)
   err=float(np.max(np.abs(W[:,0]-model.weights[a])));max_identity=max(max_identity,err);assert err<1e-12;scores[a]=W.mean(0);rawscores[a]=rawkl.mean(0)
   observed=scores[a,0];null=scores[a,1:];nullmean=float(null.mean());q95=float(np.quantile(null,.95));rank=(1+np.sum(null>=observed))/(BPERM+1)
   arows.append({'dataset':name,'fold':f,'attribute':a,'observed_mean_W':observed,'permutation_mean_W':nullmean,'permutation_q95_W':q95,'observed_minus_null_mean':observed-nullmean,'observed_over_null_mean':observed/nullmean if nullmean>1e-20 else None,'above_own_q95':observed>q95,'conditional_tail_rank':float(rank),'observed_raw_kl_bits':rawscores[a,0],'permutation_raw_kl_bits_mean':float(rawscores[a,1:].mean())})
  fs=scores.mean(0);null=fs[1:];q={'dataset':name,'fold':f,'fit_n':n,'attributes':m,'seed':seed,'observed_mean_W':float(fs[0]),'permutation_mean_W':float(null.mean()),'permutation_q025_W':float(np.quantile(null,.025)),'permutation_q95_W':float(np.quantile(null,.95)),'permutation_q975_W':float(np.quantile(null,.975)),'observed_over_null_mean':float(fs[0]/null.mean()) if null.mean()>1e-20 else None,'conditional_tail_rank':float((1+np.sum(null>=fs[0]))/(BPERM+1)),'attribute_above_own_q95':int(np.sum(scores[:,0]>np.quantile(scores[:,1:],.95,axis=1))),'observed_raw_kl_bits':float(rawscores[:,0].mean()),'permutation_raw_kl_bits_mean':float(rawscores[:,1:].mean())};frows.append(q)
  np.savez_compressed(job/f'fold_{f}_scores.npz',fitting_ids=source,permutation_indices=perms,mean_W=scores,mean_raw_KL=rawscores,normalizer=normalizer,global_label_mean=p,bandwidths=model.bandwidths)
  print('DONE',name,f,'observed',q['observed_mean_W'],'null',q['permutation_mean_W'],'ratio',q['observed_over_null_mean'],'rank',q['conditional_tail_rank'],'q95attrs',q['attribute_above_own_q95'],flush=True)
for fname,rows in [('attributes',arows),('folds',frows)]:
 with (out/(fname+'.csv')).open('w',newline='') as file:w=csv.DictWriter(file,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=[]
for name in names:
 ff=[r for r in frows if r['dataset']==name];aa=[r for r in arows if r['dataset']==name];m=ff[0]['attributes'];obs=np.mean([r['observed_mean_W'] for r in ff]);null=np.mean([r['permutation_mean_W'] for r in ff]);stable=sum(all(next(r for r in aa if r['fold']==f and r['attribute']==a)['above_own_q95'] for f in range(3)) for a in range(m))
 summary.append({'dataset':name,'attributes':m,'observed_mean_W':float(obs),'permutation_mean_W':float(null),'observed_over_null_mean':float(obs/null),'mean_attribute_fraction_above_own_q95':float(np.mean([r['attribute_above_own_q95']/m for r in ff])),'attributes_above_own_q95_all_three_folds':stable,'fold_tail_ranks':[r['conditional_tail_rank'] for r in ff],'fold_q95_counts':[r['attribute_above_own_q95'] for r in ff],'observed_raw_kl_bits':float(np.mean([r['observed_raw_kl_bits'] for r in ff])),'permutation_raw_kl_bits_mean':float(np.mean([r['permutation_raw_kl_bits_mean'] for r in ff])),'scope':'Dataset averages are descriptive; no combined p-value'} )
(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'validation.json').write_text(json.dumps({'fitting_folds':9,'permutations_per_fold':BPERM,'attribute_fold_records':len(arows),'observed_weight_max_error':max_identity,'whole_label_vectors_preserved':True,'global_label_mean_preserved':True,'validation_test_labels_used':False,'selection_or_prediction_rerun':False,'parameters_changed':False,'seconds':time.perf_counter()-started},indent=2));print('SUMMARY',json.dumps(summary,indent=2),flush=True)

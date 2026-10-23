"""Frozen subsets, fold-refitted Euclidean/L1/incoming directed neighbours.
No subset selection, outer-test evaluation or hyperparameter changes.
"""
from pathlib import Path
import sys,json,csv,gc
import numpy as np
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'prediction_interface'))
from directed_predictor import DirectedNeighborPredictor
names=['Natural Scene','SBU_3DFE','Yeast_cold'];subsetnames=['Full','Original joint','Prediction only','Guided','Empty'];methods=['Euclidean','Manhattan','Incoming directed'];rows=[];foldrows=[];diagnostics=[]
protocol={'scope':'Same three development repeat-1 n192 splits; three fixed inner folds; no outer-test evaluation or selection rerun','subset_rule':'Full, saved original joint, saved prediction-only, saved guided, empty','learners':methods,'k':5,'directed_orientation':'Training source to query target','fit':'All source weights, bandwidths, profiles and scaling refit only on each inner fitting fold','parameters':{'beta':1,'zeta':.1},'tie':'Stable fitting-object order','reference':'Separate full-set validation loss per learner','caution':'Subsets were selected using the same development folds; this is not an unbiased generalization comparison'}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
for name in names:
 job=out/name.replace(' ','_');job.mkdir(exist_ok=True)
 folder=root/'stability/repeat_1_n192' if name=='Natural Scene' else root/'cross_dataset'/name/'repeat_1_n192'
 split=json.loads((folder/'split.json').read_text());outertrain=np.array(split['train_indices']);n=len(outertrain)
 if name=='Natural Scene':
  X=np.load(root/'pilot/data/feature.npy');L=np.load(root/'pilot/data/label.npy');costs=np.array(json.loads((root/'stability/protocol.json').read_text())['costs']);prior=json.loads((root/'baselines/repeat_1_n192/results.json').read_text())
  orig=next(r['selected'] for r in json.loads((folder/'results.json').read_text()) if r['method']=='Joint guard');pure=json.loads((root/'baselines/repeat_1_n192/selection_before_test.json').read_text())['Prediction-only']
 else:
  X=np.load(root/'cross_dataset/data'/f'{name}_X.npy');L=np.load(root/'cross_dataset/data'/f'{name}_L.npy');costs=np.array(split['costs']);prior=json.loads((folder/'results.json').read_text());saved=json.loads((folder/'selection_before_test.json').read_text())['selected'];orig=saved['Joint guard'];pure=saved['Prediction-only']
 guided=json.loads((root/'predictive_revision'/name.replace(' ','_')/'selection_before_test.json').read_text())['selected']
 subsets={'Full':list(range(X.shape[1])),'Original joint':orig,'Prediction only':pure,'Guided':guided,'Empty':[]}
 (job/'fixed_subsets.json').write_text(json.dumps({'selected':subsets,'costs':costs.tolist(),'source_split':str(folder.relative_to(root))},indent=2))
 predictions={(s,m):np.zeros((n,L.shape[1])) for s in subsetnames for m in methods};pools={s:[] for s in subsetnames};provenance=[]
 for f,v in enumerate(split['fold_validation_positions']):
  val=np.array(v);fit=np.setdiff1d(np.arange(n),val);fitids=outertrain[fit];valids=outertrain[val]
  assert set(fitids).isdisjoint(valids) and set(fitids).isdisjoint(split['test_indices']) and set(valids).isdisjoint(split['test_indices'])
  model=DirectedNeighborPredictor(X[fitids],L[fitids],kinds=['numerical']*X.shape[1],k=5);truth=L[valids]
  provenance.append({'fold':f,'fit_ids':fitids.tolist(),'validation_ids':valids.tolist(),'geometry_fit_n':model.geometry.n,'fit_n':len(fit),'validation_n':len(val)})
  x=X[fitids];query=X[valids];lo=x.min(0);span=x.max(0)-lo;span[span==0]=1.;a=(x-lo)/span;b=(query-lo)/span
  for s,P in subsets.items():
   if not P:
    pred=np.tile(L[fitids].mean(0),(len(val),1));
    for m in methods:predictions[s,m][val]=pred;foldrows.append({'dataset':name,'fold':f,'subset':s,'learner':m,'validation_n':len(val),'brier':float(np.mean(np.sum((truth-pred)**2,axis=1)))})
    continue
   raw=np.abs(b[:,None,P]-a[None,:,P]);l1=raw.sum(2);eu=(raw**2).sum(2);directed=model.incoming(query,P)
   distances={'Euclidean':eu,'Manhattan':l1,'Incoming directed':directed};nn={m:np.argsort(d,axis=1,kind='stable')[:,:5] for m,d in distances.items()}
   for m in methods:
    pred=L[fitids][nn[m]].mean(1);assert np.all(pred>=0) and np.allclose(pred.sum(1),1,atol=1e-12);predictions[s,m][val]=pred
    foldrows.append({'dataset':name,'fold':f,'subset':s,'learner':m,'validation_n':len(val),'brier':float(np.mean(np.sum((truth-pred)**2,axis=1)))})
   coef=.1+.9*np.array([model.geometry.weights[a] for a in P]).mean(0)
   varying=[j for j,a in enumerate(P) if model.numeric[a][1]>0]
   geometric=.1*raw[:,:,varying].sum(2)/model.m
   information=directed-geometric
   assert information.min()>=-1e-12
   for i,pos in enumerate(val):
    E=set(nn['Euclidean'][i]);M=set(nn['Manhattan'][i]);D=set(nn['Incoming directed'][i]);directidx=nn['Incoming directed'][i];midx=nn['Manhattan'][i]
    rec={'fold':f,'validation_position':int(pos),'euclidean_directed_jaccard':len(E&D)/len(E|D),'manhattan_directed_jaccard':len(M&D)/len(M|D),'euclidean_manhattan_jaccard':len(E&M)/len(E|M),'directed_manhattan_same_set':D==M,'information_share_all_sources':float(information[i].sum()/directed[i].sum()) if directed[i].sum()>0 else 0.,'directed_mean_source_coefficient':float(coef[directidx].mean()),'manhattan_mean_source_coefficient':float(coef[midx].mean()),'directed_raw_l1_per_attribute':float(l1[i,directidx].mean()/len(P)),'manhattan_raw_l1_per_attribute':float(l1[i,midx].mean()/len(P))}
    assert rec['directed_raw_l1_per_attribute']>=rec['manhattan_raw_l1_per_attribute']-1e-12;pools[s].append(rec)
   np.savez_compressed(job/f'fold_{f}_{s.replace(" ","_")}_neighbours.npz',validation_positions=val,euclidean=fitids[nn['Euclidean']],manhattan=fitids[nn['Manhattan']],directed=fitids[nn['Incoming directed']])
  del model;gc.collect()
 (job/'fold_provenance.json').write_text(json.dumps(provenance,indent=2))
 np.savez_compressed(job/'out_of_fold_predictions.npz',**{s.replace(' ','_')+'__'+m.replace(' ','_'):pred for (s,m),pred in predictions.items()})
 local=[]
 for s,P in subsets.items():
  for m in methods:
   loss=float(np.mean(np.sum((L[outertrain]-predictions[s,m])**2,axis=1)));ref=float(np.mean(np.sum((L[outertrain]-predictions['Full',m])**2,axis=1)))
   rec={'dataset':name,'subset':s,'learner':m,'attributes':len(P),'cost':int(sum(costs[P])),'inner_brier':loss,'full_reference':ref,'passes_own_reference':loss<=ref,'relative_to_euclidean':loss/float(np.mean(np.sum((L[outertrain]-predictions[s,'Euclidean'])**2,axis=1)))-1};rows.append(rec);local.append(rec)
   oldmethod={'Full':'All attributes','Original joint':'Joint guard','Prediction only':'Prediction-only'}.get(s)
   if m=='Euclidean' and oldmethod:old=next(r for r in prior if r['method']==oldmethod);assert abs(loss-old['inner_brier'])<1e-12
   if m=='Euclidean' and s=='Guided':old=json.loads((root/'predictive_revision'/name.replace(' ','_')/'selection_before_test.json').read_text());assert abs(loss-old['inner_brier'])<1e-12
  if pools[s]:
   rr=pools[s];diag={'dataset':name,'subset':s,'queries':len(rr)}
   for k in rr[0]:
    if k not in ['fold','validation_position']:diag[k]=float(np.mean([r[k] for r in rr]))
   diagnostics.append(diag)
 (job/'results.json').write_text(json.dumps(local,indent=2));(job/'neighbour_diagnostics.json').write_text(json.dumps(pools,indent=2));print('DONE',name,[(r['subset'],r['learner'],round(r['inner_brier'],6)) for r in local],flush=True)
for fname,rr in [('results',rows),('fold_results',foldrows),('neighbour_summary',diagnostics)]:
 with (out/(fname+'.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
(out/'validation.json').write_text(json.dumps({'models':45,'fold_records':135,'outer_test_used':False,'selection_rerun':False,'euclidean_reproduction':'All fixed existing subset references within 1e-12','fold_fitting':'Nine geometry fits use only corresponding inner training IDs','comparison_scope':'Descriptive fixed-subset development diagnostic; no fresh test or unbiased generalization claim'},indent=2))

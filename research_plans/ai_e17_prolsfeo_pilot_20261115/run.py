from pathlib import Path
import sys,os,json,time
from reconstruction import *
ROOT=Path(__file__).resolve().parent

def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--parent-source',type=Path,required=True);ap.add_argument('--parent-run',type=Path,required=True);ap.add_argument('--known-data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 for name,h in json.loads((ROOT/'MANIFEST.json').read_text()).items():assert sha(ROOT/name)==h
 protocol=json.loads((ROOT/'PROTOCOL.json').read_text());assert sha(a.parent_source/'PROTOCOL.json')==protocol['parent_protocol_sha256'];sys.path.insert(0,str(a.parent_source));import runner as r;parent=r.verify();records={}
 frozen=json.loads((a.parent_run/'e11/ALL_PREDICTIONS_FROZEN.json').read_text())
 for family,fi in [('Music',5),('Painting',6),('SJAFFE',8)]:
  f=parent['stages']['e11']['families'][fi];assert f['family']==family;name=family+'/fold_0/record_PRIVATE.json';oldfile=a.parent_run/'e11'/name;assert sha(oldfile)==frozen['records'][name];old=json.loads(oldfile.read_text());X,Y=r.checked_data(a.known_data,f);fit=np.array(old['fit_indices']);held=np.array(old['validation_indices']);case=a.output/family;case.mkdir(parents=True,exist_ok=True);recordfile=case/'record_PRIVATE.json'
  if recordfile.exists():
   rec=json.loads(recordfile.read_text());assert rec['protocol_sha256']==sha(ROOT/'PROTOCOL.json') and rec['predictions_sha256']==sha(case/'predictions_PRIVATE.npz');records[str(recordfile.relative_to(a.output))]=sha(recordfile);continue
  bits,folds,detail=optimize(X[fit],Y[fit],202611170000+fi,progress=case/'progress_PRIVATE.json');pred={};state={}
  if detail['best_inner_kl'] is not None:
   state=deploy_state(X[fit],Y[fit],bits);v=predict(state,X[held]);r.valid(v,len(held),Y.shape[1]);pred['ProLSFEO_reconstruction']=v;pred['ProLSFEO_reconstruction_halfprior']=.5*v+.5*Y[fit].mean(0)
  A,Q,_,_=scale(X[fit],X[held]);aa=normalize(component_predict(A,Y[fit],Q,np.ones((Y.shape[1],X.shape[1]),bool)));pred['AA4_stable']=aa;pred['AA4_stable_halfprior']=.5*aa+.5*Y[fit].mean(0)
  np.savez_compressed(case/'predictions_PRIVATE.npz',**pred);np.savez_compressed(case/'inference_state_PRIVATE.npz',**state);np.savez_compressed(case/'genome_PRIVATE.npz',bits=bits)
  save(recordfile,{'family':family,'fold':0,'fit_indices':fit.tolist(),'validation_indices':held.tolist(),'inner_folds':[x.tolist() for x in folds],'protocol_sha256':sha(ROOT/'PROTOCOL.json'),'source_manifest_sha256':sha(ROOT/'MANIFEST.json'),'source_record_sha256':sha(oldfile),'predictions_sha256':sha(case/'predictions_PRIVATE.npz'),'states_sha256':sha(case/'inference_state_PRIVATE.npz'),'genome_sha256':sha(case/'genome_PRIVATE.npz'),'selected_prototype_count':int(bits[:len(fit)].sum()),'selected_label_feature_counts':bits[len(fit):].reshape(Y.shape[1],X.shape[1]).sum(1).tolist(),'union_feature_count':int(len(state.get('P',[]))),'numeric_array_bytes':sum(x.nbytes for x in state.values()),'details':detail});records[str(recordfile.relative_to(a.output))]=sha(recordfile);print('FROZEN',family,detail['status'],flush=True)
 save(a.output/'ALL_PREDICTIONS_FROZEN.json',{'records':records,'protocol_sha256':sha(ROOT/'PROTOCOL.json'),'source_manifest_sha256':sha(ROOT/'MANIFEST.json'),'restriction':'Three consulted development cases, single seeds; incomplete searches must be reported as incomplete diagnostics, never full paper-setting results.'})
if __name__=='__main__':main()

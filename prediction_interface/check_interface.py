from pathlib import Path
import json
import numpy as np
from directed_predictor import DirectedNeighborPredictor
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent
checks=[];maxerror=0.
# Exact orientation check on original training values for numeric, categorical,
# and mixed data, over all subsets. Also includes a constant feature.
for kinds,X in [(['numerical']*3,np.array([[0,2,5],[1,1,5],[2,3,5],[3,0,5]],object)),(['categorical']*3,np.array([['a','u','c'],['a','v','c'],['b','u','c'],['b','v','c']],object)),(['numerical','categorical','numerical'],np.array([[0,'a',5],[1,'a',5],[2,'b',5],[3,'b',5]],object))]:
 L=np.array([[.8,.2,0],[.7,.3,0],[.2,.8,0],[.1,.9,0]])
 model=DirectedNeighborPredictor(X,L,kinds=kinds,k=2)
 for mask in range(8):
  P=tuple(a for a in range(3) if mask>>a&1);actual=model.incoming(X,P).T;expected=model.geometry.directed(P);err=float(np.max(np.abs(actual-expected)));maxerror=max(maxerror,err);assert err<1e-12
  pred=model.predict(X,P);assert np.all(pred>=0) and np.allclose(pred.sum(1),1)
checks.append('24 all-subset training-source orientation checks agree with original directed geometry; probability vectors valid')
# New categories use training-global profile; querying does not update fitted weights.
X=np.array([[0,'a'],[1,'a'],[2,'b'],[3,'b']],object);L=np.array([[.8,.2],[.7,.3],[.2,.8],[.1,.9]])
model=DirectedNeighborPredictor(X,L,kinds=['numerical','categorical'],k=2);weights=[w.copy() for w in model.geometry.weights]
Q=np.array([[4,'unseen'],[1.5,'a']],object);batched=model.predict(Q,[0,1]);single=np.vstack([model.predict(Q[i:i+1],[0,1]) for i in range(len(Q))]);assert np.allclose(batched,single)
assert all(np.array_equal(w,b) for w,b in zip(weights,model.geometry.weights));assert np.allclose(model.predict(Q,[]),np.tile(L.mean(0),(2,1)))
checks.append('Unseen categories, numeric extrapolation, batching invariance, frozen weights and empty-subset global mean verified')
# Same feature-space labels are never a predict input; stable ties follow training order.
T=np.zeros((5,1));Y=np.array([[.9,.1],[.8,.2],[.2,.8],[.1,.9],[.5,.5]])
model=DirectedNeighborPredictor(T,Y,kinds=['numerical'],k=2);assert np.allclose(model.predict([[0]],[0]),Y[:2].mean(0))
checks.append('Deterministic training-order tie handling verified')
# One real saved inner fold: fit ONLY its training IDs and label vectors.
# A single fold probes the interface; no new selection or test evaluation.
split=json.loads((root/'stability/repeat_1_n192/split.json').read_text());train=np.array(split['train_indices']);val=np.array(split['fold_validation_positions'][0]);fit=np.setdiff1d(np.arange(192),val)
X=np.load(root/'pilot/data/feature.npy');L=np.load(root/'pilot/data/label.npy');source=train[fit];query=train[val]
model=DirectedNeighborPredictor(X[source],L[source],kinds=['numerical']*X.shape[1]);P=[11,86,98,128,166,281];pred=model.predict(X[query],P)
assert set(source).isdisjoint(query) and model.geometry.n==128 and pred.shape==(64,9)
assert np.all(pred>=0) and np.allclose(pred.sum(1),1,atol=1e-12)
# Altering held-out labels cannot affect a prediction that takes features only.
changed=L.copy();changed[query]=np.roll(changed[query],1,axis=1);refitted=DirectedNeighborPredictor(X[source],changed[source],kinds=['numerical']*X.shape[1]);again=refitted.predict(X[query],P);assert np.array_equal(pred,again)
assert all(np.array_equal(a,b) for a,b in zip(model.geometry.weights,refitted.geometry.weights))
checks.append('Real Natural Scene inner-fold interface: 128 fitting objects / 64 validation objects, disjoint; held-out labels are not a prediction input')
result={'checks':checks,'training_orientation_max_error':maxerror,'real_fold':{'dataset':'Natural Scene','saved_split':'repeat_1_n192','fold':0,'fit_n':128,'validation_n':64,'all_label_geometry_fit_n':model.geometry.n,'selected_subset':P,'scope':'Interface/provenance check only; not a loss comparison, selection rerun or benchmark'}}
(out/'interface_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

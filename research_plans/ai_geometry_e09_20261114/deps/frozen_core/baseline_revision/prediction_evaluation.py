"""Fixed five-neighbor distribution predictor and declared external metrics."""
import numpy as np
from scipy.spatial.distance import cdist


def predict_five_neighbors(x, labels, query, subset):
    x, labels, query = np.asarray(x,float),np.asarray(labels,float),np.asarray(query,float)
    if len(x)<5: raise ValueError('Five training objects required')
    subset=list(subset)
    if not subset:
        return np.tile(labels.mean(0),(len(query),1))
    lo=x.min(0);span=x.max(0)-lo;span[span==0]=1
    train=(x-lo)/span;test=(query-lo)/span
    d=cdist(test[:,subset],train[:,subset],metric='sqeuclidean')
    neighbors=np.argsort(d,axis=1,kind='stable')[:,:5]
    pred=labels[neighbors].mean(1)
    return pred/pred.sum(1,keepdims=True)


def metrics(truth, prediction):
    truth,prediction=np.asarray(truth,float),np.asarray(prediction,float)
    for a in (truth,prediction):
        if a.ndim!=2 or not np.isfinite(a).all() or np.any(a<0) or not np.allclose(a.sum(1),1,rtol=0,atol=1e-10):
            raise ValueError('Expected probability distributions')
    if truth.shape!=prediction.shape:raise ValueError('Shape mismatch')
    safe=np.maximum(prediction,1e-12);safe/=safe.sum(1,keepdims=True)
    middle=(truth+prediction)/2
    def relative(a,b):
        out=np.zeros_like(a);positive=a>0
        out[positive]=a[positive]*np.log(a[positive]/np.maximum(b[positive],1e-300))
        return out.sum(1)
    return dict(brier=float(np.mean(np.sum((truth-prediction)**2,axis=1))),
        tv=float(np.mean(.5*np.abs(truth-prediction).sum(1))),
        kl=float(np.mean(relative(truth,safe))),
        js=float(np.mean(.5*(relative(truth,middle)+relative(prediction,middle)))))

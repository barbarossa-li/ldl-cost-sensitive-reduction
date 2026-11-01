"""Exact-order caching/vectorization for the radius-amended GBFRS implementation.
No parameter-grid, partition, ordering, tie, quota, or metric changes.
"""
import ast,inspect
import numpy as np
from scipy.spatial.distance import cdist
VERSION='GBFRS-exact-cache-v1'
def install(repair_counter=None):
 from baseline_revision import gbfrs_reconstruction as impl
 original_lower=impl.lower;original_local=impl.local_consistency
 state=dict(lower_s=None,local_partition=None,lower_preparations=0,local_preparations=0,local_calls=0,lower_calls=0,roundoff_repairs=0,duplicate_score_reuses=0)
 def lower(r,s,alpha,exclude_self=False):
  state['lower_calls']+=1
  if state['lower_s'] is not s or state.get('alpha')!=alpha or state.get('exclude_self')!=exclude_self:
   groups={};tdmask=np.zeros(s.shape,bool)
   for i in range(len(s)):
    mask=np.ones(s.shape[1],bool)
    if exclude_self:mask[i]=False
    pd=np.flatnonzero(mask&(s[i]>0)&(s[i]<=alpha)&(s[i]<1));td=np.flatnonzero(mask&(s[i]==0));k=min(max(len(pd),1),len(td));tdmask[i,td]=True
    groups.setdefault((len(pd),k),[]).append((i,pd))
   prepared=[]
   for (npd,k),items in groups.items():
    ids=np.array([i for i,_ in items],int);pds=np.array([pd for _,pd in items],int).reshape(len(items),npd)
    prepared.append((ids,pds,k))
   state.update(lower_s=s,alpha=alpha,exclude_self=exclude_self,groups=prepared,tdmask=tdmask,lower_preparations=state['lower_preparations']+1)
  # Sorting all columns with +inf padding preserves ascending-index stable ties
  # within exactly the original td subset. Only the first k valid entries are read.
  nearest=np.argsort(np.where(state['tdmask'],-r,np.inf),axis=1,kind='stable')
  result=np.zeros(len(r))
  for ids,pds,k in state['groups']:
   if pds.shape[1]+k==0:continue
   chosen=np.concatenate([pds,nearest[ids,:k]],axis=1)
   values=(1-r[ids[:,None],chosen])*(1-s[ids[:,None],chosen])
   result[ids]=values.mean(axis=1)
  return result
 def local_consistency(x,labels,subset,partition,mode):
  state['local_calls']+=1
  if mode not in ['literal_neighborhood','purity_shortcut']:raise ValueError('Unknown local interpretation')
  prefix=tuple(subset[:-1]);a=subset[-1]
  if state['local_partition'] is not partition or state.get('local_x') is not x or state.get('local_labels') is not labels or state.get('prefix')!=prefix:
   cached=[]
   for b in partition:
    # Both original z[b] and x[b] are row-advanced C-order copies; their
    # axis-0 reductions visit each column's members in the same order.
    center=x[b].mean(0)
    diff=x[:,prefix]-center[list(prefix)]
    squared=np.add.reduce(diff*diff,axis=1)
    cached.append((b,center,squared))
   state.update(local_partition=partition,local_x=x,local_labels=labels,prefix=prefix,balls=cached,
    agreement=labels.shape[1]-cdist(labels,labels,'cityblock'),purity_value=None,local_preparations=state['local_preparations']+1)
  if mode=='purity_shortcut':
   if state['purity_value'] is None:state['purity_value']=float(np.mean([impl.purity(labels[b]) for b in partition]))
   return state['purity_value']
  total=0.;agreement=state['agreement']
  for b,center,squared in state['balls']:
   # Original F-order norm reduces selected columns from left to right.
   # Append the new column after that same prefix reduction, then sqrt.
   last=x[:,a]-center[a];distance=np.sqrt(squared+last*last)
   radius=float(distance[b].mean());neighbors=np.flatnonzero(distance<=radius)
   if not len(neighbors):
    minimum=float(distance[b].min())
    if not np.isfinite(distance).all() or not np.isfinite(radius) or not radius<minimum:raise AssertionError('Empty radius neighborhood is not the recognized roundoff boundary')
    radius=minimum;neighbors=np.flatnonzero(distance<=radius);state['roundoff_repairs']+=1
    if repair_counter is not None:repair_counter[0]+=1
   total+=float(agreement[np.ix_(b,neighbors)].mean(axis=1).sum())
  return total/len(partition)
 impl.lower=lower;impl.local_consistency=local_consistency
 # Transform only repeated candidate evaluation; all attributes remain in traces.
 source=inspect.getsource(impl.rank);tree=ast.parse(source);fn=tree.body[0]
 fn.args.args.append(ast.arg(arg='prefix_limit'));fn.args.defaults.append(ast.Constant(value=None))
 outer=next(n for n in fn.body if isinstance(n,ast.While))
 outer.test=ast.parse('remaining and (prefix_limit is None or len(selected)<prefix_limit)',mode='eval').body
 candidates=next(n for n in outer.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='a')
 candidates.body=ast.parse("key=feature_keys[a]\nif key in candidate_cache:\n _optimization_state['duplicate_score_reuses']+=1\n scores.append(candidate_cache[key][0])\n if stage=='ball':locals_.append(candidate_cache[key][1])\n continue").body+candidates.body+ast.parse("candidate_cache[key]=(scores[-1],locals_[-1] if stage=='ball' else None)").body
 outer.body.insert(outer.body.index(candidates),ast.parse('candidate_cache={}').body[0])
 pos=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='delta' for t in n.targets))
 fn.body.insert(pos,ast.parse('feature_keys=[x[:,a].tobytes() for a in range(m)]').body[0])
 ast.fix_missing_locations(tree);namespace=impl.rank.__globals__.copy()
 namespace['lower']=lower;namespace['local_consistency']=local_consistency;namespace['_optimization_state']=state
 exec(compile(tree,'<'+VERSION+'>','exec'),namespace);compiled=namespace['rank']
 def full_rank(features,labels,alpha,beta,local_mode='literal_neighborhood'):
  return compiled(features,labels,alpha,beta,local_mode,None)
 def prefix_rank(features,labels,alpha,beta,limit,local_mode='literal_neighborhood'):
  m=np.asarray(features).shape[1]
  if not isinstance(limit,int) or not 1<=limit<=m:raise ValueError('Invalid prefix limit')
  value=compiled(features,labels,alpha,beta,local_mode,limit)
  value['execution']='prefix truncation of frozen full-ranking loop'
  return value
 return dict(version=VERSION,state=state,original_lower=original_lower,original_local=original_local,optimized_lower=lower,optimized_local=local_consistency,full_rank=full_rank,prefix_rank=prefix_rank)

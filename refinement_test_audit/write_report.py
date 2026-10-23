from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent;s=json.loads((out/'results.json').read_text());methods=['Full','Empty','Prediction only before','Prediction only after','Guided before','Guided after'];names=['Natural Scene','SBU_3DFE','Yeast_cold'];summary=[];pairs=[]
for name in names:
 for method in methods:
  q=[r for r in s if r['dataset']==name and r['method']==method];v=dict(dataset=name,method=method)
  for k in ['cost','test_brier','test_tv','test_kl','test_js']:v[k+'_mean']=float(np.mean([r[k] for r in q]));v[k+'_sd']=float(np.std([r[k] for r in q],ddof=1))
  full=[next(r['test_brier'] for r in s if r['dataset']==name and r['repeat']==rep and r['method']=='Full') for rep in range(1,6)];v['brier_better_than_full']=sum(r['test_brier']<f-1e-12 for r,f in zip(q,full));v['brier_equal_full']=sum(abs(r['test_brier']-f)<=1e-12 for r,f in zip(q,full));summary.append(v)
 for method in ['Prediction only','Guided']:
  delta=[]
  for rep in range(1,6):
   b=next(r for r in s if r['dataset']==name and r['repeat']==rep and r['method']==method+' before');a=next(r for r in s if r['dataset']==name and r['repeat']==rep and r['method']==method+' after');delta.append(a['test_brier']-b['test_brier'])
  pairs.append(dict(dataset=name,method=method,mean_after_minus_before=float(np.mean(delta)),better=sum(d < -1e-12 for d in delta),equal=sum(abs(d)<=1e-12 for d in delta),worse=sum(d>1e-12 for d in delta),deltas=delta))
(out/'summary.json').write_text(json.dumps(dict(means=summary,refinement_changes=pairs),indent=2))
fig,axes=plt.subplots(1,3,figsize=(11,3.7),layout='constrained');labels=['Full','Empty','P before','P after','G before','G after'];colors=['#777777','#aaaaaa','#76abc3','#226c93','#dcb18f','#bb6138']
for ax,name in zip(axes,names):
 vals=[r['test_brier_mean'] for r in summary if r['dataset']==name];ax.bar(range(6),vals,color=colors,width=.7)
 for j,method in enumerate(methods):
  q=[r['test_brier'] for r in s if r['dataset']==name and r['method']==method];ax.scatter(j+np.linspace(-.14,.14,5),q,color='#202020',s=13,alpha=.65)
 ax.set_title(name);ax.set_xticks(range(6),labels,rotation=45,ha='right',fontsize=8);ax.set_ylabel('Outer-test Brier (lower is better)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('Bars: five-repeat means | Points: reused saved-test evaluations',fontsize=10);fig.savefig(out/'outer_prediction.pdf');fig.savefig(out/'outer_prediction.png',dpi=180);plt.close(fig)
lines=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Frozen pipeline on reused saved outer tests: five-repeat means. All prediction measures are lower-is-better.}',r'\resizebox{\linewidth}{!}{%',r'\begin{tabular}{llrrrrr}',r'\hline',r'Dataset & Method & Cost & Brier & TV & KL & JS\\',r'\hline']
short={'Full':'Full','Empty':'Empty','Prediction only before':'P before','Prediction only after':'P after','Guided before':'G before','Guided after':'G after'}
for r in summary:lines.append(f"{r['dataset'].replace('_',chr(92)+'_')} & {short[r['method']]} & {r['cost_mean']:.2f} & {r['test_brier_mean']:.6f} & {r['test_tv_mean']:.6f} & {r['test_kl_mean']:.6f} & {r['test_js_mean']:.6f}\\\\")
lines += [r'\hline',r'\end{tabular}}',r'\end{table}'];(out/'outer_table.tex').write_text('\n'.join(lines)+'\n')
header='| 数据集 | 方法 | 平均成本 | Brier | TV | KL | JS |\n|---|---|---:|---:|---:|---:|---:|\n';body=''.join(f"| {r['dataset']} | {r['method']} | {r['cost_mean']:.2f} | {r['test_brier_mean']:.6f} | {r['test_tv_mean']:.6f} | {r['test_kl_mean']:.6f} | {r['test_js_mean']:.6f} |\n" for r in summary)
changes='\n'.join(f"- {r['dataset']} {r['method']}：精修后Brier改善{r['better']}次，相同{r['equal']}次，变差{r['worse']}次。" for r in pairs)
(out/'OUTER_REPORT.md').write_text('# 冻结完整流程的外部预测诊断\n\n'+header+body+'\n'+changes+'\n\n全部选集与来源哈希在预测前保存，所有训练与测试对象分离。评估90个语义记录（包含重复子集），不重选、不改规则。Brier为主要指标，TV/KL/JS为辅助；自然对数，预测按既有协议截断1e-12后归一化。\n\nNatural Scene、SBU精修后平均Brier仍差于完整集；精修成本下降但预测变化不一致。Yeast空集优于完整集，但不证明选择方法学得了有用属性。KL等指标可能给出不同排序，不能看完结果改主要指标。内部上界可行不是外部预测保证，反复复用内部验证作搜索目标可能带来选择乐观偏差；本次不能证明这一因果解释。\n\n既有测试已用于此前诊断，仍属探索性复用。下一步优先检查内部约束的可迁移性，预先隔离训练内部的搜索与复核数据并重跑冻结流程，不依据本次测试调参或继续扩正式实验。\n')
print('records',len(s));print(json.dumps(pairs,indent=2))

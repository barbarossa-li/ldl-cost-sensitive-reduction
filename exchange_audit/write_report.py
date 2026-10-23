from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent;s=json.loads((out/'summary.json').read_text());data=list(csv.DictReader(open(out/'candidates.csv')))
fig,axes=plt.subplots(1,2,figsize=(9,3.7),layout='constrained')
for ax,name in zip(axes,['Natural Scene','SBU_3DFE']):
 for b,color in [('Prediction only','#226c93'),('Guided','#bb6138')]:
  r=next(r for r in s if r['dataset']==name and r['base']==b);q=[v for v in data if v['dataset']==name and v['base']==b];best=r['base_cost'];values=[best]
  for v in q[:r['matched_budget']]:
   if v['feasible']=='True':best=min(best,int(v['candidate_cost']))
   values.append(best)
  ax.step(range(len(values)),values,where='post',label=b,color=color,lw=2)
 ax.set_title(name);ax.set_xlabel('Exchange candidates evaluated');ax.set_ylabel('Lowest feasible cost found');ax.grid(alpha=.2);ax.legend(fontsize=9)
fig.savefig(out/'exchange_budget.pdf');fig.savefig(out/'exchange_budget.png',dpi=180);plt.close(fig)
header='| 数据集 | 子集 | 原成本 | 等预算评估数 | 可行交换 | 最低成本 |\n|---|---|---:|---:|---:|---:|\n';body=''
for r in s:
 v=r['best_matched'];body+=f"| {r['dataset']} | {r['base']} | {r['base_cost']} | {r['matched_budget']} | {r['matched_feasible']} | {v['candidate_cost'] if v else r['base_cost']} |\n"
(out/'EXCHANGE_REPORT.md').write_text('# 降成本属性交换诊断\n\n'+header+body+'\n按成本节省降序、删除属性索引、加入属性索引排序，与候选标签或损失无关。每个数据集两子集预算取候选数较小值；全部枚举结果另行保存。只评估固定子集的一删一加，不逐步采用。Yeast 空子集无候选。\n\n单删不可约仍允许降成本交换。SBU 指导子集将属性83替换为225，成本20降至11，内部损失0.027367651增至0.027401540，但低于完整集上界0.027474869；依赖度0.278345降至0.148181。不能称为预测提升，也不是独立新证据：该组合已在此前路径反事实中出现。本次完成全面邻域检查。\n\n下一步应在相同预算下，将明确的交换精修阶段分别加入预测优先和粒球指导搜索，重新执行选集流程并进行重复划分；先预先确定规则，再评价泛化，不把本次最佳组合当作正式测试结果。\n')
lines=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Equal-budget cost-reducing exchange diagnostics under the unchanged Euclidean prediction constraint.}',r'\begin{tabular}{llrrrr}',r'\hline',r'Dataset & Base & Cost & Budget & Feasible & Best cost\\',r'\hline']
for r in s:
 v=r['best_matched'];lines.append(f"{r['dataset'].replace('_',chr(92)+'_')} & {'Pred.' if r['base']=='Prediction only' else 'Guided'} & {r['base_cost']} & {r['matched_budget']} & {r['matched_feasible']} & {v['candidate_cost'] if v else r['base_cost']}\\\\")
lines += [r'\hline',r'\end{tabular}',r'\end{table}'];(out/'exchange_table.tex').write_text('\n'.join(lines)+'\n')

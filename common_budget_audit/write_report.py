from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent;s=json.loads((out/'results.json').read_text());assert len(s)==54
names=['Natural Scene','SBU_3DFE','Yeast_cold'];budgets=[5,10,20];methods=['Prediction ranking','Dependency guidance'];summary=[];paired=[]
for name in names:
 for b in budgets:
  for method in methods:
   q=[v for v in s if v['dataset']==name and v['budget']==b and v['method']==method];summary.append(dict(dataset=name,budget=b,method=method,mean_spent=float(np.mean([v['spent'] for v in q])),mean_attributes=float(np.mean([v['attributes'] for v in q])),mean_audit_brier=float(np.mean([v['audit_brier'] for v in q])),mean_inner_brier=float(np.mean([v['inner_brier'] for v in q])),inner_pass=sum(v['inner_pass_full'] for v in q),audit_pass=sum(v['audit_pass_full'] for v in q),empty_outputs=sum(v['attributes']==0 for v in q)))
  delta=[];equalcost=0
  for rep in [1,2,3]:
   p=next(v for v in s if v['dataset']==name and v['budget']==b and v['method']=='Prediction ranking' and v['repeat']==rep);g=next(v for v in s if v['dataset']==name and v['budget']==b and v['method']=='Dependency guidance' and v['repeat']==rep);delta.append(g['audit_brier']-p['audit_brier']);equalcost+=p['spent']==g['spent']
  paired.append(dict(dataset=name,budget=b,mean_guided_minus_prediction=float(np.mean(delta)),guided_better=sum(v < -1e-12 for v in delta),tied=sum(abs(v)<=1e-12 for v in delta),guided_worse=sum(v>1e-12 for v in delta),equal_actual_spending=equalcost,deltas=delta))
(out/'summary.json').write_text(json.dumps(dict(means=summary,paired=paired),indent=2))
fig,axes=plt.subplots(2,3,figsize=(10,6),layout='constrained')
for j,name in enumerate(names):
 for method,color,marker in [('Prediction ranking','#226c93','o'),('Dependency guidance','#bb6138','s')]:
  q=[v for v in summary if v['dataset']==name and v['method']==method];axes[0,j].plot(budgets,[v['mean_audit_brier'] for v in q],color=color,marker=marker,label=method,alpha=.8);axes[1,j].plot(budgets,[v['mean_spent'] for v in q],color=color,marker=marker,alpha=.8)
 ref=[v for v in s if v['dataset']==name and v['budget']==5 and v['method']=='Prediction ranking'];axes[0,j].axhline(np.mean([v['audit_full'] for v in ref]),color='#666666',ls='--',lw=1,label='Full attributes');axes[0,j].axhline(np.mean([v['audit_empty'] for v in ref]),color='#999999',ls=':',lw=1,label='Empty attributes');axes[0,j].set_title(name);axes[0,j].set_ylabel('Mean reserved-audit Brier');axes[0,j].legend(fontsize=7)
 axes[1,j].plot(budgets,budgets,color='#666666',ls='--',lw=1);axes[1,j].set_ylabel('Mean actual acquisition cost');axes[1,j].set_ylim(-.5,21)
 for i in [0,1]:axes[i,j].set_xticks(budgets);axes[i,j].set_xlabel('Common budget cap');axes[i,j].grid(alpha=.2);axes[i,j].tick_params(labelsize=8);axes[i,j].yaxis.set_major_locator(MaxNLocator(4))
fig.savefig(out/'common_budget.pdf');fig.savefig(out/'common_budget.png',dpi=180);plt.close(fig)
lines=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Common acquisition-budget ablation with shared signed bridges and internal full-reference stopping. Three-repeat audit means; spending is reported separately from the budget cap.}',r'\resizebox{\linewidth}{!}{%',r'\begin{tabular}{lrrrrrr}',r'\hline',r'Dataset & Cap & P spent & G spent & P Brier & G Brier & G better/tied/worse\\',r'\hline'];header='| 数据集 | 预算上限 | P实际成本 | G实际成本 | P复核Brier | G复核Brier | G胜/平/负 |\n|---|---:|---:|---:|---:|---:|---|\n';body=''
for name in names:
 for b in budgets:
  p=next(v for v in summary if v['dataset']==name and v['budget']==b and v['method']=='Prediction ranking');g=next(v for v in summary if v['dataset']==name and v['budget']==b and v['method']=='Dependency guidance');pair=next(v for v in paired if v['dataset']==name and v['budget']==b);counts=f"{pair['guided_better']}/{pair['tied']}/{pair['guided_worse']}";lines.append(f"{name.replace('_',chr(92)+'_')} & {b} & {p['mean_spent']:.2f} & {g['mean_spent']:.2f} & {p['mean_audit_brier']:.6f} & {g['mean_audit_brier']:.6f} & {counts}\\\\");body+=f"| {name} | {b} | {p['mean_spent']:.2f} | {g['mean_spent']:.2f} | {p['mean_audit_brier']:.6f} | {g['mean_audit_brier']:.6f} | {counts} |\n"
lines += [r'\hline',r'\end{tabular}}',r'\end{table}'];(out/'budget_table.tex').write_text('\n'.join(lines)+'\n')
(out/'BUDGET_REPORT.md').write_text('# 共同成本预算的排序消融\n\n'+header+body+'\nP为纯预测增益/成本排序，G为正预测候选内优先正依赖增益/成本。共同预算上限5/10/20，每次432搜索/144复核；内层达到完整集参考即停止，预算不足时可输出内部不可行子集。没有删除或交换精修，故这是排名机制消融，不替换正式算法。实际花费不必等于上限或彼此相同。\n\n首次仅允许正增益的尝试全部停在空集，不能区分指导；完整保留在positive_only_attempt。依据搜索内部的停滞诊断恢复原始最大有符号预测增益桥接，两种方法规则相同，没有依据复核结果优化规则。结果分别报告，不把首次尝试当成功消融。\n\n候选与预算只用搜索信息，六个选集按划分先保存再复核；不使用外部测试，不按复核挑选预算。不同预算结果共享划分并可能相同，不是独立重复。缓存复用使运行时间不是公平冷启动比较，候选请求数仅作工作量记录。胜出计数和三次均值不构成统计显著性。\n')
print(json.dumps(dict(means=summary,paired=paired),indent=2))

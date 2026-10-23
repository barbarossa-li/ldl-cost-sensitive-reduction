from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent;s=json.loads((out/'results.json').read_text());summ=[]
for name in ['Natural Scene','SBU_3DFE','Yeast_cold']:
 for method in ['Prediction only','Guided']:
  q=[r for r in s if r['dataset']==name and r['method']==method];summ.append(dict(dataset=name,method=method,before_mean_cost=float(np.mean([r['before_cost'] for r in q])),after_mean_cost=float(np.mean([r['after_cost'] for r in q])),improved=sum(r['after_cost']<r['before_cost'] for r in q),budget_limited=sum(r['stop_reason'].startswith('budget exhausted') for r in q),mean_initial_seconds=float(np.mean([r['initial_seconds'] for r in q])),mean_refinement_seconds=float(np.mean([r['refinement_seconds'] for r in q]))))
(out/'summary.json').write_text(json.dumps(summ,indent=2))
fig,axes=plt.subplots(1,2,figsize=(9,3.7),layout='constrained')
for ax,name in zip(axes,['Natural Scene','SBU_3DFE']):
 for method,color,offset in [('Prediction only','#226c93',-.09),('Guided','#bb6138',.09)]:
  q=[r for r in s if r['dataset']==name and r['method']==method];x=np.arange(1,4)+offset;before=[r['before_cost'] for r in q];after=[r['after_cost'] for r in q]
  ax.vlines(x,after,before,color=color,lw=2);ax.scatter(x,before,color=color,marker='o',facecolors='none',s=65);ax.scatter(x,after,color=color,marker='s',s=45,label=method)
 ax.set_title(name);ax.set_xticks([1,2,3]);ax.set_xlabel('Saved development repeat');ax.set_ylabel('Acquisition cost');ax.grid(alpha=.2);ax.legend(fontsize=9)
fig.suptitle('Open circle: before refinement | Square: after refinement',fontsize=10);fig.savefig(out/'refinement_cost.pdf');fig.savefig(out/'refinement_cost.png',dpi=180);plt.close(fig)
lines=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Three-repeat development refinement pilot. All final subsets pass the unchanged internal prediction constraint.}',r'\begin{tabular}{llrrrr}',r'\hline',r'Dataset & Method & Before mean & After mean & Improved & Budget stops\\',r'\hline']
for r in summ:lines.append(f"{r['dataset'].replace('_',chr(92)+'_')} & {'Pred.' if r['method']=='Prediction only' else 'Guided'} & {r['before_mean_cost']:.2f} & {r['after_mean_cost']:.2f} & {r['improved']}/3 & {r['budget_limited']}/3\\\\")
lines += [r'\hline',r'\end{tabular}',r'\end{table}'];(out/'refinement_table.tex').write_text(('\n'.join(lines)+'\n').replace('\\begin{tabular}', '\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}').replace('\\end{tabular}', '\\end{tabular}}'))
header='| 数据集 | 方法 | 原平均成本 | 精修平均成本 | 降成本次数 | 预算停止次数 |\n|---|---|---:|---:|---:|---:|\n';body=''.join(f"| {r['dataset']} | {r['method']} | {r['before_mean_cost']:.2f} | {r['after_mean_cost']:.2f} | {r['improved']}/3 | {r['budget_limited']}/3 |\n" for r in summ)
(out/'REFINEMENT_REPORT.md').write_text('# 交换精修重复划分试验\n\n'+header+body+'\n固定规则后重跑原搜索并附加相同精修；每种方法每次最多200次候选请求，包含缓存命中。此预算仅针对精修，初始搜索工作量和运行时间不相等。仅开发内部验证，不新增外部测试。\n\n每次接受严格降成本且满足完整集预测上界的候选，损失可能较精修前增大。排序按成本节省、移除/新增索引；已评估可行候选按成本、损失、索引选最佳。接受后重启。预算停止不保证局部最优；只有穷尽当前邻域且无可行降成本动作才具备该邻域的局部保证。\n\nNatural Scene 精修后预测优先两次更便宜、一次相同。SBU 指导两次更便宜、一次相同。Yeast 全为空。不能将可行性或成本下降解释为泛化改善，也不能宣称粒球指导优势。建议保持规则不变，下一步在剩余两个已保存重复划分上复核，再独立汇报完整流程的预测表现。\n')

from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent;summary=json.loads((out/'summary.json').read_text());points=json.loads((out/'fronts.json').read_text());pairs=list(csv.DictReader(open(out/'paired.csv')))
for v in summary:
 for method,key in [('Prediction ranking','prediction'),('Dependency guidance','guidance')]:
  q=[r for r in points if r['dataset']==v['dataset'] and r['method']==method and r['nondominated']];v['unique_front_'+key]=len({(r['repeat'],tuple(r['selected'])) for r in q});v['front_contexts_'+key]=len({r['repeat'] for r in q})
(out/'summary.json').write_text(json.dumps(summary,indent=2))
# Deduplicate coincident coordinates for each displayed frontier.
fig,axes=plt.subplots(3,2,figsize=(9,8),layout='constrained')
for col,name in enumerate(['Natural Scene','SBU_3DFE']):
 for row,rep in enumerate([1,2,3]):
  ax=axes[row,col];q=[v for v in points if v['dataset']==name and v['repeat']==rep]
  for method,color,marker in [('Prediction ranking','#226c93','o'),('Dependency guidance','#bb6138','s')]:
   v=[r for r in q if r['method']==method];ax.scatter([r['spent'] for r in v],[r['audit_brier'] for r in v],color=color,marker=marker,s=42,alpha=.85,zorder=4)
  empty=next(v for v in q if v['method']=='Empty');full=next(v for v in q if v['method']=='Full');ax.scatter(0,empty['audit_brier'],color='#666666',marker='D',s=35,zorder=4);ax.axhline(full['audit_brier'],color='#777777',ls='--',lw=.8)
  unique=sorted({(v['spent'],v['audit_brier']) for v in q if v['nondominated'] and v['method']!='Full'});ax.plot([v[0] for v in unique],[v[1] for v in unique],color='#777777',lw=1,alpha=.7)
  for budget in [5,10,20]:
   p=next(v for v in q if v['method']=='Prediction ranking' and v['budget']==budget);g=next(v for v in q if v['method']=='Dependency guidance' and v['budget']==budget);ax.plot([p['spent'],g['spent']],[p['audit_brier'],g['audit_brier']],color='#aaaaaa',ls=':',lw=.8,zorder=1)
  ax.set_title(f'{name} | repeat {rep}',fontsize=10);ax.set_xlim(-.6,21);ax.set_xticks([0,5,10,15,20]);ax.set_xlabel('Actual acquisition cost');ax.set_ylabel('Reserved-audit Brier');ax.grid(alpha=.15);ax.yaxis.set_major_locator(MaxNLocator(4));ax.tick_params(labelsize=8)
fig.legend(handles=[Line2D([0],[0],color='#226c93',marker='o',ls='',label='Prediction ranking'),Line2D([0],[0],color='#bb6138',marker='s',ls='',label='Dependency guidance'),Line2D([0],[0],color='#666666',marker='D',ls='',label='Empty'),Line2D([0],[0],color='#777777',ls='--',label='Full loss (high cost outside view)')],loc='outside upper center',ncol=2,fontsize=8);fig.savefig(out/'pareto_cost_loss.pdf');fig.savefig(out/'pareto_cost_loss.png',dpi=180);plt.close(fig)
lines=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Same-cap paired dominance classifications across three repeats and three caps. Nine budget cases per dataset are correlated comparisons, not nine independent trials.}',r'\begin{tabular}{lrrrrr}',r'\hline',r'Dataset & G dominates & P dominates & Tradeoff & Coincident & G strict both\\',r'\hline']
for v in summary:c=v['counts'];lines.append(f"{v['dataset'].replace('_',chr(92)+'_')} & {c['guidance dominates']} & {c['prediction dominates']} & {c['tradeoff']} & {c['coincident']} & {v['guidance_strict_both']}\\\\")
lines += [r'\hline',r'\end{tabular}',r'\end{table}'];(out/'pareto_table.tex').write_text(('\n'.join(lines)+'\n').replace('\\begin{tabular}', '\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}').replace('\\end{tabular}', '\\end{tabular}}'))
header='| 数据集 | 指导占优 | 纯预测占优 | 权衡 | 重合 | 指导严格双改善 |\n|---|---:|---:|---:|---:|---:|\n';body=''.join(f"| {v['dataset']} | {v['counts']['guidance dominates']} | {v['counts']['prediction dominates']} | {v['counts']['tradeoff']} | {v['counts']['coincident']} | {v['guidance_strict_both']} |\n" for v in summary)
(out/'PARETO_REPORT.md').write_text('# 实际成本与复核预测损失的成对分析\n\n'+header+body+'\n占优：成本不高、损失不大且至少一项严格改善。严格双改善：两项都严格改善。同预算上限下，指导没有严格双改善；其占优来自相同成本下更低损失。仅统计已有结果，不重搜、不重新预测、不修改选择。\n\n前沿范围是同一划分的6个预算方法点加完整集和空集，不是全局Pareto最优。Natural Scene去重后P与G各3个前沿选集，覆盖2与3个划分；SBU为0与3，G覆盖2个划分。Yeast全空，与空集重合，无信息性对比。\n\n跨预算可找到G严格双改善的局部例子：Natural Scene第3次G上限5，成本5/Brier0.220410，优于P上限10的成本10/Brier0.236662；SBU第2次G上限10，成本10/Brier0.035055，优于P上限20的成本20/Brier0.035322。但这些比较是已经观察到的预算点，不能据此选最优预算或构成独立确认。\n\n下一步应先明确可支持的论文主张：低成本下的预测价值与条件性指导收益；不能宣称普遍无损约简或依赖度的预测单调性。保留核心理论，整理正式算法与成本—预测目标的对应关系，并将探索诊断与正式贡献分开。\n')
print(json.dumps(summary,indent=2))

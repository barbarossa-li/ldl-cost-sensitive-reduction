from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent;s=json.loads((out/'summary.json').read_text());controls=list(csv.DictReader(open(out/'controls.csv')));aggregate=[]
for name in ['Natural Scene','SBU_3DFE','Yeast_cold']:
 for method in ['Prediction only','Guided']:
  q=[v for v in s if v['dataset']==name and v['method']==method];aggregate.append(dict(dataset=name,method=method,selected_audit_mean=float(np.mean([v['selected_audit'] for v in q])),controls_audit_mean=float(np.mean([v['control_audit_mean'] for v in q])),selected_inner_mean=float(np.mean([v['selected_inner'] for v in q])),controls_inner_mean=float(np.mean([v['control_inner_mean'] for v in q])),selected_better_than_controls_mean=sum(v['selected_minus_control_audit_mean'] < -1e-12 for v in q),empty_contexts=sum(v['degenerate_empty'] for v in q)))
(out/'aggregate.json').write_text(json.dumps(aggregate,indent=2))
fig,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained')
for i,name in enumerate(['Natural Scene','SBU_3DFE']):
 for j,method in enumerate(['Prediction only','Guided']):
  ax=axes[i,j]
  for rep in [1,2,3]:
   q=next(v for v in s if v['dataset']==name and v['method']==method and v['repeat']==rep);v=[float(r['audit_brier']) for r in controls if r['dataset']==name and r['method']==method and int(r['repeat'])==rep];ax.boxplot([v],positions=[rep],widths=.35,showfliers=True,patch_artist=True,boxprops=dict(facecolor='#bbd7e3',edgecolor='#226c93'),medianprops=dict(color='#226c93'),flierprops=dict(markersize=3));ax.scatter(rep,q['selected_audit'],marker='D',color='#bb6138',s=42,zorder=5)
  ax.set_title(name+' | '+method,fontsize=10);ax.set_xticks([1,2,3]);ax.set_xlabel('Saved development repeat');ax.set_ylabel('Reserved-audit Brier');ax.grid(axis='y',alpha=.2)
fig.legend(handles=[Line2D([0],[0],color='#226c93',lw=6,label='20 cost-profile-matched controls'),Line2D([0],[0],color='#bb6138',marker='D',ls='',label='Frozen refined subset')],loc='outside upper center',ncol=2,fontsize=9);fig.savefig(out/'matched_controls.pdf');fig.savefig(out/'matched_controls.png',dpi=180);plt.close(fig)
lines=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Reserved-audit loss against label-free controls with the same attribute count and exact cost profile. Three-repeat averages; Yeast comparisons are degenerate empty references.}',r'\begin{tabular}{llrrr}',r'\hline',r'Dataset & Method & Selected Brier & Controls mean & Better repeats\\',r'\hline']
for v in aggregate:lines.append(f"{v['dataset'].replace('_',chr(92)+'_')} & {'Pred.' if v['method']=='Prediction only' else 'Guided'} & {v['selected_audit_mean']:.6f} & {v['controls_audit_mean']:.6f} & {'--' if v['empty_contexts']==3 else str(v['selected_better_than_controls_mean'])+'/3'}\\\\")
lines += [r'\hline',r'\end{tabular}',r'\end{table}'];(out/'matched_table.tex').write_text('\n'.join(lines)+'\n')
header='| 数据集 | 方法 | 选集平均复核Brier | 匹配对照平均复核Brier | 优于对照均值次数 |\n|---|---|---:|---:|---:|\n';body=''.join(f"| {v['dataset']} | {v['method']} | {v['selected_audit_mean']:.6f} | {v['controls_audit_mean']:.6f} | {'不适用：空集' if v['empty_contexts']==3 else str(v['selected_better_than_controls_mean'])+'/3'} |\n" for v in aggregate)
(out/'MATCHED_REPORT.md').write_text('# 同成本、同属性数的标签无关对照\n\n'+header+body+'\n每个非空冻结子集生成20个唯一随机子集，严格匹配各成本等级属性数量，因而匹配总成本和属性数。先冻结全部对照，再算损失；不使用标签生成、不按复核损失择优。不在所有仅总成本相同的子集上均匀采样，这是更严格的成本结构条件对照。12个非空语境共240个对照；6个Yeast空语境仅1个相同空集，共246条语义对照记录。\n\nNatural Scene 两方法均3/3次优于对照平均复核损失；SBU预测优先2/3，指导3/3。说明当前低成本选集仍有预测价值，不支持无损约简、总体显著性或粒球指导优越性。对照和选集不具备可交换性，胜出计数不是置换p值。两方法预算不同，不能直接用各自均值比较贡献。\n\n下一步在共同成本预算上做纯预测搜索与粒球指导的消融，预先固定预算和评价规则；不继续扩大样本，也不根据复核结果改变理论权重。\n')
print(json.dumps(aggregate,indent=2))

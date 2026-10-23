from pathlib import Path
import json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=Path(__file__).resolve().parent;rr=json.loads((p/'summary.json').read_text());ar=list(csv.DictReader(open(p/'attributes.csv')))
tex=[r'\begin{center}',r'\begin{tabular}{lrrrr}',r'\hline',r'Dataset & Rank corr. & Mean improved & Full passed & Stable group\\',r'\hline']
for r in rr:tex.append(f"{r['dataset'].replace('_','-')} & {r['spearman_excess_vs_euclidean_gain']:.3f} & {r['euclidean_improves_mean']}/{r['attributes']} & {r['euclidean_passes_full_count']}/{r['attributes']} & {(str(r['stable_q95_euclidean_improves_mean'])+'/'+str(r['stable_q95_count'])) if r['stable_q95_count'] else '--'}\\\\")
tex += [r'\hline',r'\end{tabular}',r'\end{center}'];(p/'attribute_table.tex').write_text('\n'.join(tex))
report=['# 信息关联与单属性预测改善的衔接','',
'固定三个既有开发划分、3折、参数与5邻居设置，评估全部561个数值属性。每个训练折重新拟合全部标签驱动量；无外部测试评价、无阈值选择、无重新约简。','',
'主关联分数为三折训练数据中的平均W减去平均置换W；预测改善为全局均值内部验证Brier减去单属性Brier。前者不是新的权重定义，不进入模型。','',
'## 描述性结果（欧氏/曼哈顿）','',
'| 数据集 | 关联差异与预测改善的Spearman | 优于全局均值 | 达到完整集参考 | 稳定背景分位组中优于均值 |','|---|---:|---:|---:|---:|']
for r in rr:report.append(f"| {r['dataset']} | {r['spearman_excess_vs_euclidean_gain']:.3f} | {r['euclidean_improves_mean']}/{r['attributes']} | {r['euclidean_passes_full_count']}/{r['attributes']} | {(str(r['stable_q95_euclidean_improves_mean'])+'/'+str(r['stable_q95_count'])) if r['stable_q95_count'] else '--'} |")
report+=['','单数值属性下，绝对差与平方差严格保序，欧氏和曼哈顿近邻完全一致；1683个属性/折全部核对通过。因此本轮单属性的有向/欧氏差异直接来自源权重，而不是L1/L2几何变化。','',
'Natural Scene和SBU显示正的描述性排序关系，但大多数单属性仍差于全局均值。稳定背景分位组的欧氏预测改善均值分别为−0.036833和−0.003426，对应其他属性为−0.045573和−0.005569；前者平均较好，但仍不是正改善。','',
'稳定组定义沿用三折都超过自身置换95%分位，未校正多重比较，只作描述。属性相互相关、折重叠、这些划分此前已用于开发，不能把Spearman作为独立证据或宣称统计显著性。','',
'## 有向权重是否改善单属性预测','',
'| 数据集 | 有向优于欧氏的属性数 | 基本相同（1e−12） | 有向−欧氏平均Brier | 有向下优于均值 |','|---|---:|---:|---:|---:|']
for r in rr:report.append(f"| {r['dataset']} | {r['directed_better_than_euclidean']}/{r['attributes']} | {r['directed_equal_euclidean']} | {r['mean_directed_minus_euclidean_brier']:.9f} | {r['directed_improves_mean']}/{r['attributes']} |")
report+=['','差异总体较小，源权重并未明显增加优于全局均值的单属性数量。平均改善不能当作显著的预测优势，也不能推广到组合子集或外部测试。','',
'## Yeast的参考问题','',
'Yeast有23/24个欧氏单属性通过完整集参考，但0/24优于全局均值；有向下18/24通过完整集参考，同样0/24优于全局均值。通过一个较弱的完整集参考，不等于具有有效预测贡献。若空集满足约束，正代价目标下它就是零成本最优，不能为了非空约简而强行排除它。','',
'## 下一步：条件增益','',
'单属性有标签关联，不意味着单属性足以满足预测目标；单属性失败也不排除组合互补。下一步保持已有小属性集固定，计算加入或删去一个属性的验证损失变化，检查背景以上的关联分数是否对应条件增益，而不是继续扩大数据规模、改变权重或重新跑完整约简。','',
'明细attributes.csv、folds.csv和fold_provenance.json保留全部比较；原全局均值参考和Yeast已有单属性损失在1e−12内重现。正文第18节与图18收录定义、单属性保序证明与散点图。']
(p/'ATTRIBUTE_REPORT.md').write_text('\n'.join(report))
fig,axes=plt.subplots(2,3,figsize=(10,6))
for col,r in enumerate(rr):
 aa=[a for a in ar if a['dataset']==r['dataset']];x=np.array([float(a['excess_W']) for a in aa]);g=np.array([float(a['euclidean_gain_over_mean']) for a in aa]);delta=np.array([float(a['directed_minus_euclidean_brier']) for a in aa]);stable=np.array([a['above_own_q95_all_folds']=='True' for a in aa])
 for mask,color,label in [(~stable,'#8c8c8c','Other'),(stable,'#187b59','Above q95 in 3 folds')]:
  axes[0,col].scatter(x[mask],g[mask],s=12,c=color,alpha=.7,label=label);axes[1,col].scatter(x[mask],delta[mask],s=12,c=color,alpha=.7)
 axes[0,col].set_title(r['dataset'].replace('_',' '));axes[0,col].set_ylabel('Mean Brier - singleton Brier');axes[1,col].set_ylabel('Directed Brier - Euclidean Brier');axes[1,col].set_xlabel('Observed W - permutation mean W')
 for row in range(2):axes[row,col].axhline(0,color='grey',lw=.8);axes[row,col].ticklabel_format(axis='x',style='sci',scilimits=(0,0));axes[row,col].grid(alpha=.2)
axes[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(p/'association_prediction.pdf');fig.savefig(p/'association_prediction.png',dpi=180)

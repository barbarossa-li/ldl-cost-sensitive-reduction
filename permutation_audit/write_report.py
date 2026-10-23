from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent;rr=json.loads((p/'summary.json').read_text());ff=list(csv.DictReader(open(p/'folds.csv')))
tex=[r'\begin{center}',r'\begin{tabular}{lrrrr}',r'\hline',r'Dataset & Observed $W$ & Permuted $W$ & Ratio & All-fold count\\',r'\hline']
for r in rr:tex.append(f"{r['dataset'].replace('_','-')} & {r['observed_mean_W']:.7f} & {r['permutation_mean_W']:.7f} & {r['observed_over_null_mean']:.2f} & {r['attributes_above_own_q95_all_three_folds']}/{r['attributes']}\\\\")
tex += [r'\hline',r'\end{tabular}',r'\end{center}'];(p/'permutation_table.tex').write_text('\n'.join(tex))
report=['# 训练标签分布整行置换对照','',
'结论：Natural Scene与SBU存在高于置换背景的局部信息统计；SBU权重绝对值小，不等于属性关联为零。Yeast的差异较弱、跨折不稳定。这仍不证明预测贡献、因果关联或最优约简。','',
'## 冻结协议','',
'三个既有开发划分、9个内部训练折，各128个训练对象。每折199次整行标签向量置换；同一置换在该折所有属性上共享。只破坏属性与标签的对应，保持标签向量内部结构、全局标签均值、活跃标签、带宽、核矩阵、支持度、β=1、归一化及其他参数不变。不使用验证/外部测试标签，不重新约简，不计算新的预测损失。','',
'主统计量预先固定为每折所有属性与源对象的平均信息权重；属性级统计作为描述性补充。尾部秩=(1+置换统计量不小于观测统计量的次数)/200，最小分辨率为0.005，不代表真实尾部概率恰好等于0.005。','',
'## 描述性汇总','',
'| 数据集 | 真实平均W | 置换平均W | 比值 | 三折均超过各自属性背景95%分位的属性数 |','|---|---:|---:|---:|---:|']
for r in rr:report.append(f"| {r['dataset']} | {r['observed_mean_W']:.7f} | {r['permutation_mean_W']:.7f} | {r['observed_over_null_mean']:.2f} | {r['attributes_above_own_q95_all_three_folds']}/{r['attributes']} |")
report += ['','汇总均值用于描述，没有合并数据集p值。三折训练对象重叠，不能把它们当成独立重复。属性95%分位计数未进行多重比较校正，不能称为“显著属性数”，也不能直接用作约简集合。','',
'## 各折平均权重的条件尾部秩','',
'| 数据集 | 折1 | 折2 | 折3 |','|---|---:|---:|---:|']
for r in rr:report.append('| '+r['dataset']+' | '+' | '.join(f'{x:.3f}' for x in r['fold_tail_ranks'])+' |')
report += ['','Natural Scene和SBU的每折真实统计都高于本轮199次置换统计。Yeast为0.060、0.045、0.270，没有稳定分离。只有在零关联假设下标签行可交换时，这些尾部秩才具备条件置换检验解释；本轮只作开发诊断，不宣称调整后的统计显著性或人口层面的无关联。','',
'## 与前面的诊断如何衔接','',
'1. 归一化前的弱尺度与关联强度是不同问题。SBU相对于置换背景的比值比Natural Scene更大，但其绝对平均权重仍小。','2. 原置换背景保留了核平均、源自身标签影响、支持收缩和标签边际。因此观测超出背景，说明当前统计能识别部分对应关系，并非只是这些拟合机制自身的背景量。','3. 关联高于背景仍不保证加大它在距离中的作用会改善预测。前面的固定子集预测比较并没有稳定收益，不能被本轮关联结果覆盖。','4. Yeast的弱结果只针对当前边际局部统计、样本与参数；它不排除特征交互或其他预测器能够利用的信息。','',
'## 验证与复现','',
'1683个属性/折观测统计与原权重最大误差5.04×10⁻¹⁶；置换索引为每折训练索引的合法排列，标签行结构和全局均值保持不变。各折保存拟合对象索引、199组置换索引、全部属性的观测/置换权重及原始KL。','',
'run_audit.py/protocol.json给出冻结协议；attributes.csv与folds.csv给出全结果；各数据集fold_*_scores.npz保留置换分数；正文第17节和图17收录对照。','',
'## 下一步','',
'保留现有定义、参数和收缩，不因为比值较大而放大权重。下一步直接检查“关联高于背景”的属性是否在相同内部验证中提供预测改善：先固定属性级比较，不重新约简，不用测试表现选择阈值。这样可以定位关联统计与预测目标的衔接，而不是继续反复改理论。']
(p/'PERMUTATION_REPORT.md').write_text('\n'.join(report))
fig,axes=plt.subplots(1,3,figsize=(10,3.5))
for ax,r in zip(axes,rr):
 name=r['dataset'];data=[];obs=[]
 for f in range(3):
  d=np.load(p/name.replace(' ','_')/f'fold_{f}_scores.npz');score=d['mean_W'].mean(0);data.append(score[1:]);obs.append(score[0])
 ax.boxplot(data,positions=[1,2,3],widths=.45,showfliers=False,patch_artist=True,boxprops={'facecolor':'#b9cbdc'},medianprops={'color':'#457fb0'})
 ax.scatter([1,2,3],obs,c='#b04d38',marker='D',s=30,label='Observed',zorder=3);ax.set_xticks([1,2,3],['Fold 1','Fold 2','Fold 3']);ax.set_title(name.replace('_',' '));ax.set_ylabel('Mean information weight');ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.grid(axis='y',alpha=.2)
axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(p/'permutation_background.pdf');fig.savefig(p/'permutation_background.png',dpi=180)

from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent;rows=list(csv.DictReader(open(p/'results.csv')));diags=list(csv.DictReader(open(p/'neighbour_summary.csv')))
names=['Natural Scene','SBU_3DFE','Yeast_cold'];sets=['Full','Original joint','Prediction only','Guided','Empty'];methods=['Euclidean','Manhattan','Incoming directed']
def row(n,s,m):return next(r for r in rows if r['dataset']==n and r['subset']==s and r['learner']==m)
def loss(n,s,m):return float(row(n,s,m)['inner_brier'])
nonempty=[(n,s) for n in names for s in sets if int(row(n,s,methods[0])['attributes'])>0]
counts={'nonempty_subset_cases':len(nonempty),'directed_better_than_euclidean':sum(loss(n,s,methods[2])<loss(n,s,methods[0]) for n,s in nonempty),'directed_better_than_manhattan':sum(loss(n,s,methods[2])<loss(n,s,methods[1]) for n,s in nonempty),'warning':'Descriptive repeated-subset cases on three development splits; not independent trials'}
(p/'counts.json').write_text(json.dumps(counts,indent=2))
tex=[r'\begin{center}',r'\begin{tabular}{llrrr}',r'\hline',r'Dataset & Subset & Euclidean & Manhattan & Directed\\',r'\hline']
for n in names:
 for s in sets:tex.append(f"{n.replace('_','-')} & { {'Full':'Full','Original joint':'Joint','Prediction only':'Pred. only','Guided':'Guided','Empty':'Empty'}[s]} & {loss(n,s,methods[0]):.6f} & {loss(n,s,methods[1]):.6f} & {loss(n,s,methods[2]):.6f}\\\\")
tex += [r'\hline',r'\end{tabular}',r'\end{center}']
(p/'loss_table.tex').write_text('\n'.join(tex))
report=['# 固定子集的逐折邻域与预测损失诊断','',
'本轮在三个既有开发划分上固定子集，比较欧氏、未加信息权重的曼哈顿、训练源→查询目标的有向五近邻。每个预测器分别使用各内层训练折拟合的缩放和权重。没有重新约简，没有使用外层测试标签，没有修改参数。','',
'曼哈顿对照必不可少：数值型有向距离以绝对差求和，而原欧氏近邻以平方差求和。如果只对比有向与欧氏，就会把距离形式变化和标签权重变化混在一起。','',
'## 内部验证Brier','',
'| 数据集 | 固定子集 | 欧氏 | 曼哈顿 | 有向 | 有向下是否通过其完整集参考 |','|---|---|---:|---:|---:|---|']
for n in names:
 for s in sets:
  d=row(n,s,methods[2]);report.append(f"| {n} | {s} | {loss(n,s,methods[0]):.6f} | {loss(n,s,methods[1]):.6f} | {loss(n,s,methods[2]):.6f} | {'是' if d['passes_own_reference']=='True' else '否'} |")
report += ['','三个Full结果分别构成各预测器自己的参考值。旧欧氏下的可行性不能迁移：Natural Scene的预测约束子集和指导子集改用有向后失败；SBU原联合子集和预测约束子集改用有向后失败。Yeast原联合非空子集仍通过有向参考，但空集更好。','',
'10个非空固定子集情形中，有向仅3个优于欧氏、3个优于曼哈顿。这些情形共享对象且子集此前适应过同一验证折，不能视为10次独立实验，更不能视为无偏泛化证据。空集在三个预测器下完全相同。','',
'## 完整属性集的邻域与信息项','',
'| 数据集 | 有向与曼哈顿5邻居集合完全一致 | 平均Jaccard | 信息项平均距离占比 |','|---|---:|---:|---:|']
for n in names:
 d=next(r for r in diags if r['dataset']==n and r['subset']=='Full');report.append(f"| {n} | {100*float(d['directed_manhattan_same_set']):.2f}% | {float(d['manhattan_directed_jaccard']):.6f} | {100*float(d['information_share_all_sources']):.3f}% |")
report += ['','信息项占比定义为：每个验证对象到所有训练源的有向距离中，信息部分之和/总距离之和，再对验证对象平均；不是最大逐对占比，也不是标签预测贡献的百分比。','',
'SBU和Yeast上，信息项较弱，有向距离基本表现为缩放后的曼哈顿距离。Natural Scene的信息项改变更多近邻，但损失改善不一致：完整集曼哈顿0.245291优于有向0.253438；原联合子集有向0.250105优于曼哈顿0.252456；指导子集有向0.266607则差于曼哈顿0.263788。','',
'## 可证明的邻域稳定条件','',
'对数值型特征、给定查询x，将有向分数写成D_i=G_i+I_i，其中G_i为几何项、I_i≥0为信息项。若曼哈顿k邻居集合内外的G边界间隔，大于各训练源I_i的最大值与最小值之差，则有向与曼哈顿k邻居集合相同。因为任意集合外j和集合内i，都有D_j−D_i≥几何间隔−信息变化范围>0。这个充分条件不声称现有每个查询都满足它。','',
'当全部信息权重为零，有向距离退化为ζ/m乘训练缩放后的曼哈顿距离；固定的正比例不改变近邻排序。数值训练恒定属性在欧氏/曼哈顿中的外推只增加源无关偏移，也不改变排序。','',
'## 实现和证据','',
'共9个内层几何拟合，45个语义方法/子集记录、135个折损失记录；相同空集记录并不代表不同模型信息。所有原欧氏损失与保存结果在1e−12内一致。训练/验证对象互斥且均不含外部测试对象。保存了逐折训练索引、近邻索引、折外预测和逐对象邻域诊断。','',
'## 下一步','',
'暂不把有向预测器纳入正式方法，也不调ζ或增加混合参数。应先检查信息权重为何在SBU和Yeast上很弱：区分原始局部标签轮廓本就接近全局，还是带宽平滑、收缩、共同归一化导致信息尺度很小。这一步是机制分解，维持理论定义与参数不变。只有确认可解释的预测贡献后，才有理由在新预测器下重新运行约简。']
(p/'NEIGHBOUR_AUDIT_REPORT.md').write_text('\n'.join(report))
fig,axes=plt.subplots(2,3,figsize=(11,6));colors=['#457fb0','#8c8c8c','#187b59'];labels=['Full','Joint','Pred.','Guided','Empty']
for col,n in enumerate(names):
 for m,color in zip(methods,colors):axes[0,col].plot(range(5),[loss(n,s,m) for s in sets],'o-',label=m,color=color,ms=4)
 axes[0,col].set_title(n.replace('_',' '));axes[0,col].set_xticks(range(5),labels);axes[0,col].set_ylabel('Inner Brier loss');axes[0,col].grid(alpha=.2)
 for reference,color,marker in [('Euclidean','#457fb0','o'),('Manhattan','#8c8c8c','s')]:
  vals=[]
  for s in sets:
   dd=next((r for r in diags if r['dataset']==n and r['subset']==s),None);vals.append(float(dd['euclidean_directed_jaccard' if reference=='Euclidean' else 'manhattan_directed_jaccard']) if dd else float('nan'))
  axes[1,col].plot(range(5),vals,marker+'-',color=color,label='Directed vs '+reference)
 axes[1,col].set_xticks(range(5),labels);axes[1,col].set_ylim(0,1.05);axes[1,col].set_ylabel('Mean neighbour Jaccard');axes[1,col].grid(alpha=.2)
axes[0,0].legend(fontsize=7);axes[1,0].legend(fontsize=7);fig.tight_layout();fig.savefig(p/'neighbour_comparison.pdf');fig.savefig(p/'neighbour_comparison.png',dpi=180)
print(counts)

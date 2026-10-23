from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent;rr=json.loads((p/'summary.json').read_text());rows=list(csv.DictReader(open(p/'candidates.csv')))
primary=[r for r in rr if r['learner']=='Euclidean'];tex=[r'\begin{center}',r'\begin{tabular}{llrrr}',r'\hline',r'Dataset & Fixed base & Positive adds & Rank corr. & Feasible drops\\',r'\hline']
for r in primary:tex.append(f"{r['dataset'].replace('_','-')} & {'Pred. only' if r['base']=='Prediction only' else 'Guided'} & {r['positive_add_gains']}/{r['add_candidates']} & {r['spearman_excess_vs_conditional_add_gain']:.3f} & {r['deletion_preserves_reference']}/{r['deletions']}\\\\")
tex += [r'\hline',r'\end{tabular}',r'\end{center}'];text='\n'.join(tex).replace('& 0/0\\\\','& --\\\\');(p/'conditional_table.tex').write_text(text)
report=['# 固定小属性集的条件增益检查','',
'保留两个已保存基础子集：纯预测约束输出、依赖度指导输出；比较欧氏、曼哈顿和训练源→查询目标有向5NN。子集不更新，只穷举一次加入/删除。所有标签驱动量逐折拟合，未使用测试数据。','',
r'添加增益=R(P)−R(P∪{a})；删除损害=R(P\{a})−R(P)，正值表示删除后损失增加。正添加增益也意味着属性代价增加，不能直接称为更优约简。','',
'## 欧氏主比较','',
'| 数据集 | 固定基础子集 | 有正增益的添加 | 关联差异与条件增益Spearman | 保持完整集约束的删减 |','|---|---|---:|---:|---:|']
for r in primary:report.append(f"| {r['dataset']} | {r['base']} | {r['positive_add_gains']}/{r['add_candidates']} | {r['spearman_excess_vs_conditional_add_gain']:.3f} | {(str(r['deletion_preserves_reference'])+'/'+str(r['deletions'])) if r['deletions'] else '无可删属性'} |")
report += ['','SBU单属性相关约0.687，但在已有纯预测/指导子集上降为0.140/0.065。Natural Scene的条件相关分别为−0.146和0.124，与单属性相关0.331不同。边际局部关联包含的信息不一定是已有子集之外的新信息。','',
'Yeast基础子集均为空，因此这两个情形退化为相同单属性检查，不能当作两次独立结果；没有任何添加改善其全局均值预测。','',
'18个“数据集×基础子集×预测器”情形的基准损失均在1e−12内重现。欧氏版本都没有可保持预测约束的单属性删减，核对了原启发式的单删不可约性质。这不证明全局最小代价或包含极小；组合删减、交换属性仍可能存在可行解。','',
'## 其他预测器与完整结果','',
'所有三种预测器的完整比较见summary.json和candidates.csv，3366条语义候选记录。旧欧氏基础子集在新预测器下可能不满足其参考，因此记录了base_passes_reference，不能把所有条件比较称为“可行子集上的约简”。没有生成新的属性集合。','',
'## 结论及下一步','',
'信息权重具有边际关联，但不能稳定代理条件预测增益。不能因为关联高于置换背景就将其设为高优先级、更不能据此淘汰低边际信息属性。','',
'下一步维持理论结构，检查已有粒球依赖度的子集边际增益ΔΓ(P,a)与同一固定子集上的预测条件增益是否对应。Γ本来依赖当前P，是与边际W不同的量；先核查它的作用，暂不新增权重形式、混合系数或正式实验。']
(p/'CONDITIONAL_REPORT.md').write_text('\n'.join(report))
fig,axes=plt.subplots(2,3,figsize=(10,6))
for col,name in enumerate(['Natural Scene','SBU_3DFE','Yeast_cold']):
 for row,b in enumerate(['Prediction only','Guided']):
  aa=[r for r in rows if r['dataset']==name and r['base']==b and r['learner']=='Euclidean' and r['operation']=='add'];ax=axes[row,col];ax.scatter([float(r['excess_W']) for r in aa],[float(r['gain']) for r in aa],s=12,color='#457fb0',alpha=.7);ax.axhline(0,color='grey',lw=.8);ax.grid(alpha=.2);ax.set_title(name.replace('_',' ')+' / '+('Pred.' if row==0 else 'Guided'));ax.set_ylabel('Conditional inner Brier gain');ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
  if row==1:ax.set_xlabel('Observed W - permutation mean W')
fig.tight_layout();fig.savefig(p/'conditional_gain.pdf');fig.savefig(p/'conditional_gain.png',dpi=180)

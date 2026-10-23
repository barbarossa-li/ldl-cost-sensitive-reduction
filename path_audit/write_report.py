from pathlib import Path
import json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent;summary=[];allcf=[];allpaths=[]
for name in ['Natural_Scene','SBU_3DFE','Yeast_cold']:
 d=json.loads((p/name/'paths_before_test.json').read_text());cf=d['same_parent_counterfactuals'];allcf.extend({'dataset':name,**r} for r in cf)
 for mode,key in [('Guided','guided_path'),('Prediction only','pure_path')]:allpaths.extend({'dataset':name,'method':mode,**r} for r in d[key])
 g=d['guided_path'];q=d['pure_path'];summary.append({'dataset':name,'guided_additions':sum(r['phase']=='add' for r in g),'pure_additions':sum(r['phase']=='add' for r in q),'guided_deletions':sum(r['phase']=='delete' for r in g),'pure_deletions':sum(r['phase']=='delete' for r in q),'guided_final_cost':g[-1]['cost'] if g else 0,'pure_final_cost':q[-1]['cost'] if q else 0,'same_parent_divergent':sum(r['divergent'] for r in cf),'same_parent_steps':len(cf),'lower_predictive_gain_per_cost':sum(r['guided_gain_per_cost']<r['pure_gain_per_cost'] for r in cf)})
(p/'summary.json').write_text(json.dumps(summary,indent=2))
for fname,rows in [('same_parent_candidates.csv',allcf),('paths.csv',allpaths)]:
 with (p/fname).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
fig,axes=plt.subplots(2,2,figsize=(9,6))
for col,name in enumerate(['Natural_Scene','SBU_3DFE']):
 d=json.loads((p/name/'paths_before_test.json').read_text());checkpoint=json.loads((p/name/'selection_before_test.json').read_text());initial=d['same_parent_counterfactuals'][0]['guided_after_inner']+d['same_parent_counterfactuals'][0]['guided_gain']
 for key,label,color in [('guided_path','Dependency guidance','#187b59'),('pure_path','Prediction only','#457fb0')]:
  rr=d[key];steps=[0]+[r['step'] for r in rr];loss=[initial]+[r['inner_brier'] for r in rr];cost=[0]+[r['cost'] for r in rr]
  axes[0,col].plot(steps,loss,'o-',ms=3,color=color,label=label);axes[1,col].plot(steps,cost,'o-',ms=3,color=color,label=label)
  dd=[r for r in rr if r['phase']=='delete']
  axes[0,col].scatter([r['step'] for r in dd],[r['inner_brier'] for r in dd],s=70,marker='*',c=color,zorder=3);axes[1,col].scatter([r['step'] for r in dd],[r['cost'] for r in dd],s=70,marker='*',c=color,zorder=3)
 axes[0,col].axhline(checkpoint['inner_reference'],ls='--',lw=1,c='grey',label='Full-set loss limit');axes[0,col].set_title(name.replace('_',' '));axes[0,col].set_ylabel('Inner Brier loss');axes[1,col].set_ylabel('Acquisition cost');axes[1,col].set_xlabel('Search step (stars: deletions)')
 for row in range(2):axes[row,col].grid(alpha=.2)
axes[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(p/'search_paths.pdf');fig.savefig(p/'search_paths.png',dpi=180)
report=['# 依赖度指导与纯预测指导：搜索路径审计','',
'参数、可行域、代价指数λ=-1、非正预测桥接、停止条件及删减规则完全相同。两条路径唯一的选择规则差异为：正预测改善候选是否优先按正依赖度增益排序。继续使用三个既有开发划分，没有扩大正式实验。','',
'重新运行的纯预测路径与已保存版本的代价和验证损失一致，新依赖度指导输出与上一轮完全一致。两种方法共用缓存，本轮耗时不能用作公平运行时间比较。','',
'## 路径摘要','',
'| 数据集 | 依赖度指导 添加/删除 | 纯预测 添加/删除 | 最终代价 指导/纯预测 | 同一状态下选择分歧 |','|---|---:|---:|---:|---:|']
for r in summary:report.append(f"| {r['dataset']} | {r['guided_additions']}/{r['guided_deletions']} | {r['pure_additions']}/{r['pure_deletions']} | {r['guided_final_cost']}/{r['pure_final_cost']} | {r['same_parent_divergent']}/{r['same_parent_steps']} |")
report+=['',
'为避免把不同路径状态的候选直接比较，另在每个依赖度指导的当前子集上，计算纯预测规则会选择哪个属性。这是同一状态下的即时反事实比较，不是完整替代运行。Natural Scene的8次添加中6次不同，SBU的5次添加中4次不同；这些分歧均牺牲了当步预测改善/代价。不能将即时增益差异当成后续测试风险差异。','',
'## 最明确的局部证据：SBU第五步','',
'当前子集为[6,84,201,234]（索引从0开始），代价为10。','',
'| 下一属性 | 代价 | 预测损失改善 | 依赖度增益 | 添加后验证Brier | 添加后总代价 |','|---|---:|---:|---:|---:|---:|',
'| 指导选择83 | 10 | 0.000475428 | +0.008579955 | 0.027367651 | 20 |',
'| 同状态纯预测选择225 | 1 | 0.000441539 | −0.121584014 | 0.027401540 | 11 |','',
'完整集验证损失上限为0.027474869。因此两者都满足预测约束，代价11的候选并不需要保持正依赖度增益。其随后如果执行相同删减，代价只会进一步下降。代价20方案的验证损失略低，但在“满足验证约束后最小化代价”的目标下，不能用这点额外改善证明其代价合理。这不是全局最优性结果，也没有为反事实候选新增测试表现结论。','',
'## 第一选择也体现错位','',
'Natural Scene：指导首选11（代价3），预测改善/代价为0.000106105；纯预测首选177（代价1），为0.002580296，相差约24.32倍。指导的依赖度增益更高。','',
'SBU：指导首选201（代价7），预测改善/代价为0.000055034；纯预测首选223（代价4），为0.000700623，相差约12.73倍。依赖度提升更快，预测约束却更晚达标。','',
'## 删减与停止','',
'Natural Scene指导路径先到代价16、验证损失0.252562，随后按代价删去147与173，最终代价12、损失0.259264；损失上升但仍不超过上限0.259592，符合现有目标，不是实现错误。纯预测最终代价9、损失0.250780。尽管后者在训练选择指标上更好，前者已保存测试损失更低；因此不能把验证/代价优势直接推成泛化优势。','',
'Yeast空集一开始就满足预测约束，两种规则都没有发生属性排序。不能拿其零代价结果支持粒球排序贡献。','',
'## 结论与下一项检查','',
'依赖度硬下界的问题已消除，但优先依赖度增益仍可能牺牲预测改善效率并增加达标代价。在这三个开发划分上尚无稳定收益，不适合据此恢复正式实验扩展。','',
'不再追加混合系数或调整参数。下一项更有意义的检查是：有向距离诱导的邻域与实际预测器使用的邻域是否一致，以及能否在训练折内定义保留现有理论的预测映射。当前5NN只使用归一化原属性的欧氏邻域，标签驱动几何没有进入预测映射。这是两个目标衔接的结构问题；更换排序本身不能证明解决了它。','',
'完整逐步轨迹见paths.csv，同状态候选比较见same_parent_candidates.csv；每个数据集paths_before_test.json在测试评估前保存了路径。图search_paths.pdf/png展示损失与代价轨迹。']
(p/'PATH_AUDIT_REPORT.md').write_text('\n'.join(report))
print(json.dumps(summary,indent=2))

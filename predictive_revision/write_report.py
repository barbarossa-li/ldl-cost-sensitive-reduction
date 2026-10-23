from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent;rows=json.loads((p/'results.json').read_text());names=['Natural Scene','SBU_3DFE','Yeast_cold'];methods=['All attributes','Joint guard','Prediction-only','Predictive dependency guidance']
tex=['\\begin{center}','\\begin{tabular}{llrrrr}','\\hline','Dataset & Search & Attr. & Cost & Inner Brier & Test Brier\\\\','\\hline']
short={'All attributes':'Full','Joint guard':'Joint','Prediction-only':'Pred. only','Predictive dependency guidance':'Guided'}
for r in rows:
 tex.append(f"{r['dataset'].replace('_','-')} & {short[r['method']]} & {r['attributes']} & {r['cost']:.0f} & {r['inner_brier']:.6f} & {r['test_brier']:.6f}\\\\")
tex+=['\\hline','\\end{tabular}','\\end{center}',r'''All three candidate outputs pass the fixed full-set prediction guard and
are single-deletion irreducible under that guard. The Natural Scene output
has dependency 0.102931, below its old floor 0.113918; it is valid under the
revised objective. Its test loss improves relative to both saved searches,
but its cost increases and its validation loss is closer to the allowed
limit. On SBU-3DFE, guidance increases cost and does not improve test loss
relative to either search. On Yeast-cold, the empty subset passes immediately,
so no dependency ranking is performed; its improvement over joint search
is due to removing the hard feasibility floor, not evidence of a benefit
from directed geometry. The development pilot therefore validates the new
predicate and permits lower-dependency solutions, but does not establish
a stable advantage of dependency-guided ranking. Further formal benchmark
expansion remains deferred.

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{predictive_revision/pilot_comparison.pdf}
\caption{Development pilot on three previously examined splits. Points show
acquisition cost and saved test Brier loss for the two original searches
and the candidate guidance rule. Lower values are preferable on both axes.
The full-set losses are reported in the table; full-set costs lie outside
these reduced-search panels. Coincident Yeast-cold prediction-only and
guided points indicate the same zero-feature mean predictor.}
\end{figure}
''']
(p/'pilot_results.tex').write_text('\n'.join(tex))
report=['# 预测约束主导、依赖度指导的候选修订','',
'保留熵权、有向距离、粒球构造和依赖度定义。原双约束代码及结果保持原样。新版本单独实现，没有把诊断结果追溯替换为新算法结果。','',
'## 目标和规则','',
'最小化属性测试代价，约束仅为训练内部验证损失不超过完整属性集参考值。依赖度没有硬下界。','',
'1. 尚未满足预测约束时，先找出能降低内部验证损失的候选属性。','2. 其中如有正依赖度增益，按正依赖度增益乘原代价权重排序；否则按预测损失改善乘原代价权重排序。','3. 若所有候选都不能立即改善预测损失，按原始有符号预测增益作桥接，避免因互补属性而提前停止。','4. 达到预测可行后停止添加；逐个尝试删减，接受条件只检查预测约束，按代价降序并在成功删减后重新扫描。','',
'没有引入新的混合系数。此优先规则仍是启发式，不能证明比预测增益排序更好，也不保证全局最小代价。','',
'## 能证明什么','',
'最多添加m个属性必达完整集，因此终止且预测可行；最终没有满足约束的单属性删减，但不保证包含极小。对应回溯法仅按正代价下界及已可行节点剪枝，完成后返回全局最小预测可行代价，无需依赖度或预测损失的单调性。','',
'一旦采用预测可行域，精确最优解不由依赖度或λ决定。粒球和有向距离的贡献需要通过启发式路径的结果或效率体现，不能再说它们改变了该精确优化问题的可行域。','',
'80个任意非单调六属性损失/依赖度表的精确解均与穷举一致；另验证了低依赖度单属性、非正预测桥接、可行空集三个边界案例。','',
'## 小规模开发复核','',
'运行前固定三个数据集各第1次重复、训练192/测试256的既有划分，沿用相同三折、代价、λ=-1、零预测容差及5NN。子集确定并保存后才计算测试损失。测试对象此前已用于诊断，因此不能当作新的独立确认实验。','',
'| 数据集 | 方法 | 属性数 | 代价 | 验证Brier | 测试Brier |','|---|---|---:|---:|---:|---:|']
labels={'All attributes':'完整集','Joint guard':'原双约束','Prediction-only':'原预测约束','Predictive dependency guidance':'新依赖度指导'}
for r in rows:report.append(f"| {r['dataset']} | {labels[r['method']]} | {r['attributes']} | {r['cost']:.0f} | {r['inner_brier']:.6f} | {r['test_brier']:.6f} |")
report+=['',
'三个新输出都通过原完整集预测损失参考，并通过最终单删检查。Natural Scene的依赖度0.102931低于旧下界0.113918，仍被新规则接受；测试损失改善，但代价增加。SBU的代价增加且测试损失略恶化。Yeast返回空集，与原预测约束版本相同，没有发生依赖度排序。','',
'因此，本轮验证了目标和可行性规则的一致性，尚未证明依赖度指导搜索有稳定优势。Yeast改善来自取消硬下界，不能归功于有向距离或粒球排序。','',
'下一步仍应保持参数固定，优先隔离搜索排序的贡献：在相同预测可行性与删减规则下对比依赖度指导与纯预测指导，分析路径和代价，而不扩大期刊正式实验或继续增加混合参数。','',
'## 文件','',
'revised_search.py：候选启发式与预测约束回溯。check_search.py/search_checks.json：逻辑验证。run_pilot.py/protocol.json：固定开发协议。各数据集selection_before_test.json：选定子集、训练指标及路径。results.json：全部比较记录。正文第12节给出公式、规则、证明与开发结果。']
(p/'REVISION_REPORT.md').write_text('\n'.join(report))
fig,axes=plt.subplots(1,3,figsize=(10,3.2))
for ax,name in zip(axes,names):
 for method,color,marker in [('Joint guard','#b04d38','x'),('Prediction-only','#457fb0','s'),('Predictive dependency guidance','#187b59','o')]:
  r=next(r for r in rows if r['dataset']==name and r['method']==method);ax.scatter(r['cost'],r['test_brier'],s=65 if method=='Prediction-only' else 35,c=color,marker=marker,label=short[method],alpha=.8)
 ax.set_title(name.replace('_',' '));ax.set_xlabel('Acquisition cost');ax.grid(alpha=.2)
axes[0].set_ylabel('Test Brier loss');axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(p/'pilot_comparison.pdf');fig.savefig(p/'pilot_comparison.png',dpi=180)

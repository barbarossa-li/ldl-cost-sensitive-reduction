from pathlib import Path
import csv,json
p=Path(__file__).resolve().parent;root=p.parent
ss=list(csv.DictReader((p/'summary.csv').open()));paired=json.loads((p/'paired_diagnostics.json').read_text())
def get(n,m):return next(s for s in ss if int(s['train_n'])==n and s['method']==m)
def fmt(s,k,dec=4):return f"{float(s[k+'_mean']):.{dec}f} ± {float(s[k+'_sd']):.{dec}f}"
md=['# 扩大训练规模后的约束稳定性检验','', '实际完成：Natural Scene，192/384/576训练对象，每个规模5次重复，共15组划分、45个方法结果。每次外部测试256对象，内部三折，所有核心参数与容差保持不变。协议见PROTOCOL.md。', '', '## 主要结果','', '|训练数|方法|属性数（均值±SD）|模拟代价|内部Brier|外部Brier|联合可行次数|','|---:|---|---:|---:|---:|---:|---:|']
tex=[r'\section{Training-size and repeated-split stability}',r'\label{sec:stability}',r'Keeping the joint feasibility rule and every parameter fixed, we evaluate Natural Scene with 192, 384 and 576 training objects, each over five repeated random splits. Each split holds out 256 test objects. Within a repetition the test set is shared and training sets are nested. We exclude all 832 objects used in the two earlier pilots, leaving 1168 objects. Three-fold internal validation and the same five-neighbor distribution predictor are retained. Scaling is fitted within the respective training fold. Simulated costs and both zero tolerances are unchanged; no outer test result selects a parameter.',r'All feature selections are saved before test evaluation. Report means and sample standard deviations across five splits. Objects overlap across repetitions, so these standard deviations describe split sensitivity and are not independent-replicate confidence intervals. This is a single-dataset exploratory stability check, not a multi-dataset confirmation.',r'\begin{center}\small',r'\begin{tabular}{rlrrrr}\hline',r'$n$ & Method & Attributes & Cost & Inner Brier & Test Brier\\\hline']
for n in [192,384,576]:
 for m,label in [('All attributes','All'),('Dependency-only','Dep'),('Joint guard','Joint')]:
  s=get(n,m);md.append(f"|{n}|{m}|{fmt(s,'attributes',1)}|{fmt(s,'cost',1)}|{fmt(s,'inner_brier')}|{fmt(s,'test_brier')}|{s['joint_pass_count']}/5|")
  f=lambda k,d=4:fmt(s,k,d).replace(' ± ',r'$\pm$')
  tex.append(f"{n} & {label} & {f('attributes',1)} & {f('cost',1)} & {f('inner_brier')} & {f('test_brier')}\\\\")
tex += [r'\hline\end{tabular}\end{center}',r'\begin{figure}[htbp]\centering',r'\includegraphics[width=\linewidth]{stability/stability.pdf}',r'\caption{Fixed-parameter repeated-split diagnostics. Top: test Brier with descriptive standard deviations. Middle: selected attribute counts. Bottom: individual paired differences between joint and all-attribute test Brier; positive values mean worse prediction after reduction.}\label{fig:stability}\end{figure}',r'\subsection{Feasibility, predictive stability and subset stability}']
md+=['','## 配对与集合稳定性','','|训练数|联合−全部的外部Brier差值|相对差异均值|联合不劣于全部|平均Jaccard|','|---:|---:|---:|---:|---:|']
for d in paired:
 n=d['train_n'];md.append(f"|{n}|{d['outer_brier_delta_mean']:.4f} ± {d['outer_brier_delta_sd']:.4f}|{d['relative_delta_percent_mean']:+.1f}%|{d['no_worse_than_all_count']}/5|{d['pairwise_jaccard_mean']:.3f}|")
 tex.append(f"At $n={n}$, the mean paired test-Brier difference (joint minus all) is {d['outer_brier_delta_mean']:.4f}, with descriptive standard deviation {d['outer_brier_delta_sd']:.4f}. The joint output is no worse than all attributes on {d['no_worse_than_all_count']} of five splits. Its mean pairwise subset Jaccard is {d['pairwise_jaccard_mean']:.3f}.")
md += ['', '## 全属性依赖度参考与覆盖率', '', '|训练数|全属性依赖度|全属性合格覆盖率|联合解合格覆盖率|', '|---:|---:|---:|---:|']
for n in [192,384,576]:
 a=get(n,'All attributes');j=get(n,'Joint guard')
 md.append(f"|{n}|{fmt(a,'gamma')}|{fmt(a,'coverage')}|{fmt(j,'coverage')}|")
tex.append(r'The mean paired relative test-Brier increase falls from 18.2\% to 7.7\% to 5.0\%, but 14 of 15 joint outputs remain worse than all attributes. The sole improvement is only $0.000017$. In repetition four, $\Gamma(A)=0$ at both $n=384$ and $n=576$, and dependency-only search returns the empty set. This exposes a weak dependency reference; no causal benefit of the granular-ball component is established.')
md += ['', '结果解释：相对增幅是每次配对百分比增幅的平均值。192→384→576时增幅为18.2%→7.7%→5.0%，差距缩小，但14/15组仍劣于全部属性；唯一不劣组的Brier差值约−0.000017，几乎持平。平均Jaccard仅0.126→0.142→0.199，具体属性集合仍有较大划分敏感性。第四次重复的384与576规模全属性依赖度为零，仅依赖度方法返回空集；联合方法仍受预测上界约束。']
md+=['','全部15个联合输出均满足内部预测损失上界和训练依赖度下界。这说明实现稳定地执行可行性规则；它本身不证明外部预测性能保持。Jaccard是同一训练规模下五个输出集合两两交并比的平均值（10对），不应把不同划分对象当成独立推断。','', '内部损失使用相同验证折比较大量候选，可能出现选择过拟合。增加训练规模不能从理论上消除此问题。不能根据本表调整阈值后仍把这些测试集当作未使用的最终验证。','', '下一阶段宜在更多LDL数据集和多种LDL预测器上使用预先冻结的嵌套评估协议；如修改搜索准则，应另设外部确认实验。','', '原始结果results.csv包含KL、TV、JS、覆盖率、依赖度与每次属性集合；summary.csv保留各指标均值和标准差；每组目录包含划分与选择检查点。未运行全空间精确回溯，不宣称最优总代价。']
tex += [r'Every joint output satisfies both training constraints (15/15). This establishes empirical feasibility across these runs, not preservation of outer predictive performance. Repeated reuse of inner folds during feature search can overfit the selection criterion; larger training sets do not supply a theoretical generalization guarantee. Full KL, total variation, Jensen--Shannon, coverage and subset data accompany the reproducible package. No full-space exact cost certificate is claimed.',r'Further changes require a frozen nested protocol on additional LDL datasets and predictors, with separate outer confirmation.']
(p/'STABILITY_REPORT.md').write_text('\n'.join(md)+'\n');(root/'section_7_stability.tex').write_text('\n'.join(tex)+'\n')
main=root/'ASC_LDL_foundations.tex';s=main.read_text()
if '\\input{section_7_stability.tex}' not in s:s=s.replace('\\input{section_6_guarded_pilot.tex}','\\input{section_6_guarded_pilot.tex}\n\\input{section_7_stability.tex}')
main.write_text(s)
read=root/'READ_ME.txt';note='Repeated-size stability: Section 7 and stability/STABILITY_REPORT.md; 15 runs, fixed settings, 192/384/576 objects, five repeats per size. Reproduce using stability/run_stability.py then summarize.py and write_report.py.'
if note not in read.read_text():read.write_text(read.read_text()+'\n'+note+'\n')

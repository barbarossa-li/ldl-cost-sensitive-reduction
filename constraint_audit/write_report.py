from pathlib import Path
import csv
p=Path(__file__).resolve().parent;root=p.parent;counts=list(csv.DictReader((p/'counts.csv').open()));pairs=list(csv.DictReader((p/'paired.csv').open()))
text='''# 固定搜索路径下的依赖度约束审计

两个版本沿用同一条已保存的预算桥接添加路径、同样的15组划分、内部三折、预测器、代价上限与全部属性预测损失阈值。都选择最低代价可行前缀，再按最高代价优先删除并重启。唯一变化是开启版额外检查Gamma不低于全部属性参考。

本轮审计约束的作用，不审计原联合算法中Gamma增益排序的作用。共同路径无可行前缀不代表全空间无解；原联合输出是已知可行解。不改路径或将违规输出当作成功。

## 覆盖与独立作用

15组中，5组两个版本都成功，10组两个版本均没有预测可行前缀。成功的5组中，3组输出一致、2组输出不同。共同路径中3个前缀通过预测约束但被Gamma拒绝。不能把10组共同搜索失败当作约束没有作用。

|重复|训练数|关闭属性数/代价|开启属性数/代价|关闭测试Brier|开启测试Brier|
|---:|---:|---:|---:|---:|---:|
'''
for r in pairs:
 if r['changed']=='True':text+=f"|{r['repeat']}|{r['train_n']}|{r['off_attributes']}/{r['off_cost']}|{r['on_attributes']}/{r['on_cost']}|{float(r['off_brier']):.4f}|{float(r['on_brier']):.4f}|\n"
text+='''
这两组开启约束后分别降低测试Brier约0.0060、0.0048，但属性数和代价均增加，不能据此声称相同实际资源用量下有普遍收益。

## 原联合解的删除审计

对15个原联合输出逐个删除属性：7个候选仍通过预测损失约束，但仅因Gamma下界被拒绝，涉及4组划分。保存训练判断后再计算测试表现：3个删除会使外部Brier变差，保留属性有好处；另4个删除反而使外部Brier更低，约束阻止了更便宜且测试损失更低的候选。候选来自相同划分时具有相关性，不能当作独立重复。

## 判断

约束不是完全冗余，但独立预测收益不稳定。保留当前核心与冻结参数，下一步在其他真实LDL数据集复核，而不是依据已观察的测试结果调阈值。这里是探索性机制诊断，不是外部预测保持保证。

复现顺序：run_audit.py → audit_original_deletions.py → summarize.py → write_report.py。counts.csv、paired.csv、blocked_deletions.csv保留完整数值；各划分保留候选事件与选择检查点。无可行前缀的组明确记为失败，未参与成功配对的损失比较。
'''
(p/'CONSTRAINT_REPORT.md').write_text(text)
tex=r'''\section{Constraint-only audit on a frozen search path}
\label{sec:constraint-audit}
Both variants reuse the same stored budget-bridge addition sequence on each of the 15 splits, identical costs, folds, predictor, budget and full-attribute risk reference. They choose the lowest-cost risk-feasible prefix and apply the same highest-cost-first deletion/restart rule. Only the on variant additionally requires $\Gamma(P)\ge\Gamma(A)$. Addition ranking does not use dependency here.

This audits the feasibility constraint, not dependency-based ranking in the original heuristic. Path failure does not imply global infeasibility: the original joint output is known to be feasible. We do not switch paths or report violating outputs as successful. Existing test observations make this exploratory.

\begin{center}\small\begin{tabular}{rrrrrr}\hline
$n$ & Off found & On found & Changed & Prefix blocks & Blocked drops\\\hline
'''
for r in counts:tex+=' & '.join(r[k] for k in ['train_n','off_feasible','on_feasible','different_outputs','prefix_gamma_only_blocks','original_blocked_deletions'])+r'\\'+'\n'
tex+=r'''\hline\end{tabular}\end{center}
Both variants find outputs on the same five splits. On ten splits the common path has no risk-feasible prefix; these are shared search failures, not demonstrations of an inactive constraint. Among the five available pairs, three outputs are identical and two change. Three common prefixes pass risk but fail dependency.

\subsection{The two changed paired outputs}
\begin{center}\small\begin{tabular}{rrrrrr}\hline
Repeat & $n$ & Off cost & On cost & Off Brier & On Brier\\\hline
'''
for r in pairs:
 if r['changed']=='True':tex+=f"{r['repeat']} & {r['train_n']} & {r['off_cost']} & {r['on_cost']} & {float(r['off_brier']):.4f} & {float(r['on_brier']):.4f}"+r'\\'+'\n'
tex+=r'''\hline\end{tabular}\end{center}
The two changed outputs lower test Brier by about 0.0060 and 0.0048, at increased cost and cardinality. This establishes an effect of the constraint on these paths, not universal benefit at identical actual resource usage.

\subsection{Deletion decisions in the original joint outputs}
Seven deletions across four original-output splits pass risk but fail dependency. Decisions are saved before test evaluation. Three deletions worsen test Brier; four improve it and reduce cost. These correlated candidates show that the constraint blocks both harmful and beneficial changes.

\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{constraint_audit/constraint_effect.pdf}
\caption{Gamma-only blocked deletions from original outputs (top) and on-minus-off test Brier on the five available fixed-path pairs (bottom). Negative differences favor on. The ten shared path failures are explicitly reported and omitted from this paired plot.}\label{fig:constraint-audit}
\end{figure}
The constraint is not entirely redundant, but its independent predictive benefit is unstable. Preserve the current core and frozen settings for additional genuine LDL datasets; these observations do not establish predictive preservation or justify tuning thresholds on observed test data.
'''
(root/'section_9_constraint_audit.tex').write_text(tex)
main=root/'ASC_LDL_foundations.tex';s=main.read_text()
if '\\input{section_9_constraint_audit.tex}' not in s:main.write_text(s.replace('\\input{section_8_baselines.tex}','\\input{section_8_baselines.tex}\n\\input{section_9_constraint_audit.tex}'))
read=root/'READ_ME.txt';note='Constraint-only fixed-path audit: Section 9 and constraint_audit/CONSTRAINT_REPORT.md. Shared path failures, original deletion diagnostics and paired effects retained.'
if note not in read.read_text():read.write_text(read.read_text()+'\n'+note+'\n')

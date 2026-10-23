from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent
s=json.loads((out/'summary.json').read_text());primary=[r for r in s if r['learner']=='Euclidean'];data=list(csv.DictReader(open(out/'candidates.csv')))
fig,axes=plt.subplots(3,2,figsize=(9,8),layout='constrained')
for ax,r in zip(axes.flat,primary):
 q=[x for x in data if x['dataset']==r['dataset'] and x['base']==r['base'] and x['learner']=='Euclidean' and x['operation']=='add'];x=np.array([float(v['gamma_change']) for v in q]);y=np.array([float(v['gain']) for v in q]);ax.scatter(x,y,s=13,alpha=.65,color='#226c93');ax.axhline(0,color='gray',lw=.7);ax.axvline(0,color='gray',lw=.7);ax.set_title(f"{r['dataset']} | {r['base']}\nSpearman = {r['spearman_gamma_vs_prediction']:.3f}",fontsize=10);ax.set_xlabel('Training dependency gain');ax.set_ylabel('Inner Brier improvement');ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2));ax.grid(alpha=.15)
fig.savefig(out/'gamma_prediction.pdf');fig.savefig(out/'gamma_prediction.png',dpi=180);plt.close(fig)
header='| 数据集 | 固定子集 | 秩相关 | 正依赖增益数 | 同时改善预测 | 负依赖增益但改善预测 |\n|---|---|---:|---:|---:|---:|\n'
body=''.join(f"| {r['dataset']} | {r['base']} | {r['spearman_gamma_vs_prediction']:.3f} | {r['positive_gamma']} | {r['positive_both']} | {r['negative_gamma_prediction_better']} |\n" for r in primary)
(out/'GAMMA_REPORT.md').write_text('# 固定子集的依赖度边际增益诊断\n\n'+header+body+'\n依赖度按原搜索流程在全部192个开发训练样本上重新生成粒球；预测增益复用三折内部验证记录。没有新约简或外部测试评估。两个 Yeast 空子集是重复语境。\n\n依赖度边际增益与条件预测增益的对应较弱。正依赖度增益不足以保证预测改善，负依赖度增益也不能用于排除预测有益的候选。训练表示恒等式仍成立，不能提升为验证预测保证。应保留预测损失作为可行性判据，暂不把依赖度优先级作为已验证的性能贡献。下一步以等预算、相同预测约束的属性交换检查成本改进空间，再决定如何定位粒球贡献。\n')
lines=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Training dependency marginals versus conditional Euclidean prediction gains. Counts concern additions only.}',r'\begin{tabular}{llrrrr}',r'\hline',r'Dataset & Base & Rank corr. & $\Delta\Gamma>0$ & Both positive & $\Delta\Gamma<0,g_R>0$\\',r'\hline']
for r in primary:
 name=r['dataset'].replace('_',r'\_');b='Pred.' if r['base']=='Prediction only' else 'Guided'
 lines.append(f"{name} & {b} & {r['spearman_gamma_vs_prediction']:.3f} & {r['positive_gamma']} & {r['positive_both']} & {r['negative_gamma_prediction_better']}\\\\")
lines += [r'\hline',r'\end{tabular}',r'\end{table}'];(out/'gamma_table.tex').write_text(('\n'.join(lines)+'\n').replace('\\begin{tabular}', '\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}').replace('\\end{tabular}', '\\end{tabular}}'))

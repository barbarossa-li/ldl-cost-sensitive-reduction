from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent;a=json.loads((out/'audit_results.json').read_text());r=json.loads((out/'results.json').read_text());summary=[];gaps=[]
methods=['Full','Empty','Prediction only before','Prediction only after','Guided before','Guided after'];names=['Natural Scene','SBU_3DFE','Yeast_cold']
for name in names:
 for method in methods:
  q=[v for v in a if v['dataset']==name and v['method']==method];full=[next(v['audit_brier'] for v in a if v['dataset']==name and v['repeat']==rep and v['method']=='Full') for rep in range(1,6)];summary.append(dict(dataset=name,method=method,cost_mean=float(np.mean([v['cost'] for v in q])),audit_brier_mean=float(np.mean([v['audit_brier'] for v in q])),audit_brier_sd=float(np.std([v['audit_brier'] for v in q],ddof=1)),passes_full_reference=sum(v['audit_brier']<=f for v,f in zip(q,full)),empty_outputs=sum(v['attributes']==0 for v in q)))
 for method in ['Prediction only','Guided']:
  for rep in range(1,6):
   s=next(v for v in r if v['dataset']==name and v['repeat']==rep and v['method']==method);q=next(v for v in a if v['dataset']==name and v['repeat']==rep and v['method']==method+' after');full=next(v for v in a if v['dataset']==name and v['repeat']==rep and v['method']=='Full');d=s['after_loss']-s['full_reference'];e=q['audit_brier']-full['audit_brier'];gaps.append(dict(dataset=name,repeat=rep,method=method,inner_relative_loss=d,audit_relative_loss=e,relative_transfer_gap=e-d,inner_feasible=d<=0,audit_feasible=e<=0));assert d<=0
(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'transfer_gaps.json').write_text(json.dumps(gaps,indent=2))
with (out/'transfer_gaps.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(gaps[0]));w.writeheader();w.writerows(gaps)
fig,axes=plt.subplots(1,3,figsize=(10,3.6),layout='constrained')
for ax,name in zip(axes,names):
 for method,color,marker in [('Prediction only','#226c93','o'),('Guided','#bb6138','s')]:
  q=[v for v in gaps if v['dataset']==name and v['method']==method];ax.scatter([v['inner_relative_loss'] for v in q],[v['audit_relative_loss'] for v in q],color=color,marker=marker,s=40,label=method,alpha=.8)
 ax.axhline(0,color='gray',lw=.8);ax.axvline(0,color='gray',lw=.8);ax.set_title(name);ax.set_xlabel('Inner relative Brier loss',fontsize=9);ax.set_ylabel('Audit relative Brier loss',fontsize=9);ax.ticklabel_format(style='plain',useOffset=False);ax.tick_params(labelsize=7);ax.xaxis.set_major_locator(MaxNLocator(3));ax.yaxis.set_major_locator(MaxNLocator(4));ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(out/'constraint_transfer.pdf');fig.savefig(out/'constraint_transfer.png',dpi=180);plt.close(fig)
lines=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Independent-within-run development audit: five repeats, search size 144 and audit size 48. Pass counts compare the same-run full-attribute predictor.}',r'\resizebox{\linewidth}{!}{%',r'\begin{tabular}{llrrrr}',r'\hline',r'Dataset & Method & Cost & Audit Brier & Passes & Empty outputs\\',r'\hline']
short={'Full':'Full','Empty':'Empty','Prediction only before':'P before','Prediction only after':'P after','Guided before':'G before','Guided after':'G after'}
for v in summary:lines.append(f"{v['dataset'].replace('_',chr(92)+'_')} & {short[v['method']]} & {v['cost_mean']:.2f} & {v['audit_brier_mean']:.6f} & {v['passes_full_reference']}/5 & {v['empty_outputs']}/5\\\\")
lines += [r'\hline',r'\end{tabular}}',r'\end{table}'];(out/'audit_table.tex').write_text('\n'.join(lines)+'\n')
header='| 数据集 | 方法 | 平均成本 | 复核Brier | 达到完整集参考 | 空集次数 |\n|---|---|---:|---:|---:|---:|\n';body=''.join(f"| {v['dataset']} | {v['method']} | {v['cost_mean']:.2f} | {v['audit_brier_mean']:.6f} | {v['passes_full_reference']}/5 | {v['empty_outputs']}/5 |\n" for v in summary)
(out/'HOLDOUT_REPORT.md').write_text('# 搜索与训练内部复核隔离\n\n'+header+body+'\n预先固定随机划分，将每次192个开发训练样本分为144搜索、48复核；3折搜索内部验证每折96拟合/48验证。两方法完成选集及精修并保存后，训练全部144搜索样本预测48复核对象。复核标签和属性统计不参与搜索或拟合；不使用外部测试。\n\n30个精修结果全部满足内部参考，Natural Scene预测优先仅0/5传递到复核，指导1/5；SBU各1/5；Yeast各5/5且全部为空。隔离复核揭示约束的经验可靠性不足，但不证明自适应搜索偏差是唯一原因。搜索训练由192减至144、复核仅48且各次划分重叠，均限制解释。不能把复核集重新用作反复筛选目标并仍称独立复核。\n\n核心理论参数和搜索精修规则保持不变；下一步用已保存较大开发训练规模、相同3:1隔离和冻结规则检验样本量影响，不根据复核结果调参或扩大正式数据集比较。\n')
print('inner feasible',sum(v['inner_feasible'] for v in gaps),'audit feasible',sum(v['audit_feasible'] for v in gaps))

from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent
old=json.loads((root/'refinement_pilot/results.json').read_text());new=json.loads((out/'results.json').read_text());assert len(new)==12 and len(old)==18
s=sorted(old+new,key=lambda r:(r['dataset'],r['repeat'],r['method']));assert len({(r['dataset'],r['repeat'],r['method']) for r in s})==30
(out/'five_repeat_results.json').write_text(json.dumps(s,indent=2))
with (out/'five_repeat_results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(s[0]));w.writeheader();w.writerows(s)
summ=[];paired=[]
for name in ['Natural Scene','SBU_3DFE','Yeast_cold']:
 for method in ['Prediction only','Guided']:
  q=[r for r in s if r['dataset']==name and r['method']==method];summ.append(dict(dataset=name,method=method,before_mean_cost=float(np.mean([r['before_cost'] for r in q])),after_mean_cost=float(np.mean([r['after_cost'] for r in q])),after_sd_cost=float(np.std([r['after_cost'] for r in q],ddof=1)),improved=sum(r['after_cost']<r['before_cost'] for r in q),budget_limited=sum(r['stop_reason'].startswith('budget exhausted') for r in q),loss_increased=sum(r['after_loss']>r['before_loss']+1e-12 for r in q),all_feasible=all(r['after_loss']<=r['full_reference'] for r in q),mean_initial_seconds=float(np.mean([r['initial_seconds'] for r in q])),mean_refinement_seconds=float(np.mean([r['refinement_seconds'] for r in q]))))
 pairs=[]
 for rep in range(1,6):
  p=next(r for r in s if r['dataset']==name and r['repeat']==rep and r['method']=='Prediction only');g=next(r for r in s if r['dataset']==name and r['repeat']==rep and r['method']=='Guided');pairs.append(dict(repeat=rep,prediction_cost=p['after_cost'],guided_cost=g['after_cost'],guided_minus_prediction=g['after_cost']-p['after_cost']))
 paired.append(dict(dataset=name,guided_cheaper=sum(r['guided_minus_prediction']<0 for r in pairs),tied=sum(r['guided_minus_prediction']==0 for r in pairs),guided_more_expensive=sum(r['guided_minus_prediction']>0 for r in pairs),pairs=pairs))
(out/'summary.json').write_text(json.dumps(dict(methods=summ,paired=paired),indent=2))
fig,axes=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
for ax,name in zip(axes,['Natural Scene','SBU_3DFE']):
 for method,color,offset in [('Prediction only','#226c93',-.08),('Guided','#bb6138',.08)]:
  q=[r for r in s if r['dataset']==name and r['method']==method];x=np.arange(1,6)+offset;before=[r['before_cost'] for r in q];after=[r['after_cost'] for r in q];ax.vlines(x,after,before,color=color,lw=2);ax.scatter(x,before,color=color,marker='o',facecolors='none',s=55);ax.scatter(x,after,color=color,marker='s',s=40,label=method)
 ax.axvline(3.5,color='gray',ls='--',lw=1);ax.set_title(name);ax.set_xticks(range(1,6));ax.set_xlabel('Saved repeat (4–5: fixed-rule extension)');ax.set_ylabel('Acquisition cost');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Open circle: before refinement | Square: after refinement',fontsize=10);fig.savefig(out/'five_repeat_cost.pdf');fig.savefig(out/'five_repeat_cost.png',dpi=180);plt.close(fig)
lines=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Five-repeat development refinement summary. SD describes variation across overlapping saved splits, not independent uncertainty.}',r'\resizebox{\linewidth}{!}{%',r'\begin{tabular}{llrrrrr}',r'\hline',r'Dataset & Method & Before mean & After mean $\pm$ SD & Improved & Cap stops & Loss increases\\',r'\hline']
for r in summ:lines.append(f"{r['dataset'].replace('_',chr(92)+'_')} & {'Pred.' if r['method']=='Prediction only' else 'Guided'} & {r['before_mean_cost']:.2f} & {r['after_mean_cost']:.2f} $\\pm$ {r['after_sd_cost']:.2f} & {r['improved']}/5 & {r['budget_limited']}/5 & {r['loss_increased']}/5\\\\")
lines += [r'\hline',r'\end{tabular}}',r'\end{table}'];(out/'five_repeat_table.tex').write_text('\n'.join(lines)+'\n')
header='| 数据集 | 方法 | 原平均成本 | 精修成本均值±标准差 | 降成本次数 | 预算停止 | 损失较起点增加 |\n|---|---|---:|---:|---:|---:|---:|\n';body=''.join(f"| {r['dataset']} | {r['method']} | {r['before_mean_cost']:.2f} | {r['after_mean_cost']:.2f}±{r['after_sd_cost']:.2f} | {r['improved']}/5 | {r['budget_limited']}/5 | {r['loss_increased']}/5 |\n" for r in summ)
comparisons='\n'.join(f"- {r['dataset']}：指导成本更低{r['guided_cheaper']}次，相同{r['tied']}次，更高{r['guided_more_expensive']}次。" for r in paired)
(out/'FIVE_REPEAT_REPORT.md').write_text('# 固定规则的第4、5次划分复核与五次汇总\n\n'+header+body+'\n'+comparisons+'\n\n新做12次运行，与已保存18次合并；参数、接受规则和200次精修预算保持不变。所有最终子集都满足原内部验证上界。低成本允许损失较起点增大，不等于预测提升。预算耗尽不保证局部最优。划分可能重叠，标准差仅描述波动，不是独立试验的不确定性。两方法初始搜索工作量不同，仅精修预算上限相同。\n\n下一步应对已经冻结的完整选集加精修流程评估外部预测，报告完整集、空集和精修前后，不能利用测试结果再改规则。既有测试对象曾用于此前诊断，应明确开发复用性质，不标为全新确认实验。\n')
print(json.dumps(dict(methods=summ,paired=paired),indent=2))

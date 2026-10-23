from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent
base=[v for v in json.loads((root/'holdout_size_audit/audit_results.json').read_text()) if v['development_size']==576];new=json.loads((out/'audit_results.json').read_text());assert len(new)==36;s=base+new;assert len(s)==54
initial=[v for v in json.loads((root/'holdout_size_audit/results.json').read_text()) if v['development_size']==576]+json.loads((out/'results.json').read_text());assert len(initial)==18
common=json.loads((out/'common48_results.json').read_text());summary=[];paired=[];common_summary=[]
names=['Natural Scene','SBU_3DFE','Yeast_cold'];methods=['Full','Empty','Prediction only before','Prediction only after','Guided before','Guided after']
for name in names:
 for method in methods:
  q=sorted([v for v in s if v['dataset']==name and v['method']==method],key=lambda v:v['repeat']);full=[next(v['audit_brier'] for v in s if v['dataset']==name and v['repeat']==rep and v['method']=='Full') for rep in [1,2,3]];summary.append(dict(dataset=name,method=method,cost_mean=float(np.mean([v['cost'] for v in q])),audit_brier_mean=float(np.mean([v['audit_brier'] for v in q])),audit_brier_sd=float(np.std([v['audit_brier'] for v in q],ddof=1)),passes=sum(v['audit_brier']<=f for v,f in zip(q,full)),empty_outputs=sum(v['attributes']==0 for v in q)))
 for method in ['Prediction only','Guided']:
  for rep in [1,2,3]:
   q=next(v for v in s if v['dataset']==name and v['repeat']==rep and v['method']==method+' after');full=next(v['audit_brier'] for v in s if v['dataset']==name and v['repeat']==rep and v['method']=='Full');d=next(v for v in initial if v['dataset']==name and v['repeat']==rep and v['method']==method);paired.append(dict(dataset=name,repeat=rep,method=method,audit_gap=q['audit_brier']-full,cost=q['cost'],inner_gap=d['after_loss']-d['full_reference']))
  for size in [192,576]:
   gaps=[]
   for rep in [1,2,3]:
    q=next(v['brier'] for v in common if v['dataset']==name and v['repeat']==rep and v['development_size']==size and v['method']==method+' after');f=next(v['brier'] for v in common if v['dataset']==name and v['repeat']==rep and v['development_size']==size and v['method']=='Full');gaps.append(q-f)
   common_summary.append(dict(dataset=name,method=method,development_size=size,mean_gap=float(np.mean(gaps)),passes=sum(v<=0 for v in gaps),gaps=gaps))
(out/'summary.json').write_text(json.dumps(dict(means=summary,paired=paired,common48=common_summary),indent=2));(out/'three_repeat_audit.json').write_text(json.dumps(s,indent=2))
fig,axes=plt.subplots(1,3,figsize=(10,3.6),layout='constrained')
for ax,name in zip(axes,names):
 for method,color,offset in [('Prediction only','#226c93',-.05),('Guided','#bb6138',.05)]:
  q=[v for v in paired if v['dataset']==name and v['method']==method];ax.plot(np.arange(1,4)+offset,[v['audit_gap'] for v in q],marker='o' if method=='Prediction only' else 's',color=color,label=method,alpha=.8)
 ax.axhline(0,color='gray',lw=.8);ax.set_title(name);ax.set_xticks([1,2,3]);ax.set_xlabel('Saved development repeat');ax.set_ylabel('Audit Brier minus full reference');ax.yaxis.set_major_locator(MaxNLocator(4));ax.tick_params(labelsize=8);ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(out/'large_repeats.pdf');fig.savefig(out/'large_repeats.png',dpi=180);plt.close(fig)
lines=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Three-repeat frozen evaluation at search size 432 and reserved audit size 144. All means refer to development audits, not outer tests.}',r'\resizebox{\linewidth}{!}{%',r'\begin{tabular}{llrrrr}',r'\hline',r'Dataset & Method & Cost & Audit Brier $\pm$ SD & Passes & Empty outputs\\',r'\hline'];short={'Full':'Full','Empty':'Empty','Prediction only before':'P before','Prediction only after':'P after','Guided before':'G before','Guided after':'G after'}
for v in summary:lines.append(f"{v['dataset'].replace('_',chr(92)+'_')} & {short[v['method']]} & {v['cost_mean']:.2f} & {v['audit_brier_mean']:.6f} $\\pm$ {v['audit_brier_sd']:.6f} & {v['passes']}/3 & {v['empty_outputs']}/3\\\\")
lines += [r'\hline',r'\end{tabular}}',r'\end{table}'];(out/'large_table.tex').write_text('\n'.join(lines)+'\n')
header='| 数据集 | 方法 | 平均成本 | 复核Brier均值±标准差 | 达到完整集参考 | 空集次数 |\n|---|---|---:|---:|---:|---:|\n';body=''.join(f"| {v['dataset']} | {v['method']} | {v['cost_mean']:.2f} | {v['audit_brier_mean']:.6f}±{v['audit_brier_sd']:.6f} | {v['passes']}/3 | {v['empty_outputs']}/3 |\n" for v in summary)
(out/'LARGE_REPORT.md').write_text('# 576规模另两次固定规则复核\n\n'+header+body+'\n三次均为432搜索/144复核，内层每折288拟合/144验证。首次记录保留，新增第2、3次；不改变规则和200次精修预算、不使用外部测试。三次保存划分可能重叠，标准差描述波动，不是独立确认的不确定性。固定48复核对象对照另存。\n')
print(json.dumps(dict(means=summary,common48=common_summary),indent=2))

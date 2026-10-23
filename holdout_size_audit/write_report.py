from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent
old=json.loads((root/'internal_holdout_audit/audit_results.json').read_text());old=[dict(v,development_size=192,search_size=144,audit_size=48) for v in old if v['repeat']==1];new=json.loads((out/'audit_results.json').read_text());assert len(new)==36
allrows=old+new;common=json.loads((out/'common48_results.json').read_text());summary=[]
for name in ['Natural Scene','SBU_3DFE','Yeast_cold']:
 p192=json.loads((root/'internal_holdout_audit'/(name.replace(' ','_')+'_repeat_1_partition.json')).read_text());previous=p192
 for size in [192,384,576]:
  if size>192:
   current=json.loads((out/(name.replace(' ','_')+f'_repeat_1_n{size}_partition.json')).read_text());assert current['search_ids'][:len(previous['search_ids'])]==previous['search_ids'] and current['audit_ids'][:len(previous['audit_ids'])]==previous['audit_ids'];previous=current
  q=[v for v in allrows if v['dataset']==name and v['development_size']==size];full=next(v for v in q if v['method']=='Full');empty=next(v for v in q if v['method']=='Empty');cf=next(v['brier'] for v in common if v['dataset']==name and v['development_size']==size and v['method']=='Full')
  for method in ['Prediction only','Guided']:
   v=next(v for v in q if v['method']==method+' after');c=next(v['brier'] for v in common if v['dataset']==name and v['development_size']==size and v['method']==method+' after');summary.append(dict(dataset=name,development_size=size,search_size=v['search_size'],audit_size=v['audit_size'],method=method,cost=v['cost'],attributes=v['attributes'],full_audit_brier=full['audit_brier'],empty_audit_brier=empty['audit_brier'],audit_brier=v['audit_brier'],audit_gap=v['audit_brier']-full['audit_brier'],audit_pass=v['audit_brier']<=full['audit_brier'],common48_brier=c,common48_full_brier=cf,common48_gap=c-cf,common48_pass=c<=cf))
(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'all_scale_audit.json').write_text(json.dumps(allrows,indent=2))
with (out/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
fig,axes=plt.subplots(2,3,figsize=(10,6),layout='constrained')
for j,name in enumerate(['Natural Scene','SBU_3DFE','Yeast_cold']):
 for i,key in enumerate(['audit_gap','common48_gap']):
  ax=axes[i,j]
  for method,color,marker in [('Prediction only','#226c93','o'),('Guided','#bb6138','s')]:
   q=[v for v in summary if v['dataset']==name and v['method']==method];ax.plot([v['search_size'] for v in q],[v[key] for v in q],color=color,marker=marker,label=method,alpha=.8)
  ax.axhline(0,color='gray',lw=.8);ax.set_title(name+(' | expanding audit' if i==0 else ' | common 48 audit'),fontsize=10);ax.set_xticks([144,288,432]);ax.set_xlabel('Search training objects');ax.set_ylabel('Brier minus full reference');ax.tick_params(labelsize=8);ax.yaxis.set_major_locator(MaxNLocator(4));ax.grid(alpha=.2)
  if j==0:ax.legend(fontsize=8)
fig.savefig(out/'size_transfer.pdf');fig.savefig(out/'size_transfer.png',dpi=180);plt.close(fig)
lines=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Repeat-one nested scale diagnostic. P and G denote refined prediction-only and guided outputs; positive gaps are worse than full attributes.}',r'\resizebox{\linewidth}{!}{%',r'\begin{tabular}{lrrrrrr}',r'\hline',r'Dataset & Search/audit & P cost & G cost & Full Brier & P Brier & G Brier\\',r'\hline']
header='| 数据集 | 搜索/复核 | P成本 | G成本 | 完整集复核Brier | P复核Brier | G复核Brier | 固定48复核差值P/G |\n|---|---|---:|---:|---:|---:|---:|---|\n';body=''
for name in ['Natural Scene','SBU_3DFE','Yeast_cold']:
 for size in [192,384,576]:
  p=next(v for v in summary if v['dataset']==name and v['development_size']==size and v['method']=='Prediction only');g=next(v for v in summary if v['dataset']==name and v['development_size']==size and v['method']=='Guided');lines.append(f"{name.replace('_',chr(92)+'_')} & {p['search_size']}/{p['audit_size']} & {p['cost']} & {g['cost']} & {p['full_audit_brier']:.6f} & {p['audit_brier']:.6f} & {g['audit_brier']:.6f}\\\\");body+=f"| {name} | {p['search_size']}/{p['audit_size']} | {p['cost']} | {g['cost']} | {p['full_audit_brier']:.6f} | {p['audit_brier']:.6f} | {g['audit_brier']:.6f} | {p['common48_gap']:+.6f}/{g['common48_gap']:+.6f} |\n"
lines += [r'\hline',r'\end{tabular}}',r'\end{table}'];(out/'size_table.tex').write_text('\n'.join(lines)+'\n')
(out/'SIZE_REPORT.md').write_text('# 固定规则的嵌套样本规模诊断\n\n'+header+body+'\nP为预测优先加精修，G为粒球指导加精修；差值为候选损失减完整集损失，正值表示未达到参考。仅第1次已保存划分，不是规模上的重复确认。按192样本块进行固定3:1分配，嵌套搜索和复核成员保持角色；外部测试未使用。额外固定原48个复核对象，控制复核构成变化。\n\n核心参数及200次精修上限不变，内部完整集参考随搜索数据重新计算。每折拟合样本为96、192、288。不同规模的选集、权重拟合、完整集参考均会变化，故不是纯粹的单因素因果实验，也不能把连接曲线当作总体趋势。只记录冻结程序的结果，不依复核结果调规则。\n')
print(json.dumps(summary,indent=2))

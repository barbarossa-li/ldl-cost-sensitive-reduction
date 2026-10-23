from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=Path(__file__).resolve().parent;rr=json.loads((p/'summary.json').read_text())
tex=[r'\begin{center}',r'\begin{tabular}{lrrrr}',r'\hline',r'Dataset & Raw KL & Shrunk KL & $B_L$ & Mean $W$\\',r'\hline']
for r in rr:tex.append(f"{r['dataset'].replace('_','-')} & {r['raw_kl_bits']:.6f} & {r['regularized_kl_bits']:.6f} & {r['normalizer_bits_mean']:.3f} & {r['mean_weight']:.7f}\\\\")
tex += [r'\hline',r'\end{tabular}',r'\end{center}'];(p/'decomposition_table.tex').write_text('\n'.join(tex))
report=['# 信息权重机制分解：保持现有定义与参数','',
'本轮仅使用三个既有开发划分的9个内部训练折，每折128个对象；未使用验证或外部测试标签，未改变带宽、β、ζ、归一化、预测器或属性子集。重算1683个属性/折记录，与原权重最大误差1.39×10⁻¹⁶。','',
'## 三步分解','',
'令p为训练全局标签分布、u为原始核加权局部分布、η=t/(t+β)，则：','',
r'$$D_{\rm raw}=D_{\rm KL}(u\Vert p),\quad D_{\rm reg}=D_{\rm KL}(\eta u+(1-\eta)p\Vert p),\quad W=D_{\rm reg}/B_L.$$','',
'KL使用以2为底的对数，单位为bit。下面是对各属性、源对象及3折汇总的均值；B_L为3折均值，各折恒等式单独核对，不能用汇总KL除以汇总B_L当作精确汇总W。','',
'| 数据集 | 原始局部KL | 收缩后KL | B_L | 平均W |','|---|---:|---:|---:|---:|']
for r in rr:report.append(f"| {r['dataset']} | {r['raw_kl_bits']:.6f} | {r['regularized_kl_bits']:.6f} | {r['normalizer_bits_mean']:.3f} | {r['mean_weight']:.7f} |")
report += ['','SBU和Yeast在归一化前已明显弱于Natural Scene。它们的B_L反而更小，跨数据集的权重差距不能主要归因于共同归一化。以Natural Scene/Yeast比较，收缩后KL相差约212倍，而归一化W约88倍，归一化反而缩小了这次跨数据集差异。此比较不是跨数据集尺度不变性声明。','',
'## 核平均与支持收缩','',
'有效核样本数定义为N_eff=(ΣK)²/ΣK²，它是加权分散程度，不是实际近邻个数。局部相对个体KL基线的保留比，使用每个源的核加权个体KL作为Jensen上界后，汇总原始局部KL/汇总上界。它是描述性凸性比率，不表示剩余可预测信息比例，也不能证明带宽选错。','',
'| 数据集 | 平均N_eff/128 | 局部KL/核加权个体KL | 收缩KL保留比 | 平均η |','|---|---:|---:|---:|---:|']
for r in rr:report.append(f"| {r['dataset']} | {r['effective_n_mean']:.2f} | {100*r['averaging_retention']:.2f}% | {100*r['shrink_retention']:.2f}% | {r['eta_mean']:.4f} |")
report += ['','核轮廓平均覆盖较广，与局部—全局差异偏弱一致，但仍无法区分真实关联弱、宽核平均抑制信号及样本噪声。训练对象自身的平均KL也不同：Natural Scene约1.4166bit、SBU约0.1144bit、Yeast约0.0180bit，因此不同数据集原本的标签变化尺度不能忽略。','',
'平均η接近1不代表KL变化很小：原始KL可能集中在少数低支持位置，且KL不是线性函数。不能用平均η直接估算KL保留比。','',
'## 低支持位置的作用','',
'下表固定以其他对象核支持t<1划分，仅用于诊断，不改变原算法条件。原始u包含源自身标签，此时自身核质量在总支持中超过一半。','',
'| 数据集 | t<1位置占比 | 这些位置的原始KL占比 | 这些位置的收缩KL占比 |','|---|---:|---:|---:|']
for r in rr:report.append(f"| {r['dataset']} | {100*r['source_fraction_other_support_lt1']:.3f}% | {100*r['raw_kl_mass_other_support_lt1']:.2f}% | {100*r['regularized_kl_mass_other_support_lt1']:.2f}% |")
report += ['','Natural Scene仅1.50%的低支持位置贡献29.26%的原始KL，收缩后贡献2.42%。这说明部分大原始权重来自自身标签占比较高的估计，支持收缩在削弱它们；不能把取消收缩后的大权重自动当作有效属性关联。这里不声称低支持位置全部是噪声。','',
'## 可证明的关系','',
'由KL对第一个参数的凸性，有D_KL(u_i||p)≤Σ_j(K_ij/s_i)D_KL(ℓ_j||p)，以及D_KL(η_i u_i+(1−η_i)p||p)≤η_iD_KL(u_i||p)。合并即可给出W的上界。所有1683个属性/折逐源检查均符合，仅存在机器精度量级的舍入。','',
'## 下一步','',
'目前可确认弱差异在归一化前就存在，并且支持收缩具有抑制低支持自影响的作用。尚不能确认弱差异意味着属性无信息，也不能宣布核带宽错误。','',
'建议维持现有定义和参数，在各训练折内对标签分布向量做整行置换对照：保持属性、标签边际、带宽和支持不变，只破坏属性与标签的对应，比较真实权重与置换权重。先判断是否存在超出该对照的属性关联，再决定是否需要讨论局部轮廓的建模；暂不修改理论、调参或扩展正式实验。','',
'属性/折明细见attributes.csv，支持分层见support_bins.csv，9折来源见fold_provenance.json；summary.json、validation.json和run_audit.py可重现结果。正文第16节收录分解公式、凸性界和诊断图。']
(p/'WEIGHT_AUDIT_REPORT.md').write_text('\n'.join(report))
fig,axes=plt.subplots(1,3,figsize=(11,3.6));colors=['#457fb0','#8c8c8c','#187b59'];x=np.arange(3);width=.23
for j,(key,label) in enumerate([('raw_kl_bits','Raw local KL'),('regularized_kl_bits','Shrunk KL'),('mean_weight','Normalised W')]):axes[0].bar(x+(j-1)*width,[r[key] for r in rr],width,label=label,color=colors[j])
axes[0].set_yscale('log');axes[0].set_xticks(x,['Scene','SBU','Yeast']);axes[0].set_ylabel('Bits (KL) / dimensionless (W)');axes[0].set_title('Defined scale stages');axes[0].legend(fontsize=7)
axes[1].bar(x,[r['effective_n_mean'] for r in rr],color='#457fb0');axes[1].axhline(128,c='grey',ls='--',lw=1);axes[1].set_xticks(x,['Scene','SBU','Yeast']);axes[1].set_ylabel('Kernel effective sample size');axes[1].set_title('Broad fitted support');axes[1].set_ylim(0,140)
for j,(key,label) in enumerate([('source_fraction_other_support_lt1','Source fraction'),('raw_kl_mass_other_support_lt1','Raw KL share'),('regularized_kl_mass_other_support_lt1','Shrunk KL share')]):axes[2].bar(x+(j-1)*width,[100*r[key] for r in rr],width,label=label,color=colors[j])
axes[2].set_xticks(x,['Scene','SBU','Yeast']);axes[2].set_ylabel('Percent');axes[2].set_title('Low other-object support (<1)');axes[2].legend(fontsize=7)
for ax in axes:ax.grid(axis='y',alpha=.2)
fig.tight_layout();fig.savefig(p/'weight_decomposition.pdf');fig.savefig(p/'weight_decomposition.png',dpi=180)

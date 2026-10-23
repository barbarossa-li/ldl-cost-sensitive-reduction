# 两个新真实LDL数据集：冻结参数机制复核

SBU_3DFE与Yeast-cold的数据规模与官方LDL目录一致。数据来自固定提交的PyLDL公开镜像，保留原始标签分布（含真实零描述度），仅转换float64。原始MAT、转换后的NPY、来源与SHA256都保留在data/。

参数与Natural Scene阶段完全一致：beta=1、zeta=.1、kappa=.1、smin=2、alpha=1、lambda=-1，依赖度与预测损失容差均0。五近邻分布均值预测器，三折内部验证。每个数据集192/384/576训练对象，每个规模5次，测试固定256对象；同一次重复共用测试且训练嵌套，不同重复允许重叠。新数据集外部结果没有用于调参。

比较全部属性、原联合启发式、仅预测约束与同代价上限的强桥接前向基线。完整桥接路径的前缀供开启/关闭Gamma约束的共同路径消融；两个版本采用相同预算、阈值、最低代价前缀选择及删除/重启规则。保留无可行前缀的失败组，不用替代路径或违规输出补齐。

## 主要结果：均值±样本SD

|数据|训练数|方法|属性数|模拟代价|测试Brier|相对全部属性增幅均值|
|---|---:|---|---:|---:|---:|---:|
|SBU_3DFE|192|All attributes|243.0 ± 0.0|1373.0 ± 0.0|0.03090 ± 0.00232|+0.0%|
|SBU_3DFE|192|Joint guard|6.4 ± 3.6|7.2 ± 4.4|0.03373 ± 0.00301|+9.3%|
|SBU_3DFE|192|Prediction-only|1.4 ± 1.9|5.4 ± 7.4|0.03344 ± 0.00120|+8.8%|
|SBU_3DFE|192|Budget-forward bridge|2.2 ± 1.6|5.8 ± 5.2|0.03373 ± 0.00214|+9.3%|
|SBU_3DFE|192|Global mean|0.0 ± 0.0|0.0 ± 0.0|0.03247 ± 0.00162|+5.3%|
|SBU_3DFE|384|All attributes|243.0 ± 0.0|1373.0 ± 0.0|0.03003 ± 0.00197|+0.0%|
|SBU_3DFE|384|Joint guard|5.4 ± 2.9|8.4 ± 5.5|0.03188 ± 0.00086|+6.4%|
|SBU_3DFE|384|Prediction-only|4.0 ± 2.3|12.6 ± 7.2|0.03240 ± 0.00182|+8.3%|
|SBU_3DFE|384|Budget-forward bridge|2.8 ± 1.9|6.8 ± 6.3|0.03328 ± 0.00145|+11.1%|
|SBU_3DFE|384|Global mean|0.0 ± 0.0|0.0 ± 0.0|0.03245 ± 0.00158|+8.2%|
|SBU_3DFE|576|All attributes|243.0 ± 0.0|1373.0 ± 0.0|0.02806 ± 0.00156|+0.0%|
|SBU_3DFE|576|Joint guard|8.4 ± 4.6|11.0 ± 6.0|0.03153 ± 0.00133|+12.6%|
|SBU_3DFE|576|Prediction-only|3.6 ± 1.1|12.8 ± 3.7|0.03298 ± 0.00171|+17.9%|
|SBU_3DFE|576|Budget-forward bridge|3.0 ± 3.1|9.2 ± 8.5|0.03232 ± 0.00177|+15.3%|
|SBU_3DFE|576|Global mean|0.0 ± 0.0|0.0 ± 0.0|0.03239 ± 0.00159|+15.5%|
|Yeast_cold|192|All attributes|24.0 ± 0.0|127.0 ± 0.0|0.00770 ± 0.00074|+0.0%|
|Yeast_cold|192|Joint guard|1.6 ± 0.5|3.0 ± 1.4|0.00767 ± 0.00081|-0.2%|
|Yeast_cold|192|Prediction-only|0.0 ± 0.0|0.0 ± 0.0|0.00645 ± 0.00055|-16.1%|
|Yeast_cold|192|Budget-forward bridge|0.0 ± 0.0|0.0 ± 0.0|0.00645 ± 0.00055|-16.1%|
|Yeast_cold|192|Global mean|0.0 ± 0.0|0.0 ± 0.0|0.00645 ± 0.00055|-16.1%|
|Yeast_cold|384|All attributes|24.0 ± 0.0|127.0 ± 0.0|0.00727 ± 0.00059|+0.0%|
|Yeast_cold|384|Joint guard|3.0 ± 2.5|8.2 ± 8.5|0.00752 ± 0.00054|+3.6%|
|Yeast_cold|384|Prediction-only|0.0 ± 0.0|0.0 ± 0.0|0.00642 ± 0.00054|-11.6%|
|Yeast_cold|384|Budget-forward bridge|0.0 ± 0.0|0.0 ± 0.0|0.00642 ± 0.00054|-11.6%|
|Yeast_cold|384|Global mean|0.0 ± 0.0|0.0 ± 0.0|0.00642 ± 0.00054|-11.6%|
|Yeast_cold|576|All attributes|24.0 ± 0.0|127.0 ± 0.0|0.00720 ± 0.00061|+0.0%|
|Yeast_cold|576|Joint guard|2.6 ± 1.1|7.2 ± 4.1|0.00748 ± 0.00090|+3.6%|
|Yeast_cold|576|Prediction-only|0.0 ± 0.0|0.0 ± 0.0|0.00641 ± 0.00054|-11.0%|
|Yeast_cold|576|Budget-forward bridge|0.0 ± 0.0|0.0 ± 0.0|0.00641 ± 0.00054|-11.0%|
|Yeast_cold|576|Global mean|0.0 ± 0.0|0.0 ± 0.0|0.00641 ± 0.00054|-11.0%|

## 约束机制复核

|数据|训练数|关闭找到解|开启找到解|成功配对输出改变|原联合解Gamma独立阻止删除|删除后测试更差/更好|
|---|---:|---:|---:|---:|---:|---:|
|SBU_3DFE|192|4/5|3/5|3|12|7/5|
|SBU_3DFE|384|2/5|2/5|0|1|1/0|
|SBU_3DFE|576|2/5|1/5|1|13|4/9|
|Yeast_cold|192|5/5|5/5|5|4|2/2|
|Yeast_cold|384|5/5|5/5|5|9|4/5|
|Yeast_cold|576|5/5|3/5|3|1|0/1|

## 判断

SBU_3DFE at n=576: joint mean Brier is 0.03153, with mean cardinality 8.4 and cost 11.0. The mean paired relative increase over all attributes is 12.6%, and joint is no worse on 0/5 splits. Budget bridge mean Brier is 0.03232. Across all sizes, 26 original deletions are blocked solely by dependency: 12 worsen test Brier and 14 improve it.
Yeast_cold at n=576: joint mean Brier is 0.00748, with mean cardinality 2.6 and cost 7.2. The mean paired relative increase over all attributes is 3.6%, and joint is no worse on 2/5 splits. Budget bridge mean Brier is 0.00641. Across all sizes, 14 original deletions are blocked solely by dependency: 6 worsen test Brier and 8 improve it.

这些结果检验冻结机制在新数据上的可复现性，而不把一项内部约束等同于外部预测保证。只能按实际结果判断收益是否一致，不据此回调阈值。本轮模拟代价不是实际采集费用；未做全空间最优性认证。进一步工作应补已发表LDL属性选择方法和其他LDL预测器，并使用独立冻结协议。

来源：官方目录 https://palm.seu.edu.cn/xgeng/LDL/index.htm ，固定镜像 https://github.com/SpriteMisaka/PyLDL/tree/356decddc801982274a89fa37842888f56b4a2ed/datasets 。完整四项分布指标、划分、选择检查点、候选事件见结果目录。复现run_cross.py → add_mean.py → summarize.py → write_report.py。

## 跨数据集机制判断

30组共同路径审计中，关闭约束23组找到解，开启约束19组找到解；共有19组成功配对。17组输出改变，开启后的测试Brier全部变差；另外2组完全一致。7组双方均失败，4组只在开启约束后失败。不能把失败组补成有效配对。

原联合解的单属性删除中，40个候选仅被Gamma阻止：18个删除会使测试更差，22个删除反而使测试更好。候选相关，不能作为40次独立重复做显著性检验。

Yeast-cold的仅预测版本在576对象规模始终返回空集；固定全局均值预测Brier为0.00641，联合为0.00748。零属性预测更好且代价为零，因此该规模当前联合结果没有预测/代价优势。SBU_3DFE联合均值略优于桥接和均值对照，但仍比全属性平均高12.6%。

这些结果支持先检查当前硬依赖度约束与预测目标的相容性，而不是马上铺开正式多数据集比较。结论限于冻结的五近邻预测器，尚不能推广到所有LDL学习器。有向距离、粒球定义和参数本轮均未修改。

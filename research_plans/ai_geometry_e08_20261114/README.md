# E0.8 强聚类基线与 LDLKM 重建

2026-11-14。接续已公开的[文献核查](https://github.com/barbarossa-li/ldl-cost-sensitive-reduction/tree/main/research_plans/ai_literature_audit_20261114/)。本阶段先私人冻结代码和协议，再运行；公开发布尚未完成，不称GitHub预注册。与E07相同的16个已咨询族、旧训练池前min(192,ntrain)、3外层折和各3内层折；不触碰原始外部测试评分，不改写论文。

## 两个问题，分开回答

1. **固定构造比较**：保留旧纯几何二分及单次确定性20轮Lloyd，加入scikit-learn1.8.0的KMeans++(10次初始化/300轮)与标准BisectingKMeans(biggest_inertia，随机初始化10次/300轮)。同一P、U/G几何、25/50/75/100%原型预算、统一5近邻中心均值预测器，分别报告alpha0及alpha.5。每个构造、几何、先验分别用内层选择预算；所选策略之间不称同容量比较。
2. **文献预测器比较**：按[LDLKM正文](https://html.rhhz.net/tis/html/201704024.htm)式(2)–(9)重建固定K5/迭代20/p5的全中心倒距离softmax预测。另列半均值收缩适配版。这里不是作者发布代码，缩放、初始化/收敛容差、空簇和零距离约定均在PROTOCOL.json披露。固定K5与比例预算结果不强称同容量比较。

K=n统一使用单点端点，是机制比较的明确特殊情况，不冒充库原生结果。若不同特征向量少于目标K，库先按不同向量数拟合，再按声明的确定性规则补足非空组；修复和警告数完整报告。原生LDLKM保留实际非空组，不强行补K。

## 运行

```bash
python -m pip install -r requirements.txt
python check.py
python prepare_data.py --output /absolute/private/data
python runner.py run --data /absolute/private/data --output /absolute/private/e08_run
python runner.py score --data /absolute/private/data --output /absolute/private/e08_run
```

32份数据数组校验已记录哈希；已有同哈希缓存可复用。每个case的内层选择及全部预测完成后才写私有记录；全部48case冻结后才统一评分。所有源代码和协议哈希见MANIFEST.json。默认单线程数值计算；库聚类也显式限制线程。技术检查包括重复特征/常量输入、精确K/容量、两档先验满预算端点，以及刻意替换外层留出标签后选参和预测不变。

## 范围

- 旧纯几何和旧Lloyd的alpha.5、相同固定预算与cap100策略预期重现E07；不将重复证据冒充独立实验。
- G权重给每个构造，防止把加权收益只给候选；原生LDLKM按自身U/未加权规则单列。
- 报告全部族、预算、先验与阴性结果。折、先验、预算不是独立数据集。
- 只有部署数值数组的容量比较；本轮不作完整拟合/查询硬件成本优势主张。
- cost40的Full训练校准仍读取所有拟合特征，不称完整训练采集成本降低60%。
- 原始数据、行号、分组和逐对象预测只留私有输出；公开协议、代码、汇总与校验。
- 原生文献实现核验、ProLSFEO-LDL、RkNN-LDL、新独立数据确认及新稿改写尚未完成。该轮只检验强标准对照能否消除既有开发信号。

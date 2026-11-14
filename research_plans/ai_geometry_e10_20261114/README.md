# E0.10：构造加权 × 查询加权的独立交叉消融

协议和源码须先提交再计算。16个已咨询数据族、与E09相同的48个训练池留出折；不新增原始外部测试评分，不把开发结果作为独立确认，不直接改写论文主张。

## 四个单元

固定NN5的加权L1形式、最近min(5,K)原型、标签均值聚合、训练Full校准、zeta=.5、alpha0/.5和同一特征掩码，分别控制：

- UU：均匀构造、均匀查询。
- GU：信息加权构造、均匀查询。
- UG：均匀构造、信息加权查询。
- GG：信息加权构造、信息加权查询。

在固定C的两个Q单元中，分组、中心和标签完全一致。固定查询后的构造差为GU-UU和GG-UG；固定构造后的查询差为UG-UU和GG-GU；交互为GG-GU-UG+UU。负风险差有利于对应加权因素。报告全部设置，不选最优点或给开发结果加确认性p值。

构造包括几何最远对二分、KMeans++10次初始化、标准BisectingKMeans10次初始化、paper20单次随机初始化20轮适配参照；预算K5和25/50/75/100%。Full及cost40均保留，所有四单元匹配K和实际数组字节。每个族3折按留出对象数池化，再对16族等权。主展示为几何构造alpha.5的三个非单点比例预算；所有其他条件完整保留。

## 随机性与旧结果重放

为单独干预构造度量，E10在U/G构造间使用相同聚类种子。E09的KMeans++/Bisecting在G上增加过100种子偏移，因此不能将其G对角线宣称精确重放。全部E10预测冻结之后，只对共同设置中U对角线所有构造、G对角线几何/paper20做E09逐对象精确核对。完整规则见PROTOCOL.json。这里的paper20是显式重建/适配，不是原作者实现或独立新增强基线。

K=n时固定Q的构造差须逐位等于零。K<=5时两种Q都平均全部原型，查询差数学上为零，但距离排序改变浮点加法次序，容许1e-15数值误差。K5及整数K相同的比例槽位不当作独立证据。

## 运行与公开范围

```bash
python -m pip install -r requirements.txt
python check.py
python prepare_data.py --output /absolute/private/data
python runner.py run --data /absolute/private/data --output /absolute/private/e10_run
python runner.py score --data /absolute/private/data --output /absolute/private/e10_run
```

已有缓存先按PROTOCOL逐一校验32个NPY哈希，匹配则不重新下载。全部48份预测冻结后才统一评分。每折320个模型设置加均值基线，总15408条Brier。拟合/留出/原始外部测试严格分离。

源码依赖从E09冻结版本逐字复制，仅纳入代码，不复制历史划分JSON。DEPENDENCY_PROVENANCE.json记录上游代码哈希，MANIFEST.json校验本轮所有源文件。

原始数据、实际划分和逐对象预测仅存私有缓存；GitHub仅公开代码、协议、汇总结果、校验清单。数组计数不代表完整内存/运行成本；cost40仍可能使用Full训练特征。需要外部LDL强基线、尚未咨询数据集及受试者分组证据才能支撑进一步投稿主张。

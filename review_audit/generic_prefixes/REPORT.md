# 通用排序路径对照：当前九训练划分

本轮在选择前固定协议；使用与GBIP-O1相同的九个训练划分、成本向量、训练表示构造及完整集逐对象误差参考。没有读取测试标签或进行预测评价。三种方法是直接定义的通用规则，不是FSLC、GBFRS等发表算法复现。

相关度定义为每个属性与各标签维数的平方Pearson相关之和，常量列相关定义为0。分别按成本升序、相关度降序、相关度/成本降序生成完整排序，破同分规则见protocol.json。每个排序检查空集至完整集的全部m+1前缀，选最低成本可行前缀，不再进行后向删除。全部选择冻结后才读取O1结果进行比较。

| 数据集 | 排序规则 | 平均前缀成本 | 平均O1成本 | 前缀更低/相同/更高 |
|---|---|---:|---:|---|
| Natural Scene | cost_ascending | 1648.00 | 1565.67 | 0/0/3 |
| Natural Scene | squared_correlation | 1648.00 | 1565.67 | 0/0/3 |
| Natural Scene | squared_correlation_per_cost | 1645.00 | 1565.67 | 0/0/3 |
| SBU_3DFE | cost_ascending | 1369.67 | 1087.00 | 0/0/3 |
| SBU_3DFE | squared_correlation | 1373.00 | 1087.00 | 0/0/3 |
| SBU_3DFE | squared_correlation_per_cost | 1373.00 | 1087.00 | 0/0/3 |
| Yeast_cold | cost_ascending | 127.00 | 127.00 | 0/3/0 |
| Yeast_cold | squared_correlation | 127.00 | 127.00 | 0/3/0 |
| Yeast_cold | squared_correlation_per_cost | 127.00 | 127.00 | 0/3/0 |

## 九划分明细

| 数据集/重复 | 规则 | 前缀成本 | O1成本 | 前缀属性数 | 可行前缀数 | 查询数 | 秒 |
|---|---|---:|---:|---:|---:|---:|---:|
| Natural Scene/1 | cost_ascending | 1648 | 1640.0 | 294 | 1 | 295 | 39.67 |
| Natural Scene/1 | squared_correlation | 1648 | 1640.0 | 294 | 1 | 295 | 55.00 |
| Natural Scene/1 | squared_correlation_per_cost | 1648 | 1640.0 | 294 | 1 | 295 | 37.30 |
| Natural Scene/2 | cost_ascending | 1648 | 1544.0 | 294 | 1 | 295 | 46.39 |
| Natural Scene/2 | squared_correlation | 1648 | 1544.0 | 294 | 1 | 295 | 61.26 |
| Natural Scene/2 | squared_correlation_per_cost | 1648 | 1544.0 | 294 | 1 | 295 | 41.74 |
| Natural Scene/3 | cost_ascending | 1648 | 1513.0 | 294 | 1 | 295 | 86.37 |
| Natural Scene/3 | squared_correlation | 1648 | 1513.0 | 294 | 1 | 295 | 65.43 |
| Natural Scene/3 | squared_correlation_per_cost | 1639 | 1513.0 | 293 | 2 | 295 | 37.32 |
| SBU_3DFE/1 | cost_ascending | 1373 | 1241.0 | 243 | 1 | 244 | 74.95 |
| SBU_3DFE/1 | squared_correlation | 1373 | 1241.0 | 243 | 1 | 244 | 55.05 |
| SBU_3DFE/1 | squared_correlation_per_cost | 1373 | 1241.0 | 243 | 1 | 244 | 33.45 |
| SBU_3DFE/2 | cost_ascending | 1373 | 1131.0 | 243 | 1 | 244 | 27.82 |
| SBU_3DFE/2 | squared_correlation | 1373 | 1131.0 | 243 | 1 | 244 | 27.68 |
| SBU_3DFE/2 | squared_correlation_per_cost | 1373 | 1131.0 | 243 | 1 | 244 | 26.35 |
| SBU_3DFE/3 | cost_ascending | 1363 | 889.0 | 242 | 2 | 244 | 26.72 |
| SBU_3DFE/3 | squared_correlation | 1373 | 889.0 | 243 | 1 | 244 | 32.17 |
| SBU_3DFE/3 | squared_correlation_per_cost | 1373 | 889.0 | 243 | 1 | 244 | 28.77 |
| Yeast_cold/1 | cost_ascending | 127 | 127.0 | 24 | 1 | 25 | 2.35 |
| Yeast_cold/1 | squared_correlation | 127 | 127.0 | 24 | 1 | 25 | 2.41 |
| Yeast_cold/1 | squared_correlation_per_cost | 127 | 127.0 | 24 | 1 | 25 | 2.35 |
| Yeast_cold/2 | cost_ascending | 127 | 127.0 | 24 | 1 | 25 | 2.37 |
| Yeast_cold/2 | squared_correlation | 127 | 127.0 | 24 | 1 | 25 | 2.22 |
| Yeast_cold/2 | squared_correlation_per_cost | 127 | 127.0 | 24 | 1 | 25 | 2.36 |
| Yeast_cold/3 | cost_ascending | 127 | 127.0 | 24 | 1 | 25 | 2.32 |
| Yeast_cold/3 | squared_correlation | 127 | 127.0 | 24 | 1 | 25 | 2.39 |
| Yeast_cold/3 | squared_correlation_per_cost | 127 | 127.0 | 24 | 1 | 25 | 2.31 |

## 核验与解释边界

- 共27条排序路径，5076次前缀表示构造；另有9次完整参考构造及27次独立端点重构。所有端点均逐对象满足固定参考+1e-12。
- 已核对全部九组训练数组哈希、划分哈希和完整参考误差与O1一致，逐条核对返回成本为该路径最低可行前缀成本。端点重新构造验证，不构成单删不可约证明。
- 输出耗时包含该路径前缀检查与端点验证，不含一次性几何构建和共享相关计算；两进程运行。仅作运行记录，不据此断言比O1更快或搜索预算公平。
- 每条路径是固定前缀集合，O1可在非前缀子集中删减；因此比较回答实际路径能否找到更低成本可行集，不能证明某搜索策略全局优越。
- 仍为已反复开发使用的三数据集九训练划分。未做独立确认、发表方法正式复现、预测评价或显著性检验；不能据此宣布投稿就绪。
- 同一数据集的三个重复不是三个独立数据集；不把27条路径作为27个独立统计任务。

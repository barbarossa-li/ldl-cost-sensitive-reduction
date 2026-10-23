# 搜索与训练内部复核隔离

| 数据集 | 方法 | 平均成本 | 复核Brier | 达到完整集参考 | 空集次数 |
|---|---|---:|---:|---:|---:|
| Natural Scene | Full | 1648.00 | 0.265200 | 5/5 | 0/5 |
| Natural Scene | Empty | 0.00 | 0.299029 | 0/5 | 5/5 |
| Natural Scene | Prediction only before | 17.40 | 0.321073 | 0/5 | 1/5 |
| Natural Scene | Prediction only after | 15.60 | 0.327708 | 0/5 | 1/5 |
| Natural Scene | Guided before | 20.20 | 0.305335 | 1/5 | 1/5 |
| Natural Scene | Guided after | 17.60 | 0.302816 | 1/5 | 1/5 |
| SBU_3DFE | Full | 1373.00 | 0.033521 | 5/5 | 0/5 |
| SBU_3DFE | Empty | 0.00 | 0.031608 | 3/5 | 5/5 |
| SBU_3DFE | Prediction only before | 5.00 | 0.035381 | 1/5 | 1/5 |
| SBU_3DFE | Prediction only after | 4.40 | 0.035360 | 1/5 | 1/5 |
| SBU_3DFE | Guided before | 6.00 | 0.034866 | 1/5 | 1/5 |
| SBU_3DFE | Guided after | 4.20 | 0.035445 | 1/5 | 1/5 |
| Yeast_cold | Full | 127.00 | 0.007921 | 5/5 | 0/5 |
| Yeast_cold | Empty | 0.00 | 0.006770 | 5/5 | 5/5 |
| Yeast_cold | Prediction only before | 0.00 | 0.006770 | 5/5 | 5/5 |
| Yeast_cold | Prediction only after | 0.00 | 0.006770 | 5/5 | 5/5 |
| Yeast_cold | Guided before | 0.00 | 0.006770 | 5/5 | 5/5 |
| Yeast_cold | Guided after | 0.00 | 0.006770 | 5/5 | 5/5 |

预先固定随机划分，将每次192个开发训练样本分为144搜索、48复核；3折搜索内部验证每折96拟合/48验证。两方法完成选集及精修并保存后，训练全部144搜索样本预测48复核对象。复核标签和属性统计不参与搜索或拟合；不使用外部测试。

30个精修结果全部满足内部参考，Natural Scene预测优先仅0/5传递到复核，指导1/5；SBU各1/5；Yeast各5/5且全部为空。隔离复核揭示约束的经验可靠性不足，但不证明自适应搜索偏差是唯一原因。搜索训练由192减至144、复核仅48且各次划分重叠，均限制解释。不能把复核集重新用作反复筛选目标并仍称独立复核。

核心理论参数和搜索精修规则保持不变；下一步用已保存较大开发训练规模、相同3:1隔离和冻结规则检验样本量影响，不根据复核结果调参或扩大正式数据集比较。

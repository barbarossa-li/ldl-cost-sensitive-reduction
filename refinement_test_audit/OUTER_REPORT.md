# 冻结完整流程的外部预测诊断

| 数据集 | 方法 | 平均成本 | Brier | TV | KL | JS |
|---|---|---:|---:|---:|---:|---:|
| Natural Scene | Full | 1648.00 | 0.250905 | 0.455918 | 1.159174 | 0.218922 |
| Natural Scene | Empty | 0.00 | 0.302688 | 0.557889 | 0.998019 | 0.269234 |
| Natural Scene | Prediction only before | 10.00 | 0.288290 | 0.500250 | 1.335939 | 0.246175 |
| Natural Scene | Prediction only after | 5.80 | 0.285292 | 0.496044 | 1.273075 | 0.242458 |
| Natural Scene | Guided before | 13.60 | 0.286910 | 0.503117 | 1.242122 | 0.245478 |
| Natural Scene | Guided after | 9.00 | 0.286751 | 0.501554 | 1.263062 | 0.244705 |
| SBU_3DFE | Full | 1373.00 | 0.030896 | 0.151445 | 0.077716 | 0.018586 |
| SBU_3DFE | Empty | 0.00 | 0.032466 | 0.160359 | 0.081268 | 0.019179 |
| SBU_3DFE | Prediction only before | 5.40 | 0.033440 | 0.160029 | 0.083572 | 0.019845 |
| SBU_3DFE | Prediction only after | 5.00 | 0.033477 | 0.159822 | 0.083803 | 0.019891 |
| SBU_3DFE | Guided before | 8.20 | 0.033723 | 0.160864 | 0.084465 | 0.020053 |
| SBU_3DFE | Guided after | 4.60 | 0.033165 | 0.159257 | 0.082804 | 0.019662 |
| Yeast_cold | Full | 127.00 | 0.007703 | 0.067457 | 0.015685 | 0.003964 |
| Yeast_cold | Empty | 0.00 | 0.006449 | 0.061672 | 0.013150 | 0.003341 |
| Yeast_cold | Prediction only before | 0.00 | 0.006449 | 0.061672 | 0.013150 | 0.003341 |
| Yeast_cold | Prediction only after | 0.00 | 0.006449 | 0.061672 | 0.013150 | 0.003341 |
| Yeast_cold | Guided before | 0.00 | 0.006449 | 0.061672 | 0.013150 | 0.003341 |
| Yeast_cold | Guided after | 0.00 | 0.006449 | 0.061672 | 0.013150 | 0.003341 |

- Natural Scene Prediction only：精修后Brier改善3次，相同1次，变差1次。
- Natural Scene Guided：精修后Brier改善3次，相同0次，变差2次。
- SBU_3DFE Prediction only：精修后Brier改善0次，相同4次，变差1次。
- SBU_3DFE Guided：精修后Brier改善1次，相同3次，变差1次。
- Yeast_cold Prediction only：精修后Brier改善0次，相同5次，变差0次。
- Yeast_cold Guided：精修后Brier改善0次，相同5次，变差0次。

全部选集与来源哈希在预测前保存，所有训练与测试对象分离。评估90个语义记录（包含重复子集），不重选、不改规则。Brier为主要指标，TV/KL/JS为辅助；自然对数，预测按既有协议截断1e-12后归一化。

Natural Scene、SBU精修后平均Brier仍差于完整集；精修成本下降但预测变化不一致。Yeast空集优于完整集，但不证明选择方法学得了有用属性。KL等指标可能给出不同排序，不能看完结果改主要指标。内部上界可行不是外部预测保证，反复复用内部验证作搜索目标可能带来选择乐观偏差；本次不能证明这一因果解释。

既有测试已用于此前诊断，仍属探索性复用。下一步优先检查内部约束的可迁移性，预先隔离训练内部的搜索与复核数据并重跑冻结流程，不依据本次测试调参或继续扩正式实验。

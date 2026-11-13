# E0.8 public release

Published after the completed run with explicit user authorization. The protocol/source were privately frozen before computation; this is not public preregistration. Archive-internal private-delivery status is historical.

# E0.8 强基线验证结果（已公开）

2026-11-14。已完成16族、48个训练池外层验证case，各含3个内层折。文献核查报告已公开到原GitHub仓库；本E0.8源代码、协议及结果现已公开，原先私人冻结记录保留作历史。

## 结果入口

- REPORT.md：全部设置说明、主要表格及研究判断。
- SUMMARY.json、contrasts.csv、selected_policies.csv、native_LDLKM.csv：全部汇总及阴性结果。
- family_results.csv、RESULTS_PUBLIC.json：按族/折汇总的指标；文件名PUBLIC表示没有逐对象记录，不表示已经公开发布。
- EVIDENCE_AUDIT.json：私人冻结包校验、输入与预测记录的哈希和环境。
- strong_baseline_curves.png：两档先验、两档特征掩码下的风险—容量曲线。

## 当前判断

Full下的纯几何层次信号未被强标准实现完全消除；cost40下标准BisectingKMeans更有竞争力。当前不支持全面优越或原创性已经确立的主张。保留全部U/G、四构造、四预算及两档先验；不是挑最佳设置报告。

LDLKM按论文设置重建，非作者代码；零距离、缩放及初始化选择已披露。固定5中心的原生读出与统一比例预算构造不属于容量完全匹配的对照，半收缩变体单独命名。

## 干净解压后核查

```bash
# 在 results 目录中，仅复算公开汇总算术：
python verify_aggregate.py
# 在 source 目录中，运行合成技术检查：
python check.py
```

完整数据运行见 ../source/README.md。所有48份预测先冻结再统一评分。7,920个Brier以私有预测逐对象内积式独立复算；1,536项所选预算和对应预测核对通过；1,920个与E07重叠的指标精确重现。

verify_aggregate.py不从原始数据重新产生预测；本轮私有输入、新训练/验证行号、分组和逐对象预测不在本交付包中。source/deps沿用既有公开依赖所带的历史划分元数据，它们不是本轮输出。audit_and_report.py需要合法数据与完整私有运行目录，另需E07指标文件来作历史重现核对。标准库调用的内层警告未逐项留存，外层固定构造计数在报告中明确限定范围。

## 后续

先厘清Full/cost40两种几何预算下的结构差异，再准备可核验的ProLSFEO-LDL、RkNN-LDL等外部方法及独立验证。当前不改写论文、不新增原始外部测试评分。

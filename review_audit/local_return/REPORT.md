# 本地实验返回结果核验

日期：2026-10-29。三个数据集各三次重复均完成，18个方法任务，1962个参数候选、5886次训练内部拟合。Windows预检通过，九个完整参考精确一致；本轮重新核验所有参数记录及选集冻结哈希、代码哈希、折索引和损失汇总，两个驱动的共同预测参照逐值相同。未独立重跑全部预测。

压缩包内status.json仍为running：控制程序先打包再写complete状态。所有最终评价文件齐全，不能用该旧状态判定任务失败。

| 数据集 | 方法/配置 | 有效重复 | 平均成本 | 平均测试Brier | 逐对象通过/3 |
|---|---|---:|---:|---:|---:|
| Yeast_cold | literal_neighborhood | 3 | 46.00 | 0.00766158 | 0 |
| Yeast_cold | purity_shortcut | 3 | 45.33 | 0.00749332 | 0 |
| Yeast_cold | wrapped_primary | 3 | 127.00 | 0.00770576 | 3 |
| Yeast_cold | IWDLFS_paper_cv | 3 | 4.00 | 0.00772976 | 0 |
| Yeast_cold | IWDLFS_paper_fixed_08 | 3 | 4.00 | 0.00772976 | 0 |
| Yeast_cold | IWDLFS_released_fixed_08 | 3 | 4.00 | 0.00772976 | 0 |
| Yeast_cold | IWCLFS_paper_cv | 3 | 45.00 | 0.00766885 | 1 |
| Yeast_cold | IWCLFS_paper_fixed_08 | 3 | 4.00 | 0.00772976 | 0 |
| Yeast_cold | IWCLFS_released_fixed_08 | 3 | 4.00 | 0.00772976 | 0 |
| Yeast_cold | Full | 3 | 127.00 | 0.00770576 | 3 |
| Yeast_cold | Empty | 3 | 0.00 | 0.00655153 | 0 |
| Yeast_cold | GBIP-O1 | 3 | 127.00 | 0.00770576 | 3 |
| Natural_Scene | literal_neighborhood | 3 | 468.67 | 0.23766677 | 0 |
| Natural_Scene | purity_shortcut | 3 | 501.67 | 0.25517120 | 0 |
| Natural_Scene | wrapped_primary | 3 | 1648.00 | 0.24527686 | 3 |
| Natural_Scene | IWDLFS_paper_cv | 3 | 1179.00 | 0.25344709 | 0 |
| Natural_Scene | IWDLFS_paper_fixed_08 | 3 | 372.33 | 0.31822794 | 0 |
| Natural_Scene | IWDLFS_released_fixed_08 | 3 | 4.00 | 0.34456614 | 0 |
| Natural_Scene | IWCLFS_paper_cv | 3 | 1554.00 | 0.25132223 | 0 |
| Natural_Scene | IWCLFS_paper_fixed_08 | 3 | 419.00 | 0.32022349 | 0 |
| Natural_Scene | IWCLFS_released_fixed_08 | 3 | 554.67 | 0.31020770 | 0 |
| Natural_Scene | Full | 3 | 1648.00 | 0.24527686 | 3 |
| Natural_Scene | Empty | 3 | 0.00 | 0.30422414 | 0 |
| Natural_Scene | GBIP-O1 | 3 | 1565.67 | 0.24675097 | 3 |
| SBU_3DFE | literal_neighborhood | 3 | 400.33 | 0.03067575 | 0 |
| SBU_3DFE | purity_shortcut | 3 | 387.00 | 0.03091626 | 0 |
| SBU_3DFE | wrapped_primary | 3 | 1372.00 | 0.03108824 | 3 |
| SBU_3DFE | IWDLFS_paper_cv | 3 | 1197.33 | 0.03137375 | 0 |
| SBU_3DFE | IWDLFS_paper_fixed_08 | 3 | 4.00 | 0.03779684 | 0 |
| SBU_3DFE | IWDLFS_released_fixed_08 | 3 | 4.00 | 0.03779684 | 0 |
| SBU_3DFE | IWCLFS_paper_cv | 3 | 1351.67 | 0.03098366 | 2 |
| SBU_3DFE | IWCLFS_paper_fixed_08 | 3 | 4.00 | 0.03779684 | 0 |
| SBU_3DFE | IWCLFS_released_fixed_08 | 3 | 4.00 | 0.03779684 | 0 |
| SBU_3DFE | Full | 3 | 1373.00 | 0.03115156 | 3 |
| SBU_3DFE | Empty | 3 | 0.00 | 0.03227098 | 0 |
| SBU_3DFE | GBIP-O1 | 3 | 1087.00 | 0.03147972 | 3 |

解释：原方法不以O1逐对象约束为目标，未通过该约束不等于原方法失败。GBFRS为披露约定的正文重建版；IW主版恢复正文距离因子，其余发布逻辑保留。原始输出、共同保持包装、敏感性配置分别报告，不依据测试结果择优合并。

当前比较仍仅三个数据集，测试划分曾用于开发，不是独立确认。不得据此声称普遍优于已发表方法。后续扩展原稿25个主数据集并复核M2B纳入资格；当前返回包不包含该扩展。

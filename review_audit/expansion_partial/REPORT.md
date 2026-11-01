# 25数据集扩展：Movie中断前返回包核验

日期：2026-11-01。本包为数值补丁启用前的失败退出快照，不是全批完成结果。

新增17个数据集51个划分完成，连同上一批3数据集9划分，共20个主数据集60个划分完成。待完成18个划分：Movie, Flickr, Twitter, MOR-PH, Human_Gene, M2B。M2B为单列边界对照。

已核验完成新增任务的11118个参数候选、33354次内层拟合，全部源文件、划分、选择冻结和外部评价关联哈希一致，三折索引及损失汇总和选参最小值均通过。此次未重新运行预测。

## Movie中断位置

Movie第一次重复的O1已经完成：状态ok，选中188个属性，成本1074.0。该数据来自训练选择，尚无本划分冻结后的外部评价。

IWDLFS和IWCLFS各九个候选、每候选三个折，全部记录nonfinite_released_weights，故CV值为None，均无CV选中参数。此记录说明发布信息权重计算非有限；没有悄悄删除属性或修复权重，也不能当作有效低误差结果。具体是否由常量列或其他权重归一化边界触发，需要原始数组独立核算。

Movie没有任何完成的GBFRS参数检查点，报错在首个组合完成前。补丁将从这个尚未完成的组合重新计算，保留已经完成的O1与IW记录。包中没有失败时的候选属性/球成员局部快照，因此该包本身不能确定实际触发断言的邻域；已发布补丁依据同一断言的独立舍入反例，并只处理明确识别的有限舍入边界。

## 已完成新增数据集的主配置（三重复均值）

| 数据集 | 方法 | 有效次数 | 平均成本 | 平均Brier | 逐对象通过/3 |
|---|---|---:|---:|---:|---:|
| Yeast_spo | GBIP-O1 | 3 | 127.00 | 0.00955763 | 3 |
| Yeast_spo | IWDLFS_paper_cv | 3 | 82.00 | 0.00949745 | 0 |
| Yeast_spo | IWCLFS_paper_cv | 3 | 66.67 | 0.00945431 | 1 |
| Yeast_spo | GBFRS_literal_neighborhood | 3 | 44.67 | 0.00946682 | 0 |
| Yeast_alpha | GBIP-O1 | 2 | — | — | — |
| Yeast_alpha | IWDLFS_paper_cv | 3 | 4.00 | 0.00075579 | 1 |
| Yeast_alpha | IWCLFS_paper_cv | 3 | 127.00 | 0.00075766 | 3 |
| Yeast_alpha | GBFRS_literal_neighborhood | 3 | 50.00 | 0.00075336 | 2 |
| Yeast_cdc | GBIP-O1 | 3 | 127.00 | 0.00105521 | 3 |
| Yeast_cdc | IWDLFS_paper_cv | 3 | 4.00 | 0.00106558 | 0 |
| Yeast_cdc | IWCLFS_paper_cv | 3 | 45.00 | 0.00106288 | 1 |
| Yeast_cdc | GBFRS_literal_neighborhood | 3 | 48.33 | 0.00106670 | 0 |
| Yeast_diau | GBIP-O1 | 3 | 127.00 | 0.00418520 | 3 |
| Yeast_diau | IWDLFS_paper_cv | 3 | 124.33 | 0.00417416 | 2 |
| Yeast_diau | IWCLFS_paper_cv | 3 | 66.67 | 0.00415952 | 1 |
| Yeast_diau | GBFRS_literal_neighborhood | 3 | 47.67 | 0.00413682 | 1 |
| Yeast_dtt | GBIP-O1 | 3 | 127.00 | 0.00339665 | 3 |
| Yeast_dtt | IWDLFS_paper_cv | 3 | 119.33 | 0.00342709 | 1 |
| Yeast_dtt | IWCLFS_paper_cv | 3 | 127.00 | 0.00339665 | 3 |
| Yeast_dtt | GBFRS_literal_neighborhood | 3 | 54.33 | 0.00348233 | 0 |
| Yeast_elu | GBIP-O1 | 3 | 125.67 | 0.00108830 | 3 |
| Yeast_elu | IWDLFS_paper_cv | 3 | 4.00 | 0.00110329 | 0 |
| Yeast_elu | IWCLFS_paper_cv | 3 | 70.67 | 0.00108535 | 1 |
| Yeast_elu | GBFRS_literal_neighborhood | 3 | 52.33 | 0.00107496 | 0 |
| Yeast_heat | GBIP-O1 | 3 | 127.00 | 0.00506454 | 3 |
| Yeast_heat | IWDLFS_paper_cv | 3 | 82.67 | 0.00506544 | 1 |
| Yeast_heat | IWCLFS_paper_cv | 3 | 4.00 | 0.00512178 | 0 |
| Yeast_heat | GBFRS_literal_neighborhood | 3 | 50.33 | 0.00497337 | 0 |
| Yeast_spo5 | GBIP-O1 | 3 | 127.00 | 0.02266525 | 3 |
| Yeast_spo5 | IWDLFS_paper_cv | 3 | 40.67 | 0.02163359 | 0 |
| Yeast_spo5 | IWCLFS_paper_cv | 3 | 45.00 | 0.02163123 | 1 |
| Yeast_spo5 | GBFRS_literal_neighborhood | 3 | 47.00 | 0.02128209 | 0 |
| Yeast_spoem | GBIP-O1 | 3 | 127.00 | 0.02778259 | 3 |
| Yeast_spoem | IWDLFS_paper_cv | 3 | 83.00 | 0.02878814 | 0 |
| Yeast_spoem | IWCLFS_paper_cv | 3 | 86.00 | 0.02860777 | 2 |
| Yeast_spoem | GBFRS_literal_neighborhood | 3 | 50.33 | 0.02774338 | 0 |
| FG-Net | GBIP-O1 | 3 | 1412.00 | 0.08668294 | 3 |
| FG-Net | IWDLFS_paper_cv | 3 | 433.33 | 0.07550637 | 0 |
| FG-Net | IWCLFS_paper_cv | 3 | 1020.33 | 0.08700248 | 1 |
| FG-Net | GBFRS_literal_neighborhood | 3 | 421.67 | 0.08149742 | 0 |
| SJAFFE | GBIP-O1 | 3 | 1373.00 | 0.01851888 | 3 |
| SJAFFE | IWDLFS_paper_cv | 3 | 1130.33 | 0.01909308 | 0 |
| SJAFFE | IWCLFS_paper_cv | 3 | 1309.33 | 0.01880772 | 1 |
| SJAFFE | GBFRS_literal_neighborhood | 3 | 409.33 | 0.02201924 | 0 |
| Music | GBIP-O1 | 3 | 752.00 | 0.02611621 | 3 |
| Music | IWDLFS_paper_cv | 3 | 524.67 | 0.02527083 | 2 |
| Music | IWCLFS_paper_cv | 3 | 396.00 | 0.02383565 | 0 |
| Music | GBFRS_literal_neighborhood | 3 | 239.00 | 0.02454599 | 0 |
| Painting | GBIP-O1 | 2 | — | — | — |
| Painting | IWDLFS_paper_cv | 0 | — | — | — |
| Painting | IWCLFS_paper_cv | 0 | — | — | — |
| Painting | GBFRS_literal_neighborhood | 3 | 229.33 | 0.14329540 | 1 |
| SCUT_FBP | GBIP-O1 | 3 | 931.00 | 0.15510788 | 3 |
| SCUT_FBP | IWDLFS_paper_cv | 3 | 4.00 | 0.13871498 | 0 |
| SCUT_FBP | IWCLFS_paper_cv | 3 | 1687.33 | 0.15636995 | 1 |
| SCUT_FBP | GBFRS_literal_neighborhood | 3 | 504.67 | 0.15751599 | 0 |
| fbp5500 | GBIP-O1 | 3 | 2670.33 | 0.06931686 | 3 |
| fbp5500 | IWDLFS_paper_cv | 3 | 2530.67 | 0.06682565 | 0 |
| fbp5500 | IWCLFS_paper_cv | 3 | 2584.67 | 0.06812371 | 0 |
| fbp5500 | GBFRS_literal_neighborhood | 3 | 549.67 | 0.06404912 | 0 |
| emotion6 | GBIP-O1 | 3 | 852.00 | 0.20785401 | 3 |
| emotion6 | IWDLFS_paper_cv | 3 | 630.67 | 0.21337850 | 0 |
| emotion6 | IWCLFS_paper_cv | 3 | 625.67 | 0.21285058 | 0 |
| emotion6 | GBFRS_literal_neighborhood | 3 | 299.00 | 0.20936450 | 0 |
| RAF_ML | GBIP-O1 | 3 | 1162.00 | 0.20524787 | 3 |
| RAF_ML | IWDLFS_paper_cv | 3 | 4.00 | 0.20816442 | 0 |
| RAF_ML | IWCLFS_paper_cv | 3 | 4.00 | 0.20816442 | 0 |
| RAF_ML | GBFRS_literal_neighborhood | 3 | 361.67 | 0.18497791 | 0 |

全部原始行和敏感性配置保存在results.json。未完成案例不被排除后声称完整25数据集比较；方法无效输出保留缺失状态，不把少量有效重复平均成完整三重复。当前只报告描述性结果，不做重复当作独立数据集的显著性检验。

# DeepSeek 新增实验：结果与 notebook 对应

这些结果来自最新 blackwell_cu128_results (2).zip，RUN_ID 为 blackwell_cu128。当前论文尚未纳入它们。代码入口位于 [DeepSeek notebook 目录](../../notebooks/13_deepseek_extension/README.md)。本目录保留 44 份结果汇总与协议记录；来源哈希见 [support_manifest](../../docs/support_manifest.json)。

## 从哪个文件查看

| 对应实验 | notebook | 完成记录与主要结果 |
|---|---|---|
| 4.2–4.3 迁移预测 | [01 Transfer](../../notebooks/13_deepseek_extension/code/01_DeepSeek_Transfer_4_2_4_3.ipynb) | [验证完成标记](results/prediction/models/deepseek16b/validation/completion.json)、[映射差异](results/prediction/models/deepseek16b/validation/analysis/mapping_contrasts.csv)、[预测检验](results/prediction/models/deepseek16b/validation/analysis/primary_comparisons.csv) |
| 4.3 新符号验证 | 同一 01 notebook | [格式完成标记](results/prediction/models/deepseek16b/formats/completion.json)、[预测检验](results/prediction/models/deepseek16b/formats/analysis/primary_comparisons.csv)、[映射差异](results/prediction/models/deepseek16b/formats/analysis/mapping_contrasts.csv) |
| 4.4 读出与修准 | [02 Readout](../../notebooks/13_deepseek_extension/code/02_DeepSeek_Readout_4_4.ipynb) | [完成标记](results/readout/completion.json)、[1007 场景汇总](results/readout/test_summary.csv)、[304 场景修准](results/readout/correction_summary.csv)、[对比](results/readout/correction_contrasts.csv) |
| 4.5 修复确认 | [03 Repair fixed](../../notebooks/13_deepseek_extension/code/03_DeepSeek_Repair_4_5_fixed.ipynb) | [完成标记](results/repair/confirmation_capacity52_v1/completion.json)、[主检验](results/repair/confirmation_capacity52_v1/primary_tests.csv)、[修复/损坏计数](results/repair/confirmation_capacity52_v1/repair_harm_counts.csv) |
| 容量与技术边界 | 同一 fixed notebook | [52 家族修订](results/repair/confirmation_capacity52_v1/capacity_amendment.json)、[技术失败](results/repair/confirmation_capacity52_v1/technical_failures.csv)、[有效家族敏感性分析](results/repair/confirmation_capacity52_v1/valid_family_sensitivity.csv) |

[全阶段完成摘要](results/summary.json) 确认上述阶段已完成。三份带输出 notebook 记录的是此前安装失败或 64 家族容量不足的过程。修订后的完成结果应以此处文件为准，不能把中断记录误读为最终阴性实验。

## 可以怎样解释

- 内容与物理迁移验证各 96 家族，共 192 家族、3456 试次；跨符号复用内容模块 96 家族。L20/L26 的符号预测规则跨 ABC、XYZ、数字保持优势。ABC 的 matching−conflicting 准确率收益差为 20.57/18.23 个百分点；新符号下未确认对应准确率差。
- 1007 场景中，模型正确 528、读出正确 816；模型错而读出对为 361，即模型错误的 75.4%。304 场景修准使读出类别全部正确，模型正确数却从 112 降至 111：修复 3、损坏 4，净变化 −0.33 个百分点，95% 区间 [−1.97,1.32]。因此是未建立净准确率收益，不能说一个错误都没有修好。
- 类别正确不等于数值误差为零：289/304 满足数值修准容差，15 未满足。72 场景的随机控制包含 288 次随机操作，其中 2 次无效，共同有效子集为 69 场景。
- 修复确认使用 52 家族、416 呈现；基线 241/416 正确。L13 编辑后 245 正确，修复 6、损坏 2；L20 编辑后 229 正确，修复 10、损坏 22。初始准确率修复与准确率依赖均未通过确认。
- 分数指标上，L20 的初始改善为 +0.058（校正 q=0.00070），L20→L23 的依赖差为 0.01146（q=0.00327）。它们是分数层面的结果，不能替换未通过的准确率检验。
- 直接在 L23 使用真值指导编辑，得到 359/416 正确、修复 118、损坏 0；这是描述性结果，不等于统一控制下已确认的物理修复或完整依赖链。

## 必须保留的执行细节

等重题型最多只有 13 个符合约束的未见几何，原 64 家族无法生成。因此确认前修订为四个题型层各 13 家族，共 52 家族；方向和剂量不改。原版本留在 notebooks 的 versions/，实际完成目录名是 confirmation_capacity52_v1。该队列不能当作与 Qwen 的 64 家族完全等量比较。

ua/ub 两个下游条件的等范数随机移除共有 154 条技术失败：ua 为 108 条、涉及 39 家族；ub 为 46 条、涉及 25 家族；合并涉及 45 家族，仅 7 家族所有条件有效。初始修复/随机方向及 ab 的相关条件有效。应按条件报告边界，不能把所有结果一并作废，也不能隐藏这些失败。

输出头 replay 有 1/1728 不一致；主分数测量使用完整前向，不受这项 replay 近似直接替代。跨模型合并的 18 项 Holm 校正属于已有审计中的事后整合，需与各实验原主检验区分。

已有审阅材料：[10 月 26 日迁移结果复核](review_20261026/RESULTS_REVIEW_ZH.md)、[10 月 27 日结论一致性核对](claim_consistency_20261027.md)。审阅记录保留了当时使用的文件路径和论文修改建议；实验数值以对应运行结果为准。

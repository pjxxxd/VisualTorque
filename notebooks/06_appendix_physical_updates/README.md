# 附录 答案不变的物理差分迁移与分量分析

**论文位置：** app:physical-transfer；tab:physical-controls

**证据角色：** 附录主对照及同队列补充分析

24 家族的来源物理差分在 Qwen-27B L48 产生正向分数效应，主要对照差为 0.0601；准确率变化 0.26 个百分点的区间跨零。对齐、残差和条件子集分析复用该队列。

## 阅读与运行顺序

Native_PhysicalUpdate_vs_AnswerCode_Confirmation_v2 → FamilyNeutrality_StateAlignment → AlignedResidual_Factorial → Native_AlignedResidual_Robustness。v1 保存的异常属于中断记录，不作阴性科学结论。

## 文件索引

`code/` 为建议入口；`runs/` 保留实际输出和异常；`versions/` 保留不同源码。分类不是重要性排名，输出存在不等于科学假说成立。

| Notebook | 类型 | 保存状态 | 输出中识别的模型 |
|---|---|---|---|
| [VisualTorque_Qwen27B_Native_AlignedResidual_Robustness_PairedAccuracy_BF16_CPU_v1.ipynb](code/VisualTorque_Qwen27B_Native_AlignedResidual_Robustness_PairedAccuracy_BF16_CPU_v1.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_Native_AlignedResidual_Robustness_PairedAccuracy_BF16_CPU_v1.ipynb](runs/VisualTorque_Qwen27B_Native_AlignedResidual_Robustness_PairedAccuracy_BF16_CPU_v1.ipynb) | runs | 有输出（14/14 个代码单元） | 见配置 |
| [VisualTorque_Qwen27B_Native_PhysicalUpdate_AlignedResidual_Factorial_96GB_v1.ipynb](code/VisualTorque_Qwen27B_Native_PhysicalUpdate_AlignedResidual_Factorial_96GB_v1.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_Native_PhysicalUpdate_AlignedResidual_Factorial_96GB_v1.ipynb](runs/VisualTorque_Qwen27B_Native_PhysicalUpdate_AlignedResidual_Factorial_96GB_v1.ipynb) | runs | 有输出（16/16 个代码单元） | 见配置 |
| [VisualTorque_Qwen27B_Native_PhysicalUpdate_FamilyNeutrality_StateAlignment_CPU_v1.ipynb](code/VisualTorque_Qwen27B_Native_PhysicalUpdate_FamilyNeutrality_StateAlignment_CPU_v1.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_Native_PhysicalUpdate_FamilyNeutrality_StateAlignment_CPU_v1.ipynb](runs/VisualTorque_Qwen27B_Native_PhysicalUpdate_FamilyNeutrality_StateAlignment_CPU_v1.ipynb) | runs | 有输出（14/14 个代码单元） | 见配置 |
| [VisualTorque_Qwen27B_Native_PhysicalUpdate_vs_AnswerCode_Confirmation_96GB_v1.ipynb](code/VisualTorque_Qwen27B_Native_PhysicalUpdate_vs_AnswerCode_Confirmation_96GB_v1.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_Native_PhysicalUpdate_vs_AnswerCode_Confirmation_96GB_v1.ipynb](runs/VisualTorque_Qwen27B_Native_PhysicalUpdate_vs_AnswerCode_Confirmation_96GB_v1.ipynb) | runs | 保存异常：ValueError | 见配置 |
| [VisualTorque_Qwen27B_Native_PhysicalUpdate_vs_AnswerCode_Confirmation_96GB_v2.ipynb](code/VisualTorque_Qwen27B_Native_PhysicalUpdate_vs_AnswerCode_Confirmation_96GB_v2.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_Native_PhysicalUpdate_vs_AnswerCode_Confirmation_96GB_v2.ipynb](runs/VisualTorque_Qwen27B_Native_PhysicalUpdate_vs_AnswerCode_Confirmation_96GB_v2.ipynb) | runs | 有输出（15/15 个代码单元） | 见配置 |

本阅读版精简 Markdown 说明；代码单元格、参数、保存输出与元数据保留原样。完整说明见 notebook 开头链接。数据和模型环境仍按相应原协议准备。

[结论对照](../../docs/CLAIM_TO_NOTEBOOK.md) · [运行说明](../../docs/RUNNING.md) · [覆盖范围](../../docs/COVERAGE.md) · [来源清单](../../docs/source_to_copy.csv)

# 附录 LoRA 训练后的行为改善与干预变化

**论文位置：** app:training-settings；sec:finetuning

**证据角色：** 附录训练结果及机制边界

同一个 Qwen-27B LoRA adapter 在 1007 个保留场景上使六映射准确率从 52.23% 升至 88.68%；生成口径为 55.11%→87.39%。状态交换和冻结方向比较刻画训练前后变化，不等同于已确认通用力矩算法。

## 阅读与运行顺序

BF16_LoRA_WeightShortcut → PostLoRA_Heldout_Evaluation → Base_vs_LoRA_Mechanism_Discovery → Blind_Causal_Validation；其余分别为子空间、距离、风格及方向×剂量边界。多个 notebook 复用同一 adapter 和部分测试集。

## 文件索引

`code/` 为建议入口；`runs/` 保留实际输出和异常；`versions/` 保留不同源码。分类不是重要性排名，输出存在不等于科学假说成立。

| Notebook | 类型 | 保存状态 | 输出中识别的模型 |
|---|---|---|---|
| [VisualTorque_Qwen27B_AxisDose_FailureFamilies_SignedTransport_CPU_v1.ipynb](code/VisualTorque_Qwen27B_AxisDose_FailureFamilies_SignedTransport_CPU_v1.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_AxisDose_FailureFamilies_SignedTransport_CPU_v1.ipynb](runs/VisualTorque_Qwen27B_AxisDose_FailureFamilies_SignedTransport_CPU_v1.ipynb) | runs | 有输出（16/16 个代码单元） | 见配置 |
| [VisualTorque_Qwen27B_AxisDose_ReviewBundle_Analysis_CPU_v1.ipynb](code/VisualTorque_Qwen27B_AxisDose_ReviewBundle_Analysis_CPU_v1.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_AxisDose_ReviewBundle_Analysis_CPU_v1.ipynb](runs/VisualTorque_Qwen27B_AxisDose_ReviewBundle_Analysis_CPU_v1.ipynb) | runs | 有输出（14/14 个代码单元） | 见配置 |
| [VisualTorque_Qwen27B_BF16_LoRA_WeightShortcut_96GB.ipynb](code/VisualTorque_Qwen27B_BF16_LoRA_WeightShortcut_96GB.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_BF16_LoRA_WeightShortcut_96GB.ipynb](runs/VisualTorque_Qwen27B_BF16_LoRA_WeightShortcut_96GB.ipynb) | runs | 有输出（17/17 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_Base_vs_LoRA_DistanceFactorial_Audit_96GB.ipynb](code/VisualTorque_Qwen27B_Base_vs_LoRA_DistanceFactorial_Audit_96GB.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_Base_vs_LoRA_DistanceFactorial_Audit_96GB.ipynb](runs/VisualTorque_Qwen27B_Base_vs_LoRA_DistanceFactorial_Audit_96GB.ipynb) | runs | 有输出（15/15 个代码单元） | 见配置 |
| [VisualTorque_Qwen27B_Base_vs_LoRA_FrozenHandoff_Comparison_96GB.ipynb](code/VisualTorque_Qwen27B_Base_vs_LoRA_FrozenHandoff_Comparison_96GB.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_Base_vs_LoRA_FrozenHandoff_Comparison_96GB_v2.ipynb](code/VisualTorque_Qwen27B_Base_vs_LoRA_FrozenHandoff_Comparison_96GB_v2.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_Base_vs_LoRA_FrozenHandoff_Comparison_96GB_v3.ipynb](code/VisualTorque_Qwen27B_Base_vs_LoRA_FrozenHandoff_Comparison_96GB_v3.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_Base_vs_LoRA_FrozenHandoff_Comparison_96GB_v3.ipynb](runs/VisualTorque_Qwen27B_Base_vs_LoRA_FrozenHandoff_Comparison_96GB_v3.ipynb) | runs | 有输出（18/18 个代码单元） | 见配置 |
| [VisualTorque_Qwen27B_Base_vs_LoRA_Mechanism_Discovery_96GB.ipynb](code/VisualTorque_Qwen27B_Base_vs_LoRA_Mechanism_Discovery_96GB.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_Base_vs_LoRA_Mechanism_Discovery_96GB.ipynb](runs/VisualTorque_Qwen27B_Base_vs_LoRA_Mechanism_Discovery_96GB.ipynb) | runs | 有输出（17/17 个代码单元） | 见配置 |
| [VisualTorque_Qwen27B_Base_vs_LoRA_MetricVsOrdinal_Boundary_Audit_96GB.ipynb](code/VisualTorque_Qwen27B_Base_vs_LoRA_MetricVsOrdinal_Boundary_Audit_96GB.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_Base_vs_LoRA_MetricVsOrdinal_Boundary_Audit_96GB.ipynb](runs/VisualTorque_Qwen27B_Base_vs_LoRA_MetricVsOrdinal_Boundary_Audit_96GB.ipynb) | runs | 有输出（15/15 个代码单元） | 见配置 |
| [VisualTorque_Qwen27B_FrozenHandoff_BoundaryScale_Audit_CPU_v1.ipynb](code/VisualTorque_Qwen27B_FrozenHandoff_BoundaryScale_Audit_CPU_v1.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_FrozenHandoff_BoundaryScale_Audit_CPU_v2.ipynb](code/VisualTorque_Qwen27B_FrozenHandoff_BoundaryScale_Audit_CPU_v2.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_FrozenHandoff_BoundaryScale_Audit_CPU_v3.ipynb](code/VisualTorque_Qwen27B_FrozenHandoff_BoundaryScale_Audit_CPU_v3.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_FrozenHandoff_BoundaryScale_Audit_CPU_v3.ipynb](runs/VisualTorque_Qwen27B_FrozenHandoff_BoundaryScale_Audit_CPU_v3.ipynb) | runs | 有输出（11/11 个代码单元） | 见配置 |
| [VisualTorque_Qwen27B_Handoff_AxisDose_Crossover_NewGeometry_96GB_v1.ipynb](code/VisualTorque_Qwen27B_Handoff_AxisDose_Crossover_NewGeometry_96GB_v1.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_Handoff_AxisDose_Crossover_NewGeometry_96GB_v1.ipynb](runs/VisualTorque_Qwen27B_Handoff_AxisDose_Crossover_NewGeometry_96GB_v1.ipynb) | runs | 有输出（18/18 个代码单元） | 见配置 |
| [VisualTorque_Qwen27B_LocalGeometry_OracleReadout_BidirectionalResponse_96GB_v1.ipynb](code/VisualTorque_Qwen27B_LocalGeometry_OracleReadout_BidirectionalResponse_96GB_v1.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_LocalGeometry_OracleReadout_BidirectionalResponse_96GB_v1.ipynb](runs/VisualTorque_Qwen27B_LocalGeometry_OracleReadout_BidirectionalResponse_96GB_v1.ipynb) | runs | 有输出（17/17 个代码单元） | 见配置 |
| [VisualTorque_Qwen27B_Native_LoRA_AlignedUpdate_Crossover_NewGeometry_96GB_v1.ipynb](code/VisualTorque_Qwen27B_Native_LoRA_AlignedUpdate_Crossover_NewGeometry_96GB_v1.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_Native_LoRA_AlignedUpdate_Crossover_NewGeometry_96GB_v1.ipynb](runs/VisualTorque_Qwen27B_Native_LoRA_AlignedUpdate_Crossover_NewGeometry_96GB_v1.ipynb) | runs | 有输出（18/20 个代码单元） | 见配置 |
| [VisualTorque_Qwen27B_PostLoRA_Blind_Causal_Validation_96GB.ipynb](code/VisualTorque_Qwen27B_PostLoRA_Blind_Causal_Validation_96GB.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_PostLoRA_Blind_Causal_Validation_96GB.ipynb](runs/VisualTorque_Qwen27B_PostLoRA_Blind_Causal_Validation_96GB.ipynb) | runs | 有输出（20/20 个代码单元） | 见配置 |
| [VisualTorque_Qwen27B_PostLoRA_Heldout_Evaluation_96GB.ipynb](code/VisualTorque_Qwen27B_PostLoRA_Heldout_Evaluation_96GB.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_PostLoRA_Heldout_Evaluation_96GB.ipynb](runs/VisualTorque_Qwen27B_PostLoRA_Heldout_Evaluation_96GB.ipynb) | runs | 有输出（15/15 个代码单元） | 见配置 |
| [VisualTorque_Qwen27B_PostLoRA_L50_CausalSubspace_Decomposition_96GB.ipynb](code/VisualTorque_Qwen27B_PostLoRA_L50_CausalSubspace_Decomposition_96GB.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_PostLoRA_L50_CausalSubspace_Decomposition_96GB.ipynb](runs/VisualTorque_Qwen27B_PostLoRA_L50_CausalSubspace_Decomposition_96GB.ipynb) | runs | 有输出（17/17 个代码单元） | 见配置 |
| [VisualTorque_Qwen27B_PostLoRA_L50_FrozenRank4_BlindV3_96GB.ipynb](code/VisualTorque_Qwen27B_PostLoRA_L50_FrozenRank4_BlindV3_96GB.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_PostLoRA_L50_FrozenRank4_BlindV3_96GB.ipynb](runs/VisualTorque_Qwen27B_PostLoRA_L50_FrozenRank4_BlindV3_96GB.ipynb) | runs | 有输出（15/15 个代码单元） | 见配置 |
| [VisualTorque_Qwen27B_PostLoRA_L50_MetricRoute_BlindCausal_96GB.ipynb](code/VisualTorque_Qwen27B_PostLoRA_L50_MetricRoute_BlindCausal_96GB.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_PostLoRA_L50_MetricRoute_BlindCausal_96GB.ipynb](runs/VisualTorque_Qwen27B_PostLoRA_L50_MetricRoute_BlindCausal_96GB.ipynb) | runs | 有输出（15/15 个代码单元） | 见配置 |
| [VisualTorque_Qwen27B_PostLoRA_StyleCode_Crossover_Audit_96GB.ipynb](code/VisualTorque_Qwen27B_PostLoRA_StyleCode_Crossover_Audit_96GB.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_PostLoRA_StyleCode_Crossover_Audit_96GB.ipynb](runs/VisualTorque_Qwen27B_PostLoRA_StyleCode_Crossover_Audit_96GB.ipynb) | runs | 有输出（15/15 个代码单元） | 见配置 |
| [VisualTorque_Qwen27B_PostLoRA_TorqueControl_StressTest_96GB.ipynb](code/VisualTorque_Qwen27B_PostLoRA_TorqueControl_StressTest_96GB.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_PostLoRA_TorqueControl_StressTest_96GB.ipynb](runs/VisualTorque_Qwen27B_PostLoRA_TorqueControl_StressTest_96GB.ipynb) | runs | 有输出（16/16 个代码单元） | 见配置 |

本阅读版精简 Markdown 说明；代码单元格、参数、保存输出与元数据保留原样。完整说明见 notebook 开头链接。数据和模型环境仍按相应原协议准备。

[结论对照](../../docs/CLAIM_TO_NOTEBOOK.md) · [运行说明](../../docs/RUNNING.md) · [覆盖范围](../../docs/COVERAGE.md) · [来源清单](../../docs/source_to_copy.csv)

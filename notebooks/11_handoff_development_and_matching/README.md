# 历史开发 匹配队列、修复方向与序贯阻断

**论文位置：** app:repair-settings 的历史方向来源；正文独立确认见 04

**证据角色：** 历史开发、冻结方向来源与条件性复现

保留错误分型、支持集扩展、精确匹配、反向诱错、序贯阻断及复现链。旧筛选队列与正文 64 新家族确认不能混池，也不能把末层对齐统计称作阻断中介比例。

## 阅读与运行顺序

Readiness / Targeted_CommonSupport_Expansion → ExactPhysics_ErrorSpecific_Handoff → ExactPhysics_SequentialHandoff → SequentialHandoff_IndependentReplication；再衔接 04 的等幅控制与 64 家族确认。其他子空间/对称性实验为路线发现和边界诊断。

## 文件索引

`code/` 为建议入口；`runs/` 保留实际输出和异常；`versions/` 保留不同源码。分类不是重要性排名，输出存在不等于科学假说成立。

| Notebook | 类型 | 保存状态 | 输出中识别的模型 |
|---|---|---|---|
| [VisualTorque_2x2_SideCorrectness_Cohort_Readiness_A100.ipynb](code/VisualTorque_2x2_SideCorrectness_Cohort_Readiness_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_2x2_SideCorrectness_Cohort_Readiness_A100.ipynb](runs/VisualTorque_2x2_SideCorrectness_Cohort_Readiness_A100.ipynb) | runs | 有输出（15/15 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_2x2_SideCorrectness_Cohort_Readiness_A100_v2.ipynb](code/VisualTorque_2x2_SideCorrectness_Cohort_Readiness_A100_v2.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_2x2_SideCorrectness_Cohort_Readiness_A100_v3.ipynb](code/VisualTorque_2x2_SideCorrectness_Cohort_Readiness_A100_v3.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_2x2_SideCorrectness_Cohort_Readiness_A100_v3.ipynb](runs/VisualTorque_2x2_SideCorrectness_Cohort_Readiness_A100_v3.ipynb) | runs | 有输出（15/15 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_MediatorSymmetry_OrientationSpecific_Reanalysis.ipynb](code/VisualTorque_MediatorSymmetry_OrientationSpecific_Reanalysis.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_MediatorSymmetry_OrientationSpecific_Reanalysis.ipynb](runs/VisualTorque_MediatorSymmetry_OrientationSpecific_Reanalysis.ipynb) | runs | 有输出（11/11 个代码单元） | 见配置 |
| [VisualTorque_Qwen27B_CompetenceMatchedClearance_Readiness_A100.ipynb](code/VisualTorque_Qwen27B_CompetenceMatchedClearance_Readiness_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_CompetenceMatchedClearance_Readiness_A100.ipynb](runs/VisualTorque_Qwen27B_CompetenceMatchedClearance_Readiness_A100.ipynb) | runs | 有输出（16/16 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_CorrectSource_DistanceError_Rescue_A100.ipynb](code/VisualTorque_Qwen27B_CorrectSource_DistanceError_Rescue_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_CorrectSource_DistanceError_Rescue_A100.ipynb](runs/VisualTorque_Qwen27B_CorrectSource_DistanceError_Rescue_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_Distributed_FactorMediator_Subspace_A100.ipynb](code/VisualTorque_Qwen27B_Distributed_FactorMediator_Subspace_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_Distributed_FactorMediator_Subspace_A100.ipynb](runs/VisualTorque_Qwen27B_Distributed_FactorMediator_Subspace_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_ErrorConditioned_PhysicalState_Audit_A100.ipynb](code/VisualTorque_Qwen27B_ErrorConditioned_PhysicalState_Audit_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_ErrorConditioned_PhysicalState_Audit_A100.ipynb](runs/VisualTorque_Qwen27B_ErrorConditioned_PhysicalState_Audit_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_ExactPhysics_ErrorSpecific_Handoff_Confirmatory_A100.ipynb](code/VisualTorque_Qwen27B_ExactPhysics_ErrorSpecific_Handoff_Confirmatory_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_ExactPhysics_ErrorSpecific_Handoff_Confirmatory_A100.ipynb](runs/VisualTorque_Qwen27B_ExactPhysics_ErrorSpecific_Handoff_Confirmatory_A100.ipynb) | runs | 有输出（19/19 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_ExactPhysics_Handoff_Paper_Audit.ipynb](code/VisualTorque_Qwen27B_ExactPhysics_Handoff_Paper_Audit.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_ExactPhysics_Handoff_Paper_Audit.ipynb](runs/VisualTorque_Qwen27B_ExactPhysics_Handoff_Paper_Audit.ipynb) | runs | 有输出（12/12 个代码单元） | 见配置 |
| [VisualTorque_Qwen27B_ExactPhysics_SequentialHandoff_Blocking_A100.ipynb](code/VisualTorque_Qwen27B_ExactPhysics_SequentialHandoff_Blocking_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_ExactPhysics_SequentialHandoff_Blocking_A100 (1).ipynb](runs/VisualTorque_Qwen27B_ExactPhysics_SequentialHandoff_Blocking_A100%20%281%29.ipynb) | runs | 有输出（19/19 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_ExactPhysics_SequentialHandoff_Blocking_A100.ipynb](runs/VisualTorque_Qwen27B_ExactPhysics_SequentialHandoff_Blocking_A100.ipynb) | runs | 有输出（19/19 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_MatchedClearance_FlipError_RescueInduction_A100.ipynb](code/VisualTorque_Qwen27B_MatchedClearance_FlipError_RescueInduction_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_MatchedClearance_FlipError_RescueInduction_A100_v2.ipynb](code/VisualTorque_Qwen27B_MatchedClearance_FlipError_RescueInduction_A100_v2.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_MatchedClearance_FlipError_RescueInduction_A100_v2.ipynb](runs/VisualTorque_Qwen27B_MatchedClearance_FlipError_RescueInduction_A100_v2.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_MatchedFracture_Localization_ReciprocalRepair_A100.ipynb](code/VisualTorque_Qwen27B_MatchedFracture_Localization_ReciprocalRepair_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_MatchedFracture_Localization_ReciprocalRepair_A100.ipynb](runs/VisualTorque_Qwen27B_MatchedFracture_Localization_ReciprocalRepair_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_MediatorMatched_Handoff_Symmetry_A100.ipynb](code/VisualTorque_Qwen27B_MediatorMatched_Handoff_Symmetry_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_MediatorMatched_Handoff_Symmetry_A100.ipynb](runs/VisualTorque_Qwen27B_MediatorMatched_Handoff_Symmetry_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_NetTorqueToAnswer_Fracture_Rescue_A100.ipynb](code/VisualTorque_Qwen27B_NetTorqueToAnswer_Fracture_Rescue_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_NetTorqueToAnswer_Fracture_Rescue_A100.ipynb](runs/VisualTorque_Qwen27B_NetTorqueToAnswer_Fracture_Rescue_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_NetTorqueToSemantic_Handoff_Mediation_A100.ipynb](code/VisualTorque_Qwen27B_NetTorqueToSemantic_Handoff_Mediation_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_NetTorqueToSemantic_Handoff_Mediation_A100.ipynb](runs/VisualTorque_Qwen27B_NetTorqueToSemantic_Handoff_Mediation_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_RightSide_ErrorSpecific_Handoff_A100.ipynb](code/VisualTorque_Qwen27B_RightSide_ErrorSpecific_Handoff_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_RightSide_ErrorSpecific_Handoff_A100_v2.ipynb](code/VisualTorque_Qwen27B_RightSide_ErrorSpecific_Handoff_A100_v2.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_RightSide_ErrorSpecific_Handoff_A100_v2.ipynb](runs/VisualTorque_Qwen27B_RightSide_ErrorSpecific_Handoff_A100_v2.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_SemanticError_Mediation_Confirmatory_A100.ipynb](code/VisualTorque_Qwen27B_SemanticError_Mediation_Confirmatory_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_SemanticError_Mediation_Confirmatory_A100.ipynb](runs/VisualTorque_Qwen27B_SemanticError_Mediation_Confirmatory_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_SequentialHandoff_IndependentReplication_A100.ipynb](code/VisualTorque_Qwen27B_SequentialHandoff_IndependentReplication_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_SequentialHandoff_IndependentReplication_A100.ipynb](runs/VisualTorque_Qwen27B_SequentialHandoff_IndependentReplication_A100.ipynb) | runs | 有输出（24/24 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_Targeted_CommonSupport_Expansion_A100.ipynb](code/VisualTorque_Qwen27B_Targeted_CommonSupport_Expansion_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_Targeted_CommonSupport_Expansion_A100.ipynb](runs/VisualTorque_Qwen27B_Targeted_CommonSupport_Expansion_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_Torque_to_Answer_Bridge_A100.ipynb](code/VisualTorque_Qwen27B_Torque_to_Answer_Bridge_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_Torque_to_Answer_Bridge_A100.ipynb](runs/VisualTorque_Qwen27B_Torque_to_Answer_Bridge_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_RightSide_CommonSupport_Matching_Readiness.ipynb](code/VisualTorque_RightSide_CommonSupport_Matching_Readiness.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_RightSide_CommonSupport_Matching_Readiness.ipynb](runs/VisualTorque_RightSide_CommonSupport_Matching_Readiness.ipynb) | runs | 有输出（11/11 个代码单元） | 见配置 |
| [VisualTorque_Targeted_Shortcut_Error_Rescue_Induction_A100.ipynb](code/VisualTorque_Targeted_Shortcut_Error_Rescue_Induction_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Targeted_Shortcut_Error_Rescue_Induction_A100 (1).ipynb](runs/VisualTorque_Targeted_Shortcut_Error_Rescue_Induction_A100%20%281%29.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-9B |
| [VisualTorque_Targeted_Shortcut_Error_Rescue_Induction_A100 (2).ipynb](runs/VisualTorque_Targeted_Shortcut_Error_Rescue_Induction_A100%20%282%29.ipynb) | runs | 有输出（18/18 个代码单元） | google/gemma-4-12B-it |
| [VisualTorque_Targeted_Shortcut_Error_Rescue_Induction_A100.ipynb](runs/VisualTorque_Targeted_Shortcut_Error_Rescue_Induction_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |

本阅读版精简 Markdown 说明；代码单元格、参数、保存输出与元数据保留原样。完整说明见 notebook 开头链接。数据和模型环境仍按相应原协议准备。

[结论对照](../../docs/CLAIM_TO_NOTEBOOK.md) · [运行说明](../../docs/RUNNING.md) · [覆盖范围](../../docs/COVERAGE.md) · [来源清单](../../docs/source_to_copy.csv)

# 附录 来源判断、语义、字母与选项位置的区分

**论文位置：** app:content-diagnostics；Earlier Content Diagnostics；fig:vt-content；tab:vt-fixed-content

**证据角色：** 附录结果与 Section 4.2–4.3 的前期位点选择

跨问题、跨映射与深度比较区分来源判断和符号偏好；Qwen-27B 的 16 场景字母×位置实验中，字母解释的移除效应变异份额为 94.27%。这不是模型决策的解释比例。

## 阅读与运行顺序

CrossQuery_FactorDecision_Transport → ResponseState_Remapping / SemanticToLabel_Depth_Localization；跨模型看 CrossModel_CrossQuery_Transport、Gemma_Response_Depth、FixedSite_Content_Confirmation。ResponseCode_PositionFactorial 对应字母与位置的独立交叉。

## 文件索引

`code/` 为建议入口；`runs/` 保留实际输出和异常；`versions/` 保留不同源码。分类不是重要性排名，输出存在不等于科学假说成立。

| Notebook | 类型 | 保存状态 | 输出中识别的模型 |
|---|---|---|---|
| [VisualTorque_CrossModel_CrossQuery_Transport_A100.ipynb](code/VisualTorque_CrossModel_CrossQuery_Transport_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_CrossModel_CrossQuery_Transport_A100_gemma12B.ipynb](runs/VisualTorque_CrossModel_CrossQuery_Transport_A100_gemma12B.ipynb) | runs | 有输出（18/18 个代码单元） | google/gemma-4-12B-it |
| [VisualTorque_CrossModel_CrossQuery_Transport_A100_qwen27B.ipynb](runs/VisualTorque_CrossModel_CrossQuery_Transport_A100_qwen27B.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_CrossModel_CrossQuery_Transport_A100_qwen9B.ipynb](runs/VisualTorque_CrossModel_CrossQuery_Transport_A100_qwen9B.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-9B |
| [VisualTorque_CrossModel_FixedSite_Content_Confirmation_A100.ipynb](code/VisualTorque_CrossModel_FixedSite_Content_Confirmation_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_CrossModel_FixedSite_Content_Confirmation_A100_gemma.ipynb](runs/VisualTorque_CrossModel_FixedSite_Content_Confirmation_A100_gemma.ipynb) | runs | 有输出（16/16 个代码单元） | google/gemma-4-12B-it |
| [VisualTorque_CrossModel_FixedSite_Content_Confirmation_A100_qwen27B.ipynb](runs/VisualTorque_CrossModel_FixedSite_Content_Confirmation_A100_qwen27B.ipynb) | runs | 有输出（16/16 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_CrossModel_FixedSite_Content_Confirmation_A100_qwen9B.ipynb](runs/VisualTorque_CrossModel_FixedSite_Content_Confirmation_A100_qwen9B.ipynb) | runs | 有输出（16/16 个代码单元） | Qwen/Qwen3.5-9B |
| [VisualTorque_CrossQuery_FactorDecision_Transport_A100.ipynb](code/VisualTorque_CrossQuery_FactorDecision_Transport_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_CrossQuery_FactorDecision_Transport_A100 (1).ipynb](runs/VisualTorque_CrossQuery_FactorDecision_Transport_A100%20%281%29.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_CrossQuery_FactorDecision_Transport_A100.ipynb](runs/VisualTorque_CrossQuery_FactorDecision_Transport_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-9B |
| [VisualTorque_Gemma_Response_Depth_A100.ipynb](code/VisualTorque_Gemma_Response_Depth_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Gemma_Response_Depth_A100.ipynb](runs/VisualTorque_Gemma_Response_Depth_A100.ipynb) | runs | 有输出（18/18 个代码单元） | google/gemma-4-12B-it |
| [VisualTorque_L55_CodeTransition_Asymmetry_Audit.ipynb](code/VisualTorque_L55_CodeTransition_Asymmetry_Audit.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_L55_CodeTransition_Asymmetry_Audit.ipynb](runs/VisualTorque_L55_CodeTransition_Asymmetry_Audit.ipynb) | runs | 有输出（11/11 个代码单元） | 见配置 |
| [VisualTorque_Qwen27B_CrossContext_ResponseCode_Subspace_A100.ipynb](code/VisualTorque_Qwen27B_CrossContext_ResponseCode_Subspace_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_CrossContext_ResponseCode_Subspace_A100.ipynb](runs/VisualTorque_Qwen27B_CrossContext_ResponseCode_Subspace_A100.ipynb) | runs | 有输出（20/20 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_L55_CodeSubspace_Mediation_A100.ipynb](code/VisualTorque_Qwen27B_L55_CodeSubspace_Mediation_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_L55_CodeSubspace_Mediation_A100.ipynb](runs/VisualTorque_Qwen27B_L55_CodeSubspace_Mediation_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_RecipientConditioned_SemanticRecode_Rescue_A100.ipynb](code/VisualTorque_Qwen27B_RecipientConditioned_SemanticRecode_Rescue_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_RecipientConditioned_SemanticRecode_Rescue_A100.ipynb](runs/VisualTorque_Qwen27B_RecipientConditioned_SemanticRecode_Rescue_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_RecipientRecode_DoseResponse_Confirmatory_A100.ipynb](code/VisualTorque_Qwen27B_RecipientRecode_DoseResponse_Confirmatory_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_RecipientRecode_DoseResponse_Confirmatory_A100.ipynb](runs/VisualTorque_Qwen27B_RecipientRecode_DoseResponse_Confirmatory_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_ResponseCode_Alphabet_Replication_A100.ipynb](code/VisualTorque_Qwen27B_ResponseCode_Alphabet_Replication_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_ResponseCode_Alphabet_Replication_A100.ipynb](runs/VisualTorque_Qwen27B_ResponseCode_Alphabet_Replication_A100.ipynb) | runs | 有输出（20/20 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_ResponseCode_PositionFactorial_A100.ipynb](code/VisualTorque_Qwen27B_ResponseCode_PositionFactorial_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_ResponseCode_PositionFactorial_A100.ipynb](runs/VisualTorque_Qwen27B_ResponseCode_PositionFactorial_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_ResponseSubspace_DimensionTransfer_A100.ipynb](code/VisualTorque_Qwen27B_ResponseSubspace_DimensionTransfer_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_ResponseSubspace_DimensionTransfer_A100.ipynb](runs/VisualTorque_Qwen27B_ResponseSubspace_DimensionTransfer_A100.ipynb) | runs | 有输出（20/20 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_SymmetricCodeAxis_A100.ipynb](code/VisualTorque_Qwen27B_SymmetricCodeAxis_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_SymmetricCodeAxis_A100.ipynb](runs/VisualTorque_Qwen27B_SymmetricCodeAxis_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_ResponseCode_Bypass_Threshold_Audit.ipynb](code/VisualTorque_ResponseCode_Bypass_Threshold_Audit.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_ResponseCode_Bypass_Threshold_Audit.ipynb](runs/VisualTorque_ResponseCode_Bypass_Threshold_Audit.ipynb) | runs | 有输出（12/12 个代码单元） | 见配置 |
| [VisualTorque_ResponseState_Remapping_A100.ipynb](code/VisualTorque_ResponseState_Remapping_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_ResponseState_Remapping_A100 (1).ipynb](runs/VisualTorque_ResponseState_Remapping_A100%20%281%29.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_ResponseState_Remapping_A100.ipynb](runs/VisualTorque_ResponseState_Remapping_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-9B |
| [VisualTorque_SemanticToLabel_Depth_Localization_A100.ipynb](code/VisualTorque_SemanticToLabel_Depth_Localization_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_SemanticToLabel_Depth_Localization_A100 (1).ipynb](runs/VisualTorque_SemanticToLabel_Depth_Localization_A100%20%281%29.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B / Qwen/Qwen3.5-9B |
| [VisualTorque_SemanticToLabel_Depth_Localization_A100.ipynb](runs/VisualTorque_SemanticToLabel_Depth_Localization_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B / Qwen/Qwen3.5-9B |

实验设置和运行要求见各 notebook 开头的说明链接。

[结论对照](../../docs/CLAIM_TO_NOTEBOOK.md) · [运行说明](../../docs/RUNNING.md) · [覆盖范围](../../docs/COVERAGE.md) · [来源清单](../../docs/source_to_copy.csv)

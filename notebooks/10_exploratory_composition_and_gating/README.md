# 历史探索 跨背景组合、乘法规律与正确性门控

**论文位置：** 与 app:supporting-analyses 相关；不替代 Section 4.2–4.3 前瞻预测实验

**证据角色：** 历史探索与机制边界

保留跨背景复用、因素纯度、乘法组合和门控检验的完整证据。不同实验包含部分支持、支持不足或未确认结果；不能仅凭名称将其视为乘法计算或门控机制成立。

## 阅读与运行顺序

CrossContext_Factor_Recombination / ExpandedCrossContext_SourceReadiness → Final_CrossContext_Factor_Composition → Robustness_Audit；ProductLaw 与 FactorEquivalence 系列检验更强组合假说，OutcomeFlip 系列检验正确性门控。

## 文件索引

`code/` 为建议入口；`runs/` 保留实际输出和异常；`versions/` 保留不同源码。分类不是重要性排名，输出存在不等于科学假说成立。

| Notebook | 类型 | 保存状态 | 输出中识别的模型 |
|---|---|---|---|
| [VisualTorque_CrossContext_Composition_Paper_Robustness_Audit.ipynb](code/VisualTorque_CrossContext_Composition_Paper_Robustness_Audit.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_CrossContext_Composition_Paper_Robustness_Audit.ipynb](runs/VisualTorque_CrossContext_Composition_Paper_Robustness_Audit.ipynb) | runs | 有输出（12/12 个代码单元） | 见配置 |
| [VisualTorque_CrossContext_Factor_Recombination_A100.ipynb](code/VisualTorque_CrossContext_Factor_Recombination_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_CrossContext_Factor_Recombination_A100 (1).ipynb](runs/VisualTorque_CrossContext_Factor_Recombination_A100%20%281%29.ipynb) | runs | 有输出（18/18 个代码单元） | google/gemma-4-12B-it |
| [VisualTorque_CrossContext_Factor_Recombination_A100 (2).ipynb](runs/VisualTorque_CrossContext_Factor_Recombination_A100%20%282%29.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_CrossContext_Factor_Recombination_A100.ipynb](runs/VisualTorque_CrossContext_Factor_Recombination_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-9B |
| [VisualTorque_Qwen27B_CrossContext_FactorEquivalence_ProductInteraction_A100.ipynb](code/VisualTorque_Qwen27B_CrossContext_FactorEquivalence_ProductInteraction_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_CrossContext_FactorEquivalence_ProductInteraction_A100_v2.ipynb](code/VisualTorque_Qwen27B_CrossContext_FactorEquivalence_ProductInteraction_A100_v2.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_CrossContext_FactorEquivalence_ProductInteraction_A100_v3.ipynb](code/VisualTorque_Qwen27B_CrossContext_FactorEquivalence_ProductInteraction_A100_v3.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_DecisionNeutral_FactorSemantic_Mediation_A100.ipynb](code/VisualTorque_Qwen27B_DecisionNeutral_FactorSemantic_Mediation_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_DecisionNeutral_FactorSemantic_Mediation_A100.ipynb](runs/VisualTorque_Qwen27B_DecisionNeutral_FactorSemantic_Mediation_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_ExpandedCrossContext_SourceReadiness_A100.ipynb](code/VisualTorque_Qwen27B_ExpandedCrossContext_SourceReadiness_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_ExpandedCrossContext_SourceReadiness_A100.ipynb](runs/VisualTorque_Qwen27B_ExpandedCrossContext_SourceReadiness_A100.ipynb) | runs | 有输出（16/16 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_Final_CrossContext_Factor_Composition_A100.ipynb](code/VisualTorque_Qwen27B_Final_CrossContext_Factor_Composition_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_Final_CrossContext_Factor_Composition_A100.ipynb](runs/VisualTorque_Qwen27B_Final_CrossContext_Factor_Composition_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_Graded_ProductLaw_Composition_A100.ipynb](code/VisualTorque_Qwen27B_Graded_ProductLaw_Composition_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_Graded_ProductLaw_Composition_A100.ipynb](runs/VisualTorque_Qwen27B_Graded_ProductLaw_Composition_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_OutcomeFlip_ExpandedSupport_Confirmatory_A100_v2.ipynb](code/VisualTorque_Qwen27B_OutcomeFlip_ExpandedSupport_Confirmatory_A100_v2.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_OutcomeFlip_ExpandedSupport_Confirmatory_A100_v2.ipynb](runs/VisualTorque_Qwen27B_OutcomeFlip_ExpandedSupport_Confirmatory_A100_v2.ipynb) | runs | 有输出（19/20 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_OutcomeFlip_WithinContext_RouteGating_Confirmatory_A100.ipynb](code/VisualTorque_Qwen27B_OutcomeFlip_WithinContext_RouteGating_Confirmatory_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_ProductLaw_BoundaryCalibration_Audit.ipynb](code/VisualTorque_Qwen27B_ProductLaw_BoundaryCalibration_Audit.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_ProductLaw_BoundaryCalibration_Audit.ipynb](runs/VisualTorque_Qwen27B_ProductLaw_BoundaryCalibration_Audit.ipynb) | runs | 有输出（13/13 个代码单元） | 见配置 |
| [VisualTorque_Qwen27B_WeightRoute_CompetenceGating_Audit.ipynb](code/VisualTorque_Qwen27B_WeightRoute_CompetenceGating_Audit.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_WeightRoute_CompetenceGating_Audit.ipynb](runs/VisualTorque_Qwen27B_WeightRoute_CompetenceGating_Audit.ipynb) | runs | 有输出（15/15 个代码单元） | 见配置 |
| [VisualTorque_Qwen27B_WeightRoute_MappingSide_Factorial_A100.ipynb](code/VisualTorque_Qwen27B_WeightRoute_MappingSide_Factorial_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_WeightRoute_MappingSide_Factorial_A100.ipynb](runs/VisualTorque_Qwen27B_WeightRoute_MappingSide_Factorial_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_WeightRoute_WithinContext_Gating_Audit.ipynb](code/VisualTorque_Qwen27B_WeightRoute_WithinContext_Gating_Audit.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_WeightRoute_WithinContext_Gating_Audit.ipynb](runs/VisualTorque_Qwen27B_WeightRoute_WithinContext_Gating_Audit.ipynb) | runs | 有输出（15/15 个代码单元） | 见配置 |

实验设置和运行要求见各 notebook 开头的说明链接。

[结论对照](../../docs/CLAIM_TO_NOTEBOOK.md) · [运行说明](../../docs/RUNNING.md) · [覆盖范围](../../docs/COVERAGE.md) · [来源清单](../../docs/source_to_copy.csv)

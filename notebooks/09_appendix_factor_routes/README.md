# 附录 重量、距离与跨模型局部路径

**论文位置：** app:site-criteria；app:supporting-analyses

**证据角色：** 附录局部路径比较与历史发现

所测模型与位点的重量、距离作用不同；因素绑定和子空间恢复提供局部依赖证据，不能据此认定跨模型共享同一条物理计算链。

## 阅读与运行顺序

Natural_Counterfactual_Causal_Tracing → Counterfactual_Causal_Tracing_Confirmatory → Task_Token_Component_Routing；Orthogonal_ValueBinding / SameManifest_Factorial 提供对照；后续 grouped-route、large-model、relocation 与 FactorPath 检验适用范围。

## 文件索引

`code/` 为建议入口；`runs/` 保留实际输出和异常；`versions/` 保留不同源码。分类不是重要性排名，输出存在不等于科学假说成立。

| Notebook | 类型 | 保存状态 | 输出中识别的模型 |
|---|---|---|---|
| [VisualTorque_Competence_GroupedRoute_Confirmatory_A100.ipynb](code/VisualTorque_Competence_GroupedRoute_Confirmatory_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Competence_GroupedRoute_Confirmatory_A100.ipynb](runs/VisualTorque_Competence_GroupedRoute_Confirmatory_A100.ipynb) | runs | 有输出（17/17 个代码单元） | Qwen/Qwen3.5-9B |
| [VisualTorque_Competence_GroupedRoute_Confirmatory_A100.ipynb](versions/VisualTorque_Competence_GroupedRoute_Confirmatory_A100.ipynb) | versions | 源码无保存输出 | 见配置 |
| [VisualTorque_Competence_Mechanism_Audit.ipynb](code/VisualTorque_Competence_Mechanism_Audit.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Competence_Mechanism_Audit.ipynb](runs/VisualTorque_Competence_Mechanism_Audit.ipynb) | runs | 有输出（13/13 个代码单元） | 见配置 |
| [VisualTorque_Competence_Mechanism_Audit.ipynb](versions/VisualTorque_Competence_Mechanism_Audit.ipynb) | versions | 源码无保存输出 | 见配置 |
| [VisualTorque_ContextGated_WeightRoute_Confirmatory_A100.ipynb](code/VisualTorque_ContextGated_WeightRoute_Confirmatory_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_ContextGated_WeightRoute_Confirmatory_A100 (1).ipynb](runs/VisualTorque_ContextGated_WeightRoute_Confirmatory_A100%20%281%29.ipynb) | runs | 有输出（17/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_ContextGated_WeightRoute_Confirmatory_A100.ipynb](runs/VisualTorque_ContextGated_WeightRoute_Confirmatory_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-9B |
| [VisualTorque_Counterfactual_Causal_Tracing_Confirmatory_A100.ipynb](code/VisualTorque_Counterfactual_Causal_Tracing_Confirmatory_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Counterfactual_Causal_Tracing_Confirmatory_A100.ipynb](runs/VisualTorque_Counterfactual_Causal_Tracing_Confirmatory_A100.ipynb) | runs | 有输出（17/17 个代码单元） | Qwen/Qwen3.5-9B |
| [VisualTorque_Counterfactual_Causal_Tracing_Confirmatory_A100.ipynb](versions/VisualTorque_Counterfactual_Causal_Tracing_Confirmatory_A100.ipynb) | versions | 源码无保存输出 | 见配置 |
| [VisualTorque_CrossModel_CommonSupport_Audit.ipynb](code/VisualTorque_CrossModel_CommonSupport_Audit.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_CrossModel_CommonSupport_Audit.ipynb](runs/VisualTorque_CrossModel_CommonSupport_Audit.ipynb) | runs | 有输出（12/12 个代码单元） | 见配置 |
| [VisualTorque_Factorial_Robustness_Paper_Audit.ipynb](code/VisualTorque_Factorial_Robustness_Paper_Audit.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Factorial_Robustness_Paper_Audit.ipynb](runs/VisualTorque_Factorial_Robustness_Paper_Audit.ipynb) | runs | 有输出（13/13 个代码单元） | 见配置 |
| [VisualTorque_Gemma_Competence_GroupedRoute_Replication_A100.ipynb](code/VisualTorque_Gemma_Competence_GroupedRoute_Replication_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Gemma_Competence_GroupedRoute_Replication_A100.ipynb](runs/VisualTorque_Gemma_Competence_GroupedRoute_Replication_A100.ipynb) | runs | 有输出（17/17 个代码单元） | google/gemma-4-12B-it |
| [VisualTorque_Gemma_Competence_GroupedRoute_Replication_A100.ipynb](versions/VisualTorque_Gemma_Competence_GroupedRoute_Replication_A100.ipynb) | versions | 源码无保存输出 | 见配置 |
| [VisualTorque_InverseQuery_WeightRoute_Specificity_A100.ipynb](code/VisualTorque_InverseQuery_WeightRoute_Specificity_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_InverseQuery_WeightRoute_Specificity_A100 (1).ipynb](runs/VisualTorque_InverseQuery_WeightRoute_Specificity_A100%20%281%29.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_InverseQuery_WeightRoute_Specificity_A100.ipynb](runs/VisualTorque_InverseQuery_WeightRoute_Specificity_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-9B |
| [VisualTorque_LargeModel_FrozenClaim_Replication_A100.ipynb](code/VisualTorque_LargeModel_FrozenClaim_Replication_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_LargeModel_FrozenClaim_Replication_A100.ipynb](runs/VisualTorque_LargeModel_FrozenClaim_Replication_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B / Qwen/Qwen3.5-9B |
| [VisualTorque_ModelAdaptive_InformativeSite_Replication_A100.ipynb](code/VisualTorque_ModelAdaptive_InformativeSite_Replication_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_ModelAdaptive_InformativeSite_Replication_A100 (1).ipynb](runs/VisualTorque_ModelAdaptive_InformativeSite_Replication_A100%20%281%29.ipynb) | runs | 有输出（18/18 个代码单元） | google/gemma-4-12B-it |
| [VisualTorque_ModelAdaptive_InformativeSite_Replication_A100 (2).ipynb](runs/VisualTorque_ModelAdaptive_InformativeSite_Replication_A100%20%282%29.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_ModelAdaptive_InformativeSite_Replication_A100.ipynb](runs/VisualTorque_ModelAdaptive_InformativeSite_Replication_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-9B |
| [VisualTorque_Natural_Counterfactual_Causal_Tracing_A100.ipynb](code/VisualTorque_Natural_Counterfactual_Causal_Tracing_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Natural_Counterfactual_Causal_Tracing_A100.ipynb](runs/VisualTorque_Natural_Counterfactual_Causal_Tracing_A100.ipynb) | runs | 有输出（17/17 个代码单元） | Qwen/Qwen3.5-9B |
| [VisualTorque_Natural_Counterfactual_Causal_Tracing_A100.ipynb](versions/VisualTorque_Natural_Counterfactual_Causal_Tracing_A100.ipynb) | versions | 源码无保存输出 | 见配置 |
| [VisualTorque_Orthogonal_Binding_Robustness_Audit.ipynb](code/VisualTorque_Orthogonal_Binding_Robustness_Audit.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Orthogonal_Binding_Robustness_Audit.ipynb](runs/VisualTorque_Orthogonal_Binding_Robustness_Audit.ipynb) | runs | 有输出（13/13 个代码单元） | 见配置 |
| [VisualTorque_Orthogonal_ValueBinding_Control_A100.ipynb](code/VisualTorque_Orthogonal_ValueBinding_Control_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Orthogonal_ValueBinding_Control_A100 (1).ipynb](runs/VisualTorque_Orthogonal_ValueBinding_Control_A100%20%281%29.ipynb) | runs | 有输出（18/18 个代码单元） | google/gemma-4-12B-it |
| [VisualTorque_Orthogonal_ValueBinding_Control_A100.ipynb](runs/VisualTorque_Orthogonal_ValueBinding_Control_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-9B |
| [VisualTorque_Orthogonal_ValueBinding_Control_A100.ipynb](versions/VisualTorque_Orthogonal_ValueBinding_Control_A100.ipynb) | versions | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_Depth_Relocalization_Audit_A100.ipynb](code/VisualTorque_Qwen27B_Depth_Relocalization_Audit_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_Depth_Relocalization_Audit_A100.ipynb](runs/VisualTorque_Qwen27B_Depth_Relocalization_Audit_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_FactorPath_TokenComponent_A100.ipynb](code/VisualTorque_Qwen27B_FactorPath_TokenComponent_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_FactorPath_TokenComponent_A100.ipynb](runs/VisualTorque_Qwen27B_FactorPath_TokenComponent_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_Route_Relocation_Audit_A100.ipynb](code/VisualTorque_Qwen27B_Route_Relocation_Audit_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_Route_Relocation_Audit_A100.ipynb](runs/VisualTorque_Qwen27B_Route_Relocation_Audit_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen_Gemma_Mechanism_Synthesis.ipynb](code/VisualTorque_Qwen_Gemma_Mechanism_Synthesis.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen_Gemma_Mechanism_Synthesis.ipynb](runs/VisualTorque_Qwen_Gemma_Mechanism_Synthesis.ipynb) | runs | 有输出（11/11 个代码单元） | 见配置 |
| [VisualTorque_Routing_Robustness_Audit.ipynb](code/VisualTorque_Routing_Robustness_Audit.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Routing_Robustness_Audit.ipynb](runs/VisualTorque_Routing_Robustness_Audit.ipynb) | runs | 有输出（13/13 个代码单元） | 见配置 |
| [VisualTorque_Routing_Robustness_Audit.ipynb](versions/VisualTorque_Routing_Robustness_Audit.ipynb) | versions | 源码无保存输出 | 见配置 |
| [VisualTorque_SameManifest_Factorial_Composition_A100.ipynb](code/VisualTorque_SameManifest_Factorial_Composition_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_SameManifest_Factorial_Composition_A100 (1).ipynb](runs/VisualTorque_SameManifest_Factorial_Composition_A100%20%281%29.ipynb) | runs | 有输出（18/18 个代码单元） | google/gemma-4-12B-it |
| [VisualTorque_SameManifest_Factorial_Composition_A100.ipynb](runs/VisualTorque_SameManifest_Factorial_Composition_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-9B |
| [VisualTorque_SameManifest_Factorial_Composition_A100.ipynb](versions/VisualTorque_SameManifest_Factorial_Composition_A100.ipynb) | versions | 源码无保存输出 | 见配置 |
| [VisualTorque_Task_Token_Component_Routing_A100.ipynb](code/VisualTorque_Task_Token_Component_Routing_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Task_Token_Component_Routing_A100.ipynb](runs/VisualTorque_Task_Token_Component_Routing_A100.ipynb) | runs | 有输出（17/17 个代码单元） | Qwen/Qwen3.5-9B |
| [VisualTorque_Task_Token_Component_Routing_A100.ipynb](versions/VisualTorque_Task_Token_Component_Routing_A100.ipynb) | versions | 源码无保存输出 | 见配置 |

实验设置和运行要求见各 notebook 开头的说明链接。

[结论对照](../../docs/CLAIM_TO_NOTEBOOK.md) · [运行说明](../../docs/RUNNING.md) · [覆盖范围](../../docs/COVERAGE.md) · [来源清单](../../docs/source_to_copy.csv)

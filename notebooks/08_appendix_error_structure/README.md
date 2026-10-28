# 附录 物理题型、重量捷径与错误结构

**论文位置：** app:error-structure；app:site-criteria；app:supporting-analyses

**证据角色：** 附录描述性结果与竞争解释

分层统计区分左右方向、重量与距离冲突、求和与真实平衡；竞争物理线索和问法比较限定对模型错误策略的解释。

## 阅读与运行顺序

WeightShortcut_Error_Decomposition / Error_Structure 读取已有结果；Competing_Causal_Abstraction 检验竞争线索。其他策略与计算诱导研究保留为补充，不自动作正文因果依据。

## 文件索引

`code/` 为建议入口；`runs/` 保留实际输出和异常；`versions/` 保留不同源码。分类不是重要性排名，输出存在不等于科学假说成立。

| Notebook | 类型 | 保存状态 | 输出中识别的模型 |
|---|---|---|---|
| [VisualTorque_Competing_Causal_Abstraction_FinalConfirmatory_A100.ipynb](code/VisualTorque_Competing_Causal_Abstraction_FinalConfirmatory_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Competing_Causal_Abstraction_FinalConfirmatory_A100 (1).ipynb](runs/VisualTorque_Competing_Causal_Abstraction_FinalConfirmatory_A100%20%281%29.ipynb) | runs | 有输出（18/18 个代码单元） | google/gemma-4-12B-it |
| [VisualTorque_Competing_Causal_Abstraction_FinalConfirmatory_A100 (2).ipynb](runs/VisualTorque_Competing_Causal_Abstraction_FinalConfirmatory_A100%20%282%29.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Competing_Causal_Abstraction_FinalConfirmatory_A100.ipynb](runs/VisualTorque_Competing_Causal_Abstraction_FinalConfirmatory_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-9B |
| [VisualTorque_Error_Structure_CPU.ipynb](code/VisualTorque_Error_Structure_CPU.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Error_Structure_CPU.ipynb](runs/VisualTorque_Error_Structure_CPU.ipynb) | runs | 有输出（9/9 个代码单元） | 见配置 |
| [VisualTorque_LateSite_Strategy_Readout_Confirmatory_A100.ipynb](code/VisualTorque_LateSite_Strategy_Readout_Confirmatory_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_LateSite_Strategy_Readout_Confirmatory_A100 (1).ipynb](runs/VisualTorque_LateSite_Strategy_Readout_Confirmatory_A100%20%281%29.ipynb) | runs | 有输出（18/18 个代码单元） | google/gemma-4-12B-it |
| [VisualTorque_LateSite_Strategy_Readout_Confirmatory_A100 (2).ipynb](runs/VisualTorque_LateSite_Strategy_Readout_Confirmatory_A100%20%282%29.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_LateSite_Strategy_Readout_Confirmatory_A100.ipynb](runs/VisualTorque_LateSite_Strategy_Readout_Confirmatory_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-9B |
| [VisualTorque_Qwen27B_Calculation_Elicitation_A100.ipynb](code/VisualTorque_Qwen27B_Calculation_Elicitation_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_Calculation_Elicitation_A100.ipynb](runs/VisualTorque_Qwen27B_Calculation_Elicitation_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_Conditional_Strategy_PhaseMap_A100.ipynb](code/VisualTorque_Qwen27B_Conditional_Strategy_PhaseMap_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_Conditional_Strategy_PhaseMap_A100.ipynb](runs/VisualTorque_Qwen27B_Conditional_Strategy_PhaseMap_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_NetTorque_Aggregation_Invariance_A100.ipynb](code/VisualTorque_Qwen27B_NetTorque_Aggregation_Invariance_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_NetTorque_Aggregation_Invariance_A100_v2.ipynb](code/VisualTorque_Qwen27B_NetTorque_Aggregation_Invariance_A100_v2.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_NetTorque_Aggregation_Invariance_A100_v2.ipynb](runs/VisualTorque_Qwen27B_NetTorque_Aggregation_Invariance_A100_v2.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_QueryConditional_TorqueUse_A100.ipynb](code/VisualTorque_Qwen27B_QueryConditional_TorqueUse_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_QueryConditional_TorqueUse_A100.ipynb](runs/VisualTorque_Qwen27B_QueryConditional_TorqueUse_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_WeightShortcut_Error_Decomposition_A100.ipynb](code/VisualTorque_WeightShortcut_Error_Decomposition_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_WeightShortcut_Error_Decomposition_A100 (1).ipynb](runs/VisualTorque_WeightShortcut_Error_Decomposition_A100%20%281%29.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_WeightShortcut_Error_Decomposition_A100.ipynb](runs/VisualTorque_WeightShortcut_Error_Decomposition_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-9B |

实验设置和运行要求见各 notebook 开头的说明链接。

[结论对照](../../docs/CLAIM_TO_NOTEBOOK.md) · [运行说明](../../docs/RUNNING.md) · [覆盖范围](../../docs/COVERAGE.md) · [来源清单](../../docs/source_to_copy.csv)

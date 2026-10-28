# 历史汇总与其他机制试验

**论文位置：** 跨实验证据梳理；不单独生成当前 Section 4.4–Section 4.5 主结论

**证据角色：** CPU 汇总、试验性方法与外推边界

汇总 notebook 重新分析已有数据，不增加独立实验样本；机制指标预测自然错误、TorqueTangent 与更大模型试验各保留自己的方法和范围。

## 阅读与运行顺序

先生成各 notebook 声明的输入结果，再运行对应 synthesis/audit。Qwen35B 是另一架构研究，不视作正文 Qwen-27B 的直接规模复现。

## 文件索引

`code/` 为建议入口；`runs/` 保留实际输出和异常；`versions/` 保留不同源码。分类不是重要性排名，输出存在不等于科学假说成立。

| Notebook | 类型 | 保存状态 | 输出中识别的模型 |
|---|---|---|---|
| [VisualTorque_CrossModel_Results_FullAudit_CPU.ipynb](code/VisualTorque_CrossModel_Results_FullAudit_CPU.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_CrossModel_Results_FullAudit_CPU.ipynb](runs/VisualTorque_CrossModel_Results_FullAudit_CPU.ipynb) | runs | 有输出（9/9 个代码单元） | 见配置 |
| [VisualTorque_Edgewise_CausalFidelity_OOD_Readiness.ipynb](code/VisualTorque_Edgewise_CausalFidelity_OOD_Readiness.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Edgewise_CausalFidelity_OOD_Readiness.ipynb](runs/VisualTorque_Edgewise_CausalFidelity_OOD_Readiness.ipynb) | runs | 有输出（15/15 个代码单元） | 见配置 |
| [VisualTorque_Fidelity_ErrorAlignment_Reanalysis.ipynb](code/VisualTorque_Fidelity_ErrorAlignment_Reanalysis.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Mechanism_Error_Prediction_Audit.ipynb](code/VisualTorque_Mechanism_Error_Prediction_Audit.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Mechanism_Error_Prediction_Audit.ipynb](runs/VisualTorque_Mechanism_Error_Prediction_Audit.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_ModelAdaptive_Fidelity_StressPrediction_A100.ipynb](code/VisualTorque_ModelAdaptive_Fidelity_StressPrediction_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_ModelAdaptive_Fidelity_StressPrediction_A100 (1).ipynb](runs/VisualTorque_ModelAdaptive_Fidelity_StressPrediction_A100%20%281%29.ipynb) | runs | 有输出（18/18 个代码单元） | google/gemma-4-12B-it |
| [VisualTorque_ModelAdaptive_Fidelity_StressPrediction_A100 (2).ipynb](runs/VisualTorque_ModelAdaptive_Fidelity_StressPrediction_A100%20%282%29.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_ModelAdaptive_Fidelity_StressPrediction_A100.ipynb](runs/VisualTorque_ModelAdaptive_Fidelity_StressPrediction_A100.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-9B |
| [VisualTorque_Paper_Evidence_Synthesis_v1.ipynb](code/VisualTorque_Paper_Evidence_Synthesis_v1.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Paper_Evidence_Synthesis_v1.ipynb](runs/VisualTorque_Paper_Evidence_Synthesis_v1.ipynb) | runs | 有输出（18/18 个代码单元） | 见配置 |
| [VisualTorque_PathIntegrated_TorqueTangent_A100.ipynb](code/VisualTorque_PathIntegrated_TorqueTangent_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_PathIntegrated_TorqueTangent_A100.ipynb](runs/VisualTorque_PathIntegrated_TorqueTangent_A100.ipynb) | runs | 有输出（17/17 个代码单元） | Qwen/Qwen3.5-9B |
| [VisualTorque_PhysicalHandoff_FractureMap_PaperSynthesis_v2.ipynb](code/VisualTorque_PhysicalHandoff_FractureMap_PaperSynthesis_v2.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_PhysicalHandoff_FractureMap_PaperSynthesis_v2.ipynb](runs/VisualTorque_PhysicalHandoff_FractureMap_PaperSynthesis_v2.ipynb) | runs | 有输出（14/14 个代码单元） | 见配置 |
| [VisualTorque_Qwen35B_A3B_BF16_Fracture_Replication_96GB.ipynb](code/VisualTorque_Qwen35B_A3B_BF16_Fracture_Replication_96GB.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen35B_A3B_BF16_Fracture_Replication_96GB.ipynb](runs/VisualTorque_Qwen35B_A3B_BF16_Fracture_Replication_96GB.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-35B-A3B |
| [VisualTorque_Qwen35B_A3B_BF16_NetTorque_Reciprocal_Causal_96GB.ipynb](code/VisualTorque_Qwen35B_A3B_BF16_NetTorque_Reciprocal_Causal_96GB.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen35B_A3B_BF16_NetTorque_Reciprocal_Causal_96GB_v2.ipynb](code/VisualTorque_Qwen35B_A3B_BF16_NetTorque_Reciprocal_Causal_96GB_v2.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen35B_A3B_Fracture_Readiness_A100.ipynb](code/VisualTorque_Qwen35B_A3B_Fracture_Readiness_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_TorqueTangent_Pilot_A100.ipynb](code/VisualTorque_TorqueTangent_Pilot_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_TorqueTangent_Pilot_A100.ipynb](runs/VisualTorque_TorqueTangent_Pilot_A100.ipynb) | runs | 有输出（17/17 个代码单元） | Qwen/Qwen3.5-9B / google/gemma-4-12B-it |

本阅读版精简 Markdown 说明；代码单元格、参数、保存输出与元数据保留原样。完整说明见 notebook 开头链接。数据和模型环境仍按相应原协议准备。

[结论对照](../../docs/CLAIM_TO_NOTEBOOK.md) · [运行说明](../../docs/RUNNING.md) · [覆盖范围](../../docs/COVERAGE.md) · [来源清单](../../docs/source_to_copy.csv)

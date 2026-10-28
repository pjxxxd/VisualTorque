# 4.4 Qwen/Gemma 读出与数值修准

**论文位置：** 4.4 Correcting Torque Readouts Leaves Answer Errors Intact；app:readout-protocol

**证据角色：** 正文主结果及其前期诊断

统一捕获条件下，Qwen-9B/Gemma-12B 的错误中分别有 81.2%/80.4% 可被外部读出器正确判断；304 场景子集的读出类别修准到全部正确，却分别修复 0/150、0/129 个原有错误。

## 阅读与运行顺序

正文数字直接看 Aligned_Readout_Validation_GPU 的两模型 runs。重新运行前需准备历史数据和读出输入；Activation_Discovery → Activation_Confirmatory 是早期拟合链，Readout_Correction → Repeatability/Capture_Bridge 是数值诊断，不能用旧捕获数字替换正文。

## 文件索引

`code/` 为建议入口；`runs/` 保留实际输出和异常；`versions/` 保留不同源码。分类不是重要性排名，输出存在不等于科学假说成立。

| Notebook | 类型 | 保存状态 | 输出中识别的模型 |
|---|---|---|---|
| [VisualTorque_Activation_Confirmatory_A100.ipynb](code/VisualTorque_Activation_Confirmatory_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Activation_Confirmatory_A100 (1).ipynb](runs/VisualTorque_Activation_Confirmatory_A100%20%281%29.ipynb) | runs | 有输出（16/16 个代码单元） | google/gemma-4-12B-it |
| [VisualTorque_Activation_Confirmatory_A100.ipynb](runs/VisualTorque_Activation_Confirmatory_A100.ipynb) | runs | 有输出（16/16 个代码单元） | Qwen/Qwen3.5-9B |
| [VisualTorque_Activation_Confirmatory_A100.ipynb](versions/VisualTorque_Activation_Confirmatory_A100.ipynb) | versions | 源码无保存输出 | 见配置 |
| [VisualTorque_Activation_Discovery_A100.ipynb](code/VisualTorque_Activation_Discovery_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Activation_Discovery_A100.ipynb](runs/VisualTorque_Activation_Discovery_A100.ipynb) | runs | 有输出（15/15 个代码单元） | Qwen/Qwen3.5-9B |
| [VisualTorque_Activation_Discovery_A100.ipynb](versions/VisualTorque_Activation_Discovery_A100.ipynb) | versions | 源码无保存输出 | 见配置 |
| [VisualTorque_Aligned_Readout_Validation_GPU.ipynb](code/VisualTorque_Aligned_Readout_Validation_GPU.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Aligned_Readout_Validation_GPU (1).ipynb](runs/VisualTorque_Aligned_Readout_Validation_GPU%20%281%29.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-9B |
| [VisualTorque_Aligned_Readout_Validation_GPU.ipynb](runs/VisualTorque_Aligned_Readout_Validation_GPU.ipynb) | runs | 有输出（18/18 个代码单元） | google/gemma-4-12B-it |
| [VisualTorque_Causal_Torque_Intervention_A100.ipynb](code/VisualTorque_Causal_Torque_Intervention_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Causal_Torque_Intervention_A100 (1).ipynb](runs/VisualTorque_Causal_Torque_Intervention_A100%20%281%29.ipynb) | runs | 有输出（17/17 个代码单元） | Qwen/Qwen3.5-9B |
| [VisualTorque_Causal_Torque_Intervention_A100 (2).ipynb](runs/VisualTorque_Causal_Torque_Intervention_A100%20%282%29.ipynb) | runs | 有输出（17/17 个代码单元） | google/gemma-4-12B-it |
| [VisualTorque_Causal_Torque_Intervention_A100.ipynb](runs/VisualTorque_Causal_Torque_Intervention_A100.ipynb) | runs | 保存异常：AssertionError | Qwen/Qwen3.5-9B |
| [VisualTorque_CrossModel_Readout_Correction_A100.ipynb](code/VisualTorque_CrossModel_Readout_Correction_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_CrossModel_Readout_Correction_A100_gemma12B.ipynb](runs/VisualTorque_CrossModel_Readout_Correction_A100_gemma12B.ipynb) | runs | 有输出（16/16 个代码单元） | google/gemma-4-12B-it |
| [VisualTorque_CrossModel_Readout_Correction_A100_qwen27B.ipynb](runs/VisualTorque_CrossModel_Readout_Correction_A100_qwen27B.ipynb) | runs | 有输出（16/16 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_CrossModel_Readout_Correction_A100_qwen9B.ipynb](runs/VisualTorque_CrossModel_Readout_Correction_A100_qwen9B.ipynb) | runs | 有输出（16/16 个代码单元） | Qwen/Qwen3.5-9B |
| [VisualTorque_CrossModel_Readout_Correction_A100.ipynb](versions/VisualTorque_CrossModel_Readout_Correction_A100.ipynb) | versions | 源码无保存输出 | 见配置 |
| [VisualTorque_CrossModel_Readout_Repeatability_A100.ipynb](code/VisualTorque_CrossModel_Readout_Repeatability_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_CrossModel_Readout_Repeatability_A100 (1).ipynb](runs/VisualTorque_CrossModel_Readout_Repeatability_A100%20%281%29.ipynb) | runs | 有输出（13/13 个代码单元） | Qwen/Qwen3.5-9B |
| [VisualTorque_CrossModel_Readout_Repeatability_A100.ipynb](runs/VisualTorque_CrossModel_Readout_Repeatability_A100.ipynb) | runs | 有输出（13/13 个代码单元） | google/gemma-4-12B-it |
| [VisualTorque_Readout_Capture_Bridge_GPU.ipynb](code/VisualTorque_Readout_Capture_Bridge_GPU.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Readout_Capture_Bridge_GPU (1).ipynb](runs/VisualTorque_Readout_Capture_Bridge_GPU%20%281%29.ipynb) | runs | 有输出（16/16 个代码单元） | Qwen/Qwen3.5-9B |
| [VisualTorque_Readout_Capture_Bridge_GPU (2).ipynb](runs/VisualTorque_Readout_Capture_Bridge_GPU%20%282%29.ipynb) | runs | 有输出（16/16 个代码单元） | Qwen/Qwen3.5-9B |
| [VisualTorque_Readout_Capture_Bridge_GPU.ipynb](runs/VisualTorque_Readout_Capture_Bridge_GPU.ipynb) | runs | 有输出（16/16 个代码单元） | google/gemma-4-12B-it |
| [VisualTorque_Representation_Specificity_v1.ipynb](code/VisualTorque_Representation_Specificity_v1.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Representation_Specificity_v1.ipynb](runs/VisualTorque_Representation_Specificity_v1.ipynb) | runs | 有输出（13/13 个代码单元） | 见配置 |
| [VisualTorque_Representation_Specificity_v1.ipynb](versions/VisualTorque_Representation_Specificity_v1.ipynb) | versions | 源码无保存输出 | 见配置 |

本阅读版精简 Markdown 说明；代码单元格、参数、保存输出与元数据保留原样。完整说明见 notebook 开头链接。数据和模型环境仍按相应原协议准备。

[结论对照](../../docs/CLAIM_TO_NOTEBOOK.md) · [运行说明](../../docs/RUNNING.md) · [覆盖范围](../../docs/COVERAGE.md) · [来源清单](../../docs/source_to_copy.csv)

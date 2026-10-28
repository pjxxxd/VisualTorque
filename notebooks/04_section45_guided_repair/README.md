# 4.5 Qwen/Gemma 指导修复与下游依赖

**论文位置：** 4.5 Downstream Dependencies of Guided Improvements；app:repair-settings

**证据角色：** 正文主结果、候选开发与统计核查

Qwen-27B 在 64 个新家族的 512 次呈现中，L48/L49 分别修复 61/71 个错误；移除特定下游分量比等幅随机移除损失更大。Gemma L31 修复 7 个错误，但相对随机方向的准确率优势未确认。

## 阅读与运行顺序

EqualMagnitude_Blocking 提供冻结方向 → ThreePath_Confirmation 给正文独立确认；Repair_Development 提供 Gemma 候选 → Gemma12B_L31_FixedCandidate 给独立验证；Independent_Results_Audit 汇总及技术敏感性核查。较早的 Repair_Confirmation 与正文独立确认分开阅读。

## 文件索引

`code/` 为建议入口；`runs/` 保留实际输出和异常；`versions/` 保留不同源码。分类不是重要性排名，输出存在不等于科学假说成立。

| Notebook | 类型 | 保存状态 | 输出中识别的模型 |
|---|---|---|---|
| [VisualTorque_CrossModel_Repair_Confirmation_A100.ipynb](code/VisualTorque_CrossModel_Repair_Confirmation_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_CrossModel_Repair_Confirmation_A100.ipynb](runs/VisualTorque_CrossModel_Repair_Confirmation_A100.ipynb) | runs | 有输出（12/12 个代码单元） | 见配置 |
| [VisualTorque_CrossModel_Repair_Development_A100.ipynb](code/VisualTorque_CrossModel_Repair_Development_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_CrossModel_Repair_Development_A100_gemma12B.ipynb](runs/VisualTorque_CrossModel_Repair_Development_A100_gemma12B.ipynb) | runs | 有输出（14/16 个代码单元） | google/gemma-4-12B-it |
| [VisualTorque_CrossModel_Repair_Development_A100_qwen9B.ipynb](runs/VisualTorque_CrossModel_Repair_Development_A100_qwen9B.ipynb) | runs | 有输出（14/16 个代码单元） | Qwen/Qwen3.5-9B |
| [VisualTorque_CrossModel_TargetedAudit_CPU.ipynb](code/VisualTorque_CrossModel_TargetedAudit_CPU.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_CrossModel_TargetedAudit_CPU.ipynb](runs/VisualTorque_CrossModel_TargetedAudit_CPU.ipynb) | runs | 有输出（7/7 个代码单元） | 见配置 |
| [VisualTorque_Gemma12B_L31_FixedCandidate_A100.ipynb](code/VisualTorque_Gemma12B_L31_FixedCandidate_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Gemma12B_L31_FixedCandidate_A100.ipynb](runs/VisualTorque_Gemma12B_L31_FixedCandidate_A100.ipynb) | runs | 有输出（14/14 个代码单元） | google/gemma-4-12B-it |
| [VisualTorque_Independent_Results_Audit_CPU.ipynb](code/VisualTorque_Independent_Results_Audit_CPU.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Independent_Results_Audit_CPU.ipynb](runs/VisualTorque_Independent_Results_Audit_CPU.ipynb) | runs | 有输出（11/11 个代码单元） | 见配置 |
| [VisualTorque_Qwen27B_EqualMagnitude_Blocking_A100.ipynb](code/VisualTorque_Qwen27B_EqualMagnitude_Blocking_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_EqualMagnitude_Blocking_A100.ipynb](runs/VisualTorque_Qwen27B_EqualMagnitude_Blocking_A100.ipynb) | runs | 有输出（16/16 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Qwen27B_ThreePath_Confirmation_A100.ipynb](code/VisualTorque_Qwen27B_ThreePath_Confirmation_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Qwen27B_ThreePath_Confirmation_A100.ipynb](runs/VisualTorque_Qwen27B_ThreePath_Confirmation_A100.ipynb) | runs | 有输出（14/14 个代码单元） | Qwen/Qwen3.5-27B |

本阅读版精简 Markdown 说明；代码单元格、参数、保存输出与元数据保留原样。完整说明见 notebook 开头链接。数据和模型环境仍按相应原协议准备。

[结论对照](../../docs/CLAIM_TO_NOTEBOOK.md) · [运行说明](../../docs/RUNNING.md) · [覆盖范围](../../docs/COVERAGE.md) · [来源清单](../../docs/source_to_copy.csv)

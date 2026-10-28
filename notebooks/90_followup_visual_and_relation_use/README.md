# 后续探索 视觉信息、问法与物理关系使用

**论文位置：** 未纳入本次对照的正文 Section 4.4–Section 4.5

**证据角色：** 后续探索，执行状态以各文件保存记录为准

分别检验视觉信息供给、方向与平衡选项、生成问法、视觉机制迁移和 facts/products/totals 条件。不能把尚无本地输出的源码当成已生成论文结果。

## 阅读与运行顺序

各子项目按自己的说明运行。Relation_Use 的 Gemma/Qwen GPU notebook → Relation_Use_Results_Audit_CPU；VisualMechanism_Bridge → Frozen_VisualMechanism_Transfer → Results_Audit。

## 文件索引

`code/` 为建议入口；`runs/` 保留实际输出和异常；`versions/` 保留不同源码。分类不是重要性排名，输出存在不等于科学假说成立。

| Notebook | 类型 | 保存状态 | 输出中识别的模型 |
|---|---|---|---|
| [VisualTorque_Direction_Balance_Diagnostic_GPU.ipynb](code/VisualTorque_Direction_Balance_Diagnostic_GPU.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Frozen_VisualMechanism_Transfer_GPU.ipynb](code/VisualTorque_Frozen_VisualMechanism_Transfer_GPU.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Frozen_Visual_Validation_GPU.ipynb](code/VisualTorque_Frozen_Visual_Validation_GPU.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Gemma_Generation_Recheck_GPU.ipynb](code/VisualTorque_Gemma_Generation_Recheck_GPU.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Relation_Use_Results_Audit_CPU.ipynb](code/VisualTorque_Relation_Use_Results_Audit_CPU.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Relation_Use_gemma12b_GPU.ipynb](code/VisualTorque_Relation_Use_gemma12b_GPU.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Relation_Use_qwen27b_GPU.ipynb](code/VisualTorque_Relation_Use_qwen27b_GPU.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_VisualInformationSupport_GPU.ipynb](code/VisualTorque_VisualInformationSupport_GPU.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_VisualInformationSupport_GPU.ipynb](runs/VisualTorque_VisualInformationSupport_GPU.ipynb) | runs | 有输出（14/14 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_VisualMechanism_Bridge_GPU.ipynb](code/VisualTorque_VisualMechanism_Bridge_GPU.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_VisualMechanism_Results_Audit_CPU.ipynb](code/VisualTorque_VisualMechanism_Results_Audit_CPU.ipynb) | code | 源码无保存输出 | 见配置 |

本阅读版精简 Markdown 说明；代码单元格、参数、保存输出与元数据保留原样。完整说明见 notebook 开头链接。数据和模型环境仍按相应原协议准备。

[结论对照](../../docs/CLAIM_TO_NOTEBOOK.md) · [运行说明](../../docs/RUNNING.md) · [覆盖范围](../../docs/COVERAGE.md) · [来源清单](../../docs/source_to_copy.csv)

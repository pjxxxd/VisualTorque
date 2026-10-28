# 任务构造与行为基线

**论文位置：** Task and Analysis Framework；Experimental Setup

**证据角色：** 数据与方法入口

构造杠杆场景并统计模型行为，为读出和干预提供输入与基线；行为准确率本身不确定内部算法。

## 阅读与运行顺序

先准备 toolkit 与历史 Dataset v1，再运行 Behavior_A100；Behavior_Analysis_v3 汇总模型结果，v2 保留作历史版本。

## 文件索引

`code/` 为建议入口；`runs/` 保留实际输出和异常；`versions/` 保留不同源码。分类不是重要性排名，输出存在不等于科学假说成立。

| Notebook | 类型 | 保存状态 | 输出中识别的模型 |
|---|---|---|---|
| [VisualTorque_Behavior_A100.ipynb](code/VisualTorque_Behavior_A100.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Behavior_A100 (1).ipynb](runs/VisualTorque_Behavior_A100%20%281%29.ipynb) | runs | 有输出（14/14 个代码单元） | google/gemma-4-12B-it |
| [VisualTorque_Behavior_A100 (2).ipynb](runs/VisualTorque_Behavior_A100%20%282%29.ipynb) | runs | 有输出（15/15 个代码单元） | Qwen/Qwen3.5-9B |
| [VisualTorque_Behavior_A100 (3).ipynb](runs/VisualTorque_Behavior_A100%20%283%29.ipynb) | runs | 有输出（15/15 个代码单元） | google/gemma-4-12B-it |
| [VisualTorque_Behavior_A100.ipynb](runs/VisualTorque_Behavior_A100.ipynb) | runs | 有输出（14/14 个代码单元） | Qwen/Qwen3.5-9B |
| [VisualTorque_Behavior_A100.ipynb](versions/VisualTorque_Behavior_A100.ipynb) | versions | 源码无保存输出 | 见配置 |
| [VisualTorque_Behavior_Analysis_v2.ipynb](code/VisualTorque_Behavior_Analysis_v2.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Behavior_Analysis_v2.ipynb](runs/VisualTorque_Behavior_Analysis_v2.ipynb) | runs | 有输出（9/10 个代码单元） | 见配置 |
| [VisualTorque_Behavior_Analysis_v3.ipynb](code/VisualTorque_Behavior_Analysis_v3.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Behavior_Analysis_v3.ipynb](runs/VisualTorque_Behavior_Analysis_v3.ipynb) | runs | 有输出（13/13 个代码单元） | 见配置 |
| [VisualTorque_Behavior_Analysis_v3.ipynb](versions/VisualTorque_Behavior_Analysis_v3.ipynb) | versions | 源码无保存输出 | 见配置 |

实验设置和运行要求见各 notebook 开头的说明链接。

[结论对照](../../docs/CLAIM_TO_NOTEBOOK.md) · [运行说明](../../docs/RUNNING.md) · [覆盖范围](../../docs/COVERAGE.md) · [来源清单](../../docs/source_to_copy.csv)

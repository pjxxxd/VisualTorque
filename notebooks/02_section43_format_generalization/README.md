# 4.3 冻结预测器的新符号验证

**论文位置：** 4.3 Predicting Transfer Effects across Scenes and Encodings；app:prediction-results

**证据角色：** 正文主结果

ABC 上拟合的系数保持冻结，更换 XYZ 或数字后，所测位点的语义或符号偏好仍可预测干预后果；这里复用同一批 96 个验证家族。

## 阅读与运行顺序

接在 02 的 Calibration → Validation 后运行 Frozen_Format_Transfer；不要重新拟合 ABC 系数，也不要将格式变体计作独立家族。

## 文件索引

`code/` 为建议入口；`runs/` 保留实际输出和异常；`versions/` 保留不同源码。分类不是重要性排名，输出存在不等于科学假说成立。

| Notebook | 类型 | 保存状态 | 输出中识别的模型 |
|---|---|---|---|
| [VisualTorque_Frozen_Format_Transfer_GPU.ipynb](code/VisualTorque_Frozen_Format_Transfer_GPU.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Frozen_Format_Transfer_GPU (1).ipynb](runs/VisualTorque_Frozen_Format_Transfer_GPU%20%281%29.ipynb) | runs | 有输出（18/18 个代码单元） | google/gemma-4-12B-it |
| [VisualTorque_Frozen_Format_Transfer_GPU.ipynb](runs/VisualTorque_Frozen_Format_Transfer_GPU.ipynb) | runs | 有输出（18/18 个代码单元） | Qwen/Qwen3.5-27B |

本阅读版精简 Markdown 说明；代码单元格、参数、保存输出与元数据保留原样。完整说明见 notebook 开头链接。数据和模型环境仍按相应原协议准备。

[结论对照](../../docs/CLAIM_TO_NOTEBOOK.md) · [运行说明](../../docs/RUNNING.md) · [覆盖范围](../../docs/COVERAGE.md) · [来源清单](../../docs/source_to_copy.csv)

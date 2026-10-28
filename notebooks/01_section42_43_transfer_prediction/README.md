# 4.2–4.3 答案映射收益与前瞻迁移预测

**论文位置：** 4.2 Answer Encoding Changes Transfer Gains；4.3 Predicting Transfer Effects across Scenes and Encodings

**证据角色：** 正文主结果

Qwen-27B L49 的变化更符合来源语义；Qwen L63 和 Gemma-12B L37/L47 更符合来源符号。Gemma 的迁移收益随答案映射改变。物理模块未确认接收距离项带来的可靠预测增益。

## 阅读与运行顺序

每个模型先 Calibration，冻结预测器后再 Validation；两个模型分别运行。Validation 同时生成内容迁移和物理距离模块结果。

## 文件索引

`code/` 为建议入口；`runs/` 保留实际输出和异常；`versions/` 保留不同源码。分类不是重要性排名，输出存在不等于科学假说成立。

| Notebook | 类型 | 保存状态 | 输出中识别的模型 |
|---|---|---|---|
| [VisualTorque_Intervention_Prediction_Calibration_GPU.ipynb](code/VisualTorque_Intervention_Prediction_Calibration_GPU.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Intervention_Prediction_Calibration_GPU (1).ipynb](runs/VisualTorque_Intervention_Prediction_Calibration_GPU%20%281%29.ipynb) | runs | 有输出（13/13 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Intervention_Prediction_Calibration_GPU (2).ipynb](runs/VisualTorque_Intervention_Prediction_Calibration_GPU%20%282%29.ipynb) | runs | 有输出（13/13 个代码单元） | google/gemma-4-12B-it |
| [VisualTorque_Intervention_Prediction_Calibration_GPU.ipynb](runs/VisualTorque_Intervention_Prediction_Calibration_GPU.ipynb) | runs | 有输出（13/13 个代码单元） | Qwen/Qwen3.5-27B |
| [VisualTorque_Intervention_Prediction_Validation_GPU.ipynb](code/VisualTorque_Intervention_Prediction_Validation_GPU.ipynb) | code | 源码无保存输出 | 见配置 |
| [VisualTorque_Intervention_Prediction_Validation_GPU (1).ipynb](runs/VisualTorque_Intervention_Prediction_Validation_GPU%20%281%29.ipynb) | runs | 有输出（15/15 个代码单元） | google/gemma-4-12B-it |
| [VisualTorque_Intervention_Prediction_Validation_GPU.ipynb](runs/VisualTorque_Intervention_Prediction_Validation_GPU.ipynb) | runs | 有输出（15/15 个代码单元） | Qwen/Qwen3.5-27B |

实验设置和运行要求见各 notebook 开头的说明链接。

[结论对照](../../docs/CLAIM_TO_NOTEBOOK.md) · [运行说明](../../docs/RUNNING.md) · [覆盖范围](../../docs/COVERAGE.md) · [来源清单](../../docs/source_to_copy.csv)

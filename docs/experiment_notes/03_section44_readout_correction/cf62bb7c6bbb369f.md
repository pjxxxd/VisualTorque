# 2. 新旧读出捕获与批处理对照（GPU）

实验设置与运行说明。单元格编号对应来源 notebook，从 0 开始。

对应 notebook：

- [VisualTorque_Readout_Capture_Bridge_GPU.ipynb](../../../notebooks/03_section44_readout_correction/code/VisualTorque_Readout_Capture_Bridge_GPU.ipynb)
- [VisualTorque_Readout_Capture_Bridge_GPU (1).ipynb](../../../notebooks/03_section44_readout_correction/runs/VisualTorque_Readout_Capture_Bridge_GPU%20%281%29.ipynb)
- [VisualTorque_Readout_Capture_Bridge_GPU (2).ipynb](../../../notebooks/03_section44_readout_correction/runs/VisualTorque_Readout_Capture_Bridge_GPU%20%282%29.ipynb)
- [VisualTorque_Readout_Capture_Bridge_GPU.ipynb](../../../notebooks/03_section44_readout_correction/runs/VisualTorque_Readout_Capture_Bridge_GPU.ipynb)

<a id="cell-0"></a>

## 原单元格 0

# 2｜新旧读出捕获与批处理对照（GPU）

**默认 Gemma-12B；完成后保存结果，开全新的 GPU 运行时，将 MODEL_KEY 改为 qwen9b 再运行一次。不要运行 Qwen-27B。**

建议 A100 80GB 或同等级足够显存的 GPU。只统计原 19 个开发场景、每场景 6 种答案映射。批处理中可能包含原训练图片作为同批上下文，不作训练或新的测试集。

同时检查实际输入、最后 token 的位置、旧式 FP16 捕获与原生捕获，以及候选批处理上下文。旧模板默认批大小不等于历史运行已记录的批大小；本份明确保留该不确定性。

不重复之前的三次同配置测试，不扫层，不放宽原阈值。最后自动下载包含原生状态、完整明细和运行身份的 ZIP。若某部分失败，请继续运行到最后一格打包，再带回诊断信息。

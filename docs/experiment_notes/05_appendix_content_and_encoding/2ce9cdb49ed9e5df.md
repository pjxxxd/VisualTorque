# E2-B｜Gemma 的结果含义、字母与位置：七层扫描

原 notebook 的完整说明。正文精简后保留于此；单元格编号按原版、从 0 开始。

对应 notebook：

- [VisualTorque_Gemma_Response_Depth_A100.ipynb](../../../notebooks/05_appendix_content_and_encoding/code/VisualTorque_Gemma_Response_Depth_A100.ipynb)
- [VisualTorque_Gemma_Response_Depth_A100.ipynb](../../../notebooks/05_appendix_content_and_encoding/runs/VisualTorque_Gemma_Response_Depth_A100.ipynb)

<a id="cell-0"></a>

## 原单元格 0

# E2-B｜Gemma 的结果含义、字母与位置：七层扫描

默认 Gemma，沿用两个 Qwen 的 32 家族与正反重映射；完整报告七个深度，不按本轮结果挑最好层。
Gemma 有 **48 个 decoder 层**；按相对深度四舍五入（半数向上）得到零起始层号 **19、25、31、37、41、44、47**。

无需先得到 Gemma 的“末层标签效应”。本 notebook 自己检验末层，并允许未检出标签或语义阶段。
复用原 remapping 的 confirmation 清单，仅指原来的数据划分，当前研究中属于已知家族扩展。
新结果位于 `cross_model_semantic_label_depth_v1`；旧 Qwen 扫描可作为同家族历史参照。

<a id="cell-2"></a>

## 原单元格 2

## 1. Runtime and frozen depth-localization protocol

<a id="cell-8"></a>

## 原单元格 8

## 2. Common support and depth-aligned model sites

<a id="cell-11"></a>

## 原单元格 11

## 3. Multi-depth three-way response-state remapping

<a id="cell-16"></a>

## 原单元格 16

## 4. Phase localization, robustness, and synthesis

# Model-Adaptive Informative-Site Replication

实验设置与运行说明。单元格编号对应来源 notebook，从 0 开始。

对应 notebook：

- [VisualTorque_ModelAdaptive_InformativeSite_Replication_A100 (1).ipynb](../../../notebooks/09_appendix_factor_routes/runs/VisualTorque_ModelAdaptive_InformativeSite_Replication_A100%20%281%29.ipynb)
- [VisualTorque_ModelAdaptive_InformativeSite_Replication_A100 (2).ipynb](../../../notebooks/09_appendix_factor_routes/runs/VisualTorque_ModelAdaptive_InformativeSite_Replication_A100%20%282%29.ipynb)
- [VisualTorque_ModelAdaptive_InformativeSite_Replication_A100.ipynb](../../../notebooks/09_appendix_factor_routes/runs/VisualTorque_ModelAdaptive_InformativeSite_Replication_A100.ipynb)
- [VisualTorque_ModelAdaptive_InformativeSite_Replication_A100.ipynb](../../../notebooks/09_appendix_factor_routes/code/VisualTorque_ModelAdaptive_InformativeSite_Replication_A100.ipynb)

<a id="cell-0"></a>

## 原单元格 0

# VisualTorque — Model-Adaptive Informative-Site Replication

The fixed answer-site confirmation falsified a universal late route but identified an informative causal site for every model: Qwen-9B at layer-16 options, Gemma-12B at layer-41 answer, and Qwen-27B at layer-63 answer. This notebook freezes those model-specific sites before examining any new outputs.

The primary test asks whether exact torque beats distance-only reasoning at each model's independently selected informative site. Secondary tests examine multi-object summation, weight fallback, lexicographic reasoning, and equal-net recomposition. A paired congruent-control contrast tests route relocation directly. Cross-model effects are divided by each model/site's congruent transport efficacy, preventing Gemma's larger logit scale from dominating the synthesis.

All 144 families are new and geometrically disjoint from Dataset v1, the earlier competing-abstraction experiment, and the complete late-site confirmation. Semantic and nonce prompts and left/right outcomes are balanced. Three self replays, absolute raw and normalized SESOI bounds, positive-control gates, and a frozen cross-model decision tree are used throughout.

Run unchanged in fresh GPU runtimes with `MODEL_KEY` set to `qwen9b`, `gemma12b`, and `qwen27b`. Only `MODEL_KEY` may change between runs.

<a id="cell-2"></a>

## 原单元格 2

## 1. Runtime and frozen abstraction protocol

<a id="cell-8"></a>

## 原单元格 8

## 2. New strategy-diagnostic confirmation support

<a id="cell-11"></a>

## 原单元格 11

## 3. Clean strategy predictions and frozen-site interchange

<a id="cell-16"></a>

## 原单元格 16

## 4. Held-out abstraction tests and cross-model synthesis

# Competing Causal Abstractions: Final Confirmation

原 notebook 的完整说明。正文精简后保留于此；单元格编号按原版、从 0 开始。

对应 notebook：

- [VisualTorque_Competing_Causal_Abstraction_FinalConfirmatory_A100 (1).ipynb](../../../notebooks/08_appendix_error_structure/runs/VisualTorque_Competing_Causal_Abstraction_FinalConfirmatory_A100%20%281%29.ipynb)
- [VisualTorque_Competing_Causal_Abstraction_FinalConfirmatory_A100 (2).ipynb](../../../notebooks/08_appendix_error_structure/runs/VisualTorque_Competing_Causal_Abstraction_FinalConfirmatory_A100%20%282%29.ipynb)
- [VisualTorque_Competing_Causal_Abstraction_FinalConfirmatory_A100.ipynb](../../../notebooks/08_appendix_error_structure/runs/VisualTorque_Competing_Causal_Abstraction_FinalConfirmatory_A100.ipynb)
- [VisualTorque_Competing_Causal_Abstraction_FinalConfirmatory_A100.ipynb](../../../notebooks/08_appendix_error_structure/code/VisualTorque_Competing_Causal_Abstraction_FinalConfirmatory_A100.ipynb)

<a id="cell-0"></a>

## 原单元格 0

# VisualTorque — Competing Causal Abstractions: Final Confirmation

This experiment asks which high-level algorithm best predicts both a model's clean answers and its response to frozen activation interventions. Five candidates compete: exact torque, weight only, distance only, maximum single contribution, and a weight-first lexicographic rule. A sixth conditional selector is learned only on a locked calibration split and evaluated on untouched confirmation families.

The notebook generates 120 new strategy-diagnostic families that are geometrically disjoint from Dataset v1. Each family contains a base scene, its exact mirror, and an equal-net-torque recomposition. The six frozen cells isolate weight conflict, distance conflict, max-versus-sum conflict, lexicographic conflict, equal-torque recomposition, and cue-congruent controls. Clean behavior and grouped residual-stream interchange are evaluated separately; a strategy is called supported only when both agree on the confirmation split.

Run the same notebook in fresh A100 runtimes with MODEL_KEY set to qwen9b, gemma12b, and qwen27b. Never edit the manifest or gates between model runs. Captures are family-sharded and resumable.

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

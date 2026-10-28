# Qwen-27B L55 Code-Subspace Mediation

实验设置与运行说明。单元格编号对应来源 notebook，从 0 开始。

对应 notebook：

- [VisualTorque_Qwen27B_L55_CodeSubspace_Mediation_A100.ipynb](../../../notebooks/05_appendix_content_and_encoding/runs/VisualTorque_Qwen27B_L55_CodeSubspace_Mediation_A100.ipynb)
- [VisualTorque_Qwen27B_L55_CodeSubspace_Mediation_A100.ipynb](../../../notebooks/05_appendix_content_and_encoding/code/VisualTorque_Qwen27B_L55_CodeSubspace_Mediation_A100.ipynb)

<a id="cell-0"></a>

## 原单元格 0

# VisualTorque — Qwen-27B L55 Code-Subspace Mediation

The preceding experiment established that an L49 physical-side direction
controls shortcut-conflict behavior in both directions and produces a
prompt-specific downstream change at L55. This frozen follow-up asks a
stronger question: is the component of that L55 change aligned with the
current A/B/C answer code actually needed to carry the L49 effect?

For each forward/reverse label mapping, the notebook removes only the
projection of the L49-induced L55 change onto the current code-transition
direction. Equal-magnitude random-axis removal and stale-mapping removal are
mandatory controls. A code-only state tests sufficiency. The notebook first
reports literal answer-flip rates from the completed predecessor and then
runs the new subspace intervention. Sites, supports, cohorts, directions,
estimands, and gates are frozen before the new capture. Qwen-27B only.

<a id="cell-2"></a>

## 原单元格 2

## 1. Runtime and frozen code-subspace protocol

<a id="cell-8"></a>

## 原单元格 8

## 2. Frozen directions, targets, and prior literal flips

<a id="cell-11"></a>

## 原单元格 11

## 3. Path-specific L55 code-subspace interventions

<a id="cell-16"></a>

## 原单元格 16

## 4. Necessity, specificity, sufficiency, and paper result

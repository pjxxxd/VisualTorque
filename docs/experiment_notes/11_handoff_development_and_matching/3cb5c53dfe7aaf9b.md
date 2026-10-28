# Qwen-27B Exact-Physics Handoff Paper Audit

实验设置与运行说明。单元格编号对应来源 notebook，从 0 开始。

对应 notebook：

- [VisualTorque_Qwen27B_ExactPhysics_Handoff_Paper_Audit.ipynb](../../../notebooks/11_handoff_development_and_matching/runs/VisualTorque_Qwen27B_ExactPhysics_Handoff_Paper_Audit.ipynb)
- [VisualTorque_Qwen27B_ExactPhysics_Handoff_Paper_Audit.ipynb](../../../notebooks/11_handoff_development_and_matching/code/VisualTorque_Qwen27B_ExactPhysics_Handoff_Paper_Audit.ipynb)

<a id="cell-0"></a>

## 原单元格 0

# VisualTorque — Qwen-27B Exact-Physics Handoff Paper Audit

This CPU-only notebook turns the discovery and exact-physics confirmation runs into one paper-facing evidence chain. It tests the more defensible claim that a **small early error-associated route difference is amplified into a much larger late response-state and answer difference**, then triangulates that result with targeted rescue, error induction, visual-style robustness, and explicit non-claim checks.

No model is loaded and no examples are reselected. All analyses use frozen outputs produced by earlier notebooks.

<a id="cell-2"></a>

## 原单元格 2

## 1. Freeze the evidence, not a new hypothesis

The confirmation run has already fixed the model, route, intervention directions, behavioral cohorts, and exact physical matches. This notebook only asks whether those frozen results jointly support the proposed paper wording and whether any tempting stronger wording is contradicted by the data.

<a id="cell-6"></a>

## 原单元格 6

## 2. Reconstruct the matched causal contrast

Each exact-physics match contains four scenes: a left and right orientation from a pair the model solves in both orientations, and a physically identical left and right orientation from a pair where only the right orientation is usually wrong. For every match, the key contrast subtracts the ordinary left–right difference from the error-associated left–right difference. This removes any general orientation effect before asking what is special about the error condition.

<a id="cell-9"></a>

## 原单元格 9

## 3. Test amplification rather than a sharp serial break

The early and late coordinates have different natural scales, so their raw values cannot be compared directly. We divide each match-level contrast by the norm of the corresponding frozen intervention direction. The precommitted difference-in-differences remains the basic unit; normalization only makes the early and late magnitudes comparable.

<a id="cell-12"></a>

## 原单元格 12

## 4. Build a causal evidence triangle

A numerical state difference is more convincing when it predicts a directional behavioral intervention. We therefore place three observations side by side: the model's original right-side error, rescue after adding the correct route, and induction of the corresponding wrong response after reversing or replacing that route. Random directions and the left orientation serve as controls.

<a id="cell-15"></a>

## 原单元格 15

## 5. Produce the paper-facing figure and claim card

The figure deliberately separates raw endpoint effects from scale-normalized amplification. The written card reports both positive and negative checks so that a later draft cannot accidentally turn “small early difference” into “no early difference,” or a matched interaction into an unsupported right-only universal mechanism.

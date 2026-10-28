# Qwen-27B Conditional Strategy Phase Map

原 notebook 的完整说明。正文精简后保留于此；单元格编号按原版、从 0 开始。

对应 notebook：

- [VisualTorque_Qwen27B_Conditional_Strategy_PhaseMap_A100.ipynb](../../../notebooks/08_appendix_error_structure/runs/VisualTorque_Qwen27B_Conditional_Strategy_PhaseMap_A100.ipynb)
- [VisualTorque_Qwen27B_Conditional_Strategy_PhaseMap_A100.ipynb](../../../notebooks/08_appendix_error_structure/code/VisualTorque_Qwen27B_Conditional_Strategy_PhaseMap_A100.ipynb)

<a id="cell-0"></a>

## 原单元格 0

# VisualTorque — Qwen-27B Conditional Strategy Phase Map

This confirmatory experiment asks whether Qwen3.5-27B changes its internal route when simple visual cues cease to agree with torque. It does not assume that valid physical reasoning must expose a localized scalar torque node. Instead, it contrasts an outcome-changing mirror donor with a same-net-torque donor that changes cue congruence, under a frozen 2×2 design: cue-congruent versus cue-conflict scenes crossed with high versus low normalized torque margin.

The support contains 80 scene-disjoint single-stack families selected without model outputs. Every family has a base scene, its exact left-right mirror, and a cue donor with the same signed net torque and outcome but the opposite congruence class. Conflict families are balanced between weight-conflict and distance-conflict; target outcomes are balanced left/right. Qwen, prompts, layers, and token scopes are frozen. The primary estimand is a high-margin conflict-versus-congruent gain in a self-replay-centered structural-route index, with the low-margin contrast and their interaction reported separately.

<a id="cell-2"></a>

## 原单元格 2

## 1. Runtime and frozen conditional-strategy protocol

<a id="cell-8"></a>

## 原单元格 8

## 2. Model-output-blind 2×2 strategy support

<a id="cell-11"></a>

## 原单元格 11

## 3. Fixed-site physical-route and cue-route interventions

<a id="cell-16"></a>

## 原单元格 16

## 4. Conditional strategy decisions and robustness

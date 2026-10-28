# Confirmatory Activation Probe (A100)

实验设置与运行说明。单元格编号对应来源 notebook，从 0 开始。

对应 notebook：

- [VisualTorque_Activation_Confirmatory_A100 (1).ipynb](../../../notebooks/03_section44_readout_correction/runs/VisualTorque_Activation_Confirmatory_A100%20%281%29.ipynb)
- [VisualTorque_Activation_Confirmatory_A100.ipynb](../../../notebooks/03_section44_readout_correction/runs/VisualTorque_Activation_Confirmatory_A100.ipynb)
- [VisualTorque_Activation_Confirmatory_A100.ipynb](../../../notebooks/03_section44_readout_correction/code/VisualTorque_Activation_Confirmatory_A100.ipynb)

<a id="cell-0"></a>

## 原单元格 0

# VisualTorque — Confirmatory Activation Probe on one A100

This notebook performs the locked confirmatory test after activation discovery. It opens
probe_test only at the preregistered net-torque layer, refits the frozen ridge probe on
probe_train plus probe_dev, and evaluates probe_test exactly once.

Frozen primary choices:

- Qwen/Qwen3.5-9B: decoder layer 13, ridge alpha 0.01.
- google/gemma-4-12B-it: decoder layer 7, ridge alpha 100.
- Representation: mean of the six option-permutation final-prompt states.
- Primary target: signed net torque.
- Balance threshold for the derived outcome: absolute predicted torque at most 0.5.

The notebook writes resumable activation shards, point metrics, stratified scene-bootstrap
intervals, 200 shuffled-label null probes, raw-feature shortcut baselines, and a scene-level
representation–behavior linkage table. It never selects a new layer or alpha from test
results and it does not make a causal-use claim.

Run it once for Qwen and once for Gemma in fresh Colab A100 runtimes. If a genuine software
bug is found after test has been opened, fix it in a new versioned output directory and
document the change; do not silently retune this notebook.

<a id="cell-2"></a>

## 原单元格 2

## 1. Runtime, project, and frozen protocol

The first cells mount Drive, install the local toolkit, stage Dataset v1 on the Colab SSD,
and read the previous discovery artifacts before probe_test is loaded.

<a id="cell-10"></a>

## 原单元格 10

## 2. Resumable probe_test activation capture

Only the frozen candidate layer is captured. The smoke test checks that the hook-based
answer scores match the behavior adapter before the full test capture begins.

<a id="cell-15"></a>

## 原单元格 15

## 3. Locked confirmatory analysis

The next cells refit one frozen net-torque probe on probe_train plus probe_dev and report
probe_test metrics. Raw-feature baselines may fit coefficients on train plus dev, but none
of their choices are selected on probe_test.

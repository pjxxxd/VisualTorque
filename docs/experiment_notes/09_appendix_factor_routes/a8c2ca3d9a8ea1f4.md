# Natural Counterfactual Causal Tracing (A100)

实验设置与运行说明。单元格编号对应来源 notebook，从 0 开始。

对应 notebook：

- [VisualTorque_Natural_Counterfactual_Causal_Tracing_A100.ipynb](../../../notebooks/09_appendix_factor_routes/runs/VisualTorque_Natural_Counterfactual_Causal_Tracing_A100.ipynb)
- [VisualTorque_Natural_Counterfactual_Causal_Tracing_A100.ipynb](../../../notebooks/09_appendix_factor_routes/code/VisualTorque_Natural_Counterfactual_Causal_Tracing_A100.ipynb)

<a id="cell-0"></a>

## 原单元格 0

# VisualTorque — Natural Counterfactual Causal Tracing on one A100

This notebook turns the advisor's mechanistic-interpretability question into a
standard **activation-patching case study**. It does not invent a new benchmark or
estimator. Instead, it exploits Dataset v1's exact natural counterfactuals:

- `add_unit`: change weight while holding the rest of the scene fixed;
- `move_stack`: change lever-arm distance while holding block count fixed;
- `redecompose`: change weight-distance factorization while preserving torque;
- `mirror`: swap left and right while preserving unsigned scene content.

For each source/target pair, the notebook replaces target-run activations with the
source-run activations at one decoder layer and one token group. The main causal
tracing curves compare image tokens, non-image text tokens, and the final prompt
token. Two unrelated-scene controls distinguish pair-specific physical state from a
generic answer/outcome state.

The intended paper-level questions are:

1. Where are changed weight and distance signals carried?
2. Do they converge onto a shared late pathway that transports the answer?
3. Does a torque-preserving redecomposition become behaviorally invariant, or does
   factor/appearance information remain causally active?
4. Do conflict, compatible, and multi-stack problems use different pathways?

This is an exploratory causal-tracing pilot after the representation and specificity
results. It provides causal pathway evidence, not proof of a unique circuit. Run Qwen
first. If the smoke test and control gap are healthy, repeat Gemma in a fresh runtime.

<a id="cell-2"></a>

## 原单元格 2

## 1. Runtime and frozen pilot protocol

The pair sampling, layer grid, patch scopes, controls, and analysis metrics are written
to Drive before model weights are loaded. Existing output directories can only be
resumed when the complete identity is unchanged.

<a id="cell-8"></a>

## 原单元格 8

## 2. Select counterfactual pairs and load the frozen model

`intervention_test` is used as its own pair-level split. For the three
torque-changing edit families, only outcome-changing pairs enter this pilot. For
`redecompose`, source and target torque/outcome must be identical. Pair selection is
independent of model behavior.

<a id="cell-11"></a>

## 原单元格 11

## 3. Natural activation patching

All paired prompts use the same option permutation, so their token IDs, attention
mask, and image-token layout must match exactly. The only source/target input change
is the rendered image. The smoke test refuses to continue if this alignment fails.

<a id="cell-16"></a>

## 原单元格 16

## 4. Locked causal analysis

The recovery metric operates on the three centered semantic log probabilities
`[left, right, balanced]`. Centering removes an irrelevant common log-probability
offset. A recovery of 1 means that the patch reproduced the natural matched
source-minus-target output vector; 0 means no movement along that vector. Recovery is
suppressed when the natural source/target output difference is too small.

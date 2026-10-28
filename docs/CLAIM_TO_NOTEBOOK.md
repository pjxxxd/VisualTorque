# 论文结论与 notebooks 对照

本表按当前稿件 **Beyond Physical Readouts: Explaining Answer Changes in Vision–Language Models** 整理。对照源为 VisualTorque_revision/main.tex 及该目录方法、实验与附录文件，文件身份见 [paper_reference.json](paper_reference.json)。

当前顺序为 4.1 实验设置、4.2 答案编码与迁移收益、4.3 跨场景和编码的迁移预测、4.4 读出与修准、4.5 指导改善的下游依赖。C1–C9 为结论编号，表格按论文实验顺序排列。旧 Qwen/Gemma 结论与新增 DeepSeek 结果分别列出；2026-10-28 稿件尚未纳入 DeepSeek 结果。


## 正文结论

| 编号 | 论文结论及具体数字 | 文件夹 | 生成实验结果的 notebook | 使用范围 |
|---|---|---|---|---|
| C4 | **Gemma 的迁移收益随答案映射改变。** ABC 下 L37 的平均准确率增益由 +23.96 降至 +2.08 个百分点；L47 由 +22.92 转为 −4.95 个百分点。 | [`01_section42_43_transfer_prediction`](../notebooks/01_section42_43_transfer_prediction/README.md) | [Validation](../notebooks/01_section42_43_transfer_prediction/code/VisualTorque_Intervention_Prediction_Validation_GPU.ipynb) 的 matching/conflicting 映射比较 | 正文 4.2–4.3；这些是实际作答增益，不是预测器猜中选项的比例。 |
| C3 | **来源语义或符号能够提前预测迁移后果。** Qwen-27B L49 偏语义，Qwen L63 与 Gemma L37/L47 偏符号；校准选定预测器的分数误差低于不变和置信度基线。 | [`01_section42_43_transfer_prediction`](../notebooks/01_section42_43_transfer_prediction/README.md) | [Calibration](../notebooks/01_section42_43_transfer_prediction/code/VisualTorque_Intervention_Prediction_Calibration_GPU.ipynb) → [Validation](../notebooks/01_section42_43_transfer_prediction/code/VisualTorque_Intervention_Prediction_Validation_GPU.ipynb)，内容模块 | 正文 4.2–4.3、`fig:vt-intervention-predictions`、`tab:vt-forecast-main`。每模块 32 校准家族、96 验证家族；分数预测误差与物理答题正确率分开。 |
| C5 | **同一冻结规则可跨新符号使用。** ABC 的系数保持不变，更换 XYZ 或数字后，所选预测器仍优于两项基线，所测位置的语义/符号偏好保持。 | [`02_section43_format_generalization`](../notebooks/02_section43_format_generalization/README.md) | [Frozen Format Transfer](../notebooks/02_section43_format_generalization/code/VisualTorque_Frozen_Format_Transfer_GPU.ipynb)，输入来自 01 的验证结果 | 正文 4.2–4.3、`tab:vt-prediction-choices`；复用内容模块的 96 验证家族，不能算作新增独立几何。 |
| C6 | **未确认接收距离项提供额外预测增益。** 在含物理难度、接收置信度和来源分数对应的基线上增加 δw·d_r，没有可靠降低预测误差；对物理反事实答案的准确率收益也未确认。 | [`01_section42_43_transfer_prediction`](../notebooks/01_section42_43_transfer_prediction/README.md) | [Validation](../notebooks/01_section42_43_transfer_prediction/code/VisualTorque_Intervention_Prediction_Validation_GPU.ipynb)，物理距离模块 | 正文 4.2–4.3 的 recipient-distance 段落；是指定状态差分操作的边界，不是证明模型绝无距离计算。 |
| C1 | **答错时，隐藏状态仍支持正确的外部物理判断。** 1007 场景中，Qwen-9B/Gemma-12B 分别有 316/234 个“模型错、读出对”，占各自错误的 81.2%/80.4%。 | [`03_section44_readout_correction`](../notebooks/03_section44_readout_correction/README.md) | [统一捕获读出验证](../notebooks/03_section44_readout_correction/code/VisualTorque_Aligned_Readout_Validation_GPU.ipynb) | 正文 4.4、`tab:availability-use` 上半部；使用重新拟合且统一捕获条件的结果。 |
| C2 | **修准读数没有修复原有错误。** 固定 304 场景的读出类别修准后全部正确；Qwen/Gemma 分别修复 0/150、0/129 个原错误，各新增 1 个错误。54 道等重距离题的答案保持不变。 | [`03_section44_readout_correction`](../notebooks/03_section44_readout_correction/README.md) | 同一 [统一捕获读出验证](../notebooks/03_section44_readout_correction/code/VisualTorque_Aligned_Readout_Validation_GPU.ipynb) 的干预汇总 | 正文 4.4、`tab:availability-use` 下半部；“读出类别正确”不等于全部数值误差为零，不代表已修复模型原生计算。 |
| C7 | **固定编辑在新家族上改善 Qwen 答案。** L48/L49 分别使准确率由 58.0% 升至 69.9%/71.9%，修复 61/71 个错误，未损坏原来正确的 297 次呈现。 | [`04_section45_guided_repair`](../notebooks/04_section45_guided_repair/README.md) | [Qwen 三路径独立确认](../notebooks/04_section45_guided_repair/code/VisualTorque_Qwen27B_ThreePath_Confirmation_A100.ipynb) | 正文 4.5、`fig:vt-repair-dependencies`；64 家族、512 次呈现；使用真值选择编辑符号，不是可直接部署的盲修复器。 |
| C8 | **指定下游变化维持修复效果。** L48 修复后复位 L49，61 个修复只保留 6 个；同幅随机移除平均保留 59.5 个。L49→L55 比较同样支持选择性依赖。 | [`04_section45_guided_repair`](../notebooks/04_section45_guided_repair/README.md) | [Qwen 三路径独立确认](../notebooks/04_section45_guided_repair/code/VisualTorque_Qwen27B_ThreePath_Confirmation_A100.ipynb) + [独立结果 CPU 核查](../notebooks/04_section45_guided_repair/code/VisualTorque_Independent_Results_Audit_CPU.ipynb) | 正文 4.5；L48→L55 的完整技术条件未全通过，不把三段统一写成已确认完整链。 |
| C9 | **跨模型修复有边界。** Qwen-9B/Gemma 的开发扫描无候选通过联合标准；独立 Gemma L31 候选修复 7 个错误，相对随机方向准确率差为 +0.54 个百分点，95% 区间 [−0.59,1.76]。 | [`04_section45_guided_repair`](../notebooks/04_section45_guided_repair/README.md) | [Repair Development](../notebooks/04_section45_guided_repair/code/VisualTorque_CrossModel_Repair_Development_A100.ipynb) → [Gemma L31 独立候选验证](../notebooks/04_section45_guided_repair/code/VisualTorque_Gemma12B_L31_FixedCandidate_A100.ipynb) + [独立结果 CPU 核查](../notebooks/04_section45_guided_repair/code/VisualTorque_Independent_Results_Audit_CPU.ipynb) | 正文 4.5；不应写作 Gemma 完全不能被影响，也不应认定与 Qwen 为同一机制。 |


## 新增 DeepSeek：对应 4.2–4.5 的扩展

当前论文正文仍是 Qwen/Gemma 的结果。下面是 DeepSeek 实验的对应关系，不能直接把上表数值理解成已含 DeepSeek。

| 论文位置 | 新 notebook 入口 | 最新结果支持什么 | 需要保留的边界 |
|---|---|---|---|
| 4.2 答案编码与迁移收益 | [01 Transfer](../notebooks/13_deepseek_extension/code/01_DeepSeek_Transfer_4_2_4_3.ipynb) | ABC 下 L20/L26 的 matching−conflicting 收益差为 20.57/18.23 个百分点。 | XYZ/数字下未确认同样的准确率差；不可把 ABC 显著性推广到全部格式。 |
| 4.3 冻结预测与新符号 | 同一 01 notebook | L20/L26 均偏符号；相对不变基线的 MSE 降幅 ABC 为 85.5%/98.3%，XYZ 为 83.7%/97.8%，数字为 82.1%/97.0%。 | 新符号验证复用内容模块的 96 家族；预测分数与物理答对率是不同量。 |
| 4.4 读出可用性 | [02 Readout](../notebooks/13_deepseek_extension/code/02_DeepSeek_Readout_4_4.ipynb) | 1007 场景中模型正确 528，读出正确 816；模型错但读出对为 361/479，即模型错误的 75.4%。 | 与旧模型统一比较时需保持场景、提示及读出口径。 |
| 4.4 修准后果 | 同一 02 notebook | 304 场景的读出类别正确数 214→304，模型正确数 112→111；修好 3/192 个原错误、损坏 4 个，净准确率 −0.33 个百分点，95% 区间 [−1.97,1.32]。 | 支持“未建立净准确率收益”，不支持“没有任何错误被修复”。数值误差容差只在 289/304 场景满足。 |
| 4.5 指导修复 | [03 Repair fixed](../notebooks/13_deepseek_extension/code/03_DeepSeek_Repair_4_5_fixed.ipynb) | 52 家族、416 呈现；初始 L13 编辑修好 6、损坏 2，L20 编辑修好 10、损坏 22，均未通过准确率确认。 | 原 64 家族计划因等重几何容量修订为四类题型各 13 家族；方向与剂量保持冻结，不可称为同等 64 家族复现。 |
| 4.5 下游依赖 | 同一 fixed notebook | L20 修复分数改善及 L20→L23 的分数依赖通过各自主检验。 | 准确率依赖未确认；ua/ub 等范数随机移除出现 154 条技术失败，仅 7 家族全部条件有效。不能推广成完整准确率修复链。 |

完整数值、完成标记、容量修订与技术失败表均可从 [DeepSeek 结果索引](../evidence/deepseek/README.md) 打开。三份 DeepSeek runs/ 是中断过程记录，不能用来否定最新结果包中的完成状态。原 64 家族入口留在 versions/，实际确认使用 fixed 版的 confirmation_capacity52_v1。


## 打开哪份带输出 notebook

下面的 `runs/` 文件保留实际保存的输出，适合导师直接查看表格。上表链接的 `code/` 是源码入口。

| 实验 | Qwen 运行记录 | Gemma 运行记录 | 结果定位 |
|---|---|---|---|
| 4.4 统一捕获读出 | [Qwen-9B（原文件后缀 1）](../notebooks/03_section44_readout_correction/runs/VisualTorque_Aligned_Readout_Validation_GPU%20%281%29.ipynb) | [Gemma-12B（原文件无后缀）](../notebooks/03_section44_readout_correction/runs/VisualTorque_Aligned_Readout_Validation_GPU.ipynb) | 单元格索引 16 为 1007 场景汇总，18 为 304 场景修准汇总；索引从 0 开始。 |
| 4.2–4.3 前瞻验证 | [Qwen-27B 验证输出](../notebooks/01_section42_43_transfer_prediction/runs/VisualTorque_Intervention_Prediction_Validation_GPU.ipynb) | [Gemma-12B 验证输出](../notebooks/01_section42_43_transfer_prediction/runs/VisualTorque_Intervention_Prediction_Validation_GPU%20%281%29.ipynb) | 最后的汇总与图表；区分内容模块、物理模块、分数误差、选项预测和物理正确率。 |
| 4.2–4.3 跨符号验证 | [Qwen-27B 格式输出](../notebooks/02_section43_format_generalization/runs/VisualTorque_Frozen_Format_Transfer_GPU.ipynb) | [Gemma-12B 格式输出](../notebooks/02_section43_format_generalization/runs/VisualTorque_Frozen_Format_Transfer_GPU%20%281%29.ipynb) | ABC/XYZ/123 比较与跨模型多重比较校正。 |
| 4.5 独立修复 | [Qwen-27B 三路径输出](../notebooks/04_section45_guided_repair/runs/VisualTorque_Qwen27B_ThreePath_Confirmation_A100.ipynb) | [Gemma-12B L31 输出](../notebooks/04_section45_guided_repair/runs/VisualTorque_Gemma12B_L31_FixedCandidate_A100.ipynb) | repair/harm、primary tests、各路径技术检查；汇总见下项。 |
| 4.5 CPU 复核 | [独立结果完整核查输出](../notebooks/04_section45_guided_repair/runs/VisualTorque_Independent_Results_Audit_CPU.ipynb) | 同一 CPU notebook | 汇总技术失败、有效家族敏感性分析与题型描述。 |

校准目录保留了两份 Qwen 保存记录及一份 Gemma 记录，未按下载编号推断新旧或删除。核对最终预测时，应使用对应 Validation 保存的冻结预测器及其身份记录。不同模型的 `(1)` 后缀含义也不一致，必须查看模型配置。

## 附录与支持性结论

| 结论或方法 | 目录与 notebook | 对应位置与边界 |
|---|---|---|
| 任务构造与行为基线 | [`00_dataset_and_behavior`](../notebooks/00_dataset_and_behavior/README.md)；[Behavior](../notebooks/00_dataset_and_behavior/code/VisualTorque_Behavior_A100.ipynb)、[Behavior Analysis v3](../notebooks/00_dataset_and_behavior/code/VisualTorque_Behavior_Analysis_v3.ipynb)；共享实现见 `toolkit/` | 方法与实验设置。旧六映射行为口径不能直接替换正文单映射读出比较。 |
| 跨问题迁移更接近来源判断；位点内容随深度不同 | [`05_appendix_content_and_encoding`](../notebooks/05_appendix_content_and_encoding/README.md)；[Cross-query Factor/Decision](../notebooks/05_appendix_content_and_encoding/code/VisualTorque_CrossQuery_FactorDecision_Transport_A100.ipynb)、[Semantic-to-label Depth](../notebooks/05_appendix_content_and_encoding/code/VisualTorque_SemanticToLabel_Depth_Localization_A100.ipynb)、[Fixed-site Confirmation](../notebooks/05_appendix_content_and_encoding/code/VisualTorque_CrossModel_FixedSite_Content_Confirmation_A100.ipynb) | `app:content-diagnostics`、`fig:vt-content`、`tab:vt-fixed-content`；是早期内容诊断，不代替 01 的前瞻预测。 |
| 字母与选项位置可分离 | 同一 05 目录；[Response Code × Position](../notebooks/05_appendix_content_and_encoding/code/VisualTorque_Qwen27B_ResponseCode_PositionFactorial_A100.ipynb) | `fig:vt-content` 的因素分解：16 场景中，字母份额 94.27%，位置份额 0.99%；份额指移除效应的变异，非模型决策解释比例。 |
| 答案保持的物理差分产生分数效应，准确率增益未确认 | [`06_appendix_physical_updates`](../notebooks/06_appendix_physical_updates/README.md)；[Native Physical Update v2](../notebooks/06_appendix_physical_updates/code/VisualTorque_Qwen27B_Native_PhysicalUpdate_vs_AnswerCode_Confirmation_96GB_v2.ipynb)；同目录子集、对齐/残差与稳健性分析 | `app:physical-transfer`、`tab:physical-controls`；24 家族，主剂量相对未编辑 margin +0.0601；准确率 +0.26 个百分点，区间跨零。v1 是中断记录。 |
| 训练减少重量跟随错误、提高行为准确率 | [`07_appendix_lora_training`](../notebooks/07_appendix_lora_training/README.md)；[LoRA Training](../notebooks/07_appendix_lora_training/code/VisualTorque_Qwen27B_BF16_LoRA_WeightShortcut_96GB.ipynb) → [Heldout Evaluation](../notebooks/07_appendix_lora_training/code/VisualTorque_Qwen27B_PostLoRA_Heldout_Evaluation_96GB.ipynb) | `app:training-settings`；1007 场景六映射 52.23%→88.68%，单提示生成 55.11%→87.39%；同一个 adapter，非两个独立训练复现。 |
| 训练改变答案相关状态与干预响应 | 同一 07 目录；[Mechanism Discovery](../notebooks/07_appendix_lora_training/code/VisualTorque_Qwen27B_Base_vs_LoRA_Mechanism_Discovery_96GB.ipynb)、[Blind Causal Validation](../notebooks/07_appendix_lora_training/code/VisualTorque_Qwen27B_PostLoRA_Blind_Causal_Validation_96GB.ipynb)、[Axis × Dose Crossover](../notebooks/07_appendix_lora_training/code/VisualTorque_Qwen27B_Handoff_AxisDose_Crossover_NewGeometry_96GB_v1.ipynb) | `app:training-settings`；L50 状态交换与 L55 相关差异有支持，强度、子空间、度量路线等检验保留各自未通过条件。 |
| 错误和修复依赖物理题型、方向和问法 | [`08_appendix_error_structure`](../notebooks/08_appendix_error_structure/README.md)；[Error Structure](../notebooks/08_appendix_error_structure/code/VisualTorque_Error_Structure_CPU.ipynb)、[Weight Shortcut Decomposition](../notebooks/08_appendix_error_structure/code/VisualTorque_WeightShortcut_Error_Decomposition_A100.ipynb)；04 的 [独立结果 CPU 核查](../notebooks/04_section45_guided_repair/code/VisualTorque_Independent_Results_Audit_CPU.ipynb) | `app:error-structure`；子组描述不能自动变成独立确认的子组差异。 |
| 重量、距离和局部路线具有模型/位置差异 | [`09_appendix_factor_routes`](../notebooks/09_appendix_factor_routes/README.md)；[Task/Token/Component Routing](../notebooks/09_appendix_factor_routes/code/VisualTorque_Task_Token_Component_Routing_A100.ipynb)、[Value-binding Control](../notebooks/09_appendix_factor_routes/code/VisualTorque_Orthogonal_ValueBinding_Control_A100.ipynb)、[Factor Path](../notebooks/09_appendix_factor_routes/code/VisualTorque_Qwen27B_FactorPath_TokenComponent_A100.ipynb)；08 的 [Competing Causal Abstractions](../notebooks/08_appendix_error_structure/code/VisualTorque_Competing_Causal_Abstraction_FinalConfirmatory_A100.ipynb) | `app:site-criteria`、`app:supporting-analyses`；这些操作与正文末提示 token 的序贯编辑不完全相同。 |

## 为什么还保留历史与后续目录

| 目录 | 用途 | 与论文结果的关系 |
|---|---|---|
| [`10_exploratory_composition_and_gating`](../notebooks/10_exploratory_composition_and_gating/README.md) | 跨背景因素重组、乘法律边界、正确性门控 | 说明更强机制假说检验过什么；不能按文件名把 product law 或 gating 视为成立。 |
| [`11_handoff_development_and_matching`](../notebooks/11_handoff_development_and_matching/README.md) | 早期匹配、支持集扩展、修复方向发现、阻断和冻结复现 | 04 的历史输入和方法来源；旧筛选队列不可与正文 64 新家族累加。 |
| [`12_cross_study_audits_and_pilots`](../notebooks/12_cross_study_audits_and_pilots/README.md) | 多实验 CPU 汇总、TorqueTangent、机制到错误预测、35B-A3B 扩展 | 汇总不新增独立样本，另一架构研究也不直接替代正文模型确认。 |
| [`90_followup_visual_and_relation_use`](../notebooks/90_followup_visual_and_relation_use/README.md) | 视觉信息供给、问法诊断、视觉机制及 facts/products/totals 关系使用 | 未作为当前正文 4.2–4.5 的结论来源；保存状态详见目录索引。这里有本地尚无运行输出的源码，不能从代码存在推断结果成立。 |

## 证据与执行状态的区别

本次核对了论文文本、notebook 实现与保存输出，并参考项目已有的 aligned-readout、intervention-prediction、frozen-format-transfer、independent-confirmation/independent-completion 核查报告。没有重新执行模型推理或所有统计。论文中使用的四舍五入数字在此保留；查精确值应打开相应运行记录或原结果包。

原有两份保存异常的历史运行已保留：早期 `Causal_Torque_Intervention_A100.ipynb` 的 AssertionError，以及 `Native_PhysicalUpdate_vs_AnswerCode_Confirmation_96GB_v1.ipynb` 的 ValueError。它们不作为当前正文结果或有效阴性结论。另新增三份 DeepSeek 中断记录，最新完整结果另见上文。其他文件即使未保存异常，也不能仅凭执行编号认定所有科学条件通过。

[返回总览](../README.md) · [运行说明](RUNNING.md) · [所有源文件到副本的对应表](source_to_copy.csv)

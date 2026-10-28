# 下一轮：冻结预测器，测试新的答案符号

## 怎么运行

打开 `VisualTorque_Frozen_Format_Transfer_GPU.ipynb`，使用 GPU，顺序运行全部单元。

1. 默认 `MODEL_KEY='qwen27b'`，先运行 Qwen-27B。
2. 完成并导出后，使用**全新的 GPU 运行时**，改为 `MODEL_KEY='gemma12b'` 再跑一次。
3. 两次保持相同 `RUN_ID='main'`。第二个模型完成后自动汇总两个模型的 32 项主比较。

无需重新运行上一轮 Calibration 或 Validation，也不需要 Qwen-9B。辅助代码已经嵌入 notebook，不需要另传 `.py` 文件。Qwen-27B 建议 96 GB GPU；Gemma 需足够显存容纳完整 BF16，及对应模型的 Hugging Face 访问权限。不能用量化模型替代本协议。

默认读取 Drive 的 `VisualTorque/results/intervention_prediction_v1/main/`。如果该目录缺失，可上传上一轮对应模型的 `VisualTorque_Intervention_Prediction_<模型>_validation_Output.zip` 到 Colab，并在配置单元填写 `SOURCE_VALIDATION_ZIP`。**只有 validation ZIP 就够，不需要 calibration ZIP**：前者已含冻结预测器和所需验证状态。Drive 中仍需原有 `VisualTorque/toolkit`。

图像优先从上一轮 Drive 的 `images` 目录读取。ZIP 不包含原 PNG；缺失时会尝试重新渲染并核对原文件校验值。不同 Pillow/zlib 环境可能改变 PNG 文件字节，此时应通过 `SOURCE_IMAGES_OVERRIDE` 指向保留的原图目录，不放宽校验或悄悄替换图像。因此请保留上一轮 Drive 的图像目录。

可以从头运行以续跑：配置、输入、预测封存和状态均会核验，已完成部分自动复用。结果写入独立新目录，不覆盖以前的实验。不需要重新扫描历史文件或生成新几何。

## 这一次新增什么

先回答一个解释问题：最后一层干预的输出，是否与保存状态经过实际末端归一化、输出模块后的结果相符？这是一项**输出过程重算**，不是重新发现物理机制，也不是与简单预测器公平竞争的独立预测方法。保存实际模型代码、必要的小规模参数和逐次误差；不假设 BF16 在不同矩阵形状或硬件上完全一致，不根据重算结果更改预测系数。

随后回答真正的新增实验问题：**在 ABC 条件校准出的规则，换为 XYZ 或 123 后，不重新拟合，还能预测干预的后果吗？**

- 复用上一轮答案映射模块的 96 个验证家族，包括两种来源问题和三种接收映射。
- Qwen 固定 L49、L63；Gemma 固定 L37、L47，层号从 0 开始。
- 各格式的来源和接收者使用同一套符号。接收者内部的语义、字母和显示位置对应仍按原实验交叉。
- ABC 在当前运行时重新执行，作为配对参照；旧 ABC 另行比较，避免把环境差异当成格式效应。
- XYZ 仅替换答案符号；123 另把提示中的 `option letter` 改为 `option number`，其余问题、规则、选项含义和图像不变。
- 为原预测器建立固定的坐标对应：ABC 的三个数组位置对应 XYZ 或 123 的三个符号。这是标签换名，不是训练；标签坐标与屏幕显示位置仍然分开。
- 每种新格式重新读取来源和未干预接收者的分数与状态。原有系数、标准化参数、校准选择的基线保持原值。
- 全部三种格式的全部预测保存后，才执行新干预；没有拟合或重新选择层的入口。

这不是跨不同字母表的来源到接收者直接状态迁移，也不是新的独立几何验证。它检验的是**同一种预测规则能否跨答案格式使用**。这里不重跑重量—距离模块，不重复随机方向对照；不产生新的物理方向特异性主张。

## 统计和解释

主指标是三个候选分数变化的预测误差。独立单位仍为 96 个家族，同家族的字母表、映射和层不能当成更多独立样本。

XYZ 和 123 各自在两个固定层比较：

1. 语义偏好与字母偏好；
2. 字母偏好与显示位置偏好；
3. 原校准选出的基线与“不变”预测；
4. 原校准选出的基线与置信度预测。

共 16 项/模型，两模型共 32 项主比较，使用家族 bootstrap 区间、配对符号翻转检验和 Holm 校正。当前 ABC 是参照，不再混入“新格式”主检验。

同时报告具体答案预测、真正改变选项时的预测、修复/损害事件及实际分母。不出现某类事件时留空，不写成 100%。这些预测准确率不是模型物理答题准确率；答题收益单独报告。

新格式相对新 ABC 的误差变化属于描述性配对分析。区间跨零**不等于证明格式不变性**。也不以高整体预测率替代对选项确实变化的试验分析。不论结果是否迁移，都保留全部格式和家族。

## 输出

Drive 目录：

    VisualTorque/results/frozen_format_transfer_v1/main/
      models/qwen27b/validation/
      models/gemma12b/validation/

最后自动下载：

    VisualTorque_Frozen_Format_Transfer_qwen27b_Output.zip
    VisualTorque_Frozen_Format_Transfer_gemma12b_Output.zip

主要结果：

| 文件 | 内容 |
|---|---|
| `prediction_summary.csv` | 各格式预测误差、选项预测、实际变化时的准确率、答题收益 |
| `primary_comparisons.csv` | 新格式的预设主比较 |
| `cross_model_primary_comparisons.csv` | 两模型完成后的统一校正 |
| `transition_events.csv` | 修复、损害、换错答案等事件的识别与误报 |
| `format_error_change_descriptive.csv` | 新格式相对当前 ABC 的误差变化 |
| `mapping_family_gains.csv` | 同映射、冲突映射下实际和预测收益 |
| `archive_tail_replay.csv` | 旧末层状态的输出重算误差 |
| `current_tail_replay.csv` | 新干预的末端重算误差 |
| `archived_vs_current_ABC.csv` | 旧 ABC 与当前 ABC 的环境复现差异 |

ZIP 同时包含冻结预测器、输入和状态记录、干预前预测、原始干预分数、运行配置及核验清单。输出端所需的小规模权重片段仅用于核验，不包含完整模型。

最后的导出单元可在中断后单独运行，未完成时标为 `partial_not_complete`。只有全部 3,456 次目标干预和分析完成才标记完整。每个模型正常执行 2,592 次普通输入前向、3,456 次目标干预，共 6,048 次完整前向；另有较便宜的末端重算。这是操作次数，不是 GPU 耗时承诺。

## 实现和本地验证范围

沿用已有单 token 候选评分、原生 BF16 差分、精确状态重放检查。单次运行中发现同一输入的状态变化时停止，避免把不同状态的预测与干预结果配在一起。跨运行环境的旧 ABC 差异另行报告，不靠放宽重放阈值继续。

输出端链条依据 Transformers 的 Qwen3.5/Gemma4 官方实现，并使用运行时实际模块和配置；Gemma 的末端 logit softcapping 不能漏掉。参考：[Qwen3.5 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)、[Gemma4 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py)。实际运行固定上一轮使用的 Transformers 5.14.0，并保存已安装源代码身份。

本地只执行了协议与软件检查，没有运行 GPU 模型、产生新格式实验结果或测量真实运行时间。数值模拟仅用于检查程序，不进入论文结果。

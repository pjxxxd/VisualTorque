# 统一捕获条件的读出验证

打开 `VisualTorque_Aligned_Readout_Validation_GPU.ipynb`。这份 notebook 用来检查：重新在当前运行条件下拟合读出器以后，内部读数和模型答案的差距是否仍然存在，以及修准读数能否改善答案。

## 运行顺序

1. 使用全新的 GPU 运行时，建议 A100 80GB 或足够显存的同等级 GPU。默认 `MODEL_KEY='gemma12b'`，从第一格顺序运行至最后一格。
2. 保存自动下载的 `VisualTorque_Aligned_Readout_gemma12b_Output.zip` 和带输出的 notebook。
3. 断开并删除运行时，重新连接 GPU，把 `MODEL_KEY` 改成 `qwen9b`，再从第一格运行。保存对应 ZIP 和 notebook。

**本份不用运行 Qwen-27B。** 不需要再运行旧 capture bridge。

保留之前的 Drive 项目路径及 Hugging Face 访问配置。安装步骤沿用前轮环境；Gemma 仍需要账号已获模型访问权限。依赖安装失败或 GPU 不可用时先解决该步骤，再运行后续单元格。

## 固定范围

| 项目 | 本轮设置 |
|---|---|
| Gemma-12B | 原模型版本、L7、ridge alpha 100 |
| Qwen-9B | 原模型版本、L13、ridge alpha 0.01 |
| 捕获方式 | 训练、测试、干预统一 batch size 1，BF16 前向，最后一个有效提示 token |
| 读出器拟合 | 重新采集原 254 个场景 × 6 种答案映射；保留原 FP16 存储及六映射平均规则 |
| 答案与读数比较 | 原 1007 个测试场景，每个使用原来固定的一种映射；新旧读出器读取同一个新状态 |
| 数值修准 | 原 304 个诊断场景，使用真实净力矩，读数误差要求不超过 0.05 |
| 随机对照 | 原 72 个对照场景，每个 4 个固定随机方向；匹配 BF16 写入后的实际变化幅度，相对误差不超过 2% |

层号从 0 开始。254 个拟合场景包括原 235 个训练场景和 19 个开发场景，与旧读出器最终拟合范围相同。本轮不再用这些开发场景选层或选 alpha；只取其中两个固定例子检查写入过程，不将其当作独立效果证据。

不训练大模型，不做新层扫描，不重复五剂量扫描。测试集曾经被查看，本轮属于运行条件的稳健性复核。全新无缓存运行预计每个模型 3435 次前向：1524 次训练捕获、8 次开发检查、1007 次测试及 896 次干预与对照。

## 需要已有文件

挂载原项目后，notebook 会检查以下文件与样本身份：

- `data/dataset_v1.zip` 或 `data/dataset_v1/` 下原 Dataset v1 的场景、划分及图片，数据准备单元格会定位并缓存。
- `results/behavior_comparison_v3/activation_pilot_manifest.jsonl`。
- `results/<MODEL_SLUG>/activation_discovery_v1/capture_config.json`。
- `results/cross_model_readout_correction_v2/models/<MODEL_SLUG>/frozen_probe.json` 与 `protocol.json`。
- `results/cross_model_readout_correction_v2/shared_304_manifest.json` 与 `shared_1007_scene_ids.json`。
- 原 `toolkit` 代码。

不读取旧的训练或测试激活。旧读出器只作当前同一状态上的参照，实际修准和随机对照使用新拟合的读出器。

## 断点续跑与保存

结果写入独立的 `results/cross_model_aligned_readout_v1/<RUN_ID>/<MODEL_SLUG>/runs/<运行身份>/`。默认 `RUN_ID='main'`。运行身份包含模型、环境、代码与输入文件；环境改变后会自动使用不同子目录。

中途停止后，从第一格重新运行。同一身份下已完整保存的场景会自动复用；未完成的场景重新计算。恢复时不要只重复累积记录的某一格。若修改了实验配置导致冻结检查失败，保留旧目录并设置新的 `RUN_ID`。

某阶段报告未完成时，后续依赖阶段会跳过；继续执行最后一格，下载标记为 `partial` 的结果包。包中保留已完成记录、状态和失败原因。解决问题后再从第一格重新运行。若运行时已断开，文件仍保存在 Drive 上。

## 结果包看什么

- `REPORT_ZH.md`：中文摘要，直接说明读出正确但答案错误的数量、修好及改错题数。
- `RUN_COMPLETE.json`：流程是否完成、实际前向次数、输入是否改变、技术失败计数。`workflow_complete=true` 只表示流程完成。
- `readout_answer_comparison.csv`：同一当前状态上的新旧读出器与模型答案。
- `correction_summary.csv`：全部 304 场景，以及平衡题和非平衡题的修准与作答变化。
- `paired_effects.csv`：按场景配对的准确率变化、正确答案相对最高错误答案的分数变化及 95% 区间；四个随机种子先在同一场景内平均。
- `oracle_write_failures.csv`、`random_norm_failures.csv`：没有达到数值或幅度要求的尝试，全部保留。
- 原始捕获和干预记录、新读出器、状态 `.npz`、样本清单及运行身份：用于后续复核。

总体结果保留所有尝试，另列数值修准成功子集及修准成功、四个随机方向均匹配成功的共同子集。不要只报告有效子集，也不要将“没有检出提升”直接写成“完全没有作用”。原 5 个百分点改善标准配合区间解释；真实净力矩修准检验的是所选读出器方向，不能自动等同于修复了模型原生物理计算。

生成阶段完成本地合成数据检查；没有在本机运行这些大模型。实际研究结果由上述 GPU 运行产生。

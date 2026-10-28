# 下一轮实验 notebook：运行说明

这批 notebook 对应计划中的 E1、E2-A 和 E2-B，围绕两个问题补齐证据：**读出的数值修准后，模型会不会答得更好；迁移效应更接近结果含义、答案字母，还是选项位置。**

本地已完成代码、数值和分析流程检查；模型实验的完成状态与结果以 Colab 输出为准。

## Readout 在 12/15 报 Archived readout parity failed 时

这表示当前采集的隐藏状态经同一个读出器得到的数值，与旧归档相差超过 0.15；不是数值修准的 0.05 成功标准未通过。旧采集使用批量前向，新入口使用单样本前向，存在计算路径差异，但具体原因需结合诊断数据判断。

**正在运行旧 notebook 时，推荐保留当前运行时：** 打开配套 `Readout_Replace_Failed_Cell.py`，复制全部内容，替换原来报错的整个 `12/15` 代码单元格，再从该格继续。不要将补丁作为独立程序运行；它使用前面已加载的模型和已拟合的读出器。

修正版保持 0.15 的复用检查阈值和 0.05 的修准标准。旧状态无法通过数值核对，或原分片不完整时，改用当前运行方式重新采集全部 1,007 场景 × 6 映射。它按场景保存进度，保留原模型、读出器与数据清单，不按答案正确性筛选复用记录。

具体的新旧读出差异会保存在当前模型目录下 `test_reuse_v2/<身份>/parity_diagnostics.csv`；`analysis/test_capture_source.json` 记录最终使用的数据来源。补丁使用单独的决策锁与测试分片目录，不会撞上旧的 `test_reuse_identity.json` 或混入旧测试缓存。模型、checkpoint、场景或映射身份不一致仍会停止检查。

**若从头使用修正版完整 notebook：** 结果写入 `cross_model_readout_correction_v2`，避免代码更新后的协议锁与旧版冲突。用单元格补丁继续旧运行时，则继续使用原来的模型输出目录，测试分片另存。无需删除旧文件。

## 1. 文件与运行顺序

| Notebook | 默认模型 | 完成的实验 | 本轮需要运行 |
|---|---|---|---|
| `VisualTorque_CrossModel_Readout_Correction_A100.ipynb` | Qwen-9B | 净力矩读出、将读出数值修准、原五剂量、匹配随机与正交对照 | Qwen-9B、Gemma、Qwen-27B 各一次 |
| `VisualTorque_CrossModel_CrossQuery_Transport_A100.ipynb` | Gemma | 48 家族，2 种来源问题 × 5 种接收问题，统一最后一层 | Gemma、Qwen-9B 各一次；27B 可读取旧同位置结果 |
| `VisualTorque_Gemma_Response_Depth_A100.ipynb` | Gemma | 32 家族，七层扫描，正反重映射，区分结果含义、字母与位置 | Gemma 一次；两个 Qwen 的旧扫描自动作为历史参照 |

建议先运行第一份的 Qwen-9B，再完成 Gemma 的两份补齐实验；其余模型按显存安排。E1 和 E2 在逻辑上相互独立，有不同 GPU 时可以分别执行。

这三份 `.ipynb` 都包含完整代码；上传到 Colab 后不需要上传构建脚本。保持原来的 Drive 项目、数据和结果目录即可。

## 2. 开始运行

1. 将 notebook 上传至 Colab，选择 GPU 运行时。9B/Gemma 可沿用既有 A100 环境；27B 的 BF16 权重建议使用 80/96 GB 显存。
2. 在模型配置单元格修改 `MODEL_KEY`：`qwen9b`、`gemma12b` 或 `qwen27b`。
3. 从上到下运行全部单元格，挂载原 Drive 项目。Gemma 沿用原 HF 访问方式；checkpoint 从已完成实验的配置读取。
4. 换模型时使用新的运行时。中断后可以重新从头运行，已经完成的场景或家族会被读取。

项目自动识别以下两个位置之一：

```text
/content/drive/MyDrive/VisualTorque
/content/drive/MyDrive/VisualTorque/VisualTorque
```

被识别的目录需要包含 `toolkit/pyproject.toml`。安装环节沿用已有成功 notebook 的 `transformers==5.14.0`，并直接加载项目中的 toolkit 源码。

## 3. 需要保留的原文件

下列路径均相对 Drive 项目根目录。Notebook 会核对缺失或身份不一致的文件，并给出具体路径。

**E1：**

- `data/dataset_v1.zip`，或者完整 `data/dataset_v1/`。
- `results/behavior_comparison_v3/activation_pilot_manifest.jsonl`：原来的 254 个训练/开发场景。
- Qwen-9B/Gemma 各自 `results/<model_slug>/activation_discovery_v1/` 下的 `capture_config.json`、`scene_mean_activations.pt` 和 `probe_results/discovery_selection.json`。
- 已有 `activation_confirmatory_v1/`、`causal_torque_intervention_v2/` 的配置和原始分片：通过相应核对时复用。1,007 场景测试状态要求完整归档且数值核对通过，否则全部重采集；原五剂量保持原来的逐场景复用方式。

Qwen-27B 如果没有同协议 discovery 归档，会使用原 254 场景，自动完成八层候选的 ridge 开发选择；不把其他 PLS 或 LoRA 开发结果混入本次比较。

**E2：**

- `results/inverse_query_weight_route_specificity_v1/inverse_query_manifest.jsonl` 及同目录 `rendered_scenes/`。
- 层扫描还需要 `results/response_state_remapping_v1/remapping_manifest.jsonl`。
- `results/google__gemma-4-12B-it/activation_discovery_v1/capture_config.json` 用于锁定 Gemma checkpoint。
- 保留 `cross_query_factor_decision_transport_v1/` 与 `semantic_label_depth_localization_v1/`，用于读取符合条件的历史 Qwen 结果。

## 4. 这批入口中特别处理的几点

- **Qwen-9B 跨问题位置需要补齐。** 旧实验使用 L16 的选项位置；新主比较统一为最后 decoder 层的 final-prompt token，即 Qwen-9B L31、Gemma L47、Qwen-27B L63。层号均从 0 开始。
- **Gemma 按 48 层换算。** 七个扫描层为 L19、25、31、37、41、44、47，保存实际相对深度。换算时对层数四舍五入，半数向上。
- **不要求先出现期待的结果。** Gemma 无需先证明末层存在标签效应；没有检出清晰阶段时，分析与结果导出仍然完成。
- **修准成功与失败分开。** E1 以写回 BF16 后误差不超过 0.05 力矩单位为标准。完全没有开发样本达到要求时，先停在技术检查；正式运行中的个别失败保留并单列。
- **对照匹配实际写入幅度。** E1 在原 72 场景中，对每类对照使用 4 个固定方向；先在场景内平均，再计算区间。范数匹配失败，以及正交对照写回后读出变化过大的记录均单列。
- **原五剂量保持独立来源。** 使用历史五剂量结果时，与同一历史运行的零剂量比较；新增修准始终与本次新采集的零编辑配对。
- **不混合不同运行。** 新结果存入独立目录。断点恢复核对模型、输入、代码和协议；不一致时停止，不覆盖或删除旧实验。

## 5. 运行后看哪些结果

### E1

完整修正版目录：`results/cross_model_readout_correction_v2/models/<model_slug>/analysis/`。通过单元格补丁继续旧运行时，仍在该运行原来的 v1 模型目录下。

| 文件 | 用途 |
|---|---|
| `REPORT_ZH.md` | 快速阅读：多少数值修准成功、修正多少旧错误、引入多少新错误、准确率变化区间 |
| `summary_by_cohort.csv` | 304 个总体、263 个非平衡场景、41 个平衡场景分别比较 |
| `readout_1007_scenes.csv` | 六映射平均读出与固定单映射读出分别报告 |
| `paired_control_accuracy.csv` | 修准与匹配随机/正交对照的场景配对差异 |
| `tilt_margin_changes.csv` | 非平衡场景的正确方向分数变化 |
| `oracle_write_failures.csv`、`control_match_failures.csv` | 实际写入与对照匹配未达成的记录 |
| `intervention_records.json` | 完整逐场景记录，后续汇总的主要输入 |
| `test_capture_source.json` | 本轮测试状态是否来自旧归档，以及不能复用时的具体数值差异 |

三模型完成后，`cross_model_analysis/` 导出相同场景上的模型间准确率变化差异。区间按场景配对计算，不通过比较“显著/不显著”判断模型差异。

### E2

跨问题目录：`results/cross_model_cross_query_v1/`；层扫描目录：`results/cross_model_semantic_label_depth_v1/`。

- 各模型的 `analysis/REPORT.md`、结论 JSON、完整效应 CSV 和 `figures/`。
- `native_displacement_audit.csv` 记录写回模型后的实际位移，区别于计划加入的向量长度。
- `cross_model_analysis/` 中保留全部矩阵或七层轨迹、历史/新运行来源，以及共享家族上的配对差异。
- 末尾汇总单元格读取当时已经完成的模型。后续补完其他模型后，重新运行最后一个模型的 notebook 即可刷新汇总；采集阶段会从已有分片恢复。

## 6. 本轮结果与下一批实验的关系

这里复用已有场景和家族，属于同题补齐与新增诊断，不称为全新独立验证。数值修准使用真实净力矩，只检验外部读出方向的答案作用；不能直接写成模型原生物理计算被修复。

E3 的物理变化迁移，以及 E4 的答案修正与局部阻断，将依据这批结果确定各模型的开发位置、方向和适用路径。因此本包没有预填未知的修正层，也没有把“尚未检出语义阶段”改成强制三阶段流程。

## 7. 已完成的本地验证

15 项 CPU 检查通过。新增检查覆盖：读出与分数不一致后自动重新采集、完整归档复用、缺失/部分归档处理、真实身份错误仍停止，以及在旧运行时中恢复和继续采集。原来的数值、分母、映射和阴性结果导出检查也已通过。所有 notebook 已清空执行输出并通过代码语法检查。

这些是实现检查，不是模型实验。真实模型的 GPU 前向、运行时显存和完整实验结果，仍需在 Colab 执行后确认。

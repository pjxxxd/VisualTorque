# 两项收尾诊断的运行说明

这次提供两份独立 notebook：先跑 CPU 的错误结构分析，再用 GPU 对 Gemma 和 Qwen-9B 各跑一次捕获对照。原有实验文件、方向、阈值和确认状态不变。

2026-09-14 修正：GPU 输入指纹支持 Transformers 返回的 `BatchFeature`／`BatchEncoding` 等映射对象，修复正式前向前的 `Untracked prepared-input value: BatchFeature`。GPU 默认 `RUN_ID='input_mapping_fix'`，避免与原 `main` 的冻结配置冲突，并保留未完成诊断。CPU 分析已经完成，无需重跑。修正版经过本地回归测试，尚未执行真实 GPU 前向。

## 推荐顺序

| 文件 | 运行环境 | 如何选择模型 |
| --- | --- | --- |
| `VisualTorque_Error_Structure_CPU.ipynb` | Colab CPU | 不用修改模型，一次读取 Qwen-27B、Gemma |
| `VisualTorque_Readout_Capture_Bridge_GPU.ipynb` | 建议 A100 80GB 或同等足够显存 GPU | 默认 `gemma12b`，完成后换全新运行时，把 `MODEL_KEY` 改为 `qwen9b` 再跑 |

GPU notebook **不运行 Qwen-27B**。无需重跑原独立确认实验，不要求先完成 CPU notebook 才能开始 GPU 对照。

上传到 Colab 后挂载原来的 Drive，从上到下运行。保留各自默认的 `RUN_ID`：CPU 为 `main`，修正版 GPU 为 `input_mapping_fix`。`REPEAT_RUN_ID='main'` 仍指向先前重复性诊断，无需修改。两份均有进度条，最后会自动下载结果 ZIP。

## 第一份：错误结构分析

输入是原独立确认目录下的完整 `analysis/intervention_records.json`、物理场景定义、完成状态和输入身份记录：

- `results/cross_model_independent_confirmation/qwen_three_path/main/Qwen__Qwen3.5-27B__373612cb/`
- `results/cross_model_independent_confirmation/gemma_l31_candidate/main/google__gemma-4-12B-it/`

它会给出：

1. 错误时实际选择左、右还是平衡，是否与重量或距离启发式预测一致。
2. 在重量与物理答案冲突的场景中，跟随重量预测的比例；同时显示分子、分母，区分等重与不等重。
3. 正确答案为左、右时各自原有多少错误，修正多少，又造成多少新错误。
4. 同一场景镜像后的答案是否按物理要求交换左右；换答案字母后物理答案是否保持一致。
5. 修改相对原答案的一致性变化，使用同一家族的配对差值。

全部为已有数据的补充分析。区间按完整物理家族重采样，随机方向先按种子平均，不把镜像、映射或随机种子算作独立样本。没有符合条件的分母时显示缺失，不写成 0%。

输出目录：`results/cross_model_independent_confirmation/error_structure/main/`。

最终结果包：`VisualTorque_Error_Structure_Output.zip`。

若缺少一个模型或输入验证失败，另一模型仍继续；最后明确列出未完成项，不将部分运行标成完整。

## 第二份：捕获流程与批处理对照

必需输入：

- `results/cross_model_readout_correction_v2/models/<MODEL_SLUG>/` 中冻结读出器、协议和运行身份。
- `results/<MODEL_SLUG>/activation_discovery_v1/capture_config.json`。
- 原 Dataset v1 与 `results/behavior_comparison_v3/activation_pilot_manifest.jsonl`。

最好同时保留原 `scene_mean_activations.pt`、全部 `shard_*.pt`，以及上一份重复性诊断结果。缺少历史分片时会明确标注，不能据此宣称恢复了原批次。

固定模型 revision、旧读出器、原层位及原阈值；不拟合、不选层、不增加科学测试集。

运行包含三部分：

1. 从旧逐映射分片重建六映射场景均值，检查历史缓存内部是否一致。
2. 在同一当前环境中对照旧模板与当前 E1 的单张输入准备。完整比较输入张量；输入相同则只运行一次模型，同时保留原生状态和旧式 FP16 捕获，避免重复计算。
3. 对相同开发图片运行候选批处理上下文，与逐张运行及历史缓存比较。

**批大小的来源会记录。** 旧配置若未记录批大小，默认候选值来自本地旧模板：Gemma 为 2、Qwen-9B 为 4。历史分片顺序也未记录全部批次边界和恢复运行过程，因此默认不把候选批次称为历史精确重放。只有掌握历史配置时才填写 `BATCH_SIZE_OVERRIDE`。

为保留候选批次上下文，少量原训练图片可能随批前向；只统计 19 个开发场景的 114 种映射，不训练读出器，也不把这些图片当作新测试样本。运行前会打印批次数和额外上下文图片数。

单张部分通常是 114 次前向；只有两种流程准备出的输入不一致时，才分别执行，最多 228 次。另加运行前打印的候选批次数。不会重复之前的三次同配置试验，也不会全面重跑 254 个场景。

注意：批处理可能同时改变填充、图像处理或数值计算，不能仅凭它与历史缓存更接近，就认定唯一原因是数值误差。无法可靠逐图拆分的图像张量会标成未知。原模板的末尾位置读取被保留，若读到填充位置，会作为发现报告，不会悄悄改变位置让结果通过。

输出目录：`results/cross_model_independent_confirmation/capture_bridge/input_mapping_fix/<MODEL_SLUG>/`。

最终结果包分别为：

- `VisualTorque_Capture_Bridge_gemma12b_Output.zip`
- `VisualTorque_Capture_Bridge_qwen9b_Output.zip`

包内包括 114 种映射、19 个场景的完整结果、选中层的捕获状态、冻结读出器、输入指纹、批次组成、环境信息及中文报告。

## 中断、报错和继续运行

GPU 捕获按输入、模型、代码和运行环境保存独立缓存。同一身份重新运行会复用已完成部分；不同 GPU 或软件环境会进入不同缓存，避免混用。

如果某个诊断步骤提示未完成，请继续运行后面的单元格直到最后打包。模型加载失败或显存不足也会尽量写入问题记录，不会自动减小批量、修改精度或替换模型。安装依赖或挂载 Drive 本身失败时，需先解决该环境问题。

如果修改了 notebook 的冻结设置并触发配置冲突，应使用新的 `RUN_ID` 保留原输出，不删除旧目录或放宽原检查。

## 请保存的文件

- CPU 运行后的 notebook 和它的输出 ZIP。
- Gemma 运行后的 GPU notebook 和 Gemma 输出 ZIP。
- Qwen-9B 运行后的 GPU notebook 和 Qwen-9B 输出 ZIP。

即使屏幕只预览部分表格，ZIP 中仍保留完整明细。结果 ZIP 与本次提供的 notebook 发布 ZIP 是不同文件。

## 本地验证范围

构建时检查独立 notebook 的语法、依赖名称、输出为空、打包完整性，以及错误分类、家族分母、镜像/映射配对、输入差异识别、批次覆盖、场景聚合和缓存隔离等关键逻辑。测试使用明确标注的合成数据，不作为研究结果。完整 Drive 数据和真实 GPU 前向需在运行环境中验证。

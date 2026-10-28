# 下一步实验 notebook

这批 notebook 对应实验计划第 14 节。可以单独上传到 Colab；实验代码已嵌入，不需要上传旁边的 Python 源码。继续使用现有 Drive 项目、toolkit、Dataset v1 和原结果目录。

## 运行顺序

| 顺序 | 文件 | 模型与设备 | 要回答的问题 |
|---|---|---|---|
| 优先 | `VisualTorque_Qwen27B_ThreePath_Confirmation_A100.ipynb` | 固定 Qwen-27B；A100 80GB 或足够显存 | 原 L48→L49、L48→L55、L49→L55 的选择性依赖能否在新场景重现？ |
| 可并行 | `VisualTorque_CrossModel_TargetedAudit_CPU.ipynb` | CPU | 正交对照为何失败？历史错误组是否比正确组受到更强的阻断？ |
| 可选扩展 | `VisualTorque_Gemma12B_L31_FixedCandidate_A100.ipynb` | 固定 Gemma-12B；GPU | 原 L31、factor=1 的改善能否重复，并且优于随机方向？ |
| 数值诊断 | `VisualTorque_CrossModel_Readout_Repeatability_A100.ipynb` | 默认 Gemma；需要时在全新运行时切换为 Qwen-9B | 相同配置下，原读出器的读数和答案是否稳定？ |

若只运行一个新增科学实验，先运行 Qwen-27B。CPU 核查不依赖新实验完成。已有相同配置的完整重复性诊断时，不必重复运行第 4 份。

**这批第 4 份是读出诊断，不是上一批的修复开发。Qwen-27B 不需要运行上一批第 4 份，也不在这批第 4 份中运行。** 两份科学确认 notebook 的模型已固定，无需修改模型名。

## 开始前需要哪些已有文件

挂载 Drive 后自动寻找 `MyDrive/VisualTorque` 或 `MyDrive/VisualTorque/VisualTorque`，以包含 `toolkit/pyproject.toml` 的目录为项目根。沿用已验证的推理依赖和模型 revision，默认不量化、不使用磁盘卸载。

- Qwen-27B：上一批第 2 份完成后的 `results/cross_model_followup_v1/equal_blocking/main/`，其中包含 `selected_route.json`、`frozen_directions.npz` 和 `analysis/RUN_COMPLETE.json`。
- Gemma：上一批第 4 份完成后的 `results/cross_model_followup_v1/repair_development/main/google__gemma-4-12B-it/`，需要 `development_identity.json`、`selected_route.json`、`capture/` 中原训练状态和完成记录。原来的 `eligible=False` 保留，不能手动改成 True。新入口从 235 个训练缓存按原 PLS4 公式重建候选；不使用 19 个开发场景的状态拟合。因为旧实验没有导出选中向量，新入口记录重建环境与输入，不能声称做了原向量的逐位比对。
- CPU 核查：三个模型的 `results/cross_model_readout_correction_v2/models/<模型>/analysis/intervention_records.json`，以及上述 Qwen 等幅度阻断的完整记录。缺失内容进入覆盖表，其他可用部分继续分析。
- 读出诊断：对应模型原 E1 的 `frozen_probe.json`、`protocol.json`、`runtime_identity.json`，以及原 `activation_pilot_manifest.jsonl` 和 Dataset v1。原 discovery 状态归档存在时，另行比较历史场景平均读数。

没有这些原训练缓存时，不要重新采集一套状态并冒充旧候选；保留报错及缺失路径，以便针对实际文件修复。

## 两份新场景实验如何运行

打开对应 notebook，选 GPU，直接从上到下运行。`RUN_ID='main'` 和 `SOURCE_RUN_ID='main'` 默认对应现有结果；只有旧实验使用了其他运行名，才修改 `SOURCE_RUN_ID`。

每份默认 64 个物理家族，每家族含两个镜像、两种渲染、两种选项映射，共 512 个场景与映射组合。主统计样本数是 **64 个家族**，不能写成 512 个独立样本。固定样本量后，不根据输出扩样或重新选点。

生成时重新索引 Dataset v1 和当前 Drive 的已登记场景，排除物理配置及其镜像，包括旧固定位置确认和已经登记的旧第五份修复确认。Qwen 与 Gemma 默认使用各自不同的未见家族；后运行的实验也排除已生成的新家族，因此不需要先一起启动。它们可以比较趋势，但本批没有预定直接的模型间成对检验。

**改变 `RUN_ID` 不等于获得新数据。** 已有支持集会被识别；若固定物理范围内剩余家族不足，程序会在加载模型前停止，不会偷偷换范围或复用场景。

新实验目录：

```text
results/cross_model_independent_confirmation/
  qwen_three_path/main/Qwen__Qwen3.5-27B__373612cb/
  gemma_l31_candidate/main/google__gemma-4-12B-it/
  targeted_audit/main/
  readout_repeatability/main/<模型>/
```

旧目录只作为输入。每个场景与映射的全部条件完成后保存一个分片；中断后使用相同 notebook 和配置重新运行，已完成的分片会复用。修改实现、方向、环境或支持集后，原缓存不能混用。若在一个场景处理中断，最多重做该场景与映射的条件。

程序核对原实验与当前的模型 revision、Torch、Transformers、CUDA、dtype、batch、层路径和 toolkit 文件身份。若环境不同，先恢复原环境，不要删除锁文件绕过检查。原结果使用的 Torch 为 2.11.0+cu128，Transformers 为 5.14.0；以各自保存的 runtime 为准。

## Qwen-27B 的具体对照

保留 L48 原剂量、L49 原含义方向（factor=1）和 L55 的六个字母转换方向。层号均为原实验的 **0 起始 block index**。

每个场景与映射有 35 条条件记录，共 17,920 条：

- 当前无编辑基线、L48 修正、L49 修正、L55 单独修正，以及 L48 修正→L49 阻断→L55 修正。
- L48 和 L49 各四个实际同幅度随机修正。
- 三组关系各有指定阻断、四个实际同幅度随机移除，以及一个固定种子的随机坐标复位。
- L49/L55 各有单独复位及一个随机坐标单独复位，用于技术核查。

四个随机种子用于主随机对照；原随机坐标复位只保留一个固定种子，作为次要操作对照。实际幅度既在单次写入内部核对，也与对应目标条件真正执行时的位移交叉核对，容差沿用 2%。失败记录仍保留在主尝试表，并另列全部技术检查通过的家族。

主比较为：L48 修正相对基线、相对随机；L49 修正相对基线、相对随机；三条路径的“随机移除−指定阻断”。每个比较分别报告准确率与答案分数，共 14 项。只在相应起点修正成立后，判断该路径是否存在选择性依赖；三条路径独立判定，不互相替代。

## Gemma 的停止条件

每个场景与映射只有六个条件：无编辑、指定 L31 修正、四个实际同幅度随机方向，共 3,072 条记录。不运行剂量或层搜索。

主比较为指定方向相对基线、相对随机平均；各报告准确率和分数，共四项。只有两个准确率比较都通过、对照也有效，才标记可设计后续独立的 L31→L47 实验。当前 notebook 不自动启动下一轮。不通过则结束此候选；若只有分数作用，只报告分数作用。

## 读出诊断怎么运行

每个模型一个全新 GPU 运行时。默认 `MODEL_KEY='gemma12b'`；完成后如需 Qwen-9B，重新开运行时并改为 `qwen9b`。

使用原 E1 提示和读出器，每个旧开发场景的六种映射独立前向三次，共 342 次。保存状态为原生 BF16 数值的精确 float32 副本；缓存只用于恢复已完成的整组重复，不会把同一个缓存返回三次冒充重复。

报告单映射与六映射均值的读出波动、答案变化和分数变化。历史缓存比较单独报告，并保留原捕获配置；不能把旧新差异自动归因于存储压缩。它是数值诊断，不是新增科学测试集。

## 跑完后带回哪些结果

每份最终单元格会显示主要表格并输出目录。建议保存执行后的 notebook，并带回对应输出目录，至少包含：

- `analysis/REPORT_ZH.md`、`analysis/RUN_COMPLETE.json`。
- 科学确认的 `analysis/primary_tests.csv`、`repair_and_harm_counts.csv`、`technical_failures.csv`、`subgroup_descriptive.csv`、`intervention_records.json`。
- CPU 核查的 `orthogonal_failure_causes.csv`、`historical_group_interactions.csv`、`historical_bypass_vs_late_only.csv`、`coverage.csv`；这些位于 CPU 输出目录本身。
- 诊断的 `analysis/repeatability_summary.csv`、`within_runtime_pairs.csv`、`within_runtime_scene_means.csv`、`historical_scene_mean_comparison.csv`。

科学确认保存了当前无编辑候选分数和答案，不再用旧筛选标签代替当前基线。“修好/损坏数量”按场景与映射计数，随机条件先对种子平均；区间按家族计算。分数为正确候选相对最高错误候选的差，和旧阻断的两侧分数差分开标注。

预定主比较采用家族 bootstrap 95% 区间及配对符号翻转检验（差值符号可交换的零假设），在各自固定检验族内做 BH 校正。通过条件是区间下限大于 0 且 q<0.05，并满足相应技术对照。有效家族敏感性分析和子组描述不替代主总体判定。监督式真值干预不等于可以直接部署的推理算法。

## 本地验证范围

构建脚本保留在项目 notebooks 目录；验证覆盖新场景排除与恢复、三组关系的条件配对、14/4 项统计与判定、Gemma 训练缓存恢复、旧组间比较及重复诊断的数据流。合成检查仅验证代码，不是实验结果。

本地没有对应 GPU 权重和 Drive 完整原始缓存；真实模型前向、运行时间和新科学结果需要在 Colab 执行后确定。

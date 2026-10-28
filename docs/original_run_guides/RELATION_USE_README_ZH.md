# 下一轮：物理信息如何进入最终判断

本包准备两本已分别配置模型的 GPU notebook，以及一本 CPU 汇总 notebook。辅助代码全部嵌入；运行时不用上传 `.py` 文件。

## 运行顺序

1. `VisualTorque_Relation_Use_gemma12b_GPU.ipynb`：先跑 Gemma-12B，建议 A100 40 GB 或更大。
2. `VisualTorque_Relation_Use_qwen27b_GPU.ipynb`：在全新的 runtime 跑 Qwen-27B，建议 80/96 GB GPU。
3. `VisualTorque_Relation_Use_Results_Audit_CPU.ipynb`：上传两份结果 ZIP 后汇总；不需要 GPU。也可以先检查一份或部分结果。

两本 GPU notebook 都直接从头顺序运行，已分别设好 `MODEL_KEY`。默认 `RUN_ID='main'`，两模型保持一致。每本先运行开发，再运行独立验证；无需在两本之间传递新的校准文件。

这一轮不运行新的激活干预，不需要再跑之前的 readout、calibration 或 repair notebook。

## 需要准备什么

沿用 Drive 上包含 `toolkit/pyproject.toml` 的 VisualTorque 项目。自动检查：

- `/content/drive/MyDrive/VisualTorque`
- `/content/drive/MyDrive/VisualTorque/VisualTorque`

找不到时，在第一个配置单元填 `PROJECT_ROOT_OVERRIDE`。

历史场景排除从已有 Frozen_Visual_Validation 结果读取。以下任意一种即可：

- Drive 原目录 `VisualTorque/results/frozen_visual_validation_v1/main/` 中的 `historical_exclusion.json` 和 `shared/frozen_validation/support.json` 仍在；
- 任一模型的 `VisualTorque_Frozen_Visual_Validation_gemma12b_Output.zip` 或 `VisualTorque_Frozen_Visual_Validation_qwen27b_Output.zip`；放在 `/content`、Drive 原结果目录或项目根目录，或用 `SOURCE_VISUAL_ZIP` 指定准确路径。

两模型的旧场景相同，因此两本 notebook 可用同一份 ZIP。只读取命名的清单并核对 ZIP 校验值，不递归扫描 Drive，也不载入旧激活文件。

可以通过 `EXTRA_HISTORY_FILES` 提供其他已经运行过的场景 JSON/JSONL 清单，两模型须填写相同列表。新场景仅保证与实际读取的历史记录不重叠，不声称排除了未提供的历史场景。

Gemma 仍需要相应 Hugging Face 权限和 Colab 的 `HF_TOKEN`。沿用既有 checkpoint revision、Transformers 5.14.0、Pillow 11.3.0、单输入 BF16 和关闭内部 thinking 的聊天模板。显式计算条件要求模型在可见答案中给出短计算，未打开另一个隐藏推理模式。

## 实验内容

所有条件使用同一组场景和规则，分四级提供信息：

| 级别 | 提供的信息 | 主要剩余操作 |
|---|---|---|
| image | 带位置编号的图片，无重量、距离数值 | 获取变量、形成局部力矩、求和、比较 |
| facts | 图片＋每个位置的正确重量、距离 | 形成局部力矩、求和、比较 |
| products | 再提供每个重物正确的重量×距离 | 求和、比较 |
| totals | 再提供左右正确的总力矩 | 比较、表达答案 |

每一级包含三种循环平衡的 A/B/C 映射，以及不提供字母选项的语义回答。完整变量条件另外要求输出简短 JSON 计算：四个乘积、两侧总和、最终判断。

完整变量和总力矩各有一个纯文字语义回答对照。独立的组件问答分别检查图片中的重量—位置读取、文字条件的单项乘法、已知局部力矩后的求和与比较。这三项在独立上下文中运行，答案不传给主任务，也不把单独问答能力当作原判断内部过程的直接观测。

## 场景与规模

| 家族类型 | 开发 | 独立验证 |
|---|---:|---:|
| 交换重量位置后，正确方向翻转 | 8 | 32 |
| 交换重量位置后，正确方向保持 | 8 | 32 |
| 重量方向与正确倾斜方向冲突 | 4 | 16 |
| 左右等重，距离不同 | 2 | 8 |
| 真实平衡 | 2 | 8 |
| 合计 | 24 | 96 |

交换保留每侧的重量集合、距离集合和位置编号，仅改变重量放在哪个位置。每个场景都有左右镜像。开发与验证、不同家族之间排除相同几何及其左右镜像；开发和验证场景在模型运行前一起冻结。

每个场景执行 19 个判断回答，每家族另执行 3 个独立组件回答。每模型开发 1,592 次生成，验证 6,368 次，共 **7,960 次生成**。两模型合计 15,920 次；生成次数不是独立样本数。多数是短回答，不捕获全层激活，也不逐选项重复前向。不承诺未经 GPU 实测的运行小时数。

## 如何看结果

主要指标是交换前后两个场景都答对，在镜像和同一种作答方式的映射内配对，再按物理家族平均。翻转与保持两类题分别报告，同时报告二者合并的主要比较。

每模型有五项主要比较，共十项统一 Holm 校正：

1. 完整变量下，直接语义回答相对字母回答；
2. 语义回答下，完整变量相对图片；
3. 语义回答下，局部力矩相对完整变量；
4. 语义回答下，两侧总力矩相对局部力矩；
5. 完整变量下，显式计算相对直接语义回答。

开发结果仅用于描述和测量检查，不是验证证据。纯文字、分题型、组件检查、结构化计算和镜像结果用于解释主要结果；不把所有附加结果都作为另一个显著性搜索空间。模型尚未齐全时，主要比较按缺失模型的五个检验 p=1 处理，CPU notebook 收齐后重新计算同一十项范围。

所有回答都实际生成，保存原始字符串、token、结束标记、输入身份和提示。预先固定的解析器处理已知通道包装，未完成、无法识别或含糊的答案计为错误，并单独报告。不会从回答中搜索某个方向词，也不会用首位置候选分数替代答案。某个比较的成对有效回答率低于 95% 时标记 `measurement_needs_review`，不能直接把格式失败解释成物理失败。

某一级信息支持改善表现，说明这种中间信息有助于作答；它不是对某个神经模块的唯一定位。两模型可能呈现不同瓶颈。组件全对但主问题错误的统计要求主问题答案可被有效解析；无法识别的主问题回答另列。

优先查看：

- `validation/analysis/summary.csv`：各级信息支持的正确率、成对正确率和家族区间。
- `validation/analysis/primary_comparisons.csv`：五项主要比较及测量标记。
- `validation/analysis/generation_quality.csv`：有效回答率、结束率。
- `validation/analysis/component_integration.csv`：同一基础场景的组件检查与完整判断。
- `validation/analysis/structured_calculation.csv`：显式计算中的乘积、求和和答案是否一致正确。
- CPU 汇总的 `cross_model_primary.csv`、`relation_support_summary.png`、`audit_report.json`。

## 断点续跑与输出

完整结果写入 Drive：

`VisualTorque/results/relation_use_v1/main/`

每个回答先保存到 Colab 本地，每个场景完成后写入 Drive。普通中断可从头重跑并跳过已保存回答；runtime 被清除时最多重做当前场景的 19 个回答。换模型须换干净 runtime。模型、软件、实现或实验配置发生变化时不能把结果并入已有运行。

最后一个导出单元会保留 Drive ZIP，并复制到 `/content` 供下载：

- `VisualTorque_Relation_Use_gemma12b_Output.zip`
- `VisualTorque_Relation_Use_qwen27b_Output.zip`

中断后可运行导出单元保存部分结果；重新连接且没有 `RUN` 对象时，先执行到场景冻结单元即可。CPU notebook 会将未完成阶段明确标为部分结果，不会替它生成完整结论。

## 本地验证范围

交付前检查几何与镜像、支持级别、实际回答解析、家族配对统计、断点缓存、ZIP 校验、CPU 重算和 notebook 格式。解析器还用本项目保存的两模型历史生成字符串核验已知包装形式。

断点和汇总的端到端测试使用明确标记的模拟回答，不是研究结果。本地没有加载大模型或运行 GPU 推理。完整实验仍需在相应 GPU 上执行。

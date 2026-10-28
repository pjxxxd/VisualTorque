# Gemma 实际生成复核

## 运行

1. 将 `VisualTorque_Gemma_Generation_Recheck_GPU.ipynb` 上传 Colab，使用全新 A100 40GB/80GB 运行时。从头执行。无需上传辅助 Python 文件。
2. 挂载原来的 Drive 项目，保留 toolkit。需要可访问 `google/gemma-4-12B-it` 的 Hugging Face 账号；沿用原 `HF_TOKEN`。
3. 默认读取 `VisualTorque/results/frozen_visual_validation_v1/main/VisualTorque_Frozen_Visual_Validation_gemma12b_Output.zip`。也可将这一份 ZIP 上传 `/content`，或在配置格填写 `SOURCE_ZIP` 的完整路径。只需 Gemma ZIP，不需要 Qwen 或四份旧开发/诊断 ZIP。
4. 先执行 40 条配对检查。程序显示原始字符串、token、通道边界、输入一致性和旧分数复跑差异。通过后继续完成全部 1,440 条实际生成；已检查的 40 条会直接复用。
5. 最后返回 `VisualTorque_Gemma_Generation_Recheck_Output.zip`。默认保存在 Drive 的 `results/gemma_generation_recheck_v1/main/`，并尝试浏览器下载；设 `DOWNLOAD_OUTPUT=False` 可仅保存。

本 notebook 只运行 Gemma。它不准备或运行后续内部干预实验。

## 比较内容

沿用原验证的 96 个物理家族、240 个场景呈现和三种 ABC 映射；仅保留当前图片条件。原问法 720 条，中性问法 720 条。所有图片、提示和模型 revision 固定，完整 BF16、batch=1、greedy、最多 16 个新 token。原模型模板和 source ZIP 不修改。

只增加生成记录与审计信息，不强迫模型先输出某段前缀，不改变问题，不按物理正确率调整解析规则。首批 40 条额外做一次与旧代码相同的候选分数 forward，作为测量诊断；不把这些旧评分混入新的生成正确率。

## 解析与解释

每条保存完整原始 continuation、生成 token IDs、逐 token 字符串和分数、实际输入 token、提示末尾、旧 adapter 的解析结果，以及固定新规则的结果。

- **严格单字母**：去掉明确的回合结束标记后，只能是 A、B 或 C。
- **格式归一化后的单字母**：严格回复，或者固定、完整的通道包装形式后恰好一个字母。旧样本中的多余 `thought`/关闭通道包装会明确标记，不伪装成正常回复。
- **无效**：未结束回合、多个字母、额外答案文字、未知/不完整/嵌套通道结构。绝不搜索最后一个字母。

格式归一化后的指标是异常发现后明确规定的新测量定义，不证明包装输出来自正常的 final channel，也不证明模型没有内部推理。两种指标、包装比例和无效比例并列展示。所有计划输入保留在分母中；无效回复计为没有成功答对，而非被删除。

首批检查只判断输入一致和输出能否按固定规则无歧义归类，不要求物理题答对，也不要求与旧候选选择相同。未知格式会阻止昂贵的完整运行，并自动导出部分证据。后续遇到无效回复则保留，并将完成状态标为有待解释的输出。

## 统计与输出

以物理家族为单位配对，镜像、映射和交换伙伴不是独立样本。主要复核三个指标：总体正确率、真实平衡题正确率、本应倾斜题误答平衡的比例。家族 bootstrap 区间、配对符号翻转检验及三项 Holm 校正单独保存。

这是同一批旧家族上的事后测量复核，不能写成新的独立确认。旧实验 14 项主要与 8 项次要检验不覆盖、不删项重新校正，也不把两种计分方法混合。

主要导出：

- `shared/source_snapshot.json`：源 ZIP、旧运行和协议身份。
- `shared/inputs.json`、`shared/support.json`、`shared/images/`：本次全部固定输入。
- `run/runtime_identity.json`、`source_environment_comparison.json`：当前环境、模板、特殊 token 和生成配置。
- `run/families/*.json`：原始生成记录和断点。
- `run/analysis/preflight_comparison.csv`、`preflight_token_trace.csv`、`preflight_report.json`：40 条配对技术检查和逐 token 轨迹。
- `run/analysis/generation_trials.csv`、`family_metrics.csv`：逐次和家族指标。
- `run/analysis/accuracy_and_format_summary.csv`、`response_forms.csv`：准确率与格式分布。
- `run/analysis/paired_wording_comparisons.csv`、`repairs_and_harms.csv`：配对变化。
- `run/analysis/measurement_status.json`、`run/completion.json`：数据完整性及解释范围。
- ZIP 根目录 `export_manifest.json`：状态与文件哈希。

## 中断与环境

可从头续跑，已保存的生成不会重复计数。前 40 条逐条保存，完整运行按家族保存；可捕获的异常/中断会保存当前家族并导出部分 ZIP。强制重启或断电时最多重做未保存的当前家族。读图使用 `/content` 缓存，避免每次从 Drive 读取图片。

推荐沿用本次源验证的 A100 40GB。GPU 或 Torch 等变化会记录；模型、模板、toolkit 和预处理关键字段须与原档案一致。同一 RUN_ID 下运行身份改变时会停止，防止混合环境。如确需改变运行身份，应使用新的 RUN_ID 完整运行，保留旧结果。

`complete` 表示采集与分析已完成。是否存在包装或无效回复必须看 `measurement_status.json` 和严格/归一化指标，不能只看完成标记。

## 本地验证范围

交付前在 CPU 上检查源 ZIP 身份与全部 1,440 个输入、固定解析规则、断点恢复、统计分母、导出清单和 notebook 语法。合成答案只用于临时目录中的软件测试，不是研究结果。未在本地加载 Gemma 或执行真实 GPU 推理。

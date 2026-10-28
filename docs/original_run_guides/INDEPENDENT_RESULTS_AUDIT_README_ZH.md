# 下一步：只运行这一份 CPU 核查 notebook

文件：`VisualTorque_Independent_Results_Audit_CPU.ipynb`。

**选择普通 CPU 运行时，从上到下运行一次。无需 GPU，无需修改模型名。** 它读取上轮已经完成并保存到 Drive 的结果，不启动新的实验采集。

## 为什么是这一步

上一轮已确认 Qwen-27B 的 L48→L49、L49→L55；L48→L55 的统计差异通过，但随机对照技术检查未全部通过。具体失败条数、幅度和有效家族数保存在文件里，没有显示在上一份 notebook 中。

本次把这些细节直接读出，同时查看此前只展示了前五行的完整读出比较和物理条件分组。先把已有证据看全，再决定是否需要技术修正或重新采集。

## 直接怎么运行

1. 上传本 notebook 到 Colab，选择普通 CPU 运行时。
2. 从上到下运行，挂载原来的 Google Drive。
3. 默认三个运行名称均为 `main`，与这次结果一致，不需要修改。
4. 最后自动下载 `VisualTorque_Independent_Results_Audit_Output.zip`。
5. 保存执行后的 notebook，将它与下载的结果 ZIP 一起带回。

若浏览器没有触发自动下载，可以从最后显示的 Drive 路径或 Colab 左侧 Files 下载 ZIP；无需重新跑前面的单元格。要关闭自动下载，将配置单元格的 `AUTO_DOWNLOAD` 改为 False。

## 它检查哪些内容

- **输入与原判定一致性**：核对完成记录、文件哈希、64 家族的条件完整性和当前基线配对，重新计算原 14/4 项主结果。原确认结论保留。
- **Qwen 失败原因**：逐条区分单次写入内的实际范数不匹配、跨前向相对真实目标位移的不匹配、单独复位变化、已保存字段与原规则不一致。显示失败条件、随机种子、涉及家族、实际位移和误差；local/cross 两类计数可能重叠。
- **技术有效家族的敏感性**：分开呈现全体家族、该路径的全部技术对照有效家族及无效家族。后两者是事后描述，不能用来把原先的 False 改为 True。
- **物理条件与修好/损坏**：查看重量相同但距离不同、重量与距离冲突、多个物体求和、线索一致四类场景；保留左右、原正确/错误、渲染和映射分组的完整 CSV。子组区间按家族计算，不将镜像/映射当独立样本，也不基于子组重新选点。
- **完整读出差异**：读取 Gemma 和 Qwen-9B 各 19 场景、三次重复的完整历史比较，区分旧新差异、当前存储往返变化、读出类别变化与当前重复性。不会把 57 行当作 57 个独立场景，也不会仅凭配置差异认定原因。

## 使用的已有目录

自动寻找 `MyDrive/VisualTorque` 或 `MyDrive/VisualTorque/VisualTorque`，继续使用包含 `toolkit/pyproject.toml` 的原项目根。

```text
results/cross_model_independent_confirmation/
  qwen_three_path/main/Qwen__Qwen3.5-27B__373612cb/
  gemma_l31_candidate/main/google__gemma-4-12B-it/
  readout_repeatability/main/google__gemma-4-12B-it/
  readout_repeatability/main/Qwen__Qwen3.5-9B/
```

两份科学实验需要完整原输出目录，包括 `protocol.json`、`capture_identity.json`、`reference.json`、`frozen_directions.npz`、`cohort/support.json`、`cohort/images.json`、`analysis/intervention_records.json`、`analysis/primary_tests.csv` 和 `analysis/RUN_COMPLETE.json`。这些均由上一轮 notebook 保存。

读出诊断读取其原 `protocol.json`、`repeat_identity.json`、`analysis/` 下的完整 CSV 与完成记录，并核对原 E1 目录 `results/cross_model_readout_correction_v2/models/<模型>/` 下的冻结预测器、协议和运行身份。不会加载 Torch、模型权重或 discovery 的 `.pt` 状态文件。

如果某部分缺失或不一致，会列入覆盖表，其他可用部分继续完成。不要通过修改旧完成状态或删除原锁文件绕过问题。

## 输出和需要带回的文件

新输出在：

```text
results/cross_model_independent_confirmation/completion_audit/main/
```

最重要的是最终自动下载的 **`VisualTorque_Independent_Results_Audit_Output.zip`**。其中包含：

- `REPORT_ZH.md`、`AUDIT_COMPLETE.json` 和覆盖表；
- Qwen 失败条件汇总、逐条失败明细、原始 `write_audits` 和相关家族物理配置；
- 原完成状态、原主结果、原有效家族敏感性文件，以及逐项核对结果；
- 三路径原判定拆解、路径技术有效组的事后敏感性与家族贡献；
- 两模型完整物理分组表；
- 两份读出诊断的全部 57 行历史比较、19 场景汇总和可用捕获配置。

最后一个单元格直接打印完整的关键结论 JSON，包括 L48→L55 失败条数、原因类别及涉及家族数。较长的明细会展示部分行，并完整打包，不再只依赖 notebook 表格预览。

此核查目录每次运行刷新派生报告。结果包只包含本次登记的输出文件，不混入之前部分运行留下的文件；原科学实验目录不写入、不覆盖。

## 验证范围

本地检查覆盖实际幅度失败分类、原统计与判定复核、家族分组、三次重复按 19 场景汇总、缺失输入处理，以及完整报告打包流程。使用合成数据验证代码，不将这些检查当成研究结果。

完整 Drive 记录需要在运行后才能核对；本 notebook 没有任何 GPU 前向或自动重跑入口。

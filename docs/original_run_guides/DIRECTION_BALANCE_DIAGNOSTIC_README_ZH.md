# 左右方向与平衡选项诊断：运行说明

现在运行 `VisualTorque_Direction_Balance_Diagnostic_GPU.ipynb`。辅助代码已嵌入，只需上传这一个文件；仍需原项目的 toolkit。这里不运行上一轮的 validation，也不重跑全部开发实验。

## 运行顺序

1. 在 Colab 使用上次成功的 GPU 设置，保留默认 `MODEL_KEY='qwen27b'`、`RUN_ID='main'` 和 `SOURCE_RUN_ID='main'`，从头运行。
2. notebook 自动读取原 Drive 目录下两份信息支持实验的 development 输出 ZIP。如果移动过 ZIP，在 `SOURCE_ZIPS` 中填写对应文件的确切路径，无需递归寻找或解压整个 Drive。
3. Qwen 导出后，使用全新 GPU 运行时，将 `MODEL_KEY` 改为 `'gemma12b'`，其他设置保持不变，再从头运行。不要只在已加载 Qwen 的运行时改模型名称。
4. 返回两份 `VisualTorque_Direction_Balance_Diagnostic_*_Output.zip` 后，再决定是否进入独立验证。

默认原输入位置是：

    VisualTorque/results/visual_information_support_v1/main/
      VisualTorque_VisualInformationSupport_qwen27b_development_Output.zip
      VisualTorque_VisualInformationSupport_gemma12b_development_Output.zip

沿用原模型 checkpoint、Transformers 5.14.0、Pillow 11.3.0、BF16、单输入评分、关闭 thinking 和 16 token 上限的短答案生成。Qwen-27B 建议使用上次成功的 96 GB 级别 GPU，不用量化或 CPU 卸载替代；Gemma 保持相应模型访问权限和 HF_TOKEN。

## 本轮要区分什么

| 观察 | 控制方式 | 能帮助回答的问题 |
|---|---|---|
| Qwen 常回答左侧 | 明显左倾和右倾的简单场景，成对镜像、轮换答案字母 | 是否连简单方向判断也存在偏好 |
| 两模型真实平衡题全错 | 完全对称场景，以及重量和距离互相抵消的场景 | 简单平衡识别与需要计算的平衡是否不同 |
| equal 可能影响作答 | 保持问题不变，只改平衡选项表达 | 结果是否依赖该选项的措辞 |
| tip 的问法可能偏向倾斜 | 保持修改后的选项不变，只改问题为 What happens to the balance scale? | 提问方式是否影响平衡回答 |
| 当前提示与旧提示不同 | 同一物理场景使用旧图像提示、当前图像提示、图像完整变量和纯文字完整变量 | 作答异常是否依赖输入形式或提示 |
| 新运行可能与旧运行有差别 | 原样复跑四个旧平衡家族的一小部分输入 | 旧现象是否复现，输入和分数是否发生变化 |

旧提示与当前提示的比较同时改变编号、信息表和上下文，不能称为单独的编号消融。图像完整变量与纯文字完整变量使用相同文字，只改变图像是否输入。辅助数值不含力矩乘积、总和或正确答案。

## 规模和条件

8 个手工选定的简单家族，每类 2 个：

- 相同距离、重量不同：应该朝较重侧倾斜。
- 相同重量、距离不同：应该朝离支点更远的一侧倾斜。
- 左右完全对称：一个双堆场景和一个四堆场景。
- 重量与距离抵消：一个双堆场景和一个四堆场景，左右力矩相等。

每家族 2 个镜像呈现、3 种循环 ABC 映射、4 种输入形式、3 种表达，共 576 次评分。

| 表达 | 问题 | 平衡选项 |
|---|---|---|
| original | Which way will the balance scale tip? | equal |
| explicit_balance | 同上 | balanced (neither side tips) |
| neutral_question | What happens to the balance scale? | 同上一行 |

对四个旧平衡家族，每家族保留两个镜像和一组按身份固定选取的答案映射，复跑旧图像提示、图像完整变量、纯文字完整变量三种条件，额外 24 次评分。选择过程不按旧模型的分数或答案筛选。

合计每模型 **600 次评分＋120 次短答案生成检查**。生成检查覆盖每类简单控制预先固定的一个家族、两个镜像、四种输入、三种表达及一个映射，共 96 次，另加全部 24 次旧输入复跑。

这是刻意构造的诊断控制，不是新的泛化测试。原有简单几何可以重复；旧平衡题本来就是有意复跑。完全对称场景的镜像共享同一物理几何，镜像和映射均不增加独立样本数。

## 如何读结果

先看 `measurement_checks.json` 和 `archived_replay_comparison.csv`。如果同一旧题的处理后输入或候选 token 不一致，不能把分数差归因于模型行为；需要先解释环境、模板或输入变化。分数漂移会记录，不用任意阈值中断整份实验。旧结果从不覆盖。

随后看分组正确率、左/右/平衡回答比例，以及镜像两题同时正确的比例。只看总体正确率会掩盖固定回答某一方向的问题，因此这里不把全部简单题和旧题混成一个总分，也不输出小样本显著性检验。

- 若简单方向和对称平衡题正常，而原来四堆平衡题仍然失败，问题更可能与复杂任务要求有关，但仍不能仅凭此唯一定位内部机制。
- 若只改平衡选项或问题表达就出现大变化，提示依赖本身需要保留。不能自动选择分数最高的表达并把它包装成原协议结果。
- 若所有表达下简单题也异常，先检查输入解释、答案映射及模板设置，再决定验证协议。
- 简单控制通过不等于正式验证通过。确认协议后，仍需独立家族验证；若改变原实验协议，应使用新的 RUN_ID 和对应开发结果。

本轮不运行任何隐藏状态干预，也不据此宣称找到了物理信息的内部传递路径。

## 断点与输出

新目录：

    VisualTorque/results/direction_balance_diagnostic_v1/main/
      shared/diagnostic/
      models/qwen27b/diagnostic/
      models/gemma12b/diagnostic/

每六条评分保存一次，正常异常退出会保存已经完成的条目。从头运行可恢复；更换模型、实现、输入或运行环境时会核验配置，防止混合不同设置。最后一格可在配置和辅助函数加载完毕后单独导出，部分结果会标记为 `partial_not_complete`。

自动下载：

    VisualTorque_Direction_Balance_Diagnostic_qwen27b_Output.zip
    VisualTorque_Direction_Balance_Diagnostic_gemma12b_Output.zip

主要输出包括：

- `diagnostic_summary.csv`：按控制类型、输入条件、表达汇总，含正确率及方向/字母偏好。
- `family_metrics.csv`、`mirror_pairs.csv`、`mapping_invariance.csv`：家族结果、镜像配对与映射稳定性。
- `wording_pairs.csv`：只改变选项文字、只改变问题文字的逐输入比较。
- `generation_agreement.csv`：直接生成与候选评分的一致性，保留格式异常。
- `archived_replay_comparison.csv`：原输入、候选计划、分数、答案的一致性。
- `source_environment_comparison.json`：本次环境与原开发档案的差异。
- 原始候选分数、完整提示、场景图、输入张量指纹、来源快照及导出哈希。

第二个模型完成后的 ZIP 另含跨模型汇总表。两模型没有完成前，只提示等待另一模型，不影响当前模型导出。

## 本地检查范围

检查物理真值、镜像、选项单因素变化、来源 ZIP 校验、旧输入原样复用、数量、分组统计、续跑和部分导出，以及 notebook 单元语法。模拟回答仅用于检查程序，不作为实测结果。本机没有运行真实模型 GPU 前向。

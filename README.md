# VisualTorque：论文实验 notebooks（精简阅读版）

本目录在 2026-10-28 整理版上精简了 Markdown 说明，对照当前稿件 **Beyond Physical Readouts: Explaining Answer Changes in Vision–Language Models**（VisualTorque_revision）。共收录 **351 份不同内容的 notebooks、15 个实验分组**，比上一版新增 7 份 DeepSeek notebooks。原始文件和完整整理版均保留。计算代码、图表、保存输出不变；详细协议由 notebook 开头的链接进入。

修改范围见 [精简说明](docs/SIMPLIFICATION.md)。

先看 [论文结论与 notebook 对照](docs/CLAIM_TO_NOTEBOOK.md)，再按下面的目录打开实验。新增 DeepSeek 的完整结果与限定条件见 [DeepSeek 结果说明](evidence/deepseek/README.md)；更新记录见 [CHANGELOG](docs/CHANGELOG_20261028.md)。

## 按论文与实验用途查找

| 目录 | 实验内容 | notebook 数 |
|---|---|---:|
| [00_dataset_and_behavior](notebooks/00_dataset_and_behavior/README.md) | 任务构造与行为基线 | 11 |
| [01_section42_43_transfer_prediction](notebooks/01_section42_43_transfer_prediction/README.md) | 4.2–4.3 答案映射收益与前瞻迁移预测 | 7 |
| [02_section43_format_generalization](notebooks/02_section43_format_generalization/README.md) | 4.3 冻结预测器的新符号验证 | 3 |
| [03_section44_readout_correction](notebooks/03_section44_readout_correction/README.md) | 4.4 Qwen/Gemma 读出与数值修准 | 29 |
| [04_section45_guided_repair](notebooks/04_section45_guided_repair/README.md) | 4.5 Qwen/Gemma 指导修复与下游依赖 | 15 |
| [05_appendix_content_and_encoding](notebooks/05_appendix_content_and_encoding/README.md) | 附录 来源判断、语义、字母与选项位置的区分 | 39 |
| [06_appendix_physical_updates](notebooks/06_appendix_physical_updates/README.md) | 附录 答案不变的物理差分迁移与分量分析 | 10 |
| [07_appendix_lora_training](notebooks/07_appendix_lora_training/README.md) | 附录 LoRA 训练后的行为改善与干预变化 | 40 |
| [08_appendix_error_structure](notebooks/08_appendix_error_structure/README.md) | 附录 物理题型、重量捷径与错误结构 | 22 |
| [09_appendix_factor_routes](notebooks/09_appendix_factor_routes/README.md) | 附录 重量、距离与跨模型局部路径 | 55 |
| [10_exploratory_composition_and_gating](notebooks/10_exploratory_composition_and_gating/README.md) | 历史探索 跨背景组合、乘法规律与正确性门控 | 28 |
| [11_handoff_development_and_matching](notebooks/11_handoff_development_and_matching/README.md) | 历史开发 匹配队列、修复方向与序贯阻断 | 50 |
| [12_cross_study_audits_and_pilots](notebooks/12_cross_study_audits_and_pilots/README.md) | 历史汇总与其他机制试验 | 24 |
| [13_deepseek_extension](notebooks/13_deepseek_extension/README.md) | 新增 DeepSeek-VL2-small：4.2–4.5 扩展与恢复版 | 7 |
| [90_followup_visual_and_relation_use](notebooks/90_followup_visual_and_relation_use/README.md) | 后续探索 视觉信息、问法与物理关系使用 | 11 |

正文阅读顺序是 01（4.2–4.3 迁移与预测）→02（4.3 跨符号）→03（4.4 读出与修准）→04（4.5 指导修复）。13 单独收录新增 DeepSeek 扩展；当前本地论文正文尚未纳入这些结果。

## 每个目录里放什么

- code/：推荐源码入口，共 159 份；不表示已经在本机重跑。
- runs/：保留输出的运行记录，共 176 份；包括中途失败记录，需结合目录说明判断。
- versions/：不同内容的历史或替代版本，共 16 份；没有因不属于主结论而删去。

DeepSeek 的三份下载 notebook 保存了安装或容量报错，但最新结果压缩包确认各阶段已完成。完成证据保存在 evidence/deepseek/results/ 的 44 份精简结果文件中。新增结果中，读出修准修好 3 题、损坏 4 题，因此不能把原 Qwen/Gemma 的“零错误被修复”照搬为三模型共同结论。

## 运行、来源与核验

[运行指引](docs/RUNNING.md) 分开说明原模型与 DeepSeek 的环境、输入和运行顺序。support/deepseek_source/ 包含 DeepSeek notebooks 内嵌的完整实验源码；support/deepseek_recovery/ 保留 52 家族修订实现。toolkit/ 与 support/notebook_modules/ 保留共享实现。

[来源表](docs/source_to_copy.csv) 将 481 个原路径映射到 351 份副本；130 个字节完全一致的重复路径合并到同一副本，不同版本全部保留。[覆盖范围](docs/COVERAGE.md) 说明扫描位置和无法访问的目录，未声称覆盖整台电脑所有文件或压缩包。

可运行 python scripts/verify_package.py 复核文件完整性，核验结果见 [VALIDATION](docs/VALIDATION.md)。本次整理未重新执行 GPU 实验。仓库包含代码、保存输出与精简结果；模型权重、完整中间缓存和全部外部数据需另行取得，详见运行指引。

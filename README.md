# VisualTorque

**Beyond Physical Readouts: Explaining Answer Changes in Vision–Language Models**

基于可控杠杆场景，研究视觉语言模型中的物理信息读出、跨场景迁移与答案变化。仓库包含 Qwen、Gemma 和 DeepSeek 的实验代码、运行记录及结果分析。

[论文结论与实验对照](docs/CLAIM_TO_NOTEBOOK.md) · [运行说明](docs/RUNNING.md) · [实验设置](docs/EXPERIMENT_NOTES.md)

## 实验目录

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
| [13_deepseek_extension](notebooks/13_deepseek_extension/README.md) | DeepSeek-VL2-small：4.2–4.5 模型扩展 | 7 |
| [90_followup_visual_and_relation_use](notebooks/90_followup_visual_and_relation_use/README.md) | 后续探索 视觉信息、问法与物理关系使用 | 11 |


正文实验依次对应 01、02、03、04 目录。DeepSeek 扩展位于 13，结果与协议修订见 [DeepSeek 结果说明](evidence/deepseek/README.md)。

各实验目录包含：

- code/：运行入口。
- runs/：带输出的运行记录，包括标明状态的中断记录。
- versions/：历史版本和替代实现。

## 运行

模型环境、数据要求和执行顺序见 [运行说明](docs/RUNNING.md)。各 notebook 开头链接到相应实验设置。Qwen/Gemma 与 DeepSeek 使用不同的依赖环境。

toolkit/ 提供数据生成和共享模型接口；support/ 包含实验模块与 DeepSeek 恢复脚本。结果汇总位于 evidence/，运行中的输出写入 results/。

模型权重、完整数据和激活缓存需另行准备。保留的运行输出可用于查看结果，重新计算实验需要对应的输入文件。

## 文件索引

仓库包含 351 份 notebooks，分为 15 组。文件来源、版本关系及校验方式见 [Notebook 文件说明](docs/NOTEBOOKS.md)。

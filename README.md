# VisualTorque

论文 **Beyond Physical Readouts: Explaining Answer Changes in Vision–Language Models** 的实验代码。

我们用可控的杠杆场景研究视觉语言模型中的物理信息读出、状态迁移与答案变化。

[论文结论与实验对照](docs/CLAIM_TO_NOTEBOOK.md) · [运行说明](docs/RUNNING.md) · [实验设置](docs/EXPERIMENT_NOTES.md)

## 实验

| 实验 | 内容 |
|---|---|
| [数据与行为基线](notebooks/00_dataset_and_behavior/README.md) | 杠杆场景生成与模型作答基线 |
| [迁移预测](notebooks/01_section42_43_transfer_prediction/README.md) | 答案映射对迁移收益的影响，以及跨场景预测（§4.2–4.3） |
| [跨符号验证](notebooks/02_section43_format_generalization/README.md) | 固定预测器，在新的答案符号下验证（§4.3） |
| [物理量读出与修准](notebooks/03_section44_readout_correction/README.md) | 隐藏状态中的物理信息，以及修准读数后的答案变化（§4.4） |
| [指导修复](notebooks/04_section45_guided_repair/README.md) | 固定编辑的修复效果与下游依赖（§4.5） |
| [内容与编码](notebooks/05_appendix_content_and_encoding/README.md) | 来源语义、答案字母与选项位置的区分 |
| [物理量更新](notebooks/06_appendix_physical_updates/README.md) | 答案不变时的物理差分迁移与分量分析 |
| [LoRA 训练](notebooks/07_appendix_lora_training/README.md) | 训练后的行为改善与干预变化 |
| [错误结构](notebooks/08_appendix_error_structure/README.md) | 物理题型、重量捷径与错误分布 |
| [因素与路径](notebooks/09_appendix_factor_routes/README.md) | 重量、距离及其跨模型局部路径 |
| [组合与门控](notebooks/10_exploratory_composition_and_gating/README.md) | 跨背景组合、乘法规律与正确性门控的早期探索 |
| [修复方案开发](notebooks/11_handoff_development_and_matching/README.md) | 匹配队列、修复方向与序贯阻断 |
| [跨实验分析](notebooks/12_cross_study_audits_and_pilots/README.md) | 结果核查与机制试验 |
| [DeepSeek-VL2-small](notebooks/13_deepseek_extension/README.md) | 迁移、读出与修复实验的模型扩展 |
| [视觉信息与关系使用](notebooks/90_followup_visual_and_relation_use/README.md) | 视觉信息、问法与物理关系的后续实验 |

各目录的 `code/` 提供运行入口，`runs/` 保存运行输出，`versions/` 保留旧版本。论文结论对应的 notebook 及结果位置见[实验对照](docs/CLAIM_TO_NOTEBOOK.md)；DeepSeek 的结果和实验设置调整见[结果说明](evidence/deepseek/README.md)。

## 运行

按[运行说明](docs/RUNNING.md)准备模型环境和输入数据，再执行对应的 notebook。具体设置见各 notebook 开头的链接。Qwen/Gemma 与 DeepSeek 使用不同的依赖环境。

`toolkit/` 包含场景生成和模型接口，`support/` 放共用模块与 DeepSeek 恢复脚本，`evidence/` 保存结果汇总。

模型权重、完整数据和激活缓存需另行准备。查看已有结果可直接打开 `runs/` 中的 notebook；重新运行则需要相应的输入文件。

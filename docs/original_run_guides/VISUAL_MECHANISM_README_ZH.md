# 下一轮：视觉信息与冻结干预解释

本包提供两本 GPU notebook 和一本 CPU 汇总 notebook。辅助代码已嵌入，运行时无需上传 Python 文件。

## 推荐顺序

1. `VisualTorque_VisualMechanism_Bridge_GPU.ipynb`：先 Gemma-12B，再用全新 GPU runtime 运行 Qwen-27B。
2. `VisualTorque_Frozen_VisualMechanism_Transfer_GPU.ipynb`：先 Gemma-12B，再用全新 GPU runtime 运行 Qwen-27B。
3. `VisualTorque_VisualMechanism_Results_Audit_CPU.ipynb`：收齐输出后重算、核验和合并。没有 GPU 要求，也可先审计部分已完成结果。

第一本比较图片和图片＋完整重量/距离，运行内容鉴别和物理重组两个模块。第二本比较标准图、空心块、平移缩放；包含内容鉴别、行为、重量位置交换和独立变量读取。保持已有层、原始 BF16 差分、单位剂量和冻结预测器。

只运行 `gemma12b`、`qwen27b`。不需要 qwen9b，不需要重跑 Calibration / Validation。两本 notebook 和两个模型使用相同 `RUN_ID='main'`；切换模型必须换干净 runtime。

## 输入文件

第一本每次只需该模型的：

- `VisualTorque_Intervention_Prediction_gemma12b_validation_Output.zip`，或
- `VisualTorque_Intervention_Prediction_qwen27b_validation_Output.zip`。

第二本还需要该模型的：

- `VisualTorque_Frozen_Visual_Validation_gemma12b_Output.zip`，或
- `VisualTorque_Frozen_Visual_Validation_qwen27b_Output.zip`。

默认依次检查 `/content`、Drive 项目原结果目录和项目根目录。已有 ZIP 仍在原 Drive 目录时不用重新上传。若找不到，配置单元填 `SOURCE_PREDICTION_ZIP` 和第二本的 `SOURCE_VISUAL_ZIP` 的准确路径。第一本的 `SOURCE_VISUAL_ZIP` 留空。

需要 Drive 已有 `VisualTorque/toolkit/pyproject.toml` 与先前使用的 toolkit。不会扫描整个 Drive，也不会解压所有历史结果。源 ZIP 只读，已消费成员按导出清单核验。

## 显存、环境和续跑

- Gemma-12B：建议 A100 40 GB 或更大显存，需相应模型访问权限。
- Qwen-27B：建议 80/96 GB GPU，完整 BF16；不能改量化、CPU/disk offload。普通 A100 40 GB 不够容纳完整 BF16 权重及运行开销。
- 沿用固定模型 revision，安装 Transformers 5.14.0、Pillow 11.3.0；记录处理器、模板、GPU 与 toolkit 身份。
- 从头顺序运行。图片在 Colab 本地生成；干预逐次本地保存，每个家族/条件完成后写入 Drive，减少大量小文件写入。中断后同配置重跑会校验并跳过已完成部分；如果整个 runtime 被删除，最多重做尚未写入 Drive 的当前家族。
- 若模型、实现、GPU 环境或输入身份改变，请保留已有结果，用新 `RUN_ID`；不要删除身份检查来混合结果。
- 最后一个导出单元可单独运行，打包当前完成部分。重新连接且尚未创建 `RUN` 时，先执行到源结果读取/协议冻结单元即可导出，不必先加载模型。

## 规模

第一本每模型 192 个家族、2 个输入条件：6,912 次目标干预和 6,912 次随机对照生成，另加来源、未干预接收者与自然反事实生成。

第二本每模型 96 个内容家族、3 种画法：3,456 次目标干预和 3,456 次随机对照生成；另有 96 个已有视觉家族的 2,160 次行为回答、720 次变量读取及来源/接收者生成。

第二本会重新运行当前协议下的标准图作为同运行时参照，未把旧分数冒充新基线。所有候选分数和实际生成在一次带缓存的生成中取得，不为每个选项单独前向。GPU 耗时取决于硬件，运行前表格会显示目标和对照数，不承诺未经实测的小时数。

## 输出

Drive：`VisualTorque/results/visual_mechanism_frozen_v1/main/`。

每个模型、阶段独立保存；最后生成以下四个 ZIP：

- `VisualTorque_VisualMechanism_gemma12b_bridge_Output.zip`
- `VisualTorque_VisualMechanism_qwen27b_bridge_Output.zip`
- `VisualTorque_VisualMechanism_gemma12b_visual_Output.zip`
- `VisualTorque_VisualMechanism_qwen27b_visual_Output.zip`

ZIP 包括输入身份、图片、原冻结预测器、未干预状态、预测封存、实际写入、生成 token、逐次结果和统计表。CPU notebook 可以只上传这些 ZIP；输出 `VisualTorque_VisualMechanism_Audit_Output.zip`。

先看 `analysis/key_results_summary.csv`、`condition_changes_descriptive.csv` 和 `measurement_status.json`。第二本另看 `panel_summary.csv`、`binding_pairs.csv`、`reading_summary.csv`。`primary_comparisons.csv` 是预先固定的主要比较；两模型齐全后的跨模型结果在 `analysis/bridge_cross_model_primary.csv` 或 `visual_cross_model_primary.csv`。

## 结果含义

1. 所有预测先于目标干预封存；不重新拟合、调剂量、找更好层或筛掉失败家族。
2. 这轮使用已观察过的家族，新增的是支持条件和画法，不是新的独立几何。
3. 统一提示增加了位置表格框架，所以标准图片条件也需重新测量。与旧 43.1% / 62.8% 等数值不能直接混算。
4. “预测干预后答案”和“把物理题答对”分开统计。主要比较为分数变化预测误差；实际生成、修复和损害分别报告。
5. 干预只写入生成预填充时的最后一个 prompt token，不反复干预后续生成 token。原始第一步 logits 与实际生成步骤都保存。
6. 完整字符串解析保留通道包装、格式失败和截断。若第一步不是答案位置，`measurement_needs_review` 会标记；不能把那个位置的候选评分直接解释为实际行为。
7. 读取重量/距离是独立问答，不是直接观测原判断内部的感知状态。
8. 统计单位是物理家族。A 阶段计划 40 项跨模型主要比较，B 阶段计划 24 项，分开 Holm 校正。尚缺另一模型时用 p=1 补齐计划范围，并明确标记 pending。

生成同时保存未经生成处理器修改的 logits 和实际选择 token 的 scores；两者含义参见 [Transformers 官方生成说明](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)。预测仍以原始候选 log probability 为目标，不把经过生成处理器修改的分数替换进去。

## 本地验证范围

交付前使用真实来源 ZIP 检查输入读取与冻结预测数值，检查渲染、响应解析、干预位置、断点缓存和统计重算。CPU 测试与模拟接口不是 GPU 研究结果；本地不加载或运行两个大模型。首次 GPU 运行会完成预填充生成与零写入一致性检查。

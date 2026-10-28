# 方向交叉 × 剂量匹配 × 新几何确认

实验设置与运行说明。单元格编号对应来源 notebook，从 0 开始。

对应 notebook：

- [VisualTorque_Qwen27B_Handoff_AxisDose_Crossover_NewGeometry_96GB_v1.ipynb](../../../notebooks/07_appendix_lora_training/runs/VisualTorque_Qwen27B_Handoff_AxisDose_Crossover_NewGeometry_96GB_v1.ipynb)
- [VisualTorque_Qwen27B_Handoff_AxisDose_Crossover_NewGeometry_96GB_v1.ipynb](../../../notebooks/07_appendix_lora_training/code/VisualTorque_Qwen27B_Handoff_AxisDose_Crossover_NewGeometry_96GB_v1.ipynb)

<a id="cell-0"></a>

## 原单元格 0

# VisualTorque — 方向交叉 × 剂量匹配 × 新几何确认

这一轮针对已经观察到的现象：LoRA 中固定 Native L48/L49 干预的输出效应减弱，L55 直接干预仍然有效。区分 **剂量变化、旧方向适配变化、后续传递变化**。

## 设计

- **L48 保留已验证的 Native 物理轴**；本轮不拟合 LoRA L48 轴，不声称检验整条 LoRA 通路的重定位。
- L49 语义轴和 L55 六种答案码转换轴，在原来同一套 32 个开发支持家族上，用完全相同的算法分别得到 Native／LoRA 版本。Native 重建须通过历史分数、逐家族范数和保存轴方向的校准。实际 Native 干预直接读取已完成 v3 的轴；LoRA 轴统一到对应 Native 范数，避免把拟合范数当作坐标差异。
- 接收模型（Native／LoRA）、**注入轴来源**、**阻断轴来源**分开交叉。L48→L49、L48→L55 使用固定 L48 输入；L49→L55 的两套注入轴都测试。相同随机阻断轴跨模型、轴来源、剂量共用；每个输入保留自钳制对照。
- 加性剂量为 −1、−0.5、+0.5、+1 倍。负剂量检验反方向响应；主要阻断实验固定 +1 倍。零剂量是同一模型自己的 clean 基线。
- **绝对范数**使用 Native 历史幅度；**相对范数匹配**为 `D × ||h_clean,recipient|| / ||h_clean,Native||`，按同一图片／映射／层配对。规则在确认推理前冻结，实际剂量只由 clean 状态范数决定，不由答案或干预效果决定。两种方案的 Native 注入相同。分别保存目标范数、BF16 payload 范数、加法舍入后的实际状态位移，不能把 clamp 轴范数当成被移除剂量。

## 新数据与结论范围

原来的 900 个 single-stack 场景已用于 Native 分析，所以本轮排除它们的**全部几何及镜像**，同时排除 SFT train/dev、所有方向拟合支持、旧 paired cohort。默认选择 **32 个新物理家族**：最大单侧砝码数为 6 或 7 各 16 个，距离仍为 1–6 格，两个镜像、两个答案映射、一个预定绘图样式。依据物理线索冲突类型和力矩差分层选取，绝不筛选模型成功或失败例。

这是**扩展重量范围上的确认**，不是原分布总体准确率。6 块与 7 块分别报告；配色、层、方向算法、剂量和样本数不依据确认结果调整。旧场景的 Native 重放仅是数值/实现校验，不计入新确认统计。32 个物理家族是独立统计单位，256 个 clean 输入并不等于 256 个独立样本。

主要检验共六项（统一 Holm 校正）：相对范数匹配、+1 倍、Native 注入轴下，三个 route 的 LoRA−Native 特异阻断损失；以及三个 route 的“接收模型 × 阻断轴来源”交互。输入轴交叉、完整剂量曲线、signed downstream projection、6/7 块与映射分层为预定次级分析，区间为逐项 95%，不能据此事后另立主要终点。

## 运行

Colab **96 GB GPU**，建议重启会话后 Run all。复用现有 BF16 base、LoRA adapter、v3 原始结果、方向支持图片和 Dataset v1，不重新训练。本轮输出独立目录 `results/qwen27b_handoff_axis_dose_crossover_newgeometry_v1/`，不会改动旧结果。

默认 **256 次 clean + 17,920 次干预评分**，另有开发方向捕获和技术校验，约为上轮干预数量的三倍。逐输入并在输入内部逐条件保存，断线后 Run all 校验续跑；第一组实测耗时后显示预计剩余时间。不要在运行中修改配置或复用不同协议的缓存。Native 两种尺度重复计算，保留这一重复作为实现一致性检查。

如必要输入缺失，前置单元会列出路径；如果环境校准不通过，保存诊断后停止，不放宽精度门槛。固定 `highest / TF32=False`，继续用 transformers 5.14.0、peft 0.20.0。

**离线验证不等于 GPU 实验完成。** 此文件交付时未运行 27B 推理。最终结果无论显著与否都会导出；不会以“某模型显著、另一模型不显著”代替交互检验，也不把干预响应解释成自然行为改善的因果中介。

# 视觉信息支持独立验证：运行说明

运行 `VisualTorque_Frozen_Visual_Validation_GPU.ipynb`。辅助代码已经内嵌，只需上传该 notebook；继续使用 Drive 原项目的 toolkit。

## 现在怎么运行

1. 默认 `MODEL_KEY='gemma12b'`、`RUN_ID='main'`，从头运行。
2. 完成并导出后，换全新 GPU 运行时，将 `MODEL_KEY` 改为 `'qwen27b'`，其他设置保持不变，再从头运行。
3. 两模型各返回一份 `VisualTorque_Frozen_Visual_Validation_*_Output.zip`。第二个模型的 ZIP 另有跨模型联合比较。

这是新的独立验证 notebook，不用切换旧 notebook 的 PHASE，也不用重跑开发或诊断。数量、种子、输入条件与问法已经固定；不根据中途结果增减样本、调整提示或选择模型。

沿用上次成功的 GPU 环境、固定模型 revision、完整 BF16、batch=1、关闭 thinking、Transformers 5.14.0、Pillow 11.3.0。Qwen-27B 使用可容纳全量权重的 GPU；上次成功的是 96 GB 级别。Gemma 保留相应模型权限和 HF_TOKEN。

## 四份输入 ZIP

默认自动读取：

    VisualTorque/results/visual_information_support_v1/main/
      VisualTorque_VisualInformationSupport_gemma12b_development_Output.zip
      VisualTorque_VisualInformationSupport_qwen27b_development_Output.zip
    VisualTorque/results/direction_balance_diagnostic_v1/main/
      VisualTorque_Direction_Balance_Diagnostic_gemma12b_Output.zip
      VisualTorque_Direction_Balance_Diagnostic_qwen27b_Output.zip

若文件已移动，在配置格的 `SOURCE_ZIPS` 中填写各 ZIP 的确切路径。如果原运行目录名不是 main，对应修改 `SOURCE_DEVELOPMENT_RUN_ID` 或 `SOURCE_DIAGNOSTIC_RUN_ID`。

四份文件只读。首次核验各文件哈希、原实验版本及来源关系，冻结来源快照；续跑只重新核对这些文件的整包哈希，不递归扫描 Drive。保留原 ZIP 文件不变即可。

## 固定规模与场景隔离

| 物理家族类型 | 数量 |
|---|---:|
| 重量、距离与真实方向一致 | 12 |
| 左右力矩相等，真正平衡 | 12 |
| 左右重量集合相同，距离不同 | 24 |
| 重量规则与真实倾斜方向相反 | 24 |
| 交换重量—距离对应关系，使正确方向反转 | 24 |
| 总计 | 96 |

普通家族各两张镜像，交换家族含交换前后及其镜像，共 240 个场景呈现。每张运行六条件、三种循环 ABC 映射和两种问法：

**每模型 8,640 次评分＋240 次短答案生成检查。**

生成检查在每类预先按家族身份选两个家族，取 base 及其镜像、一个固定映射、六条件和两问法。它覆盖抽查输入，不代表对所有自由生成或推理模式的验证。

排除范围包括开发 ZIP 已记录的历史几何、开发场景、诊断场景及它们的镜像。如果原信息支持 notebook 已生成 `shared/validation/support.json`，会自动加入排除清单。其他已用几何可在首次运行前通过 `EXTRA_EXCLUSION_FILES` 指定 JSON/JSONL 清单。未提供的 ZIP 或未记录的数据不在排重证明范围内。

一旦开始验证，不再改变排除清单；否则会破坏两模型共享的冻结输入。统计单位为 96 个物理家族，不能把镜像、映射或两种问法当作独立样本。96 是预算下固定的规模，不构成统计功效保证。

## 六条件与两种问法

六条件沿用开发实验：带编号图像、图像＋重量、图像＋距离、图像＋完整变量、纯文字完整变量、旧图像提示参照。

所有条件都提供相同通用力矩规则，场景变量只按条件公开，不提供力矩乘积、左右总力矩、净力矩或答案。四种带编号图像条件使用相同图像；完整图文与完整文字的文字部分相同，只改变图像输入。

| 问法 | 问题 | 平衡选项 |
|---|---|---|
| original | Which way will the balance scale tip? | equal |
| neutral_question | What happens to the balance scale? | balanced (neither side tips) |

中性版本作为预设的组合变化，与原文并行比较。这里不能分别估计单个词和问题句的因果作用，也不会因为中性问法某项得分高就选它替代原文。旧图像提示参照与当前提示的差别同样包含多个因素，不是纯编号消融。

## 固定统计方案

原问法保留每模型七项主要比较：

1. 重量支持相对图像基线的总体准确率变化。
2. 距离支持相对图像基线的总体变化。
3. 完整变量支持相对图像基线的总体变化。
4. 完整信息效果减去两个单独信息效果的四条件交互。
5. 等重距离题中，距离支持相对图像基线的变化。
6. 重量冲突题中，完整变量支持相对图像基线的变化。
7. 交换题中，完整变量支持对“双题全部正确”的影响。

以物理家族为单位计算配对 bootstrap 95% 区间及双侧符号翻转检验；非零家族差值不超过 16 个时使用精确枚举，否则 10,000 次随机翻转。两模型共 14 项主要比较统一做 Holm 校正。

另有每模型四项预设的次要问法比较，均为中性版本减去原文：

- 图像基线下的总体准确率。
- 图像基线下，真正平衡题的准确率。
- 图像基线下，本应倾斜的题目误答平衡的比例；该项越低越好。
- 完整图文条件下，交换题两张全部正确的比例。

两模型共 8 项作为独立的次要检验族做 Holm 校正。主要族与次要族分别控制，不宣称合并 22 项的全局错误率保证。中性问法下七项信息支持效果另报配对区间，作描述性分析，不额外输出其显著性检验。

一个模型完成时只显示模型内校正，跨模型结论以两个模型完成后的联合表为准。未显著不代表没有效应；样本全对或全错时 bootstrap 区间可能退化，不能据此宣称总体能力界限。

## 除准确率外保留什么

- 物理方向偏好与 A/B/C 字母偏好。
- 真平衡题回答平衡的比例，以及倾斜题误答平衡的比例。
- 镜像两题同时正确、语义是否随镜像翻转。
- 交换前后两题全部正确；只改答案不算通过。
- 同场景三个答案映射下，物理含义是否一致。
- 提供信息以及改变问法分别纠正了多少错误、又新增多少错误；错误与正确的原始分母均保留。
- 正常短答案生成与候选评分的一致性，包括无效输出。
- 原始分数、实际提示、图像、输入 token、预处理张量指纹、模型与环境身份。

总体正确率按家族等权计算，交换家族不会因为多一对场景获得双倍权重。逐次纠错／损害计数另列，不与家族平均百分比混用。固定回答左侧、右侧或平衡的基线也会导出，帮助判断平均分是否掩盖方向偏好。

## 续跑与结果位置

输出独立保存在：

    VisualTorque/results/frozen_visual_validation_v1/main/
      shared/frozen_validation/
      models/gemma12b/frozen_validation/
      models/qwen27b/frozen_validation/

每六条评分保存一次，异常退出时保存已完成条目。从头运行会核验后跳过已完成输入。运行时被强制销毁可能丢失最近尚未同步的几条，需要重算；不需要重跑整个实验。

最后一格可以在前面的配置和辅助函数加载后导出部分结果；未完成的包明确标记为 `partial_not_complete`，不会生成完整验证结论。请保留相同运行环境以续跑；实现、输入或环境身份不一致时不混入旧记录。

完整包文件名：

    VisualTorque_Frozen_Visual_Validation_gemma12b_Output.zip
    VisualTorque_Frozen_Visual_Validation_qwen27b_Output.zip

主要统计文件：`accuracy_and_answer_bias.csv`、`primary_original_comparisons.csv`、`secondary_wording_comparisons.csv`、`neutral_information_effects_descriptive.csv`、`true_and_false_balance.csv`、`binding_pair_trials.csv`、`mirror_pairs.csv`、`repairs_and_harms.csv`。第二个模型完成后还有 `cross_model_primary_original.csv` 和 `cross_model_secondary_wording.csv`。

这轮不运行训练、隐藏状态干预或新的解释搜索。首先完成两模型的这套固定验证，再根据全部预设比较更新文章。

## 本地验证范围

使用真实开发和诊断档案检查来源关系、几何排重、冻结重用、原问法一致性、两问法图像一致性、数值公开范围、统计分母和配对指标、断点恢复、部分／完整导出、跨模型校正及 notebook 代码语法。合成回答只用于程序检查，不作为实验结果。本地未加载模型或执行真实 GPU 前向。

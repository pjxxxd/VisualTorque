# VisualTorque Toolkit

这部分代码负责生成和审计 VisualTorque Dataset v1。它把场景图像、精确物理真值、问题、反事实关系和实验划分放在同一套可验证的数据契约中。

## 已实现

- 自动根据场景几何判定七类单栈题型或多栈题型。
- 穷举有界单栈空间，并确定性抽样镜像闭合的多栈空间。
- 生成和验证加减砝码、移动、镜像、等力矩重分解反事实。
- 为结果和中间物理量生成问题；结果题覆盖全部六种选项排列。
- 将 probe train/dev/test 和 intervention dev/test 分开。
- 保留 unit-distance 组合与部分多栈结构作为 OOD probe test。
- 对 JSONL、物理真值、答案、划分、反事实和 PNG 文件进行落盘后复验。

## 使用

在本目录安装后：

```text
pip install -e .
visual-torque generate --output data/dataset_v1
visual-torque validate data/dataset_v1
```

快速构建较小数据集：

```text
visual-torque generate --output data/smoke_v1 --max-units 2 --max-distance 3 --multi-max-units 2 --multi-scenes 20
```

输出目录包含：

```text
dataset_config.json
scenes.jsonl
counterfactual_pairs.jsonl
questions.jsonl
splits.json
statistics.json
images/*.png
```

构建器拒绝覆盖非空目录，避免混入旧版本记录。

## 行为基线

行为实验代码位于 `visual_torque_toolkit.behavior`，支持：

- Qwen3.5-9B 和 Gemma 4 12B 的统一 Hugging Face 接口；
- 显式规则图片条件、显式规则符号条件和无规则图片条件；
- A/B/C 合法答案的批量条件似然评分；
- 少量 greedy generation 格式审计；
- 中断后从 `predictions.jsonl` 自动续跑；
- 分题型、选项排列、图像/符号、反事实和启发式分析。
- 跨模型的六排列语义聚合、场景级 bootstrap、感知/计算诊断和 activation pilot 清单。

GPU 环境安装：

```text
pip install -e ".[behavior]"
```

先运行一个小规模审计：

```text
visual-torque-behavior run \
  --dataset data/dataset_v1 \
  --output results/qwen35/smoke \
  --model Qwen/Qwen3.5-9B \
  --conditions image_rule symbolic_rule \
  --question-kinds outcome \
  --scenes-per-stratum 1 \
  --generation-every 1
```

完整主实验应使用单独输出目录，设 `--generation-every 0`，并移除场景数量限制。Colab 的推荐执行顺序和 A100 参数见项目根目录的 `notebooks/VisualTorque_Behavior_A100.ipynb`。

两个模型完成后，使用 `notebooks/VisualTorque_Behavior_Analysis_v2.ipynb` 生成六排列聚合的场景级比较、paired bootstrap、感知/计算诊断和 activation discovery manifest。机制实验的假设、数据隔离和可支持的论断层级见项目根目录的 `ANALYSIS_PROTOCOL.md`。

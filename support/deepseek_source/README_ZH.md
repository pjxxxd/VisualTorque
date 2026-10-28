# DeepSeek-VL2-Small：Section 4.2–4.5 实验包

模型为 `deepseek-ai/deepseek-vl2-small`，约 16B 总参数，27 个语言解码层。所有层号从 0 开始。此包准备第三个模型家族的实验；不含已经测得的 DeepSeek 科学结果。

## 直接运行

1. 解压总 ZIP，将 `01_DeepSeek_Transfer_4_2_4_3.ipynb` 上传 Colab，选择 GPU。新版使用支持 Blackwell 的 CUDA 12.8 运行库；建议约 80GB 或更多显存，全 BF16。安装后自动检查实际 GPU 运算。未实现量化或 CPU 卸载，也没有在本地实测完整 16B GPU 前向或峰值显存。
2. 修改首个代码单元的 `PROJECT_ROOT`，默认 `/content/drive/MyDrive/VisualTorque`。保留新版 `RUN_ID='blackwell_cu128'`，依次运行所有单元。首次自动下载并建立独立的 Python 3.11.11 环境，即使 Colab 的 Python 是 3.13 也无需手动切换。模型在后续阶段下载；原 Qwen/Gemma 环境不受影响。
3. 用同一个项目目录和 RUN_ID 运行 `02_DeepSeek_Readout_4_4.ipynb`，再运行 `03_DeepSeek_Repair_4_5.ipynb`。第三份使用第一份的 **32 个 content 校准家族**构建答案方向，因此至少先完成第一份的校准。
4. 每份最后一格可以导出结果。即使某阶段失败，也可单独运行最后一格，导出已有记录和错误信息。请同时保存运行后的 notebook。

每个 notebook 内嵌全部实验代码和 toolkit，不需要另外上传 Python 模块。`experiment_source.zip` 是同一份代码的普通源码包，可在 Linux GPU 服务器运行。

### 旧版安装报错的修复

如果旧版在安装单元报 `CalledProcessError`，且 traceback 中出现 `/usr/lib/python3.13/`，这是 Colab 使用 Python 3.13，而旧安装器只接受 Python 3.10–3.12 导致的。此时实验尚未开始。

请上传新版 notebook，从第一个代码单元重新运行，使用原来的 `PROJECT_ROOT` 和新版默认的 `RUN_ID='blackwell_cu128'`。新版通过固定版本的 uv 自动准备 Python 3.11.11，环境目录为 `/content/visualtorque_deepseek_py311_cu128`，无需修改 Colab 内核或删除旧环境。三份 notebook 均已更新，后续也请使用新版。

安装和实验的详细输出会实时显示在 notebook，同时追加保存到 Drive 的 `results/deepseek_vl2_small/<RUN_ID>/logs/`。安装日志是 `install.log`，其他阶段为 `<stage>.log`。若仍遇到问题，保存运行后的 notebook 或这个日志即可保留具体原因。只有安装成功后才可执行实验和结果导出单元。

### RTX PRO 6000 Blackwell 的 CUDA 报错修复

第二份报错记录中的显卡是 RTX PRO 6000 Blackwell Server Edition（sm_120，约 96GB），旧 PyTorch 2.6.0/cu124 不包含适用的 CUDA kernel。`cuda.is_available()` 能识别显卡，但不能证明实际运算可用，因此旧检查通过后，第一次 `torch.zeros_like` 仍报 `no kernel image is available for execution on the device`。

新版固定为 PyTorch 2.9.1、torchvision 0.24.1、xformers 0.0.33.post2、CUDA 12.8；保持 Transformers 4.38.2。每次安装或复用环境时，都先实际执行布尔 mask、BF16 矩阵乘法和官方图像注意力（729 token、16 heads、每头 72 维），通过后才允许加载完整模型。检查不下载模型权重；结果保存在 `logs/gpu_check.json`。

**请保留新版的 `RUN_ID='blackwell_cu128'`，三份 notebook 一致，不要改回 `main`。** 旧运行已经留下与 cu124 环境绑定的身份文件，不能直接覆盖。新版自动寻找 `results/deepseek_vl2_small/main/prediction/support.json`，存在时复用其中已冻结的家族及相邻 `images/`，并逐文件核对 `image_manifest.json`；不会把旧模型状态或预测缓存混入新结果，也不需要删除旧记录。此次附件停在首个评分路径检查，尚未完成校准。

## Drive 中需要的原实验输入

```text
VisualTorque/
  data/dataset_v1/                  # 或 data/dataset_v1.zip
    scenes.jsonl
    questions.jsonl
    splits.json
    images/...
  results/behavior_comparison_v3/activation_pilot_manifest.jsonl
  results/cross_model_readout_correction_v2/shared_1007_scene_ids.json
  results/cross_model_readout_correction_v2/shared_304_manifest.json
```

预检查会明确列出缺失输入。254 个拟合场景必须仍为 235 个训练场景、19 个开发场景。304 和 72 子集直接读取原共享清单，不能用“模型表现较明显”的新样本替代。

第一份优先恢复上述旧 DeepSeek `main` 中已冻结的家族；如果不存在且没有手动指定 `SOURCE_SUPPORT`，再查找 `results/intervention_prediction_v1/main/support.json` 及其 `images/`、`image_manifest.json`。存在时复用物理家族与图片。如果原目录在其他位置，把 `SOURCE_SUPPORT` 设为对应 `support.json` 的完整路径。两处都不存在时，按原生成器重新生成 32+96 个家族，并在报告中标明这是**相同协议的新几何**，不能与旧模型逐家族配对。复用旧 DeepSeek 样本本身不代表它与 Gemma/Qwen 的几何相同，须追溯源样本的 provenance。被压缩、未展开的旧实验几何不在自动排除范围内。

## 覆盖哪些结论

| 部分 | 固定协议 | 主要输出 |
|---|---|---|
| 4.2 映射是否改变迁移收益 | content：32 校准 + 96 验证家族；L20、L26；matching 与两类 conflict；单位剂量原始 BF16 状态差 | 映射下的基线、修好/损坏次数、净收益、家族区间、matching−conflicting 比较 |
| 4.3 效果是否可预测 | content 与 physical 各 32+96；原预测器和随机对照；验证预测全部封存后才施加验证干预 | 意义/符号/位置、置信度、物理与自然图像对照的误差和多重比较；答案变化事件数 |
| 4.3 新符号 | 同 96 个 content 验证几何；ABC、XYZ、1/2/3；重新测源/接收者状态和分数，但冻结 ABC 系数 | 三种格式的预测与选择指标、对应输出头重放的数值误差 |
| 4.4 读出与修准 | 235 训练/19 开发选层与 ridge alpha；254 重拟合；1007 测试；304 修准，72 场景×4 随机对照 | 读出 MAE、读出与模型答案分歧、数值修准成功率、答案收益及损坏 |
| 4.5 修复与移除 | 训练集拟合 PLS4；开发集选 upstream 层/剂量；固定 outcome-oriented L20、symbol-oriented L23；64 新家族×8 展示 | 三条局部路径、匹配实际 BF16 幅度的随机对照、修好/损坏、家族统计与依赖判定 |

4.2–4.3 的 L20 深度为 21/27，接近原 Gemma L37 的 38/48；L26 是最后一层。这是**预先选定的相对深度对应**，不代表已经找到了 DeepSeek 的最佳功能位置。content 每层 576、physical 每层 1152 个验证目标干预；格式测试另有 3456 个目标干预。

预测器的比例系数在 DeepSeek 自己的校准家族上拟合；该实验检验解释规则能否在第三模型中成立，不是将 Gemma 的数值系数零样本迁移到 DeepSeek。读出与修复也都重新拟合，不能复用其他模型的方向向量。

4.4 扫描全部 27 层，ridge alpha 为 0.01、0.1、1、10、100、1000，仅根据 19 场景开发集的 R² 选择。之后冻结，在同一单图/BF16 条件下评估并修准。没有可复用的“旧 DeepSeek probe”，因此这里不伪造新旧 probe 比较。数值修准迭代上限 32，目标误差 0.05，结果中保留未达到容差的全部尝试。

4.5 upstream 候选为 L10、L13、L16、L19，剂量为训练状态范数中位数的 0.5% 或 1%。方向仅用训练场景拟合，开发集每场景六种映射选候选。outcome-oriented 与 symbol-oriented 方向用本轮 content 校准家族重新构建，不使用旧模型向量。所有层、方向、剂量在确认集之前固定。

如果没有 upstream 候选通过开发集门槛，仍保存这个事实，并对固定网格中按预定规则选出的候选做一次独立确认。不会自动扩大网格、换剂量或重抽样。确认集有 64 个物理家族，每个包含镜像、渲染和映射组合，35 条件，共 17920 条条件记录。此处的 U/A/B 只用于代码表示 torque-supervised、outcome-oriented、symbol-oriented 三种方向，论文术语不改变。

## 如何读结果

结果独立保存到：

```text
results/deepseek_vl2_small/blackwell_cu128/
  REPORT_ZH.md
  summary.json
  prediction/models/deepseek16b/
    calibration/
    validation/analysis/mapping_gain_summary.csv
    validation/analysis/mapping_contrasts.csv
    validation/analysis/primary_comparisons.csv
    validation/analysis/transition_events.csv
    formats/analysis/
  readout/
    development_grid.csv
    frozen_probe.json
    test_summary.csv
    correction_summary.csv
    correction_contrasts.csv
    completion.json
  repair/
    development_selection.csv
    selected_route.json
    confirmation/primary_tests.csv
    confirmation/repair_harm_counts.csv
    confirmation/technical_failures.csv
    confirmation/completion.json
```

先看样本数、原始错误数和技术失败，再看效应大小、95% 区间与校正后的比较。预测与映射比较使用各自预定比较族的 Holm 校正；4.5 的 7 个对比×2 个指标共 14 项使用 BH 校正。新 DeepSeek 实验单独报告，不直接沿用旧双模型的校正 p 值。随机种子、映射和镜像不是额外独立样本。

4.5 `completion.json` 分别给出 accuracy 和 margin 的 `repair_confirmed` 与 `selective_dependence_confirmed`。只有初始方向相对未编辑和随机方向的修复优势成立，且移除与随机移除的差异成立、技术检查通过，才将对应局部依赖标为成立。不能把分数效果当作准确率修复，也不能把三个局部关系直接写成完整物理计算链。

## 断点续跑与运行身份

已完成的场景或条件组会缓存。模型版本、代码、输入图像、prompt、环境或冻结向量不一致时停止，避免混合结果；不要直接删除身份检查来续跑。`RUN_ID` 相同且环境相同时可从头重跑并复用缓存。完成标记只在预定阶段全部结束后写入。

模型 revision：`6033e16432a1d771cf9fe4a6f894ff5e5e1459af`。
官方代码 revision：`ef9f91e2b6426536b83294c11742c27be66361b1`。
使用独立的 Python 3.11.11、uv 0.6.9 和 Transformers 4.38.2。CUDA 12.8 配套版本为 Torch 2.9.1、torchvision 0.24.1、xformers 0.0.33.post2；安装官方源码时跳过其旧依赖解析，避免覆盖固定版本。完整 CUDA 模型仍需在目标 GPU 上验证。版本依据：[PyTorch 的 Blackwell 支持说明](https://pytorch.org/blog/pytorch-2-7/)、[PyTorch 固定版本安装表](https://pytorch.org/get-started/previous-versions/)、[xFormers 版本对应表](https://github.com/facebookresearch/xformers/blob/main/CHANGELOG.md)。解释器配置方式参考 [uv 官方文档](https://docs.astral.sh/uv/guides/install-python/)。

## 本地验证的边界

本地软件检查使用小型合成张量，覆盖候选符号/前缀、图像 mask 对齐、最后 token 的 BF16 写入、异常后 hook 清理、数值读出修准、35 条件路由、随机实际幅度检查、预测封存和缓存身份、开发集选层。它们是软件测试，不是 DeepSeek 实验结果。此电脑没有可用 CUDA 环境，尚未运行 16B 模型的完整前向与真实数据实验。

另已在 CPU 上运行固定版本的官方处理器和 tokenizer，确认三种格式有相同的空评分前缀、每个标签均为单 token；使用官方代码的小型随机初始化 MoE 检查了 BF16 层输出与 norm/head 重放。详细范围见源码包的 `VALIDATION.md`。

官方接口参考：[DeepSeek-VL2 仓库](https://github.com/deepseek-ai/DeepSeek-VL2)、[模型卡](https://huggingface.co/deepseek-ai/deepseek-vl2-small)。

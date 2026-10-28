# 运行与结果查看指引

## 先查看保存结果

从 [结论对照](CLAIM_TO_NOTEBOOK.md) 进入 runs/ 或 [DeepSeek 结果汇总](../evidence/deepseek/README.md)。notebooks 保留计算代码和原始输出，因此路径、旧名称、下载编号及历史报错也保留。code/ 表示源码入口；执行编号或无异常记录不等于科学条件全部通过。

## Qwen / Gemma 主实验

先查看 [原始运行指引索引](original_run_guides/README.md) 及各实验目录的 README。

1. 准备模型访问权限、GPU 与相应数据/输入结果。
2. 迁移预测运行 01 目录的 Calibration，再运行 Validation。
3. 跨符号运行 02 目录，读取前一步冻结的预测器，不重新拟合。
4. 读出修准在 03 目录独立运行；确认模型配置为对应 Qwen-9B 或 Gemma-12B。
5. 修复确认使用 04 目录固定候选与三路径入口，并配合 CPU 核查。

根 requirements.txt 保留原项目依赖；相关原环境记录含 Torch 2.11.0+cu128、Transformers 5.14.0。各 notebook 的安装单元格与原始指引仍是具体环境依据。历史 notebook 所需 toolkit、模块、数据和上游产物可能不同，不能直接假定任意文件都可从空目录一次运行。

## 新增 DeepSeek-VL2-small

使用 13_deepseek_extension/code/ 下三个入口，顺序为 01 Transfer → 02 Readout → 03 Repair fixed。对应源码见 support/deepseek_source/，恢复实现见 support/deepseek_recovery/。

DeepSeek 使用独立环境。当前入口的兼容组合为 Python 3.11.11、Torch 2.9.1、torchvision 0.24.1、xformers 0.0.33.post2、CUDA 12.8、Transformers 4.38.2；按 notebook 内嵌安装步骤建立环境，不要把根 requirements.txt 的 Transformers 5.14.0 安装进该环境。旧 runs/ 的 Python 3.13 安装失败与 CUDA 12.4/Blackwell 报错已保留供追溯。

恢复现有实验时，保持 RUN_ID=blackwell_cu128，并使 notebook 指向原 Drive 结果目录。03 fixed 会读取原来冻结的方向和剂量，在 confirmation_capacity52_v1 写入 52 家族新确认。原先 64 家族容量不足，不能通过把固定版输出改名为原目录来声称完成原计划。重新复现实验时需由 01/02 和 03 的开发步骤生成所需上游文件。

最新结果包已完成全部阶段；“completed”与“统计检验通过”是两件事。DeepSeek 准确率修复与依赖未获确认；技术失败与有效家族分析仍需报告，详见 [结果说明](../evidence/deepseek/README.md)。

## 外部输入与发布范围

notebooks 保留原 Colab/Drive 路径。使用前按配置单元格设置自己的路径、模型访问及输入；避免直接批量替换整个文件。模型权重、原始图片数据、激活缓存与完整逐试次结果不随仓库提供。data/ 和 results/ 的说明保留相关位置；结果汇总可用于阅读和对照，不能替代重算全部统计所需的原始产物。

只做文件完整性核查时，Python 标准库即可运行 python scripts/verify_package.py。它不执行 notebook，不需要 GPU，也不会检查模型效果能否复现。

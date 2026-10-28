# 2026-10-28 更新记录

本次新建 VisualTorque_GitHub_20261028，上一版 VisualTorque_GitHub 保留。此前 474 个源路径均未改变；原 344 份不同内容的 notebooks 全部保留。本次补充 7 份不同内容，现为 481 个源路径、351 份副本、15 个实验分组。

## 新增 notebooks

| 文件 | 角色 | 说明 |
|---|---|---|
| [01_DeepSeek_Transfer_4_2_4_3.ipynb](../notebooks/13_deepseek_extension/code/01_DeepSeek_Transfer_4_2_4_3.ipynb) | code | 迁移预测与跨格式入口；runs 中同名文件是 Python 安装失败记录。 |
| [02_DeepSeek_Readout_4_4.ipynb](../notebooks/13_deepseek_extension/code/02_DeepSeek_Readout_4_4.ipynb) | code | 读出拟合、测试与数值修准入口。 |
| [03_DeepSeek_Repair_4_5_fixed.ipynb](../notebooks/13_deepseek_extension/code/03_DeepSeek_Repair_4_5_fixed.ipynb) | code | 推荐确认入口：保持已冻结方向，改用 52 家族容量修订。 |
| [01_DeepSeek_Transfer_4_2_4_3 (1).ipynb](../notebooks/13_deepseek_extension/runs/01_DeepSeek_Transfer_4_2_4_3%20%281%29.ipynb) | runs | 早期 CUDA/Blackwell 不兼容的中断记录。 |
| [01_DeepSeek_Transfer_4_2_4_3.ipynb](../notebooks/13_deepseek_extension/runs/01_DeepSeek_Transfer_4_2_4_3.ipynb) | runs | 迁移预测与跨格式入口；runs 中同名文件是 Python 安装失败记录。 |
| [03_DeepSeek_Repair_4_5.ipynb](../notebooks/13_deepseek_extension/runs/03_DeepSeek_Repair_4_5.ipynb) | runs | 原 64 家族入口或容量不足的中断记录；查看 role 列。 |
| [03_DeepSeek_Repair_4_5.ipynb](../notebooks/13_deepseek_extension/versions/03_DeepSeek_Repair_4_5.ipynb) | versions | 原 64 家族入口或容量不足的中断记录；查看 role 列。 |

## 按当前论文重排目录

| 上一版 | 新版 |
|---|---|
| 01_rq1_readout_and_correction | 03_section44_readout_correction |
| 02_rq2_transfer_prediction | 01_section42_43_transfer_prediction |
| 03_rq2_format_generalization | 02_section43_format_generalization |
| 04_rq3_guided_repair | 04_section45_guided_repair |

文件级旧路径到新路径见 [path_changes.json](path_changes.json)。原整理版只调整副本位置；本阅读版另外精简 Markdown，计算代码与保存输出不变。

## 补充材料

- 新增 DeepSeek 完整结果压缩包中的 44 份精简结果文件、完成标记、52 家族容量修订及技术失败记录。
- 从实验源码压缩包复制 54 个源文件，另保留修复恢复脚本与原始运行指引。
- 增加当前论文与新增实验的对应关系，并明确“零错误被修复”只适用于原 Qwen/Gemma 结果。
- 保留开发、阴性结果、探索和后续实验，不将它们自动视为“不关键”而删除。

本次没有改写原件、清除历史输出、重新执行 GPU 实验或上传 GitHub。

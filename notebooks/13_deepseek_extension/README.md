# DeepSeek-VL2-small：4.2–4.5 模型扩展

**论文位置：** 对应当前稿件 4.2–4.5；2026-10-28 稿件尚未纳入该模型结果

**证据角色：** 新增已完成实验、历史中断记录及协议修订

最新结果包确认迁移预测、跨符号验证、读出修准及 52 家族修复确认已完成。符号预测有支持；读出纠正修好 3、损坏 4，未建立净准确率收益；L13/L20 初始准确率修复未确认，L20→L23 有分数层面的依赖。

## 阅读与运行顺序

01_DeepSeek_Transfer → 02_DeepSeek_Readout → 03_DeepSeek_Repair_4_5_fixed。保持 RUN_ID=blackwell_cu128；fixed 读取已冻结方向，使用 52 家族修订，不重新选点。原 64 家族版置于 versions，三份中断运行置于 runs。已完成结果见 evidence/deepseek，不能仅看中断 notebook 判断完成性。

## 文件索引

`code/` 为建议入口；`runs/` 保留实际输出和异常；`versions/` 保留不同源码。分类不是重要性排名，输出存在不等于科学假说成立。

| Notebook | 类型 | 保存状态 | 输出中识别的模型 |
|---|---|---|---|
| [01_DeepSeek_Transfer_4_2_4_3.ipynb](code/01_DeepSeek_Transfer_4_2_4_3.ipynb) | code | 源码无保存输出 | 见配置 |
| [01_DeepSeek_Transfer_4_2_4_3 (1).ipynb](runs/01_DeepSeek_Transfer_4_2_4_3%20%281%29.ipynb) | runs | 保存异常：RuntimeError | deepseek-ai/DeepSeek-VL2.git / deepseek-ai/deepseek-vl2-small |
| [01_DeepSeek_Transfer_4_2_4_3.ipynb](runs/01_DeepSeek_Transfer_4_2_4_3.ipynb) | runs | 保存异常：CalledProcessError | 见配置 |
| [02_DeepSeek_Readout_4_4.ipynb](code/02_DeepSeek_Readout_4_4.ipynb) | code | 源码无保存输出 | 见配置 |
| [03_DeepSeek_Repair_4_5_fixed.ipynb](code/03_DeepSeek_Repair_4_5_fixed.ipynb) | code | 源码无保存输出 | 见配置 |
| [03_DeepSeek_Repair_4_5.ipynb](runs/03_DeepSeek_Repair_4_5.ipynb) | runs | 保存异常：RuntimeError | deepseek-ai/DeepSeek-VL2.git / deepseek-ai/deepseek-vl2-small |
| [03_DeepSeek_Repair_4_5.ipynb](versions/03_DeepSeek_Repair_4_5.ipynb) | versions | 源码无保存输出 | 见配置 |

三份 runs 是旧安装、CUDA 兼容性、64 家族容量中断记录。最新完整结果在 [DeepSeek 结果索引](../../evidence/deepseek/README.md)，其中 4.5 实际使用恢复版 52 家族协议。不能把中断记录当作当前实验失败，也不能把完成标记当成修复门槛通过。

实验设置和运行要求见各 notebook 开头的说明链接。

[结论对照](../../docs/CLAIM_TO_NOTEBOOK.md) · [运行说明](../../docs/RUNNING.md) · [覆盖范围](../../docs/COVERAGE.md) · [来源清单](../../docs/source_to_copy.csv)

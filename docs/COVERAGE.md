# Notebook 来源与收录范围

| 来源 | 文件路径数 |
|---|---:|
| VisualTorque 项目中的 notebooks | 435 |
| Downloads 中 VisualTorque 开头的 notebooks | 43 |
| Downloads 中 DeepSeek notebooks | 3 |
| 合计 | 481 |
| 按文件内容去重后的 notebooks | 351 |

130 个完全相同的来源路径合并到同一仓库文件；不同内容的版本分别保留。按用途分为 159 份运行入口、176 份运行记录和 16 份历史版本。

完整路径与哈希见 [来源表](source_to_copy.csv) 和 [文件清单](notebook_manifest.json)。来源范围限于可读取的项目目录和相关下载文件；25 个临时、缓存或演示目录无法读取，记录见 [coverage_scan.json](coverage_scan.json)。

DeepSeek 源码与结果分别来自 experiment_source.zip 和 blackwell_cu128_results (2).zip。结果包不含 notebook。压缩包身份与所收录文件见 [archive_sources.json](archive_sources.json) 和 [support_manifest.json](support_manifest.json)。

模型权重、大型激活缓存和完整结果压缩包未纳入仓库。

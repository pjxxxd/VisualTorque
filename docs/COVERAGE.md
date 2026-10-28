# 扫描与收录范围

| 范围 | 原路径数 |
|---|---:|
| 当前 VisualTorque 工作区内可读取、已展开的 notebooks，排除整理输出目录 | 435 |
| Downloads 中 VisualTorque 开头的 notebooks | 43 |
| Downloads 中本次新增 DeepSeek notebooks | 3 |
| 合计 | 481 |
| 按完整文件 SHA-256 去重后的副本 | 351 |
| 合并到同一副本的完全相同路径 | 130 |

相对于上一版，474 个已有源文件均未改变；新增 7 个源文件也都是新内容。159 份 code、176 份 runs、16 份 versions 均已复制，未按是否支持论文结论删去文件。完整来源见 [source_to_copy.csv](source_to_copy.csv)，内容身份见 [notebook_manifest.json](notebook_manifest.json)。

此次扫描不是整台电脑的无遗漏盘点。工作区内 25 个演示文稿/临时/缓存目录无法读取，详细位置见 [coverage_scan.json](coverage_scan.json)。未扫描其他硬盘，也未穷举所有压缩包。Downloads 仅检查项目相关文件名。

另定向读取了 experiment_source.zip 与最新 blackwell_cu128_results (2).zip，分别取得 DeepSeek 源码和完成结果。最新结果包不含 notebook。压缩包身份见 [archive_sources.json](archive_sources.json)，所选成员与副本哈希见 [support_manifest.json](support_manifest.json)。未将整个结果包、模型权重或大型中间缓存复制进发布目录。

这些计数说明本次整理覆盖了哪些可见文件；不能用未收录或未出现在主结果目录来判断某项实验不重要。

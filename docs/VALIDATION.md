# 核验说明

本阅读版仅修改 Markdown 说明。notebook_manifest.json 记录精简副本的 sha256 与原文件的 source_sha256；来源清单保留全部 481 个路径。

核验结果见 [validation.json](validation.json)。核验包括：

- 351 份精简副本与完整整理版逐一比较：5,591 个代码单元格、保存输出、执行编号和元数据一致。
- 全部原 notebook、两份原始压缩包及 210 份支持文件的来源与原清单匹配。上传准备仅为 .gitattributes 增加了禁止换行转换的规则，其原哈希另存为 source_sha256。
- 精简版文件哈希、来源映射、文档和 notebook 内的说明链接有效。
- 历史整理包与原件保持原样。5 份保存异常的运行记录保留。

核验不执行 notebook，也不重算模型结果。说明文字改动见 [SIMPLIFICATION](SIMPLIFICATION.md)。

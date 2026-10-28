# 文件完整性检查

运行 python scripts/verify_package.py，检查 notebook 格式、文件哈希、来源映射和文档链接。

已有核验记录见 [validation.json](validation.json)。核验覆盖 351 份 notebooks、210 份支持文件与配套说明。5,591 个代码单元格的源代码、输出、执行编号及元数据与来源记录一致；5 份包含异常输出的历史运行仍保留。

notebook_manifest.json 中 sha256 对应仓库文件，source_sha256 对应来源文件。Git 配置禁止自动转换换行符，以保持文件哈希一致。

本检查不执行模型推理或重新计算统计结果。外部数据与模型环境要求见 [运行说明](RUNNING.md)。

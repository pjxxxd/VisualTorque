# 文件完整性检查

运行 python scripts/verify_package.py，检查 notebook 格式、文件哈希、来源映射和文档链接。

已有核验记录见 [validation.json](validation.json)。核验覆盖 351 份 notebooks、210 份原有支持文件与配套说明。5 份包含异常输出的历史运行仍保留。

与来源逐格比较时，允许的代码变更仅为 [通用哈希函数](runtime_helpers.json) 的等价导入及首个辅助单元格。原有 5,591 个代码单元格中的实验逻辑、参数、断言、数值结果和图像保持原样；长哈希的显示被缩短，完整值另存于单元格元数据。来源单元格编号与当前编号的对应关系见 [notebook_provenance.json](notebook_provenance.json)。

python scripts/test_notebook_checks.py 检查哈希函数兼容性、异常检测和输出格式。安装 IPython、ipykernel 后，还会启动临时 Jupyter 内核检查连续执行、单文件运行、富文本输出及 DeepSeek 日志保留。

notebook_manifest.json 中 sha256 对应仓库文件，source_sha256 对应来源文件。Git 配置禁止自动转换换行符，以保持文件哈希一致。

本检查不执行模型推理或重新计算统计结果。外部数据与模型环境要求见 [运行说明](RUNNING.md)。

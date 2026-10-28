# Notebook 文件说明

notebooks/ 按论文实验和研究主题分组。code/ 提供运行入口，runs/ 保存执行输出，versions/ 保存历史版本。

详细设置与统计方法见 [实验说明索引](EXPERIMENT_NOTES.md)。每份 notebook 开头也有对应链接。

## 来源与版本

- [notebook_manifest.json](notebook_manifest.json)：文件清单、所属实验及 SHA-256。
- [source_to_copy.csv](source_to_copy.csv)：来源路径与仓库路径的对应关系。
- [notebook_provenance.json](notebook_provenance.json)：来源文件与仓库文件的哈希、单元格编号对应关系及配套说明路径。
- [path_changes.json](path_changes.json)：实验目录调整前后的路径。

sha256 记录仓库文件，source_sha256 记录来源文件。Markdown 说明与实验设置分开存放；代码、参数和保存输出与来源 notebook 一致。

文件完整性检查见 [VALIDATION](VALIDATION.md)。

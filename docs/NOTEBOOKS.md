# Notebook 文件说明

notebooks/ 按论文实验和研究主题分组。code/ 提供运行入口，runs/ 保存执行输出，versions/ 保存历史版本。

详细设置与统计方法见 [实验说明索引](EXPERIMENT_NOTES.md)。每份 notebook 开头也有对应链接。

## 来源与版本

- [notebook_manifest.json](notebook_manifest.json)：文件清单、所属实验及 SHA-256。
- [source_to_copy.csv](source_to_copy.csv)：来源路径与仓库路径的对应关系。
- [notebook_provenance.json](notebook_provenance.json)：来源文件与仓库文件的哈希、单元格编号对应关系及配套说明路径。
- [path_changes.json](path_changes.json)：实验目录调整前后的路径。

sha256 记录仓库文件，source_sha256 记录来源文件。详细实验说明单独存放，notebook 保留执行步骤。

通用文件校验放在 [visualtorque_checks.py](../support/visualtorque_checks.py)。长哈希在输出中缩写；变量、结果文件和日志仍使用完整值。已有输出的完整哈希保存在对应单元格的 metadata.visualtorque.display_hashes 中。

首个代码单元格加载辅助模块，标记为折叠显示，并附带模块副本以支持单文件上传 Colab。修改模块后，用 python scripts/notebook_runtime_tools.py --sync 更新 notebook 内的副本。

文件完整性检查见 [VALIDATION](VALIDATION.md)。

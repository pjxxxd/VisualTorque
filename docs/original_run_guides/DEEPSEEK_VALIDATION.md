# 软件验证记录

检查日期：2026-10-26。

- 本地 CPU PyTorch 2.6.0 软件检查：7/7 通过。
- 原官方处理器、tokenizer 与固定 revision 的配置：实际 CPU 执行通过。
- ABC token IDs 为 32/33/34；XYZ 为 55/56/57；1/2/3 为 16/17/18。三种格式的公共评分前缀均为空。
- 合成 PNG 经官方多图块处理器处理，图像 token mask 与输入序列对齐，最后评分 token 为文本。
- 官方 DeepSeek 解码器代码的小型随机初始化 MoE：BF16 前向与最后 norm/head 重放通过。该模型仅用于接口检查，不是 DeepSeek-VL2-Small 权重，输出不作为研究证据。
- 三份 notebook 的代码单元可解析，初始输出为空；内嵌源码及独立源码包一致性在打包时检查。

## Python 3.13 安装修复

用户提供的运行记录在安装单元退出，traceback 显示 Colab 使用 Python 3.13；内嵌旧版 `install.py` 在建立环境前即断言 Python 必须为 3.10–3.12。新版通过 uv 准备独立的 Python 3.11.11，并把子进程输出实时显示、保存到 Drive 的日志文件。实验算法与参数未修改。

- 安装修复回归检查：4/4 通过，覆盖模拟 Python 3.13 主环境的解释器配置、不覆盖不兼容的已有环境、安装失败不写完成标记，以及真实失败子进程的 stdout/stderr 显示和日志留存。
- 实际解释器配置检查通过：在本地 Windows 的 Python 3.13.2 中调用新版 `ensure_python`，自动下载并创建 Python 3.11.11 环境，执行新解释器确认版本一致；没有下载模型权重。此检查验证跨 Python 版本的配置路径，不替代 Colab/Linux GPU 验证。
- 上述检查不代表 Linux CUDA 依赖或 16B 模型已完成运行。

## Blackwell / CUDA kernel 修复

第二份运行记录确认：Python 3.11.11 环境安装、数据预检查、16B checkpoint 分片加载已通过，但旧 cu124 PyTorch 对 RTX PRO 6000 Blackwell（sm_120）发出架构不兼容警告；随后在首个评分路径检查的 `torch.zeros_like` 报 `no kernel image is available`。尚未完成实际模型的第一次评分前向或校准。

新版固定 Torch 2.9.1、torchvision 0.24.1、xformers 0.0.33.post2 / CUDA 12.8，独立环境和默认 RUN_ID 均与旧版分开。复用旧 main 的家族和图片，保留其 upstream provenance，不复用旧状态/预测缓存。样本数、层号、干预剂量、估计器及检验方法不变。

安装器新增真实 GPU 检查（布尔 mask、BF16 GEMM、官方视觉 attention 的 729×16×72 BF16 输入及重复前向），每次复用已有环境也会执行；检查通过前不写新的安装完成标记。测试代码随包交付，但本地没有 CUDA，未执行这些真实 GPU 检查；它们将在 Colab 安装单元执行。

- 安装与恢复流程回归检查：9/9 通过，包括设备可见但架构不兼容的拒绝、实际 GPU 检查失败不写完成标记、已有安装必须复查 GPU、新旧运行目录隔离及已冻结家族的复用。这些使用模拟设备信息/子进程检查，不能代替真实 GPU 执行。
- 已实际安装隔离的 Torch 2.9.1+cpu / torchvision 0.24.1+cpu，在本地 CPU 上重跑软件检查：7/7 通过。
- 新版 Torch 下，官方处理器/tokenizer 和小型随机初始化 MoE 的 BF16 前向、最后 norm/head 重放均通过，重放最大误差 0。该检查未加载 16B 权重，也未执行 CUDA/xFormers 视觉 kernel。
- 对照此次附件的内嵌源码：所有数值 kernels、toolkit、模型 adapter、readout 和 repair 文件逐字节一致，保留原有历史读取加速；prediction 仅追加上游样本来源记录。

未验证：完整 16B checkpoint 的 CUDA 图像前向、目标 GPU 峰值显存与运行时间、正式数据上的效果、跨模型结论。目标运行时会再次检查实际模型的零编辑重放和评分路径。

软件测试不是实验结果。只有对应阶段的 `completion.json` 生成后，才可把该阶段视为完整运行；具体结论仍以统计输出和技术失败记录为准。

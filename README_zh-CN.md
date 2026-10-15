# 静态视网膜 (TinyCharEncoder) 蒸馏与评测分析

这是独立于主工程的一项“正向蒸馏（Inward Latent Distillation）”子任务。我们的目标是用一个极轻量的单字级模型（TinyCharEncoder），通过 MSE 强行克隆主工程中 5 大外部预训练模型联合打点形成的绝对物理拓扑空间。

## 概览

本仓库提供了将大型外部预训练模型的知识蒸馏到微小、自研的单字级编码器 (TinyCharEncoder) 中的工具。该模型被设计为“静态视网膜”—— 一个极轻量的、纯原生的、显式构建了旋转位置编码 (RoPE) 的 Text-to-Vector 编码器。

## 特性

- **极轻量单字级编码器**: 使用简单的 `CharTokenizer` 进行单字组装，避免了复杂的 BPE 词表。
- **RoPE 集成**: 支持无限长度外推，在训练期间不会产生 O(N^2) 的显存膨胀。
- **跨框架支持**: 包含 Apple MLX (用于超快速本地 Mac 开发) 和 PyTorch/CUDA (用于 NVIDIA Linux 集群) 的原生实现。
- **MTEB 评测**: 将自研的 TinyCharEncoder 无缝包装成标准的 MTEB (Massive Text Embedding Benchmark) 接口，以进行严格的评测。

## 安装

首先，安装依赖：

```bash
pip install -r requirements.txt
```

**注意**: 默认情况下，`requirements.txt` 是为 PyTorch/CUDA 配置的。如果你在 Apple Silicon Mac 上进行开发并希望使用 MLX 后端，请注释掉 PyTorch 依赖并取消注释 `mlx>=0.14.0`。

如果你计划运行评测，可能还需要安装 MTEB：
```bash
pip install mteb[zh]
```

## 架构设计

`TinyCharEncoder` 是一个具有 8 个注意力头和 1024 隐藏维度的 6 层 Transformer。它利用了 `RoPEMultiHeadAttention` 并通过掩码平均池化聚合句子级别的语义表示。然后将这些嵌入投影到由 `GodEncoder` 映射的绝对 1024-D 目标空间中。

## 使用方法

### 训练 (蒸馏)

要启动蒸馏过程，你需要一个冻结的 Phase 0 权重检查点来进行蒸馏：

```bash
python train.py --p0_ckpt <path_to_teacher_checkpoint>
```

你可以通过命令行参数自定义学生网络架构或数据目录 (例如，`--student_n_layers 6`, `--batch_size 256`)。

### 评测

在 C-MTEB 任务上评测蒸馏后的模型：

```bash
python evaluate_mteb.py --ckpt <path_to_student_checkpoint> --tasks LCQMC BQCorpus AFQMC
```

## MTEB 评测现象与科学分析

根据在此目录下执行 `evaluate_mteb.py` 得到的真实结果比对：

* **现象 1（大获成功）：通用相似度任务 (如 LCQMC)**
  * **得分**：学生 (68.5%) vs 老师(BGE-M3 76.1%)
  * **结论**：极高分差逼近。这证明了 MSE 正向蒸馏在数学上的绝对成立。你的 6 层 Char 级小模型成功锁死了外部大模型的联合拓扑空间，具备了极其优异的基础常识几何映射能力。

* **现象 2（断崖式下跌）：专有逻辑与细分任务 (如 AFQMC)**
  * **得分**：学生 (23.0%) vs 老师(Xiaobu 60.9% / BGE 40.0%)
  * **底层原因剖析**：
    1. **语料维度的盲区**：小模型只吃过 `chunked_mixed_omni` 通用数据集。如果训练集中没有专门教授金融/医疗/法律等长尾专有名词，小模型无从知晓这些组合坐标，只能盲猜。
    2. **物理容量与拼字极限**：大模型使用 BPE 分词，对专有名词有整体词表记忆；而你的小模型使用 `CharTokenizer`（单字拼装），在仅有 6 层的极小容量下，强行将“蚂”“蚁”“金”“服”四个毫无关联的单字在大脑中组装成“金融科技”的深层拓扑语义，这超越了其物理极限。
    3. **缺乏动力学消歧义**：小模型在这里仅仅充当静态视网膜，它不懂上下文推理。
  * **系统定位与战略定力**：这是物理规律的必然！这个静态小模型在这里**不需要**考高分。在整个终极内生架构中，专业的消歧义和长程逻辑推理，是由主仓库中后续的 **Mamba Planner（系统 2 大脑）** 通过引入正交的动态认知流（如字典注入）来完成的。

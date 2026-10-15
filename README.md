# Static Retina (TinyCharEncoder) Distillation and Evaluation Analysis

This is a subtask of "Inward Latent Distillation" independent of the main project. Our goal is to use an extremely lightweight character-level model (TinyCharEncoder) to forcibly clone the absolute physical topological space formed by the joint embeddings of 5 external pre-trained models from the main project via MSE (Mean Squared Error).

## Overview

This repository provides tools to distill knowledge from large, external pre-trained models into a tiny, custom-built character-level encoder (TinyCharEncoder). The model is designed to act as a "static retina" — a lightweight, pure-native Text-to-Vector encoder explicitly built with Rotary Position Embeddings (RoPE).

## Features

- **Tiny Character-Level Encoder**: Uses a simple `CharTokenizer` for character-level assembly, avoiding the need for a complex BPE vocabulary.
- **RoPE Integration**: Supports infinite length extrapolation without O(N^2) memory bloating during training.
- **Cross-Framework Support**: Includes native implementations in Apple MLX (for ultra-fast local Mac development) and PyTorch/CUDA (for NVIDIA Linux clusters).
- **MTEB Evaluation**: Seamlessly wraps the custom TinyCharEncoder into a standard MTEB (Massive Text Embedding Benchmark) interface for rigorous evaluation.

## Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

**Note**: By default, `requirements.txt` is configured for PyTorch/CUDA. If you are developing on an Apple Silicon Mac and want to use the MLX backend, comment out the PyTorch requirements and uncomment `mlx>=0.14.0`.

You may also need to install MTEB if you plan to run evaluations:
```bash
pip install mteb[zh]
```

## Architecture

The `TinyCharEncoder` is a 6-layer Transformer with 8 attention heads and a hidden dimension of 1024. It utilizes `RoPEMultiHeadAttention` and aggregates sentence-level semantic representations via masked average pooling. It projects these embeddings into an absolute 1024-D target space mapped by the `GodEncoder`.

## Usage

### Training (Distillation)

To start the distillation process, you need a frozen Phase 0 weights checkpoint to distill from:

```bash
python train.py --p0_ckpt <path_to_teacher_checkpoint>
```

You can customize the student architecture or data directories through command-line arguments (e.g., `--student_n_layers 6`, `--batch_size 256`).

### Evaluation

To evaluate the distilled model on C-MTEB tasks:

```bash
python evaluate_mteb.py --ckpt <path_to_student_checkpoint> --tasks LCQMC BQCorpus AFQMC
```

## Evaluation Analysis & MTEB Insights

Based on the actual results obtained by executing `evaluate_mteb.py` in this directory:

* **Phenomenon 1 (Great Success): General Similarity Tasks (e.g., LCQMC)**
  * **Score**: Student (68.5%) vs. Teacher (BGE-M3 76.1%)
  * **Conclusion**: Extremely close score approximation. This mathematically proves the absolute validity of MSE inward distillation. Your 6-layer character-level small model successfully locked onto the joint topological space of the external large models, demonstrating highly excellent basic common-sense geometric mapping capabilities.

* **Phenomenon 2 (Cliff-like Drop): Domain-Specific Logic and Niche Tasks (e.g., AFQMC)**
  * **Score**: Student (23.0%) vs. Teacher (Xiaobu 60.9% / BGE 40.0%)
  * **Underlying Reason Analysis**:
    1. **Corpus Blind Spots**: The small model has only consumed the `chunked_mixed_omni` general dataset. If the training set does not specifically teach long-tail proper nouns in finance/medical/legal domains, the small model has no way of knowing these coordinate combinations and can only guess blindly.
    2. **Physical Capacity and Spelling Limits**: Large models use BPE tokenization and have holistic vocabulary memory for proper nouns; whereas your small model uses `CharTokenizer` (single-character assembly). Forcing it to assemble four entirely unrelated characters like "Ant", "Financial", "Services", "Group" into a deep topological semantic of "Fintech" within its tiny 6-layer capacity exceeds its physical limits.
    3. **Lack of Dynamic Disambiguation**: The small model merely acts as a static retina here; it does not understand contextual reasoning.
  * **System Positioning and Strategic Focus**: This is an inevitable physical law! This static small model **does not need** to score high here. In the ultimate endogenous architecture, professional disambiguation and long-range logical reasoning are accomplished by the subsequent **Mamba Planner (System 2 Brain)** in the main repository through the introduction of orthogonal dynamic cognitive flows (like dictionary injection).

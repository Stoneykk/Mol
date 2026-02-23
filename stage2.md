# Stage 2 — MoLFormer-c3-1.1B 复现与官方对比

## 1. 目标

在 6 个 MoleculeNet 数据集上复现 MoLFormer-c3-1.1B 的 finetune 结果，使用与 [ChemBERTa-3](https://github.com/deepforestsci/chemberta3) 官方相同的 **DeepChem scaffold split (80/10/10)**，实现结果的直接横向对比。

---

## 2. MoLFormer 架构

### 2.1 关键配置

| 配置项 | 值 |
|--------|-----|
| 预训练数据量 | 1.1B 分子 (100% ZINC20 + 100% PubChem) |
| 实际参数量 | 46.8M (注: "1.1B" 指预训练分子数，非参数量) |
| 架构 | 12层 Transformer, 12 heads, hidden_size=768 |
| 特殊设计 | Linear Attention + Rotary Positional Embeddings |
| Tokenizer | BPE, vocab_size=2362, max_length=202 |
| Pooling | Masked average pooling（非 CLS token） |
| Classification Head | 2层 Linear + 残差跳连 |

### 2.2 Linear Attention 机制

标准 Attention 复杂度为 O(n²)，MoLFormer 使用 Linear Attention 降低到 O(nd)：

```
标准: softmax(QK^T / sqrt(d)) V
Linear: (Q' @ (K'^T @ V)) / norm
  其中 Q', K' = FeatureMap(Q, K)  # 随机正交投影 + ReLU kernel
```

使用 `num_random_features=32` 个正交随机投影权重，在长序列上显著加速。

### 2.3 Rotary Positional Embeddings (RoPE)

不使用传统的可学习位置 embedding，而是通过旋转矩阵将位置信息直接注入 Q 和 K：

```python
Q_pos[i] = rotate(Q[i], θ * i)
K_pos[j] = rotate(K[j], θ * j)
```

优势：相对位置编码 + 外推能力强（可处理比训练时更长的序列）。

---

## 3. 复现方案

### 3.1 与官方的异同

| 维度 | 我们的实现 | ChemBERTa-3 官方 |
|------|-----------|----------------|
| 预训练权重 | DeepChem/MoLFormer-c3-1.1B (HuggingFace) | 相同 |
| 框架 | HuggingFace AutoModel + PyTorch | DeepChem wrapper |
| 优化器 | AdamW | FusedLAMB (NVIDIA apex) |
| 数据 split | DeepChem scaffold split (80/10/10) | **相同** |
| 实验次数 | 3 seeds (triplicate) | 3 seeds (triplicate) |

### 3.2 训练配置

- **回归任务**：100 epochs, batch_size=32, lr=3e-5, weight_decay=0.01
- **分类任务**：10 epochs, batch_size=32, lr=3e-5, weight_decay=0.01
- **参数分组**：bias/LayerNorm/Embedding 设为 no_decay
- **梯度裁剪**：max_norm=1.0
- **Triplicate**：每个数据集跑 3 seeds (0, 1, 2)，取平均 ± 标准差

---

## 4. 实验结果与官方对比

### 4.1 官方 c3-MoLFormer-1.1B 结果（DeepChem scaffold split）

数据来源：[ChemBERTa-3 GitHub](https://github.com/deepforestsci/chemberta3) `results/images/Deepchem-splits-benchmark1.png`

**分类（ROC-AUC ↑）**

| Dataset | c3-MoLFormer-1.1B (官方) |
|---------|------------------------|
| BBBP    | 0.735 ± 0.019          |
| Tox21   | 0.723 ± 0.012          |
| ClinTox | 0.839 ± 0.013          |

**回归（RMSE ↓）**

| Dataset       | c3-MoLFormer-1.1B (官方) |
|---------------|------------------------|
| ESOL          | 0.829 ± 0.019          |
| FreeSolv      | 0.572 ± 0.023          |
| Lipophilicity | 0.728 ± 0.016          |

### 4.2 我们的复现结果

> **待服务器跑完后填充**

**分类（ROC-AUC ↑）**

| Dataset | 官方 c3-MoLFormer | 我们的复现 | 差异 |
|---------|-------------------|-----------|------|
| BBBP    | 0.735 ± 0.019     | TBD       | TBD  |
| Tox21   | 0.723 ± 0.012     | TBD       | TBD  |
| ClinTox | 0.839 ± 0.013     | TBD       | TBD  |

**回归（RMSE ↓）**

| Dataset       | 官方 c3-MoLFormer | 我们的复现 | 差异 |
|---------------|-------------------|-----------|------|
| ESOL          | 0.829 ± 0.019     | TBD       | TBD  |
| FreeSolv      | 0.572 ± 0.023     | TBD       | TBD  |
| Lipophilicity | 0.728 ± 0.016     | TBD       | TBD  |

### 4.3 可视化对比

> **待结果出来后生成**

---

## 5. 差异分析

> **待结果出来后分析**

预期的差异来源：
1. **优化器不同**：AdamW vs FusedLAMB — 预计影响较小
2. **框架不同**：HF AutoModel vs DeepChem wrapper — 可能影响 tokenization/padding 细节
3. **超参数**：lr/epochs/batch_size 可能与官方不完全一致
4. **随机种子**：split 相同但训练种子不同

合理的差异范围：分类 ±0.02~0.03 AUC, 回归 ±0.05~0.10 RMSE

---

## 6. 实现细节与技术挑战

### 6.1 HuggingFace Transformers 版本兼容性

IBM 的 MoLFormer 远程代码（`configuration_molformer.py`, `modeling_molformer.py`）使用了已废弃的 API：
- `transformers.onnx.OnnxConfig`（新版移除）
- `transformers.pytorch_utils.find_pruneable_heads_and_indices`（新版移除）

**解决方案**：锁定 `transformers==4.38.2`（2024 年初稳定版），与 IBM 代码完全兼容。

### 6.2 DeepChem Scaffold Split

使用 `deepchem.splits.ScaffoldSplitter` 生成 80/10/10 的 train/val/test 划分，与 ChemBERTa-3 官方使用完全相同的分割算法。脚本 `scripts/generate_deepchem_splits.py` 从现有数据中提取 SMILES + targets，重新按 DeepChem scaffold 算法分割。

### 6.3 Docker 部署

服务器环境无 conda/pip，使用 Docker 容器化：
```dockerfile
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
```

运行流程：
1. Docker 启动时自动生成 DeepChem scaffold split 文件
2. 依次运行 6 个数据集的 benchmark
3. 结果通过 volume 映射保存到宿主机

---

## 7. 文件结构

```
Mol_Regression/
├── molformer/
│   ├── __init__.py
│   ├── finetune.py                # 回归 finetune (100 epochs, 3 seeds)
│   └── finetune_cls.py            # 分类 finetune (10 epochs, 3 seeds)
├── scripts/
│   └── generate_deepchem_splits.py # 生成 DeepChem scaffold split
├── data/
│   ├── *_dc_split.csv             # DeepChem scaffold split 文件（运行时生成）
│   └── *_v2_split.csv             # 旧的 chemprop v2 split 文件
├── run_molformer_benchmark.sh     # 一键跑分脚本
├── Dockerfile                     # GPU 服务器 Docker 镜像
├── requirements.txt               # Python 依赖
├── stage2.md                      # 本文档
└── output/molformer_benchmark/    # 结果输出
```

---

*文档更新时间: 2026-02-23*  
*参考版本: transformers 4.38.2, DeepChem/MoLFormer-c3-1.1B*

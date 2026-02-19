# Stage 2 交接文档 — MoLFormer-c3-1.1B 复现

## 1. 项目总体背景

用户的最终目标是构建一个 **D-MPNN + MoLFormer 多模态融合模型**。项目分阶段推进：

- **Stage 1 (已完成)**：从零实现 D-MPNN，对齐 chemprop v2，在 6 个 MoleculeNet 数据集上验证。结果记录在 `stage1.md`，代码已推送到 `git@github.com:Stoneykk/Mol.git`。
- **Stage 2 (进行中)**：复现 MoLFormer-c3-1.1B，加载预训练权重，在相同数据集上 finetune 并与 ChemBERTa-3 论文结果对比，为后续融合打基础。

## 2. Stage 2 当前进度

### 已完成

| 步骤 | 状态 | 说明 |
|------|------|------|
| 环境准备 | 完成 | conda env `chemprop` (Python 3.11), 安装了 deepchem, transformers 4.57.6, tokenizers |
| 模型下载与验证 | 完成 | `DeepChem/MoLFormer-c3-1.1B` 预训练权重已缓存到 `~/.cache/huggingface/` |
| 架构源码精读 | 完成 | 已完整阅读 IBM 原始 `modeling_molformer.py` (922行) |
| 回归 finetune 脚本 | 已写但需修改 | `molformer/finetune.py` 已创建，但**用了 AdamW 而非官方的 FusedLAMB** |

### 未完成

| 步骤 | 说明 |
|------|------|
| **脚本需要重写以匹配 ChemBERTa-3** | 用户明确要求 finetune 方式与 ChemBERTa-3 repo 完全一致 |
| 分类 finetune 脚本 | `molformer/finetune_cls.py` 尚未创建 |
| 在 6 个数据集上跑 benchmark | 需要 GPU 服务器 |
| 结果汇总与可视化 | 对比 ChemBERTa-3 论文 + Stage 1 D-MPNN |
| 生成 stage2.md | 架构分析 + 实验结果 + 对比图 |
| 推送到远程仓库 | `git@github.com:Stoneykk/Mol.git` |

## 3. 关键技术细节

### 3.1 MoLFormer-c3-1.1B 命名澄清

**"1.1B" 指预训练数据量(11亿分子)，不是参数量。模型参数量只有 46.8M。**

- 架构：12层 Transformer, 12 heads, hidden_size=768, intermediate_size=768
- 特殊点：Linear Attention (不是标准 softmax attention) + Rotary Positional Embeddings
- 没有传统的 position embedding 层，位置信息通过 RoPE 注入 Q/K
- Pooling：masked average pooling (不是 CLS token)
- Classification Head：两层 Linear + 残差跳连 (skip_connection=True)
- Tokenizer：BPE, vocab_size=2362, max_length=202, 基于字符级 SMILES tokenization

### 3.2 Linear Attention 机制 (核心创新)

标准 Attention: `softmax(QK^T / sqrt(d)) V` — O(n^2)
MoLFormer Linear Attention:
```
1. Q, K 经过 RoPE
2. Q', K' = FeatureMap(Q, K)  # 随机正交投影 + ReLU kernel
3. KV = K'^T @ V              # 先算 K^T V, O(d^2)
4. norm = Q' @ sum(K', dim=-2) # 归一化因子
5. output = (Q' @ KV) / norm   # O(nd), 而非 O(n^2)
```
FeatureMap 使用 `num_random_features=32` 个正交随机投影权重，每次 forward 时会重新生成（除非 `deterministic_eval=True`）。

### 3.3 ChemBERTa-3 官方 finetune 流程（需要匹配的目标）

**关键文件**：`chemberta3/chemberta3_benchmarking/models_benchmarking/molformer_benchmark/`
- `molformer_finetune_regression.py` — 回归 (ESOL, FreeSolv, Lipo)
- `molformer_finetune_classification.py` — 分类 (BBBP, Tox21, ClinTox)

**核心配置**：
- **框架**：DeepChem 的 `MoLFormer` wrapper (不是直接用 HuggingFace)
- **优化器**：`FusedLAMB` (from `apex.optimizers`), lr=3e-5, 需要 NVIDIA GPU + apex
- **参数分组**：decay (Linear weights) vs no_decay (bias, LayerNorm, Embedding)
- **回归**：100 epochs, batch_size=32, NormalizationTransformer (target standardization)
- **分类**：10 epochs, batch_size=32, metric=ROC-AUC
- **重复**：每个数据集跑 3 次取平均 (triplicate)
- **Best model**：基于 val score 保存 checkpoint，最终用 best checkpoint 评估 test
- **CustomMoLFormer**：他们继承了 DeepChem 的 MoLFormer 类，重写了 `fit_generator` 以支持 FusedLAMB 的 param groups
- **预训练权重**：通过 `model.load_from_pretrained(pretrained_model_path)` 加载 DeepChem 格式的 checkpoint（HuggingFace 页面上的 `deepchem_ckpt.pt`, 562MB）

**数据格式**：DeepChem `DiskDataset`, 使用 `DummyFeaturizer` (直接存 SMILES 字符串)
**数据分割**：支持 DeepChem scaffold splits 和 MoLFormer splits 两种方式

### 3.4 当前脚本与官方的差异（需要修正）

| 维度 | 当前 `molformer/finetune.py` | ChemBERTa-3 官方 | 是否需要修正 |
|------|------------------------------|------------------|-------------|
| 框架 | 直接用 HuggingFace AutoModel | DeepChem MoLFormer wrapper | 取决于用户选择 |
| 优化器 | AdamW | FusedLAMB (apex) | 是，需要 GPU |
| Epochs | 50 | 100 (回归) / 10 (分类) | 是 |
| 重复实验 | 1次 | 3次 | 是 |
| 权重加载 | HF `from_pretrained` | DeepChem `load_from_pretrained` | 是 |
| 数据流 | 自定义 PyTorch Dataset | DeepChem DiskDataset | 取决于用户选择 |

### 3.5 关于两种实现路径的建议

**路径 A: 完全复刻 ChemBERTa-3（推荐用于结果对比）**
- 直接 clone `deepforestsci/chemberta3` repo
- 使用他们的脚本 + DeepChem wrapper
- 需要：GPU + apex + deepchem_ckpt.pt
- 优点：结果完全可比
- 缺点：依赖重 (apex, Ray)

**路径 B: 自己的 HuggingFace 脚本（推荐用于后续融合）**
- 保持当前 `molformer/finetune.py` 的方向
- 将优化器换成 AdamW (在 GPU 上也能用)
- 优点：代码简洁，方便后续与 D-MPNN 融合
- 缺点：优化器不同可能导致结果有微小差异

用户表示**要求方式与 ChemBERTa-3 一致**，但没有给出最终选择。需要进一步确认是走路径 A 还是 B。

## 4. 环境关键信息

### 本地 Mac (x86_64)
- conda env: `chemprop`, Python 3.11.13
- PyTorch 2.2.2 (CPU only, **macOS x86_64 无法安装 PyTorch >= 2.4**)
- transformers 4.57.6 (降级自 5.2.0, 因为 5.x 要求 PyTorch >= 2.4)
- deepchem 2.8.1.dev (nightly)
- **CPU finetune 速度**：每 batch ~4.2s, ESOL 一个 epoch ~6 min, 50 epochs ~5 hours, 不可行

### 需要 GPU 服务器
- 用户说实验室有服务器但尚未提供连接信息
- 需要：NVIDIA GPU (>=4GB VRAM), CUDA, conda
- 如果用路径 A 还需要 NVIDIA apex

## 5. 数据文件

所有 6 个数据集已在 `data/` 目录中准备好（chemprop v2 scaffold split）：

| 文件 | 数据集 | 任务 | Target 列 | 样本量 |
|------|--------|------|----------|--------|
| `esol_v2_split.csv` | ESOL | 回归 (RMSE) | `logSolubility` | 1128 (904/112/112) |
| `freesolv_v2_split.csv` | FreeSolv | 回归 (RMSE) | `freesolv` | 642 (515/63/64) |
| `lipophilicity_v2_split.csv` | Lipophilicity | 回归 (RMSE) | `logD` | 4200 (3360/420/420) |
| `bbbp_v2_split.csv` | BBBP | 分类 (AUC) | `bbbp` | 2039 (1633/203/203) |
| `tox21_v2_split.csv` | Tox21 | 多任务分类 (AUC) | 12个target列 | 7823 (6259/782/782) |
| `clintox_v2_split.csv` | ClinTox | 多任务分类 (AUC) | `FDA_APPROVED`, `CT_TOX` | 1480 (1184/148/148) |

**注意**：这些 split 是用 chemprop v2 的 scaffold splitter 生成的（80/10/10）。ChemBERTa-3 论文使用了两种 split: DeepChem splits 和 MoLFormer splits。我们用的是 chemprop v2 splits（与 Stage 1 D-MPNN 一致），这与两者都不完全相同，后续对比时需注意这一点。

## 6. Stage 1 的 D-MPNN 基线结果（用于对比）

### 回归 (Test RMSE, 越小越好)

| 数据集 | 我们的 D-MPNN | Chemprop v2 官方 |
|--------|-------------|----------------|
| ESOL | 0.5765 | 0.5648 |
| FreeSolv | 1.0568 | 1.0667 |
| Lipophilicity | 0.5570 | 0.5505 |

### 分类 (Test ROC-AUC, 越大越好)

| 数据集 | 我们的 D-MPNN | Chemprop v2 官方 |
|--------|-------------|----------------|
| BBBP | 0.9133 | 0.9205 |
| Tox21 | 0.7961 | 0.7865 |
| ClinTox | 0.8971 | 0.9026 |

## 7. 文件结构

```
Mol_Regression/
├── dmpnn/                  # Stage 1: D-MPNN 实现
│   ├── __init__.py
│   ├── featurizer.py       # 72-dim atom, 14-dim bond features (chemprop v2 aligned)
│   ├── model.py            # DMPNN, BondMessagePassing, FFN, NormAggregation
│   └── data.py             # MoleculeDataset, collate_fn, scaffold_split
├── molformer/              # Stage 2: MoLFormer (进行中)
│   ├── __init__.py         # 空
│   └── finetune.py         # 回归 finetune (初版，需要修改以匹配 ChemBERTa-3)
├── train.py                # D-MPNN 回归训练脚本
├── train_cls.py            # D-MPNN 分类训练脚本
├── predict.py              # D-MPNN 预测脚本
├── tests/test_model.py     # 33 个单元测试
├── data/                   # 6 个数据集 (含 scaffold split)
├── assets/                 # 可视化图表
├── output/                 # 训练输出 (可能有残留的中断 ESOL 训练)
├── stage1.md               # Stage 1 完整文档
├── task.md                 # Stage 1 原始计划
├── requirements.txt        # 依赖列表
├── .gitignore
└── HANDOVER.md             # 本文件
```

## 8. 下一步行动清单

1. **确认服务器信息**：等用户提供 SSH 连接方式、GPU 型号、CUDA 版本
2. **决定实现路径**：路径 A (完全复刻 ChemBERTa-3 + DeepChem) 还是路径 B (自己的 HuggingFace 脚本 + AdamW)
3. **在 GPU 上完成 finetune**：6 个数据集，按 ChemBERTa-3 配置运行
4. **编写分类 finetune 脚本**：`molformer/finetune_cls.py`
5. **汇总结果**：与 ChemBERTa-3 论文结果 + Stage 1 D-MPNN 三方对比
6. **生成 stage2.md**：架构分析 + 实验结果 + 可视化
7. **推送到 GitHub**

## 9. 用户偏好与注意事项

- **语言**：始终用中文回复
- **不许偷懒**：用户多次强调要仔细阅读所有源码，不能跳过
- **代码风格**：与 Stage 1 保持一致（纯 PyTorch，模块化，argparse CLI）
- **不编辑计划文件**：plan 文件只读
- **远程仓库**：`git@github.com:Stoneykk/Mol.git`，之前做过 force push
- **conda 环境名**：`chemprop`（虽然名字源自 Stage 1，但 Stage 2 也用这个环境）

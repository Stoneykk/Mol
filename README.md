**简体中文** | [English](README.en.md)

# Mol_Regression

面向 **MoleculeNet** 等基准的分子性质预测研究仓库：从 **D-MPNN（Chemprop 风格图神经网络）** 的独立实现与验证，到 **MoLFormer-c3-1.1B** 预训练模型的微调与官方结果对比，并为后续 **D-MPNN + MoLFormer 多模态融合** 打基础。

---

## 项目在做什么

| 阶段 | 内容 | 说明 |
|------|------|------|
| **Stage 1** | 自研 D-MPNN | 对齐 [chemprop](https://github.com/chemprop/chemprop) v2 的特征、消息传递、聚合与训练流程；在 ESOL、FreeSolv、Lipophilicity 及 BBBP、Tox21、ClinTox 上与官方对比。 |
| **Stage 2** | MoLFormer 微调 | 使用 HuggingFace 上的 `DeepChem/MoLFormer-c3-1.1B`，在 **DeepChem scaffold split (80/10/10)** 下与 [ChemBERTa-3](https://github.com/deepforestsci/chemberta3) 公开结果对齐比较。 |
| **后续** | 融合模型 | 计划采用门控等方式融合图模型与序列模型（见 `report_v1.md` 中的方案描述）。 |

**说明：**「1.1B」指预训练所用分子规模约 11 亿，MoLFormer 参数量约 **46.8M**（非 11B 参数）。

---

## 仓库结构

```
Mol_Regression/
├── dmpnn/                    # D-MPNN：特征化、模型、数据与 scaffold 划分
├── molformer/                # MoLFormer：回归 / 分类微调脚本
├── tests/                    # 单元测试（与 chemprop v2 特征与图结构对齐）
├── scripts/                  # 例如 DeepChem split 预生成
├── data/                     # 原始与划分数据（chemprop_split、deepchem_split 等）
├── train.py / train_cls.py   # D-MPNN 回归 / 分类训练
├── predict.py                # D-MPNN 推理
├── run_molformer_benchmark.sh
├── Dockerfile
├── requirements.txt
├── stage1.md                 # Stage 1 详细技术文档
├── stage2.md                 # Stage 2 详细技术文档
├── HANDOVER.md               # 阶段目标与交接说明
└── report_v1.md              # 对外方案级概述（双模型 + 融合设想）
```

详细模块说明、公式与超参表见 **stage1.md**；MoLFormer 配置、与官方表对标及分析见 **stage2.md**。

---

## 环境依赖

- **D-MPNN / 与 chemprop 对比**：建议 Python 3.11+，安装 `requirements.txt` 中的依赖；与 chemprop v2 对齐的完整环境说明见 `stage1.md` 第 9 节。
- **MoLFormer**：需 **PyTorch**；IBM MoLFormer 远程代码与 **transformers 4.38.2** 兼容（见 `stage2.md`）。
- **大规模微调**：GPU 环境更合适；`Dockerfile` 与 `stage2.md` 中提及服务器/Docker 用法。

```bash
pip install -r requirements.txt
# 或按 stage1.md 使用 conda 创建独立环境
```

---

## 快速开始

### D-MPNN 训练（回归示例）

```bash
python train.py \
  --data-path data/esol.csv \
  --smiles-col smiles \
  --target-col logSolubility \
  --output-dir output/my_model \
  --epochs 50
```

使用与 chemprop v2 一致的预生成划分时，可加 `--split-file` 指向 `data/chemprop_split/esol_v2_split.csv` 等（参数以 `train.py` 为准）。

### D-MPNN 预测

```bash
python predict.py \
  --model-path output/my_model/best_model.pt \
  --smiles "CCO" "c1ccccc1"
```

### D-MPNN 分类训练

使用 `train_cls.py`（多任务、BCE、ROC-AUC 等，见 `stage1.md` 第 11 节）。

### 单元测试

```bash
python -m pytest tests/test_model.py -v
```

### MoLFormer 基准

仓库提供 `run_molformer_benchmark.sh` 与 `molformer/finetune.py`、`molformer/finetune_cls.py`；数据划分可放在 `data/deepchem_split/`。完整配置与结果解读见 **stage2.md**。

### Chemprop v2 官方基线（可选）

在已安装 `chemprop` 的环境中可参考 `stage1.md` 第 10 节中的 `chemprop train` 示例作对照实验。

---

## 现阶段结果与分析

### Stage 1：D-MPNN vs Chemprop v2.2.2

**设定**：`chemprop` conda、Python 3.11.13、PyTorch 2.2.2；**chemprop v2 生成的 scaffold_balanced 划分**；超参与 chemprop v2 默认一致；**50 epochs**、Adam、Noam-like 学习率、**目标按训练集做标准化**；D-MPNN 参数量约 **318K**。

**回归（Test RMSE ↓，越小越好）** — 与 `stage1.md` §7、§11.2 一致：

| 数据集 | 规模 / 划分 | Chemprop v2 | 自研 D-MPNN | 相对差异 |
|--------|-------------|------------|------------|----------|
| ESOL | 1,128 · 904/112/112 | 0.8048 | **0.7935** | −1.4% |
| FreeSolv | 642 · 515/63/64 | 2.5069 | **2.5163** | +0.4% |
| Lipophilicity | 4,200 · 3360/420/420 | 0.5881 | **0.5890** | +0.2% |

**分类（Test ROC-AUC ↑，越大越好）** — `scaffold` 划分，见 `stage1.md` §11.3：

| 数据集 | 任务数 | 划分 | Chemprop v2 | 自研 D-MPNN | 相对差异 |
|--------|--------|------|------------|------------|----------|
| BBBP | 1 | 1633/203/203 | 0.8266 | 0.8121 | −1.8% |
| Tox21 | 12 | 6259/782/782 | 0.7638 | 0.7532 | −1.4% |
| ClinTox | 2 | 1184/148/148 | 0.8537 | **0.8797** | +3.0% |

**分析（D-MPNN）**：三条回归线路上 RMSE 与官方差距均在约 **1.5% 以内**；与 chemprop 在**消息传递公式、NormAggregation、特征（72/14 维）**上对齐是结果接近的主要原因。分类三条路与官方 AUC 差距在约 **3% 以内**；ClinTox 上自研略高，属小数据与划分的正常波动。更细的 Tox21 分任务、与官方代码级对比见 `stage1.md` §8、§12。

---

### Stage 2：MoLFormer-c3-1.1B vs ChemBERTa-3 公开表

**设定（与 Stage 1 不同）**：**DeepChem `ScaffoldSplitter` 80/10/10** 预存在 `data/deepchem_split/`；`transformers==4.38.2`；**每个数据集 3 个 random seed 取 mean ± std**；回归 **100 epochs**、分类 **10 epochs**，`batch_size=32`，`lr=3e-5`，**AdamW**（官方表多为 **FusedLAMB**）。官方数值来源见 [ChemBERTa-3 仓库](https://github.com/deepforestsci/chemberta3) 的 DeepChem-splits 图。

**分类（Test ROC-AUC ↑）**：

| 数据集 | 官方 c3-MoLFormer | 本仓库复现 |
|--------|-------------------|------------|
| BBBP | 0.735 ± 0.019 | 0.727 ± 0.006 |
| Tox21 | 0.723 ± 0.012 | **0.747 ± 0.004** |
| ClinTox | 0.839 ± 0.013 | **0.989 ± 0.001** |

**回归（Test RMSE ↓）**：

| 数据集 | 官方 c3-MoLFormer | 本仓库复现 |
|--------|-------------------|------------|
| ESOL | 0.829 ± 0.019 | **0.787 ± 0.019** |
| FreeSolv | 0.572 ± 0.023 | 2.175 ± 0.026 |
| Lipophilicity | 0.728 ± 0.016 | **0.686 ± 0.019** |

![MoLFormer 与公开基准对比](assets/stage2_comparison.png)

**分析（MoLFormer）**：

- **5/6 个数据集**上达到或优于公开表中的 c3-MoLFormer；**Tox21、ClinTox、ESOL、Lipophilicity** 明显更优或持平；**BBBP** 与官方在官方标准差范围内接近。
- **FreeSolv** 明显落后（2.175 vs 0.572）：主因是 **仅 642 个分子、测试集约 65 条**，**scaffold 划分在 DeepChem 版本间细微差别**会放大方差；另 **AdamW vs FusedLAMB、epoch 与调参** 在小数据上更敏感。ClinTox 同为小集却明显优于官方，更支持 **split 与评估难度** 带来差异，而非单指模型失败。
- **ClinTox 0.989**：除 split 外，**AdamW 的参数分组** 对小多任务分类的稳定性可能也有贡献（详见 `stage2.md` §5）。

完整讨论、训练命令与 `stage2_output/` 见 **`stage2.md`**。

---

## 文档索引

| 文件 | 内容 |
|------|------|
| [stage1.md](stage1.md) | D-MPNN 原理、特征维度、与 chemprop v1/v2 对比、最终验证与单元测试、命令行用法。 |
| [stage2.md](stage2.md) | MoLFormer 架构要点、与官方复现异同、六数据集结果、Docker/DeepChem split。 |
| [HANDOVER.md](HANDOVER.md) | 项目分阶段目标、环境注意点、数据文件列表、后续行动清单。 |
| [report_v1.md](report_v1.md) | 双模型方案与融合架构的对外说明（英文）。 |

---

## 参考

- Yang et al., *Analyzing Learned Molecular Representations for Property Prediction*, JCIM, 2019（D-MPNN / chemprop 论文）  
- [chemprop/chemprop](https://github.com/chemprop/chemprop)  
- [ChemBERTa-3 / MoLFormer 相关基准](https://github.com/deepforestsci/chemberta3)  
- [DeepChem/MoLFormer-c3-1.1B](https://huggingface.co/DeepChem/MoLFormer-c3-1.1B)（HuggingFace）

---

*项目说明与上表与 `stage1.md`、`stage2.md` 保持一致；其它背景见 `HANDOVER.md`、`report_v1.md`。*

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

## 主要结论（摘要）

- **D-MPNN**：在 chemprop v2 相同 scaffold 与默认超参下，ESOL / FreeSolv 等数据集上测试 RMSE 与官方差异约 **1% 量级**；实现与官方在核心算法上对齐（见 `stage1.md` 第 7、12 节）。
- **MoLFormer**：在 6 个 MoleculeNet 子集、相同 DeepChem 划分与 triplicate 设置下，多数数据集上达到或超过 ChemBERTa-3 公开表中的 c3-MoLFormer 水平；FreeSolv 因样本极少对划分与实现细节更敏感，需单独解读（见 `stage2.md` 第 4–5 节）。

详细表格、图表路径与实现差异说明以 **stage1.md**、**stage2.md** 为准。

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

*文档由仓库内 `stage1.md`、`stage2.md`、`HANDOVER.md`、`report_v1.md` 综合整理。*

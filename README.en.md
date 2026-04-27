[简体中文](README.md) | **English**

# Mol_Regression

A research repository for **molecular property prediction** on benchmarks such as **MoleculeNet**: an independent **D-MPNN** (Chemprop-style graph neural network) implementation with validation, **MoLFormer-c3-1.1B** fine-tuning against published baselines, and groundwork for a future **D-MPNN + MoLFormer** multimodal fusion.

---

## What this project is

| Stage | Focus | Description |
|------|--------|-------------|
| **Stage 1** | D-MPNN from scratch | Align featurization, message passing, aggregation, and training with [chemprop](https://github.com/chemprop/chemprop) v2; compare on ESOL, FreeSolv, Lipophilicity, BBBP, Tox21, and ClinTox. |
| **Stage 2** | MoLFormer fine-tuning | Use `DeepChem/MoLFormer-c3-1.1B` on HuggingFace; compare with [ChemBERTa-3](https://github.com/deepforestsci/chemberta3) numbers under a **DeepChem scaffold split (80/10/10)**. |
| **Next** | Fusion model | Planned gated (or similar) fusion of graph and sequence encoders; see `report_v1.md` for a high-level proposal. |

**Note:** “1.1B” refers to ~1.1B molecules used in pretraining; MoLFormer has about **46.8M** parameters (not 1.1B parameters).

---

## Repository layout

```
Mol_Regression/
├── dmpnn/                    # D-MPNN: featurization, model, data, scaffold splits
├── molformer/                # MoLFormer: regression / classification fine-tune scripts
├── tests/                    # Unit tests (aligned with chemprop v2 features and graphs)
├── scripts/                  # e.g. pre-generated DeepChem splits
├── data/                     # Raw and split data (chemprop_split, deepchem_split, etc.)
├── train.py / train_cls.py   # D-MPNN regression / classification
├── predict.py                # D-MPNN inference
├── run_molformer_benchmark.sh
├── Dockerfile
├── requirements.txt
├── stage1.md                 # Stage 1 technical write-up
├── stage2.md                 # Stage 2 technical write-up
├── HANDOVER.md               # Phased goals and handover notes
└── report_v1.md              # External-facing solution overview (dual models + fusion)
```

Module details, equations, and hyperparameter tables: **stage1.md**. MoLFormer settings, official comparisons, and analysis: **stage2.md**.

---

## Requirements

- **D-MPNN / chemprop baselines:** Python 3.11+ recommended; install from `requirements.txt`. Full environment notes aligned with chemprop v2: `stage1.md`, Section 9.
- **MoLFormer:** **PyTorch** required; IBM’s remote code is compatible with **transformers 4.38.2** (see `stage2.md`).
- **Large-scale fine-tuning:** A **GPU** is recommended; see `Dockerfile` and `stage2.md` for server / Docker notes.

```bash
pip install -r requirements.txt
# or use conda as described in stage1.md
```

---

## Quick start

### D-MPNN training (regression)

```bash
python train.py \
  --data-path data/esol.csv \
  --smiles-col smiles \
  --target-col logSolubility \
  --output-dir output/my_model \
  --epochs 50
```

To use a pre-generated split matching chemprop v2, pass `--split-file` (e.g. `data/chemprop_split/esol_v2_split.csv`); see `train.py` for available flags.

### D-MPNN prediction

```bash
python predict.py \
  --model-path output/my_model/best_model.pt \
  --smiles "CCO" "c1ccccc1"
```

### D-MPNN classification

Use `train_cls.py` (multi-task, BCE, ROC-AUC, etc.; see `stage1.md`, Section 11).

### Unit tests

```bash
python -m pytest tests/test_model.py -v
```

### MoLFormer benchmark

The repo includes `run_molformer_benchmark.sh` and `molformer/finetune.py`, `molformer/finetune_cls.py`; split files can live under `data/deepchem_split/`. See **stage2.md** for configuration and how to read results.

### Chemprop v2 official baseline (optional)

If `chemprop` is installed, see the `chemprop train` example in `stage1.md`, Section 10.

---

## Current results and analysis

### Stage 1: D-MPNN vs chemprop v2.2.2

**Protocol:** `chemprop` env, Python 3.11.13, PyTorch 2.2.2; **scaffold_balanced split from chemprop v2**; default chemprop v2 hyperparameters; **50 epochs**, Adam, Noam-like LR, **target standardization (train mean/std)**; D-MPNN has ~**318K** parameters.

**Regression (test RMSE ↓)** — as in `stage1.md` §7 and §11.2:

| Dataset | Size / split | Chemprop v2 | Our D-MPNN | Δ |
|---------|--------------|------------|------------|---|
| ESOL | 1,128 · 904/112/112 | 0.8048 | **0.7935** | −1.4% |
| FreeSolv | 642 · 515/63/64 | 2.5069 | **2.5163** | +0.4% |
| Lipophilicity | 4,200 · 3360/420/420 | 0.5881 | **0.5890** | +0.2% |

**Classification (test ROC-AUC ↑)** — `stage1.md` §11.3:

| Dataset | #tasks | Split | Chemprop v2 | Our D-MPNN | Δ |
|---------|--------|-------|------------|------------|---|
| BBBP | 1 | 1633/203/203 | 0.8266 | 0.8121 | −1.8% |
| Tox21 | 12 | 6259/782/782 | 0.7638 | 0.7532 | −1.4% |
| ClinTox | 2 | 1184/148/148 | 0.8537 | **0.8797** | +3.0% |

**Takeaway (D-MPNN):** RMSE is within about **1.5%** on the three regression sets; AUC is within about **3%** on the three classification sets. This matches full alignment of **message passing, NormAggregation, and 72/14-d features** with chemprop v2. Per-endpoint Tox21 tasks and a code-level comparison to the official stack are in `stage1.md` §8 and §12.

---

### Stage 2: MoLFormer-c3-1.1B vs published ChemBERTa-3

**Note:** This line uses a **DeepChem `ScaffoldSplitter` 80/10/10** (files under `data/deepchem_split/`), not the same file-wise split as Stage 1. **`transformers==4.38.2`**; **3 seeds (mean ± std)**; reg **100** / cls **10** epochs, `batch_size=32`, `lr=3e-5`, **AdamW** (published table often uses **FusedLAMB**). Reference numbers: ChemBERTa-3 DeepChem-splits material.

**Classification (test ROC-AUC ↑):**

| Dataset | Official c3-MoLFormer | This repo |
|--------|------------------------|-----------|
| BBBP | 0.735 ± 0.019 | 0.727 ± 0.006 |
| Tox21 | 0.723 ± 0.012 | **0.747 ± 0.004** |
| ClinTox | 0.839 ± 0.013 | **0.989 ± 0.001** |

**Regression (test RMSE ↓):**

| Dataset | Official c3-MoLFormer | This repo |
|--------|------------------------|-----------|
| ESOL | 0.829 ± 0.019 | **0.787 ± 0.019** |
| FreeSolv | 0.572 ± 0.023 | 2.175 ± 0.026 |
| Lipophilicity | 0.728 ± 0.016 | **0.686 ± 0.019** |

![MoLFormer vs published benchmark](assets/stage2_comparison.png)

**Takeaway (MoLFormer):** **5/6** sets match or beat the table; **Tox21, ClinTox, ESOL, Lipophilicity** are clearly at or above reference; **BBBP** is within the published std. **FreeSolv** lags (2.175 vs 0.572): **642 molecules / ~65 test points** make scaffold splits and toolchain versions a large lever; **AdamW vs FusedLAMB** and budget (epochs, tuning) also matter on tiny data. Strong **ClinTox** with weak **FreeSolv** is consistent with **split difficulty** varying across small sets, not a single “model is broken” story. **AdamW param groups** may also help small classification tasks (see `stage2.md` §5).

For full detail, see **`stage2.md`** and `stage2_output/`.

---

## Documentation index

| File | Contents |
|------|----------|
| [stage1.md](stage1.md) | D-MPNN theory, feature sizes, v1/v2 chemprop notes, validation, unit tests, CLI. |
| [stage2.md](stage2.md) | MoLFormer design, vs-official setup, six-dataset results, Docker / DeepChem splits. |
| [HANDOVER.md](HANDOVER.md) | Phased roadmap, environment notes, data file list, follow-up checklist. |
| [report_v1.md](report_v1.md) | Client-facing solution overview in English (dual models + fusion idea). |

---

## References

- Yang et al., *Analyzing Learned Molecular Representations for Property Prediction*, JCIM, 2019 (D-MPNN / chemprop)  
- [chemprop/chemprop](https://github.com/chemprop/chemprop)  
- [ChemBERTa-3 / related MoLFormer benchmarks](https://github.com/deepforestsci/chemberta3)  
- [DeepChem/MoLFormer-c3-1.1B](https://huggingface.co/DeepChem/MoLFormer-c3-1.1B) (HuggingFace)

---

*Narrative and tables above follow `stage1.md` and `stage2.md`; see also `HANDOVER.md` and `report_v1.md`. Chinese version: [README](README.md).*

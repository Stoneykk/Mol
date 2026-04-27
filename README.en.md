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

## Key results (summary)

- **D-MPNN:** With the same scaffold and default hyperparameters as chemprop v2, test RMSE on ESOL / FreeSolv and related sets is within about **1%** of the official run; the core algorithm matches the reference (see `stage1.md`, Sections 7 and 12).
- **MoLFormer:** On six MoleculeNet subsets with the same DeepChem split and triplicate runs, results mostly match or beat the c3-MoLFormer row in the ChemBERTa-3 table; **FreeSolv** is very small and sensitive to split/version details—interpret separately (see `stage2.md`, Sections 4–5).

Full tables, figure paths, and implementation caveats: **stage1.md** and **stage2.md**.

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

*This file is a companion to the Chinese [README](README.md), synthesized from `stage1.md`, `stage2.md`, `HANDOVER.md`, and `report_v1.md`.*

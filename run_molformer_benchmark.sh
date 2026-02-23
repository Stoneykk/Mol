#!/bin/bash
# ==============================================================================
# MoLFormer-c3-1.1B Benchmark — 6 MoleculeNet datasets × 3 seeds
#
# Uses DeepChem scaffold splits (80/10/10) to match ChemBERTa-3 official setup.
#
# Usage:
#   chmod +x run_molformer_benchmark.sh
#   ./run_molformer_benchmark.sh
#
# Expected GPU time: ~2-3 hours total
#   - Regression (3 datasets × 100 epochs × 3 seeds)
#   - Classification (3 datasets × 10 epochs × 3 seeds)
# ==============================================================================

set -e

N_RUNS=3
SEED=0
BATCH_SIZE=32
OUTPUT_BASE="output/molformer_benchmark"

echo "=============================================="
echo " MoLFormer-c3-1.1B Benchmark (DeepChem split)"
echo " Runs per dataset: ${N_RUNS}"
echo " Base seed: ${SEED}"
echo " Output: ${OUTPUT_BASE}/"
echo "=============================================="
echo ""

# Check GPU availability
python -c "import torch; g=torch.cuda.is_available(); print(f'CUDA available: {g}'); g or print('WARNING: running on CPU, will be very slow!')"
echo ""

# ==============================================================================
# Regression tasks (100 epochs, metric: RMSE)
# ==============================================================================

echo ">>> [1/6] ESOL (regression)"
python molformer/finetune.py \
    --split-file data/deepchem_split/esol_dc_split.csv \
    --target-col logSolubility \
    --output-dir "${OUTPUT_BASE}/esol" \
    --epochs 100 \
    --batch-size "${BATCH_SIZE}" \
    --n-runs "${N_RUNS}" \
    --seed "${SEED}"

echo ""
echo ">>> [2/6] FreeSolv (regression)"
python molformer/finetune.py \
    --split-file data/deepchem_split/freesolv_dc_split.csv \
    --target-col freesolv \
    --output-dir "${OUTPUT_BASE}/freesolv" \
    --epochs 100 \
    --batch-size "${BATCH_SIZE}" \
    --n-runs "${N_RUNS}" \
    --seed "${SEED}"

echo ""
echo ">>> [3/6] Lipophilicity (regression)"
python molformer/finetune.py \
    --split-file data/deepchem_split/lipophilicity_dc_split.csv \
    --target-col logD \
    --output-dir "${OUTPUT_BASE}/lipophilicity" \
    --epochs 100 \
    --batch-size "${BATCH_SIZE}" \
    --n-runs "${N_RUNS}" \
    --seed "${SEED}"

# ==============================================================================
# Classification tasks (10 epochs, metric: ROC-AUC)
# ==============================================================================

echo ""
echo ">>> [4/6] BBBP (classification, 1 task)"
python molformer/finetune_cls.py \
    --split-file data/deepchem_split/bbbp_dc_split.csv \
    --target-cols bbbp \
    --output-dir "${OUTPUT_BASE}/bbbp" \
    --epochs 10 \
    --batch-size "${BATCH_SIZE}" \
    --n-runs "${N_RUNS}" \
    --seed "${SEED}"

echo ""
echo ">>> [5/6] Tox21 (classification, 12 tasks)"
python molformer/finetune_cls.py \
    --split-file data/deepchem_split/tox21_dc_split.csv \
    --target-cols NR-AR NR-AR-LBD NR-AhR NR-Aromatase NR-ER NR-ER-LBD \
                  NR-PPAR-gamma SR-ARE SR-ATAD5 SR-HSE SR-MMP SR-p53 \
    --output-dir "${OUTPUT_BASE}/tox21" \
    --epochs 10 \
    --batch-size "${BATCH_SIZE}" \
    --n-runs "${N_RUNS}" \
    --seed "${SEED}"

echo ""
echo ">>> [6/6] ClinTox (classification, 2 tasks)"
python molformer/finetune_cls.py \
    --split-file data/deepchem_split/clintox_dc_split.csv \
    --target-cols FDA_APPROVED CT_TOX \
    --output-dir "${OUTPUT_BASE}/clintox" \
    --epochs 10 \
    --batch-size "${BATCH_SIZE}" \
    --n-runs "${N_RUNS}" \
    --seed "${SEED}"

# ==============================================================================
# Final summary
# ==============================================================================

echo ""
echo "=============================================="
echo " All 6 benchmarks complete!"
echo " Results saved to: ${OUTPUT_BASE}/"
echo "=============================================="

python << PYEOF
import json, os

base = "$OUTPUT_BASE"
print()
print("=" * 60)
print("  FINAL RESULTS SUMMARY")
print("=" * 60)
print()
print("  Regression (Test RMSE, lower is better):")
print("  -----------------------------------------")
for ds in ["esol", "freesolv", "lipophilicity"]:
    f = os.path.join(base, ds, "summary.json")
    if os.path.exists(f):
        s = json.load(open(f))
        m, sd = s["test_rmse_mean"], s["test_rmse_std"]
        print(f"    {ds:15s}  {m:.4f} +/- {sd:.4f}")
    else:
        print(f"    {ds:15s}  (not found)")
print()
print("  Classification (Test ROC-AUC, higher is better):")
print("  -------------------------------------------------")
for ds in ["bbbp", "tox21", "clintox"]:
    f = os.path.join(base, ds, "summary.json")
    if os.path.exists(f):
        s = json.load(open(f))
        m, sd = s["test_auc_mean"], s["test_auc_std"]
        print(f"    {ds:15s}  {m:.4f} +/- {sd:.4f}")
    else:
        print(f"    {ds:15s}  (not found)")
print()
print("=" * 60)
PYEOF

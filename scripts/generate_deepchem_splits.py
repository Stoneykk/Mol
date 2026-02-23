"""Generate DeepChem scaffold splits (80/10/10) for 6 MoleculeNet datasets.

Uses deepchem.splits.ScaffoldSplitter — the same splitter used by ChemBERTa-3.
Reads SMILES + targets from existing *_v2_split.csv files (ignoring their split column),
applies DeepChem scaffold split, and writes *_dc_split.csv files.

Usage:
    pip install deepchem
    python scripts/generate_deepchem_splits.py
"""

import os
import sys

import numpy as np
import pandas as pd

try:
    import deepchem as dc
    from deepchem.splits import ScaffoldSplitter
except ImportError:
    print("ERROR: deepchem is required. Install with: pip install deepchem")
    sys.exit(1)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

DATASETS = {
    "esol": {
        "src": "esol_v2_split.csv",
        "out": "esol_dc_split.csv",
        "smiles_col": "smiles",
        "target_cols": ["logSolubility"],
    },
    "freesolv": {
        "src": "freesolv_v2_split.csv",
        "out": "freesolv_dc_split.csv",
        "smiles_col": "smiles",
        "target_cols": ["freesolv"],
    },
    "lipophilicity": {
        "src": "lipophilicity_v2_split.csv",
        "out": "lipophilicity_dc_split.csv",
        "smiles_col": "smiles",
        "target_cols": ["logD"],
    },
    "bbbp": {
        "src": "bbbp_v2_split.csv",
        "out": "bbbp_dc_split.csv",
        "smiles_col": "smiles",
        "target_cols": ["bbbp"],
    },
    "tox21": {
        "src": "tox21_v2_split.csv",
        "out": "tox21_dc_split.csv",
        "smiles_col": "smiles",
        "target_cols": [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
            "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
            "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
        ],
    },
    "clintox": {
        "src": "clintox_v2_split.csv",
        "out": "clintox_dc_split.csv",
        "smiles_col": "smiles",
        "target_cols": ["FDA_APPROVED", "CT_TOX"],
    },
}


def generate_split(name: str, cfg: dict):
    src_path = os.path.join(DATA_DIR, cfg["src"])
    out_path = os.path.join(DATA_DIR, cfg["out"])

    df = pd.read_csv(src_path)
    cols = [cfg["smiles_col"]] + cfg["target_cols"]
    df = df[cols].copy()

    smiles = df[cfg["smiles_col"]].values
    y = df[cfg["target_cols"]].values.astype(np.float64)

    dataset = dc.data.NumpyDataset(
        X=np.zeros((len(smiles), 1)),
        y=y,
        ids=smiles,
    )

    splitter = ScaffoldSplitter()
    train_ds, val_ds, test_ds = splitter.train_valid_test_split(
        dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1,
    )

    split_map = {}
    for smi in train_ds.ids:
        split_map[smi] = "train"
    for smi in val_ds.ids:
        split_map[smi] = "val"
    for smi in test_ds.ids:
        split_map[smi] = "test"

    df["split"] = df[cfg["smiles_col"]].map(split_map)
    assert df["split"].notna().all(), f"Some SMILES not assigned a split in {name}"

    df.to_csv(out_path, index=False)

    n_train = (df["split"] == "train").sum()
    n_val = (df["split"] == "val").sum()
    n_test = (df["split"] == "test").sum()
    print(f"  {name:15s}  total={len(df):5d}  train={n_train}  val={n_val}  test={n_test}  -> {cfg['out']}")


def main():
    print("Generating DeepChem scaffold splits (80/10/10)...")
    print(f"Data directory: {DATA_DIR}")
    print()

    for name, cfg in DATASETS.items():
        generate_split(name, cfg)

    print()
    print("Done! All *_dc_split.csv files generated.")


if __name__ == "__main__":
    main()

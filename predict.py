"""Predict molecular properties from SMILES using a trained D-MPNN model."""

import argparse
import sys

import torch
import numpy as np

from dmpnn.model import DMPNN
from dmpnn.featurizer import MolGraphFeaturizer, BatchMolGraph


def predict(model_path: str, smiles_list: list, d_h: int = 300, depth: int = 3) -> np.ndarray:
    """Load a trained D-MPNN model and predict properties for a list of SMILES."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DMPNN(d_h=d_h, depth=depth, n_tasks=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    feat = MolGraphFeaturizer()
    mol_graphs = []
    valid_indices = []

    for i, smi in enumerate(smiles_list):
        mg = feat(smi)
        if mg is not None:
            mol_graphs.append(mg)
            valid_indices.append(i)
        else:
            print(f"Warning: Could not parse SMILES '{smi}'", file=sys.stderr)

    if not mol_graphs:
        return np.array([])

    bmg = BatchMolGraph.from_mol_graphs(mol_graphs)
    bmg.to(device)

    with torch.no_grad():
        preds = model(bmg).cpu().numpy()

    results = np.full((len(smiles_list), preds.shape[1]), np.nan)
    for i, idx in enumerate(valid_indices):
        results[idx] = preds[i]

    return results


def main():
    parser = argparse.ArgumentParser(description="Predict with D-MPNN")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--smiles", type=str, nargs="+", help="SMILES strings to predict")
    parser.add_argument("--input-file", type=str, help="CSV file with SMILES column")
    parser.add_argument("--smiles-col", type=str, default="smiles")
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--hidden-dim", type=int, default=300)
    parser.add_argument("--depth", type=int, default=3)

    args = parser.parse_args()

    if args.smiles:
        smiles_list = args.smiles
    elif args.input_file:
        import pandas as pd
        df = pd.read_csv(args.input_file)
        smiles_list = df[args.smiles_col].tolist()
    else:
        parser.error("Provide either --smiles or --input-file")
        return

    preds = predict(args.model_path, smiles_list, args.hidden_dim, args.depth)

    if args.output_file:
        import pandas as pd
        df_out = pd.DataFrame({"smiles": smiles_list, "prediction": preds.flatten()})
        df_out.to_csv(args.output_file, index=False)
        print(f"Predictions saved to {args.output_file}")
    else:
        for smi, pred in zip(smiles_list, preds.flatten()):
            print(f"{smi}\t{pred:.4f}")


if __name__ == "__main__":
    main()

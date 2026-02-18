"""Dataset and DataLoader utilities for D-MPNN training."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from dmpnn.featurizer import MolGraph, BatchMolGraph, MolGraphFeaturizer


class MoleculeDataset(Dataset):
    """Dataset of (MolGraph, target) pairs."""

    def __init__(
        self,
        smiles: List[str],
        targets: np.ndarray,
        featurizer: Optional[MolGraphFeaturizer] = None,
    ):
        self.featurizer = featurizer or MolGraphFeaturizer()
        self.mol_graphs: List[MolGraph] = []
        self.targets_list: List[np.ndarray] = []

        for smi, y in zip(smiles, targets):
            mg = self.featurizer(smi)
            if mg is not None:
                self.mol_graphs.append(mg)
                self.targets_list.append(y)

        self.targets_arr = np.array(self.targets_list, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.mol_graphs)

    def __getitem__(self, idx: int) -> Tuple[MolGraph, np.ndarray]:
        return self.mol_graphs[idx], self.targets_arr[idx]


def collate_fn(batch: List[Tuple[MolGraph, np.ndarray]]) -> Tuple[BatchMolGraph, torch.Tensor]:
    """Collate function for DataLoader."""
    mgs, targets = zip(*batch)
    bmg = BatchMolGraph.from_mol_graphs(mgs)
    targets_tensor = torch.tensor(np.array(targets), dtype=torch.float32)
    if targets_tensor.dim() == 1:
        targets_tensor = targets_tensor.unsqueeze(1)
    return bmg, targets_tensor


def build_dataloader(
    dataset: MoleculeDataset,
    batch_size: int = 50,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def scaffold_split(
    smiles: List[str],
    targets: np.ndarray,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
) -> Tuple[
    Tuple[List[str], np.ndarray],
    Tuple[List[str], np.ndarray],
    Tuple[List[str], np.ndarray],
]:
    """Split data by Murcko scaffold to avoid data leakage."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
    from collections import defaultdict

    scaffold_to_indices = defaultdict(list)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            scaffold_to_indices[scaffold].append(i)
        else:
            scaffold_to_indices[""].append(i)

    rng = np.random.RandomState(seed)
    scaffold_sets = list(scaffold_to_indices.values())
    rng.shuffle(scaffold_sets)

    n = len(smiles)
    train_cutoff = int(train_frac * n)
    val_cutoff = train_cutoff + int(val_frac * n)

    train_idx, val_idx, test_idx = [], [], []
    for group in scaffold_sets:
        if len(train_idx) + len(group) <= train_cutoff:
            train_idx.extend(group)
        elif len(train_idx) + len(val_idx) + len(group) <= val_cutoff:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    if not val_idx:
        val_idx = test_idx[:len(test_idx) // 2]
        test_idx = test_idx[len(test_idx) // 2:]

    return (
        ([smiles[i] for i in train_idx], targets[train_idx]),
        ([smiles[i] for i in val_idx], targets[val_idx]),
        ([smiles[i] for i in test_idx], targets[test_idx]),
    )

"""
Molecular graph featurizer for D-MPNN.

Converts SMILES strings into directed molecular graphs with atom and bond features,
aligned with chemprop v2's featurization scheme (72-dim atom, 14-dim bond).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType, BondType, BondStereo


ATOMIC_NUM_CHOICES = list(range(1, 37)) + [53]
DEGREE_CHOICES = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGE_CHOICES = [-1, -2, 1, 2, 0]
CHIRAL_TAG_CHOICES = [0, 1, 2, 3]
NUM_HS_CHOICES = [0, 1, 2, 3, 4]
HYBRIDIZATION_CHOICES = [
    HybridizationType.S,
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP2D,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2,
]
STEREO_CHOICES = [0, 1, 2, 3, 4, 5]

ATOM_FDIM = (
    len(ATOMIC_NUM_CHOICES) + 1   # +1 for unknown
    + len(DEGREE_CHOICES) + 1
    + len(FORMAL_CHARGE_CHOICES) + 1
    + len(CHIRAL_TAG_CHOICES) + 1
    + len(NUM_HS_CHOICES) + 1
    + len(HYBRIDIZATION_CHOICES) + 1
    + 1   # aromaticity
    + 1   # scaled mass
)  # = 72

BOND_FDIM = (
    1     # is_null
    + 4   # bond type (single, double, triple, aromatic)
    + 1   # conjugated
    + 1   # in ring
    + len(STEREO_CHOICES) + 1  # stereo + unknown
)  # = 14

_ATOMIC_NUM_MAP = {num: i for i, num in enumerate(ATOMIC_NUM_CHOICES)}


def _onek_encoding_unk(value, choices: list) -> List[int]:
    """One-hot encoding with an extra "unknown" bin at the end."""
    encoding = [0] * (len(choices) + 1)
    try:
        idx = choices.index(value)
    except ValueError:
        idx = len(choices)
    encoding[idx] = 1
    return encoding


def _onek_encoding_unk_map(value, choice_map: dict, total_len: int) -> List[int]:
    """One-hot encoding using a pre-built dict for O(1) lookup."""
    encoding = [0] * total_len
    idx = choice_map.get(value, len(choice_map))
    encoding[idx] = 1
    return encoding


def atom_features(atom: Optional[Chem.rdchem.Atom]) -> List[Union[int, float]]:
    """Compute atom feature vector (72 dims), matching chemprop v2."""
    if atom is None:
        return [0] * ATOM_FDIM

    features = (
        _onek_encoding_unk_map(atom.GetAtomicNum(), _ATOMIC_NUM_MAP, len(ATOMIC_NUM_CHOICES) + 1)
        + _onek_encoding_unk(atom.GetTotalDegree(), DEGREE_CHOICES)
        + _onek_encoding_unk(atom.GetFormalCharge(), FORMAL_CHARGE_CHOICES)
        + _onek_encoding_unk(int(atom.GetChiralTag()), CHIRAL_TAG_CHOICES)
        + _onek_encoding_unk(int(atom.GetTotalNumHs()), NUM_HS_CHOICES)
        + _onek_encoding_unk(int(atom.GetHybridization()), HYBRIDIZATION_CHOICES)
        + [1 if atom.GetIsAromatic() else 0]
        + [atom.GetMass() * 0.01]
    )
    return features


def bond_features(bond: Optional[Chem.rdchem.Bond]) -> List[Union[int, float]]:
    """Compute bond feature vector (14 dims), matching chemprop v2."""
    if bond is None:
        return [1] + [0] * (BOND_FDIM - 1)

    bt = bond.GetBondType()
    features = [
        0,
        bt == BondType.SINGLE,
        bt == BondType.DOUBLE,
        bt == BondType.TRIPLE,
        bt == BondType.AROMATIC,
        bond.GetIsConjugated() if bt is not None else 0,
        bond.IsInRing() if bt is not None else 0,
    ]
    features += _onek_encoding_unk(int(bond.GetStereo()), STEREO_CHOICES)
    return features


@dataclass(frozen=True)
class MolGraph:
    """A directed molecular graph representation for D-MPNN."""

    V: np.ndarray
    """Atom feature matrix, shape (n_atoms, atom_fdim)"""
    E: np.ndarray
    """Bond feature matrix, shape (n_bonds*2, bond_fdim), each bond gives 2 directed edges"""
    edge_index: np.ndarray
    """Edge list in COO format, shape (2, n_bonds*2)"""
    rev_edge_index: np.ndarray
    """Maps each directed edge to its reverse, shape (n_bonds*2,)"""
    n_atoms: int = field(init=False)
    n_edges: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "n_atoms", self.V.shape[0])
        object.__setattr__(self, "n_edges", self.E.shape[0])


@dataclass
class BatchMolGraph:
    """Batched molecular graphs for efficient processing."""

    V: Tensor
    E: Tensor
    edge_index: Tensor
    rev_edge_index: Tensor
    batch: Tensor
    n_mols: int

    @classmethod
    def from_mol_graphs(cls, mgs: Sequence[MolGraph]) -> "BatchMolGraph":
        Vs, Es = [], []
        edge_indexes, rev_edge_indexes, batch_indexes = [], [], []
        n_atoms_cum, n_edges_cum = 0, 0

        for i, mg in enumerate(mgs):
            Vs.append(mg.V)
            Es.append(mg.E)
            edge_indexes.append(mg.edge_index + n_atoms_cum)
            rev_edge_indexes.append(mg.rev_edge_index + n_edges_cum)
            batch_indexes.extend([i] * mg.n_atoms)
            n_atoms_cum += mg.n_atoms
            n_edges_cum += mg.n_edges

        return cls(
            V=torch.from_numpy(np.concatenate(Vs)).float(),
            E=torch.from_numpy(np.concatenate(Es)).float(),
            edge_index=torch.from_numpy(np.hstack(edge_indexes)).long(),
            rev_edge_index=torch.from_numpy(np.concatenate(rev_edge_indexes)).long(),
            batch=torch.tensor(batch_indexes, dtype=torch.long),
            n_mols=len(mgs),
        )

    def to(self, device: torch.device) -> "BatchMolGraph":
        self.V = self.V.to(device)
        self.E = self.E.to(device)
        self.edge_index = self.edge_index.to(device)
        self.rev_edge_index = self.rev_edge_index.to(device)
        self.batch = self.batch.to(device)
        return self


class MolGraphFeaturizer:
    """Converts SMILES strings into MolGraph objects."""

    def __init__(self, atom_fdim: int = ATOM_FDIM, bond_fdim: int = BOND_FDIM):
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim

    def __call__(self, smiles: str) -> Optional[MolGraph]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return self.featurize(mol)

    def featurize(self, mol: Chem.Mol) -> MolGraph:
        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()

        if n_atoms == 0:
            V = np.zeros((1, self.atom_fdim), dtype=np.float32)
            E = np.zeros((0, self.bond_fdim), dtype=np.float32)
            edge_index = np.zeros((2, 0), dtype=np.int64)
            rev_edge_index = np.zeros(0, dtype=np.int64)
            return MolGraph(V, E, edge_index, rev_edge_index)

        V = np.array(
            [atom_features(mol.GetAtomWithIdx(i)) for i in range(n_atoms)],
            dtype=np.float32,
        )

        E = np.empty((2 * n_bonds, self.bond_fdim), dtype=np.float32)
        src_list, dst_list = [], []

        for i, bond in enumerate(mol.GetBonds()):
            bf = bond_features(bond)
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

            # Forward edge u -> v
            E[2 * i] = bf
            src_list.append(u)
            dst_list.append(v)

            # Reverse edge v -> u
            E[2 * i + 1] = bf
            src_list.append(v)
            dst_list.append(u)

        edge_index = np.array([src_list, dst_list], dtype=np.int64)
        # rev_edge_index: edge i's reverse is i+1 if even, i-1 if odd
        rev_edge_index = np.arange(2 * n_bonds, dtype=np.int64).reshape(-1, 2)[:, ::-1].ravel().copy()

        return MolGraph(V, E, edge_index, rev_edge_index)

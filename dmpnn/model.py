"""
D-MPNN (Directed Message Passing Neural Network) model.

Implements the architecture from:
  Yang et al., "Analyzing Learned Molecular Representations for Property Prediction"
  JCIM 2019. https://doi.org/10.1021/acs.jcim.9b00237

Architecture:
  BondMessagePassing -> MeanAggregation -> FFN Predictor
"""

import torch
import torch.nn as nn
from torch import Tensor

from dmpnn.featurizer import BatchMolGraph, ATOM_FDIM, BOND_FDIM


class BondMessagePassing(nn.Module):
    """D-MPNN encoder that passes messages along directed bonds.

    Key insight vs standard MPNN: messages propagate along directed edges,
    and when aggregating incoming messages for a bond v->w, the reverse
    bond w->v is excluded to prevent information from flowing back along
    the same edge.

    Math:
        H_0[vw] = W_i([x_v || e_vw])           # initial edge hidden state
        H = tau(H_0)
        for t in 1..depth-1:
            M_all[v] = sum_{u in N(v)} H[uv]    # sum all incoming
            M[vw] = M_all[v] - H[wv]            # exclude reverse
            H[vw] = tau(H_0[vw] + W_h(M[vw]))   # update with residual
        m_v = sum_{w in N(v)} H[wv]              # aggregate to atoms
        h_v = tau(W_o([x_v || m_v]))             # atom hidden state
    """

    def __init__(
        self,
        d_v: int = ATOM_FDIM,
        d_e: int = BOND_FDIM,
        d_h: int = 300,
        depth: int = 3,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_h = d_h
        self.depth = depth

        self.W_i = nn.Linear(d_v + d_e, d_h, bias=bias)
        self.W_h = nn.Linear(d_h, d_h, bias=bias)
        self.W_o = nn.Linear(d_v + d_h, d_h)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    @property
    def output_dim(self) -> int:
        return self.d_h

    def forward(self, bmg: BatchMolGraph) -> Tensor:
        """Encode a batch of molecular graphs into atom-level representations.

        Returns: Tensor of shape (total_atoms, d_h)
        """
        V, E = bmg.V, bmg.E
        src, dst = bmg.edge_index[0], bmg.edge_index[1]
        rev = bmg.rev_edge_index

        # Initial edge hidden states: concat source atom features with bond features
        H_0 = self.W_i(torch.cat([V[src], E], dim=1))  # (n_edges, d_h)
        H = self.act(H_0)

        for _ in range(1, self.depth):
            # Aggregate all messages incoming to each atom
            M_all = torch.zeros(
                V.size(0), self.d_h, dtype=H.dtype, device=H.device
            )
            idx = dst.unsqueeze(1).expand(-1, self.d_h)
            M_all.scatter_add_(0, idx, H)

            # Message for edge v->w: all incoming to v minus reverse edge w->v
            M = M_all[src] - H[rev]

            # Update with residual connection
            H = self.act(H_0 + self.W_h(M))
            H = self.dropout(H)

        # Aggregate edge hidden states to atom level
        m_v = torch.zeros(V.size(0), self.d_h, dtype=H.dtype, device=H.device)
        idx = dst.unsqueeze(1).expand(-1, self.d_h)
        m_v.scatter_add_(0, idx, H)

        # Compute final atom hidden states
        h_v = self.act(self.W_o(torch.cat([V, m_v], dim=1)))
        h_v = self.dropout(h_v)

        return h_v  # (total_atoms, d_h)


class MeanAggregation(nn.Module):
    """Aggregate atom-level representations to molecule-level via mean pooling."""

    def forward(self, H: Tensor, batch: Tensor) -> Tensor:
        """
        H: (total_atoms, d_h)
        batch: (total_atoms,) — molecule index for each atom
        Returns: (n_mols, d_h)
        """
        n_mols = batch.max().item() + 1
        idx = batch.unsqueeze(1).expand(-1, H.size(1))

        h_sum = torch.zeros(n_mols, H.size(1), dtype=H.dtype, device=H.device)
        h_sum.scatter_add_(0, idx, H)

        counts = torch.zeros(n_mols, dtype=H.dtype, device=H.device)
        counts.scatter_add_(0, batch, torch.ones_like(batch, dtype=H.dtype))
        counts = counts.clamp(min=1).unsqueeze(1)

        return h_sum / counts


class NormAggregation(nn.Module):
    """Aggregate atom-level representations by dividing by a fixed constant.

    This is chemprop v2's default aggregation: sum over atoms, then divide
    by a fixed norm (default 100) instead of the actual atom count.
    """

    def __init__(self, norm: float = 100.0):
        super().__init__()
        self.norm = norm

    def forward(self, H: Tensor, batch: Tensor) -> Tensor:
        n_mols = batch.max().item() + 1
        idx = batch.unsqueeze(1).expand(-1, H.size(1))

        h_sum = torch.zeros(n_mols, H.size(1), dtype=H.dtype, device=H.device)
        h_sum.scatter_add_(0, idx, H)

        return h_sum / self.norm


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "leakyrelu": lambda: nn.LeakyReLU(0.1),
    "prelu": nn.PReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
}


def get_activation(name: str) -> nn.Module:
    factory = _ACTIVATIONS.get(name.lower())
    if factory is None:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(_ACTIVATIONS)}")
    return factory()


class FFN(nn.Sequential):
    """Block-structured feed-forward network following chemprop v2's MLP design.

    Each block is an nn.Sequential, enabling ``ffn[:i]`` slicing to extract
    intermediate learned representations (fingerprints).

    Architecture (n_layers=1, default):
        Block 0: Linear(input_dim → hidden_dim)
        Block 1: Activation → Dropout → Linear(hidden_dim → output_dim)

    Architecture (n_layers=2):
        Block 0: Linear(input_dim → hidden_dim)
        Block 1: Activation → Dropout → Linear(hidden_dim → hidden_dim)
        Block 2: Activation → Dropout → Linear(hidden_dim → output_dim)
    """

    def __init__(
        self,
        input_dim: int = 300,
        output_dim: int = 1,
        hidden_dim: int = 300,
        n_layers: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        dims = [input_dim] + [hidden_dim] * n_layers + [output_dim]
        blocks = [nn.Sequential(nn.Linear(dims[0], dims[1]))]
        for d_in, d_out in zip(dims[1:-1], dims[2:]):
            blocks.append(
                nn.Sequential(get_activation(activation), nn.Dropout(dropout), nn.Linear(d_in, d_out))
            )
        super().__init__(*blocks)

    @property
    def input_dim(self) -> int:
        return self[0][-1].in_features

    @property
    def output_dim(self) -> int:
        return self[-1][-1].out_features

    def encode(self, x: Tensor, i: int = -1) -> Tensor:
        """Extract the i-th intermediate representation.

        Examples:
            i=1: output after block 0 (first Linear)
            i=-1: output after all blocks except the last (final hidden)
        """
        blocks = list(self.children())
        for block in blocks[:i]:
            x = block(x)
        return x


class DMPNN(nn.Module):
    """Complete D-MPNN model: BondMessagePassing -> MeanAggregation -> FFN.

    Default hyperparameters match chemprop v2:
      d_h=300, depth=3, bias=False, dropout=0.0, ffn_layers=1,
      aggregation=norm(100)
    """

    def __init__(
        self,
        d_v: int = ATOM_FDIM,
        d_e: int = BOND_FDIM,
        d_h: int = 300,
        depth: int = 3,
        bias: bool = False,
        dropout: float = 0.0,
        ffn_hidden_dim: int = 300,
        ffn_n_layers: int = 1,
        ffn_dropout: float = 0.0,
        n_tasks: int = 1,
        aggregation: str = "norm",
        aggregation_norm: float = 100.0,
        activation: str = "relu",
    ):
        super().__init__()
        self.encoder = BondMessagePassing(d_v, d_e, d_h, depth, bias, dropout)
        if aggregation == "mean":
            self.aggregation = MeanAggregation()
        else:
            self.aggregation = NormAggregation(aggregation_norm)
        self.ffn = FFN(d_h, n_tasks, ffn_hidden_dim, ffn_n_layers, ffn_dropout, activation)

    def fingerprint(self, bmg: BatchMolGraph) -> Tensor:
        """Learned molecular fingerprint (encoder + aggregation)."""
        h_atoms = self.encoder(bmg)
        return self.aggregation(h_atoms, bmg.batch)

    def encode(self, bmg: BatchMolGraph, i: int = -1) -> Tensor:
        """Extract the i-th intermediate FFN representation."""
        return self.ffn.encode(self.fingerprint(bmg), i)

    def forward(self, bmg: BatchMolGraph) -> Tensor:
        """
        Returns: (n_mols, n_tasks) predictions
        """
        return self.ffn(self.fingerprint(bmg))

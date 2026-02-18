"""Unit tests for D-MPNN featurizer and model (aligned with chemprop v2)."""

import numpy as np
import torch
import pytest
from rdkit import Chem

from dmpnn.featurizer import (
    MolGraphFeaturizer,
    BatchMolGraph,
    MolGraph,
    atom_features,
    bond_features,
    ATOM_FDIM,
    BOND_FDIM,
)
from dmpnn.model import DMPNN, BondMessagePassing, MeanAggregation, NormAggregation, FFN


# ── Featurizer Tests ───────────────────────────────────────────────────


class TestAtomFeatures:
    def test_dimension(self):
        mol = Chem.MolFromSmiles("C")
        af = atom_features(mol.GetAtomWithIdx(0))
        assert len(af) == ATOM_FDIM == 72

    def test_none_atom(self):
        af = atom_features(None)
        assert len(af) == ATOM_FDIM
        assert all(v == 0 for v in af)

    def test_different_atoms(self):
        mol = Chem.MolFromSmiles("CO")
        af_c = atom_features(mol.GetAtomWithIdx(0))
        af_o = atom_features(mol.GetAtomWithIdx(1))
        assert af_c != af_o

    def test_matches_chemprop_v2(self):
        from chemprop.featurizers import MultiHotAtomFeaturizer

        cp_af = MultiHotAtomFeaturizer.v2()
        for smi in ["c1ccccc1N", "CCO", "C=CC#N", "[Na+].[Cl-]"]:
            mol = Chem.MolFromSmiles(smi)
            for i in range(mol.GetNumAtoms()):
                ours = atom_features(mol.GetAtomWithIdx(i))
                theirs = cp_af(mol.GetAtomWithIdx(i)).tolist()
                assert len(ours) == len(theirs), f"Dim mismatch for {smi} atom {i}"
                for j, (a, b) in enumerate(zip(ours, theirs)):
                    assert abs(a - b) < 1e-6, (
                        f"SMILES={smi}, Atom {i}, idx {j}: ours={a} != theirs={b}"
                    )


class TestBondFeatures:
    def test_dimension(self):
        mol = Chem.MolFromSmiles("CC")
        bf = bond_features(mol.GetBondWithIdx(0))
        assert len(bf) == BOND_FDIM == 14

    def test_none_bond(self):
        bf = bond_features(None)
        assert len(bf) == BOND_FDIM
        assert bf[0] == 1  # is_null flag

    def test_matches_chemprop_v2(self):
        from chemprop.featurizers import MultiHotBondFeaturizer

        cp_bf = MultiHotBondFeaturizer()
        for smi in ["C=CC#N", "c1ccccc1", "CCO"]:
            mol = Chem.MolFromSmiles(smi)
            for i in range(mol.GetNumBonds()):
                ours = bond_features(mol.GetBondWithIdx(i))
                theirs = cp_bf(mol.GetBondWithIdx(i)).tolist()
                assert len(ours) == len(theirs), f"Dim mismatch for {smi} bond {i}"
                for j, (a, b) in enumerate(zip(ours, theirs)):
                    assert abs(float(a) - float(b)) < 1e-6, (
                        f"SMILES={smi}, Bond {i}, idx {j}: ours={a} != theirs={b}"
                    )


class TestMolGraphFeaturizer:
    @pytest.fixture
    def feat(self):
        return MolGraphFeaturizer()

    def test_ethane(self, feat):
        mg = feat("CC")
        assert mg.n_atoms == 2
        assert mg.n_edges == 2  # 1 bond -> 2 directed edges
        assert mg.V.shape == (2, 72)
        assert mg.E.shape == (2, 14)
        assert mg.edge_index.shape == (2, 2)
        assert mg.rev_edge_index.shape == (2,)

    def test_benzene(self, feat):
        mg = feat("c1ccccc1")
        assert mg.n_atoms == 6
        assert mg.n_edges == 12  # 6 bonds -> 12 directed edges

    def test_rev_edge_index_correctness(self, feat):
        mg = feat("CCO")
        ei = mg.edge_index
        rev = mg.rev_edge_index
        for i in range(mg.n_edges):
            j = rev[i]
            assert ei[0, i] == ei[1, j]
            assert ei[1, i] == ei[0, j]

    def test_invalid_smiles(self, feat):
        assert feat("invalid_smiles") is None

    def test_single_atom(self, feat):
        mg = feat("[Na+]")
        assert mg.n_atoms == 1
        assert mg.n_edges == 0

    def test_matches_chemprop_v2_molgraph(self, feat):
        """Verify our MolGraph matches chemprop v2's SimpleMoleculeMolGraphFeaturizer."""
        from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

        cp_feat = SimpleMoleculeMolGraphFeaturizer()
        for smi in ["CCO", "c1ccccc1", "C=CC#N"]:
            mol = Chem.MolFromSmiles(smi)
            ours = feat(smi)
            theirs = cp_feat(mol, None, None)

            np.testing.assert_allclose(ours.V, theirs.V, atol=1e-5, err_msg=f"{smi}: V mismatch")
            np.testing.assert_allclose(ours.E, theirs.E, atol=1e-5, err_msg=f"{smi}: E mismatch")
            np.testing.assert_array_equal(ours.edge_index, theirs.edge_index, err_msg=f"{smi}: edge_index mismatch")
            np.testing.assert_array_equal(ours.rev_edge_index, theirs.rev_edge_index, err_msg=f"{smi}: rev_edge_index mismatch")


class TestBatchMolGraph:
    def test_batching(self):
        feat = MolGraphFeaturizer()
        mgs = [feat("CC"), feat("CCC"), feat("CCCC")]
        bmg = BatchMolGraph.from_mol_graphs(mgs)

        assert bmg.V.shape[0] == 2 + 3 + 4  # total atoms
        assert bmg.E.shape[0] == 2 + 4 + 6  # total directed edges
        assert bmg.n_mols == 3
        assert bmg.batch.shape[0] == 9  # total atoms

    def test_batch_indices(self):
        feat = MolGraphFeaturizer()
        mgs = [feat("CC"), feat("O")]
        bmg = BatchMolGraph.from_mol_graphs(mgs)
        assert bmg.batch.tolist() == [0, 0, 1]

    def test_to_device(self):
        feat = MolGraphFeaturizer()
        bmg = BatchMolGraph.from_mol_graphs([feat("CC")])
        bmg.to(torch.device("cpu"))
        assert bmg.V.device.type == "cpu"


# ── Model Tests ────────────────────────────────────────────────────────


class TestBondMessagePassing:
    def test_output_shape(self):
        feat = MolGraphFeaturizer()
        bmg = BatchMolGraph.from_mol_graphs([feat("CC"), feat("c1ccccc1")])
        mp = BondMessagePassing(d_h=64, depth=3)
        h = mp(bmg)
        assert h.shape == (8, 64)  # 2 + 6 atoms, d_h=64

    def test_no_edges(self):
        feat = MolGraphFeaturizer()
        mg = feat("[Na+]")
        bmg = BatchMolGraph.from_mol_graphs([mg])
        mp = BondMessagePassing(d_h=64, depth=3)
        h = mp(bmg)
        assert h.shape == (1, 64)


class TestMeanAggregation:
    def test_aggregation(self):
        agg = MeanAggregation()
        H = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        batch = torch.tensor([0, 0, 1])
        result = agg(H, batch)
        assert result.shape == (2, 2)
        torch.testing.assert_close(result[0], torch.tensor([2.0, 3.0]))
        torch.testing.assert_close(result[1], torch.tensor([5.0, 6.0]))


class TestNormAggregation:
    def test_aggregation(self):
        agg = NormAggregation(norm=100.0)
        H = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        batch = torch.tensor([0, 0, 1])
        result = agg(H, batch)
        assert result.shape == (2, 2)
        torch.testing.assert_close(result[0], torch.tensor([4.0 / 100, 6.0 / 100]))
        torch.testing.assert_close(result[1], torch.tensor([5.0 / 100, 6.0 / 100]))


class TestFFN:
    def test_output_shape(self):
        ffn = FFN(input_dim=64, output_dim=1, hidden_dim=32, n_layers=2)
        x = torch.randn(4, 64)
        out = ffn(x)
        assert out.shape == (4, 1)

    def test_properties(self):
        ffn = FFN(input_dim=128, output_dim=3, hidden_dim=64, n_layers=1)
        assert ffn.input_dim == 128
        assert ffn.output_dim == 3

    def test_block_structure(self):
        ffn = FFN(input_dim=64, output_dim=1, hidden_dim=32, n_layers=2)
        assert len(ffn) == 3  # 3 blocks: Linear, Act+Drop+Linear, Act+Drop+Linear

    def test_encode_slicing(self):
        ffn = FFN(input_dim=64, output_dim=1, hidden_dim=32, n_layers=2)
        x = torch.randn(4, 64)
        h = ffn.encode(x, i=1)
        assert h.shape == (4, 32)  # after first block: Linear(64→32)
        h2 = ffn.encode(x, i=2)
        assert h2.shape == (4, 32)  # after second block: ReLU→Drop→Linear(32→32)

    def test_activations(self):
        for act in ["relu", "leakyrelu", "prelu", "tanh", "elu"]:
            ffn = FFN(input_dim=32, output_dim=1, hidden_dim=16, activation=act)
            out = ffn(torch.randn(2, 32))
            assert out.shape == (2, 1), f"Failed for activation={act}"

    def test_invalid_activation(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            FFN(input_dim=32, output_dim=1, activation="invalid")


class TestDMPNN:
    @pytest.fixture
    def model(self):
        return DMPNN(d_h=64, ffn_hidden_dim=32, n_tasks=1)

    @pytest.fixture
    def bmg(self):
        feat = MolGraphFeaturizer()
        return BatchMolGraph.from_mol_graphs(
            [feat("CC"), feat("c1ccccc1O"), feat("CCO")]
        )

    def test_forward_shape(self, model, bmg):
        out = model(bmg)
        assert out.shape == (3, 1)

    def test_gradient_flow(self, model, bmg):
        model.train()
        out = model(bmg)
        loss = out.sum()
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"

    def test_multitask(self):
        model = DMPNN(d_h=64, ffn_hidden_dim=32, n_tasks=3)
        feat = MolGraphFeaturizer()
        bmg = BatchMolGraph.from_mol_graphs([feat("CC")])
        out = model(bmg)
        assert out.shape == (1, 3)

    def test_deterministic(self, model, bmg):
        model.eval()
        with torch.no_grad():
            out1 = model(bmg)
            out2 = model(bmg)
        torch.testing.assert_close(out1, out2)

    def test_param_count(self):
        model = DMPNN(d_h=300, ffn_hidden_dim=300, n_tasks=1)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0
        # W_i: (72+14)*300=25800, W_h: 300*300=90000, W_o: (72+300)*300+300=111900
        # FFN (n_layers=1): 300*300+300 + 300*1+1 = 90601
        expected_approx = 25800 + 90000 + 111900 + 90601
        assert abs(n_params - expected_approx) < 1000, f"Got {n_params}, expected ~{expected_approx}"

    def test_fingerprint(self, model, bmg):
        fp = model.fingerprint(bmg)
        assert fp.shape == (3, 64)  # 3 molecules, d_h=64

    def test_encode(self, model, bmg):
        h = model.encode(bmg, i=1)
        assert h.shape == (3, 32)  # after first FFN block: Linear(64→32)

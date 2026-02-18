"""Train D-MPNN on ESOL or other molecular property datasets."""

import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from dmpnn.model import DMPNN
from dmpnn.featurizer import BatchMolGraph
from dmpnn.data import MoleculeDataset, build_dataloader, scaffold_split


def noam_lr_lambda(warmup_steps: int, total_steps: int, init_lr: float, max_lr: float, final_lr: float):
    """Noam-like learning rate schedule matching chemprop v1."""
    def lr_lambda(step):
        step = max(1, step)
        if step <= warmup_steps:
            lr = init_lr + (max_lr - init_lr) * step / warmup_steps
        else:
            lr = max_lr * (final_lr / max_lr) ** ((step - warmup_steps) / (total_steps - warmup_steps))
        return lr / max_lr
    return lr_lambda


def evaluate(
    model: DMPNN,
    dataloader,
    device: torch.device,
    target_mean: float = 0.0,
    target_std: float = 1.0,
) -> float:
    """Evaluate model and return RMSE in original scale."""
    model.eval()
    preds_all, targets_all = [], []

    with torch.no_grad():
        for bmg, targets in dataloader:
            bmg.to(device)
            preds = model(bmg)
            preds_all.append(preds.cpu().numpy())
            targets_all.append(targets.numpy())

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)

    preds_all = preds_all * target_std + target_mean
    targets_all = targets_all * target_std + target_mean

    rmse = np.sqrt(np.mean((preds_all - targets_all) ** 2))
    return rmse


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    df = pd.read_csv(args.data_path)
    smiles = df[args.smiles_col].tolist()
    targets = df[args.target_col].values.astype(np.float32)

    # Split
    if args.split_file:
        df_split = pd.read_csv(args.split_file)
        train_mask = df_split["split"] == "train"
        val_mask = df_split["split"] == "val"
        test_mask = df_split["split"] == "test"
        train_smi = df_split.loc[train_mask, args.smiles_col].tolist()
        train_y = df_split.loc[train_mask, args.target_col].values.astype(np.float32)
        val_smi = df_split.loc[val_mask, args.smiles_col].tolist()
        val_y = df_split.loc[val_mask, args.target_col].values.astype(np.float32)
        test_smi = df_split.loc[test_mask, args.smiles_col].tolist()
        test_y = df_split.loc[test_mask, args.target_col].values.astype(np.float32)
    else:
        (train_smi, train_y), (val_smi, val_y), (test_smi, test_y) = scaffold_split(
            smiles, targets, seed=args.seed
        )
    print(f"Train: {len(train_smi)}, Val: {len(val_smi)}, Test: {len(test_smi)}")

    # Standardize targets using training set statistics
    target_mean = train_y.mean()
    target_std = train_y.std()
    if target_std < 1e-8:
        target_std = 1.0
    print(f"Target normalization: mean={target_mean:.4f}, std={target_std:.4f}")

    train_y_norm = (train_y - target_mean) / target_std
    val_y_norm = (val_y - target_mean) / target_std
    test_y_norm = (test_y - target_mean) / target_std

    # Datasets
    train_dataset = MoleculeDataset(train_smi, train_y_norm)
    val_dataset = MoleculeDataset(val_smi, val_y_norm)
    test_dataset = MoleculeDataset(test_smi, test_y_norm)

    train_loader = build_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = build_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = build_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    model = DMPNN(
        d_h=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
        ffn_hidden_dim=args.ffn_hidden_dim,
        ffn_n_layers=args.ffn_n_layers,
        ffn_dropout=args.dropout,
        n_tasks=1,
        aggregation=args.aggregation,
        aggregation_norm=args.aggregation_norm,
        activation=args.activation,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer & scheduler
    optimizer = Adam(model.parameters(), lr=args.max_lr)
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    lr_fn = noam_lr_lambda(warmup_steps, total_steps, args.init_lr, args.max_lr, args.final_lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_fn)

    loss_fn = nn.MSELoss()

    # Training loop
    best_val_rmse = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for bmg, targets_batch in train_loader:
            bmg.to(device)
            targets_batch = targets_batch.to(device)

            preds = model(bmg)
            loss = loss_fn(preds, targets_batch)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        val_rmse = evaluate(model, val_loader, device, target_mean, target_std)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Val RMSE: {val_rmse:.4f} | Best: {best_val_rmse:.4f}")

    # Load best model and evaluate on test
    model.load_state_dict(best_state)
    test_rmse = evaluate(model, test_loader, device, target_mean, target_std)
    print(f"\n{'='*50}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Best Val RMSE: {best_val_rmse:.4f}")
    print(f"{'='*50}")

    # Save model and results
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(best_state, os.path.join(args.output_dir, "best_model.pt"))

    results = {
        "test_rmse": float(test_rmse),
        "best_val_rmse": float(best_val_rmse),
        "train_size": len(train_smi),
        "val_size": len(val_smi),
        "test_size": len(test_smi),
        "n_params": n_params,
        "args": vars(args),
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return test_rmse


def main():
    parser = argparse.ArgumentParser(description="Train D-MPNN")
    parser.add_argument("--data-path", type=str, default="data/esol.csv")
    parser.add_argument("--smiles-col", type=str, default="smiles")
    parser.add_argument("--target-col", type=str, default="logSolubility")
    parser.add_argument("--output-dir", type=str, default="output/dmpnn")
    parser.add_argument("--hidden-dim", type=int, default=300)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--ffn-hidden-dim", type=int, default=300)
    parser.add_argument("--ffn-n-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--init-lr", type=float, default=1e-4)
    parser.add_argument("--max-lr", type=float, default=1e-3)
    parser.add_argument("--final-lr", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-file", type=str, default=None, help="CSV with 'split' column")
    parser.add_argument("--aggregation", type=str, default="norm", choices=["mean", "norm"])
    parser.add_argument("--aggregation-norm", type=float, default=100.0)
    parser.add_argument("--grad-clip", type=float, default=None, help="Max gradient norm (None=no clipping)")
    parser.add_argument("--activation", type=str, default="relu",
                        choices=["relu", "leakyrelu", "prelu", "tanh", "elu"])

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train(args)


if __name__ == "__main__":
    main()

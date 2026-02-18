"""Train D-MPNN for classification tasks (BBBP, Tox21, ClinTox, etc.)."""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import roc_auc_score

from dmpnn.model import DMPNN
from dmpnn.featurizer import BatchMolGraph
from dmpnn.data import MoleculeDataset, build_dataloader, scaffold_split


def noam_lr_lambda(warmup_steps, total_steps, init_lr, max_lr, final_lr):
    def lr_lambda(step):
        step = max(1, step)
        if step <= warmup_steps:
            lr = init_lr + (max_lr - init_lr) * step / warmup_steps
        else:
            lr = max_lr * (final_lr / max_lr) ** (
                (step - warmup_steps) / (total_steps - warmup_steps)
            )
        return lr / max_lr
    return lr_lambda


def masked_bce_loss(preds, targets):
    """BCEWithLogitsLoss that ignores NaN targets."""
    mask = ~torch.isnan(targets)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=preds.device, requires_grad=True)
    return nn.functional.binary_cross_entropy_with_logits(
        preds[mask], targets[mask]
    )


def evaluate_auc(model, dataloader, device, n_tasks):
    """Evaluate model and return per-task ROC-AUC and mean ROC-AUC."""
    model.eval()
    preds_all, targets_all = [], []

    with torch.no_grad():
        for bmg, targets in dataloader:
            bmg.to(device)
            preds = torch.sigmoid(model(bmg))
            preds_all.append(preds.cpu().numpy())
            targets_all.append(targets.numpy())

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)

    aucs = []
    for i in range(n_tasks):
        mask = ~np.isnan(targets_all[:, i])
        y_true = targets_all[mask, i]
        y_pred = preds_all[mask, i]
        if len(np.unique(y_true)) < 2:
            aucs.append(float("nan"))
        else:
            aucs.append(roc_auc_score(y_true, y_pred))

    valid_aucs = [a for a in aucs if not np.isnan(a)]
    mean_auc = np.mean(valid_aucs) if valid_aucs else float("nan")
    return mean_auc, aucs


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(args.data_path)
    target_cols = args.target_cols

    n_tasks = len(target_cols)
    print(f"Task type: classification, {n_tasks} task(s): {target_cols}")

    smiles = df[args.smiles_col].tolist()
    targets = df[target_cols].values.astype(np.float32)

    if args.split_file:
        df_split = pd.read_csv(args.split_file)
        train_mask = df_split["split"] == "train"
        val_mask = df_split["split"] == "val"
        test_mask = df_split["split"] == "test"
        train_smi = df_split.loc[train_mask, args.smiles_col].tolist()
        train_y = df_split.loc[train_mask, target_cols].values.astype(np.float32)
        val_smi = df_split.loc[val_mask, args.smiles_col].tolist()
        val_y = df_split.loc[val_mask, target_cols].values.astype(np.float32)
        test_smi = df_split.loc[test_mask, args.smiles_col].tolist()
        test_y = df_split.loc[test_mask, target_cols].values.astype(np.float32)
    else:
        (train_smi, train_y), (val_smi, val_y), (test_smi, test_y) = scaffold_split(
            smiles, targets, seed=args.seed
        )
    print(f"Train: {len(train_smi)}, Val: {len(val_smi)}, Test: {len(test_smi)}")

    train_dataset = MoleculeDataset(train_smi, train_y)
    val_dataset = MoleculeDataset(val_smi, val_y)
    test_dataset = MoleculeDataset(test_smi, test_y)

    train_loader = build_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = build_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = build_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = DMPNN(
        d_h=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
        ffn_hidden_dim=args.ffn_hidden_dim,
        ffn_n_layers=args.ffn_n_layers,
        ffn_dropout=args.dropout,
        n_tasks=n_tasks,
        aggregation=args.aggregation,
        aggregation_norm=args.aggregation_norm,
        activation=args.activation,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = Adam(model.parameters(), lr=args.max_lr)
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    lr_fn = noam_lr_lambda(warmup_steps, total_steps, args.init_lr, args.max_lr, args.final_lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_fn)

    best_val_auc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for bmg, targets_batch in train_loader:
            bmg.to(device)
            targets_batch = targets_batch.to(device)

            preds = model(bmg)
            loss = masked_bce_loss(preds, targets_batch)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        val_auc, _ = evaluate_auc(model, val_loader, device, n_tasks)

        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
                f"Val AUC: {val_auc:.4f} | Best: {best_val_auc:.4f}"
            )

    model.load_state_dict(best_state)
    test_auc, per_task_auc = evaluate_auc(model, test_loader, device, n_tasks)
    print(f"\n{'='*50}")
    print(f"Test ROC-AUC (mean): {test_auc:.4f}")
    for col, auc in zip(target_cols, per_task_auc):
        status = f"{auc:.4f}" if not np.isnan(auc) else "N/A (single class)"
        print(f"  {col}: {status}")
    print(f"Best Val AUC: {best_val_auc:.4f}")
    print(f"{'='*50}")

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(best_state, os.path.join(args.output_dir, "best_model.pt"))

    results = {
        "test_auc_mean": float(test_auc),
        "test_auc_per_task": {
            col: float(auc) if not np.isnan(auc) else None
            for col, auc in zip(target_cols, per_task_auc)
        },
        "best_val_auc": float(best_val_auc),
        "train_size": len(train_smi),
        "val_size": len(val_smi),
        "test_size": len(test_smi),
        "n_tasks": n_tasks,
        "n_params": n_params,
        "args": vars(args),
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return test_auc


def main():
    parser = argparse.ArgumentParser(description="Train D-MPNN (Classification)")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--smiles-col", type=str, default="smiles")
    parser.add_argument("--target-cols", type=str, nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, default="output/dmpnn_cls")
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
    parser.add_argument("--split-file", type=str, default=None)
    parser.add_argument("--aggregation", type=str, default="norm", choices=["mean", "norm"])
    parser.add_argument("--aggregation-norm", type=float, default=100.0)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--activation", type=str, default="relu",
                        choices=["relu", "leakyrelu", "prelu", "tanh", "elu"])

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train(args)


if __name__ == "__main__":
    main()

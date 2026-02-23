"""Finetune MoLFormer-c3-1.1B on classification tasks (BBBP, Tox21, ClinTox).

Uses HuggingFace AutoModel + AdamW with DeepChem scaffold splits (80/10/10).
Supports multi-task binary classification with NaN-masked BCE loss.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from functools import partial
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification


PRETRAINED_MODEL = "DeepChem/MoLFormer-c3-1.1B"


class SMILESDataset(Dataset):
    def __init__(self, smiles: list, targets: np.ndarray):
        self.smiles = smiles
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx], self.targets[idx]


def collate_fn(batch, tokenizer, max_length=202):
    smiles_list, targets = zip(*batch)
    tokens = tokenizer(
        list(smiles_list),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return tokens, torch.stack(targets)


def build_param_groups(model, weight_decay):
    """Separate decay vs no_decay parameters (following ChemBERTa-3 convention)."""
    no_decay_keywords = ["bias", "LayerNorm", "layernorm", "layer_norm", "embedding"]
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(kw in name for kw in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def masked_bce_loss(logits, targets):
    """BCEWithLogitsLoss that ignores NaN targets (for multi-task with missing labels)."""
    mask = ~torch.isnan(targets)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    return nn.functional.binary_cross_entropy_with_logits(logits[mask], targets[mask])


def evaluate_auc(model, dataloader, device, n_tasks):
    """Evaluate model and return (mean_auc, per_task_aucs)."""
    model.eval()
    preds_all, targets_all = [], []

    with torch.no_grad():
        for tokens, targets in dataloader:
            tokens = {k: v.to(device) for k, v in tokens.items()}
            logits = model(**tokens).logits
            preds_all.append(torch.sigmoid(logits).cpu().numpy())
            targets_all.append(targets.numpy())

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)

    if preds_all.ndim == 1:
        preds_all = preds_all.reshape(-1, 1)
    if targets_all.ndim == 1:
        targets_all = targets_all.reshape(-1, 1)

    aucs = []
    for i in range(n_tasks):
        mask = ~np.isnan(targets_all[:, i])
        y_true = targets_all[mask, i]
        y_pred = preds_all[mask, i]
        if len(np.unique(y_true)) < 2:
            aucs.append(float("nan"))
        else:
            aucs.append(roc_auc_score(y_true, y_pred))

    valid = [a for a in aucs if not np.isnan(a)]
    mean_auc = float(np.mean(valid)) if valid else float("nan")
    return mean_auc, aucs


def single_run(args, seed):
    """Run a single classification experiment with the given seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_cols = args.target_cols
    n_tasks = len(target_cols)

    print(f"\n{'='*60}")
    print(f"Run seed={seed} | Device: {device} | Tasks: {n_tasks}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL, trust_remote_code=True)

    df = pd.read_csv(args.split_file)
    splits = {}
    for s in ("train", "val", "test"):
        sub = df[df["split"] == s]
        splits[s] = (sub[args.smiles_col].tolist(),
                     sub[target_cols].values.astype(np.float32))

    train_smi, train_y = splits["train"]
    val_smi, val_y = splits["val"]
    test_smi, test_y = splits["test"]
    print(f"Train: {len(train_smi)}, Val: {len(val_smi)}, Test: {len(test_smi)}")

    train_dataset = SMILESDataset(train_smi, train_y)
    val_dataset = SMILESDataset(val_smi, val_y)
    test_dataset = SMILESDataset(test_smi, test_y)

    collate = partial(collate_fn, tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate, num_workers=0)

    config = AutoConfig.from_pretrained(PRETRAINED_MODEL, trust_remote_code=True)
    config.num_labels = n_tasks

    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL, config=config,
        trust_remote_code=True, ignore_mismatched_sizes=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,} (trainable: {trainable:,})")

    optimizer = AdamW(build_param_groups(model, args.weight_decay), lr=args.lr)

    best_val_auc = -1.0
    best_state = None
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for tokens, targets_batch in train_loader:
            tokens = {k: v.to(device) for k, v in tokens.items()}
            targets_batch = targets_batch.to(device)

            logits = model(**tokens).logits
            loss = masked_bce_loss(logits, targets_batch)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        val_auc, _ = evaluate_auc(model, val_loader, device, n_tasks)

        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 2 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f} | "
                  f"Best: {best_val_auc:.4f} | {elapsed:.0f}s")

    model.load_state_dict(best_state)
    test_auc, per_task_auc = evaluate_auc(model, test_loader, device, n_tasks)
    elapsed = time.time() - t0

    print(f"\nSeed {seed} done: Test AUC = {test_auc:.4f}, "
          f"Best Val AUC = {best_val_auc:.4f}, Time = {elapsed:.0f}s")
    for col, auc in zip(target_cols, per_task_auc):
        print(f"  {col}: {auc:.4f}" if not np.isnan(auc) else f"  {col}: N/A")

    run_dir = os.path.join(args.output_dir, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)
    torch.save(best_state, os.path.join(run_dir, "best_model.pt"))

    results = {
        "seed": seed,
        "test_auc_mean": float(test_auc),
        "test_auc_per_task": {
            col: float(auc) if not np.isnan(auc) else None
            for col, auc in zip(target_cols, per_task_auc)
        },
        "best_val_auc": float(best_val_auc),
        "elapsed_seconds": elapsed,
        "train_size": len(train_smi),
        "val_size": len(val_smi),
        "test_size": len(test_smi),
        "n_tasks": n_tasks,
        "n_params": n_params,
    }
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return test_auc, best_val_auc, per_task_auc


def main():
    parser = argparse.ArgumentParser(
        description="Finetune MoLFormer on classification tasks")
    parser.add_argument("--split-file", type=str, required=True)
    parser.add_argument("--smiles-col", type=str, default="smiles")
    parser.add_argument("--target-cols", type=str, nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, default="output/molformer_cls")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--n-runs", type=int, default=3,
                        help="Number of runs with different seeds")
    parser.add_argument("--seed", type=int, default=0, help="Base seed")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    target_cols = args.target_cols
    n_tasks = len(target_cols)
    all_test_auc, all_val_auc, all_per_task = [], [], []

    for i in range(args.n_runs):
        test_auc, val_auc, per_task = single_run(args, args.seed + i)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_per_task.append(per_task)

    mean_t, std_t = float(np.mean(all_test_auc)), float(np.std(all_test_auc))

    print(f"\n{'='*60}")
    print(f"Summary ({args.n_runs} runs)")
    print(f"{'='*60}")
    for i, (ta, va) in enumerate(zip(all_test_auc, all_val_auc)):
        print(f"  Run {i+1} (seed {args.seed+i}): Test AUC={ta:.4f}, Val AUC={va:.4f}")
    print(f"\n  Test AUC: {mean_t:.4f} +/- {std_t:.4f}")

    per_task_arr = np.array(all_per_task)
    per_task_summary = {}
    print(f"\n  Per-task AUC (mean +/- std):")
    for j, col in enumerate(target_cols):
        vals = per_task_arr[:, j]
        valid = vals[~np.isnan(vals)]
        if len(valid) > 0:
            m, s = float(np.mean(valid)), float(np.std(valid))
            print(f"    {col}: {m:.4f} +/- {s:.4f}")
            per_task_summary[col] = {"mean": m, "std": s}
        else:
            print(f"    {col}: N/A")
            per_task_summary[col] = None
    print(f"{'='*60}")

    summary = {
        "dataset": os.path.basename(args.split_file),
        "target_cols": target_cols,
        "n_runs": args.n_runs,
        "test_auc_mean": mean_t,
        "test_auc_std": std_t,
        "test_auc_per_run": [float(r) for r in all_test_auc],
        "val_auc_per_run": [float(r) for r in all_val_auc],
        "per_task_summary": per_task_summary,
        "args": vars(args),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

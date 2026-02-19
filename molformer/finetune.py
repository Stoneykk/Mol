"""Finetune MoLFormer-c3-1.1B on regression tasks (ESOL, FreeSolv, Lipophilicity).

Path B: HuggingFace AutoModel + AdamW (not ChemBERTa-3's DeepChem + FusedLAMB).
Differences from ChemBERTa-3 official:
  - Optimizer: AdamW instead of FusedLAMB (apex)
  - Data split: chemprop v2 scaffold split (consistent with Stage 1 D-MPNN)
  - Framework: pure PyTorch instead of DeepChem wrapper
"""

# --- Compatibility shim: IBM MoLFormer's remote code imports transformers.onnx,
# which was removed in newer transformers versions. We create a dummy module so
# the import succeeds (OnnxConfig is never actually used in our pipeline). ---
import sys, types
if "transformers.onnx" not in sys.modules:
    _onnx = types.ModuleType("transformers.onnx")
    _onnx.OnnxConfig = type("OnnxConfig", (), {})
    sys.modules["transformers.onnx"] = _onnx

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


def evaluate(model, dataloader, device, target_mean=0.0, target_std=1.0):
    """Evaluate and return RMSE in original scale."""
    model.eval()
    preds_all, targets_all = [], []

    with torch.no_grad():
        for tokens, targets in dataloader:
            tokens = {k: v.to(device) for k, v in tokens.items()}
            logits = model(**tokens).logits.squeeze(-1)
            preds_all.append(logits.cpu().numpy())
            targets_all.append(targets.numpy())

    preds = np.concatenate(preds_all) * target_std + target_mean
    targs = np.concatenate(targets_all) * target_std + target_mean
    return float(np.sqrt(np.mean((preds - targs) ** 2)))


def single_run(args, seed):
    """Run a single training experiment with the given seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Run seed={seed} | Device: {device}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL, trust_remote_code=True)

    df = pd.read_csv(args.split_file)
    splits = {}
    for s in ("train", "val", "test"):
        sub = df[df["split"] == s]
        splits[s] = (sub[args.smiles_col].tolist(),
                     sub[args.target_col].values.astype(np.float32))

    train_smi, train_y = splits["train"]
    val_smi, val_y = splits["val"]
    test_smi, test_y = splits["test"]
    print(f"Train: {len(train_smi)}, Val: {len(val_smi)}, Test: {len(test_smi)}")

    target_mean = float(train_y.mean())
    target_std = float(train_y.std())
    if target_std < 1e-8:
        target_std = 1.0
    print(f"Target normalization: mean={target_mean:.4f}, std={target_std:.4f}")

    norm = lambda y: (y - target_mean) / target_std
    train_dataset = SMILESDataset(train_smi, norm(train_y))
    val_dataset = SMILESDataset(val_smi, norm(val_y))
    test_dataset = SMILESDataset(test_smi, norm(test_y))

    collate = partial(collate_fn, tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate, num_workers=0)

    config = AutoConfig.from_pretrained(PRETRAINED_MODEL, trust_remote_code=True)
    config.problem_type = "regression"
    config.num_labels = 1

    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL, config=config,
        trust_remote_code=True, ignore_mismatched_sizes=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,} (trainable: {trainable:,})")

    optimizer = AdamW(build_param_groups(model, args.weight_decay), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state = None
    patience_counter = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for tokens, targets_batch in train_loader:
            tokens = {k: v.to(device) for k, v in tokens.items()}
            targets_batch = targets_batch.to(device)

            logits = model(**tokens).logits.squeeze(-1)
            loss = loss_fn(logits, targets_batch)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        val_rmse = evaluate(model, val_loader, device, target_mean, target_std)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Val RMSE: {val_rmse:.4f} | "
                  f"Best: {best_val_rmse:.4f} | {elapsed:.0f}s")

        if args.patience > 0 and patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    test_rmse = evaluate(model, test_loader, device, target_mean, target_std)
    elapsed = time.time() - t0

    print(f"\nSeed {seed} done: Test RMSE = {test_rmse:.4f}, "
          f"Best Val RMSE = {best_val_rmse:.4f}, Time = {elapsed:.0f}s")

    run_dir = os.path.join(args.output_dir, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)
    torch.save(best_state, os.path.join(run_dir, "best_model.pt"))

    results = {
        "seed": seed,
        "test_rmse": float(test_rmse),
        "best_val_rmse": float(best_val_rmse),
        "elapsed_seconds": elapsed,
        "train_size": len(train_smi),
        "val_size": len(val_smi),
        "test_size": len(test_smi),
        "n_params": n_params,
    }
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return test_rmse, best_val_rmse


def main():
    parser = argparse.ArgumentParser(description="Finetune MoLFormer on regression tasks")
    parser.add_argument("--split-file", type=str, required=True)
    parser.add_argument("--smiles-col", type=str, default="smiles")
    parser.add_argument("--target-col", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="output/molformer")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping patience (0 = disabled)")
    parser.add_argument("--n-runs", type=int, default=3,
                        help="Number of runs with different seeds")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed (runs use seed, seed+1, ...)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    all_test, all_val = [], []
    for i in range(args.n_runs):
        test_rmse, val_rmse = single_run(args, args.seed + i)
        all_test.append(test_rmse)
        all_val.append(val_rmse)

    mean_t, std_t = np.mean(all_test), np.std(all_test)
    print(f"\n{'='*60}")
    print(f"Summary ({args.n_runs} runs)")
    print(f"{'='*60}")
    for i, (t, v) in enumerate(zip(all_test, all_val)):
        print(f"  Run {i+1} (seed {args.seed+i}): Test RMSE={t:.4f}, Val RMSE={v:.4f}")
    print(f"\n  Test RMSE: {mean_t:.4f} +/- {std_t:.4f}")
    print(f"{'='*60}")

    summary = {
        "dataset": os.path.basename(args.split_file),
        "target_col": args.target_col,
        "n_runs": args.n_runs,
        "test_rmse_mean": float(mean_t),
        "test_rmse_std": float(std_t),
        "test_rmse_per_run": [float(r) for r in all_test],
        "val_rmse_per_run": [float(r) for r in all_val],
        "args": vars(args),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

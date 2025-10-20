"""
This file is a Benign-only autoencoder trainer (unsupervised).

What it does:
1) Splits the provided training data into a fit split (benign only) and a validation split.
2) Trains an autoencoder to reconstruct benign traffic (Huber loss, AdamW).
3) Scores validation and test samples using reconstruction error (MSE per row).
4) Chooses a decision threshold on validation scores to hit a target false-positive rate (FPR).
5) Saves the trained weights and the threshold to disk, and returns test scores + threshold.

Inputs:
- X_train, X_test: numpy arrays of features
- y_train: binary labels (0=BENIGN, 1=ATTACK)
- model_dir, dataset: where to save model files

Outputs:
- test_scores: 1D numpy array (higher = more anomalous)
- threshold: Decision threshold tuned at the target FPR

Helpers:
- _threshold_at_fpr(): Pick a threshold so that FPR ~= target_fpr using the benign-score quantile (higher=more anomalous).
- make_hidden_dims(): Build progressive hidden layer widths, halving each step.
"""

from __future__ import annotations
import os
from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from autoencoder import Autoencoder
from utils.io import get_dataset_dir


def _threshold_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 0.01) -> float:
    benign = scores[y_true == 0]
    if benign.size == 0:
        raise ValueError("No benign samples available to compute FPR-based threshold.")
    q = np.clip(1.0 - target_fpr, 0.0, 1.0)
    return float(np.quantile(benign, q))


def make_hidden_dims(base_width: int, depth: int) -> tuple[int, ...]:
    dims, w = [], base_width
    for _ in range(depth):
        dims.append(int(w))
        w = max(8, w // 2)
    return tuple(dims)


BEST = {
    "depth": 4,
    "base_width": 512,
    "latent_dim": 64,
    "dropout": 0.47888664475907516,
    "norm": "layer",
    "lr": 7.958041872057088e-04,
    "batch_size": 512,
    "epochs": 29,
    "fpr_target": 0.014764847022716774,
}


def train_and_score_autoencoder(
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        *,
        model_dir: str,
        dataset: str,
) -> Tuple[np.ndarray, float]:
    model_output_dir = get_dataset_dir(model_dir, dataset)
    os.makedirs(model_output_dir, exist_ok=True)
    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    mask_benign = (y_train_split == 0)
    if not mask_benign.any():
        raise ValueError("No benign samples in training split.")
    X_benign_train = X_train_split[mask_benign]

    X_benign_train_t = torch.tensor(X_benign_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_split, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(X_benign_train_t, X_benign_train_t),
        batch_size=BEST["batch_size"],
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )

    input_dim = X_benign_train.shape[1]
    hidden_dims = make_hidden_dims(BEST["base_width"], BEST["depth"])
    model = Autoencoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        latent_dim=BEST["latent_dim"],
        dropout=BEST["dropout"],
        norm=BEST["norm"],
    ).to(compute_device)

    opt = optim.AdamW(model.parameters(), lr=BEST["lr"], weight_decay=1e-5)
    crit = nn.SmoothL1Loss(beta=1.0)

    model.train()
    for ep in range(BEST["epochs"]):
        ep_loss = 0.0
        for (xb, _) in train_loader:
            xb = xb.to(compute_device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            recon = model(xb)
            loss = crit(recon, xb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            ep_loss += loss.item()
        print(f"AE Epoch {ep + 1}/{BEST['epochs']} — loss: {ep_loss / len(train_loader):.6f}")

    model.eval()
    with torch.no_grad():
        recon_val = model(X_val_t.to(compute_device)).cpu().numpy()
        val_scores = np.mean((X_val_split - recon_val) ** 2, axis=1)

        recon_te = model(X_test_t.to(compute_device)).cpu().numpy()
        test_scores = np.mean((X_test - recon_te) ** 2, axis=1)

    threshold = _threshold_at_fpr(y_val_split, val_scores, target_fpr=BEST["fpr_target"])
    val_auc = roc_auc_score(y_val_split, val_scores)
    val_ap = average_precision_score(y_val_split, val_scores)
    print(
        f"AE val — ROC-AUC: {val_auc:.4f}  AP: {val_ap:.4f}  "
        f"thr@FPR≈{BEST['fpr_target']:.2%}: {threshold:.4e}"
    )

    model_path = os.path.join(model_output_dir, "autoencoder.pth")
    threshold_path = os.path.join(model_output_dir, "autoencoder_threshold.npy")
    torch.save(model.state_dict(), model_path)
    np.save(threshold_path, np.array([threshold]))
    print(f"💾 Saved Autoencoder weights → {model_path}")
    print(f"💾 Saved threshold → {threshold_path}")

    return test_scores, threshold


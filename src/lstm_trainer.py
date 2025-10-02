from __future__ import annotations
"""
Minimal LSTM Trainer (PyTorch) 
----------------------------------------------------
- If you later want real temporal modeling (sliding windows or per-flow sequences),
  we can extend this with a tiny windowing helper without changing `train.py`.
"""

import os
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from utils.io import get_dataset_dir


# ------------------------------- Dataset ------------------------------------ #

class TabularAsSeqDataset(Dataset):
    """Wraps 2D tabular data as (B, T=1, F) sequences for an LSTM."""
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = X.astype(np.float32)
        self.y = None if y is None else y.astype(np.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = self.X[idx]
        x = np.expand_dims(x, axis=0)  # (F,) -> (T=1, F)
        if self.y is None:
            return x
        return x, self.y[idx]


# --------------------------------- Model ------------------------------------ #

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):  # x: (B, T=1, F)
        out, (h_n, c_n) = self.lstm(x)
        h_last = h_n[-1]  # (B, H)
        logits = self.head(h_last).squeeze(1)  # (B,)
        return logits


# ----------------------------- Train routine -------------------------------- #

@dataclass
class LSTMConfig:
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 12
    batch_size: int = 512
    seed: int = 42
    num_workers: int = 0


def _set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _pos_weight(y: np.ndarray) -> float:
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    return 1.0 if pos == 0 else max(neg / max(pos, 1.0), 1.0)


def train_and_score_lstm(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: Optional[np.ndarray],
    *,
    model_dir: str,
    dataset: str,
    hidden_dim: int = 64,
    num_layers: int = 1,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 12,
    batch_size: int = 512,
    seed: int = 42,
    num_workers: int = 0,
) -> np.ndarray:
    """
    Train a minimal LSTM classifier on tabular features (as T=1 sequences) and
    return anomaly scores `sigmoid(logits)` on X_test (higher = more anomalous).

    Saves model to models/<dataset>/lstm.pt.
    Prints ROC-AUC and AP if y_test provided.
    """
    _set_seed(seed)

    out_dir = get_dataset_dir(model_dir, dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = X_train.shape[1]
    model = LSTMClassifier(input_dim, hidden_dim, num_layers, dropout).to(device)

    train_ds = TabularAsSeqDataset(X_train, y_train)
    test_ds = TabularAsSeqDataset(X_test, y_test if y_test is not None else None)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_w = _pos_weight(y_train)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))

    # ----------------------------- Train loop ----------------------------- #
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = yb.size(0)
            total_loss += float(loss.item()) * bs
            n += bs
        print(f"Epoch {epoch:02d} | train_loss={total_loss / max(n,1):.4f}")

    # ----------------------------- Inference ------------------------------ #
    model.eval()
    all_logits = []
    with torch.no_grad():
        for batch in test_loader:
            if y_test is None:
                xb = batch
            else:
                xb, _ = batch
            xb = xb.to(device)
            logits = model(xb)
            all_logits.append(logits.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    scores = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

    # Report
    if y_test is not None:
        try:
            auc = roc_auc_score(y_test, scores)
            ap = average_precision_score(y_test, scores)
            print(f"LSTM — ROC-AUC: {auc:.4f}  AP: {ap:.4f}")
        except Exception:
            pass

    # Save model
    model_path = os.path.join(out_dir, "lstm.pt")
    torch.save(model.state_dict(), model_path)
    print(f"💾 Saved LSTM → {model_path}")

    return scores

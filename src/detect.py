"""
detect.py — Inference helpers for Network Anomaly Detection

Responsibilities:
- Load scaler, model weights, and threshold for a given dataset.
- Score new samples with the Autoencoder (and optionally Isolation Forest).
- Return anomaly scores and binary predictions using the saved threshold.
"""

from __future__ import annotations
import os
import json
from typing import Tuple, Dict
import numpy as np
import joblib
import torch
import torch.nn as nn
from autoencoder import Autoencoder  # fallback to old path
from utils.io import get_dataset_dir


# ------------------------------- utils ------------------------------------- #
def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _batched_recon(model: nn.Module, X: np.ndarray, *, batch_size: int = 1024) -> np.ndarray:
    """
    Run model forward in batches (CPU->device->CPU), returns reconstructed X.
    Assumes X is already float32 and scaled, shape (N, D).
    """
    device = next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(device=device, dtype=torch.float32, non_blocking=True)
            recon = model(xb).float().cpu().numpy()
            out.append(recon)
    return np.concatenate(out, axis=0) if out else np.empty_like(X)


# ------------------------------ loader ------------------------------------- #
class AEDetector:
    """
    Encapsulates Autoencoder inference:
    - Loads scaler, AE config, weights, and threshold from models/<dataset>/
    - Provides .score(X) -> anomaly_scores, .predict(X) -> (scores, labels)
    """
    def __init__(self, model_dir: str, dataset: str):
        self.dataset = dataset
        self.paths = _artifact_paths(model_dir, dataset)

        # Load scaler
        self.scaler = joblib.load(self.paths["scaler"])
        # Load config (architecture, etc.)
        with open(self.paths["cfg"], "r") as f:
            cfg = json.load(f)

        self.hidden_dims = tuple(int(x) for x in cfg["hidden_dims"])
        self.latent_dim = int(cfg["latent_dim"])
        self.dropout = float(cfg["dropout"])
        self.norm = cfg.get("norm", "batch")
        self.input_dim = int(cfg["input_dim"])

        # Build model and load weights
        self.model = Autoencoder(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
            norm=self.norm,
        ).to(_device())

        state = torch.load(self.paths["weights"], map_location=_device())
        self.model.load_state_dict(state)

        # Load threshold
        self.threshold = float(np.load(self.paths["thr"]).item())

    def score(self, X: np.ndarray, *, batch_size: int = 1024) -> np.ndarray:
        """
        Returns reconstruction MSE per row (higher = more anomalous).
        X can be raw features; we internally apply saved scaler.
        """
        # Make finite & scale with the TRAIN scaler (parity with training)
        X = np.nan_to_num(X.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()  # if X is DataFrame
                          if hasattr(X, "to_numpy") else np.nan_to_num(X, copy=False)).astype(np.float32)

        X_scaled = self.scaler.transform(X).astype(np.float32)
        recon = _batched_recon(self.model, X_scaled, batch_size=batch_size)
        errs = np.mean((X_scaled - recon) ** 2, axis=1)
        return errs

    def predict(self, X: np.ndarray, *, batch_size: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (scores, labels) with labels = (scores >= threshold).astype(int)
        """
        scores = self.score(X, batch_size=batch_size)
        labels = (scores >= self.threshold).astype(np.int32)
        return scores, labels


class IFDetector:
    """
    Optional Isolation Forest detector:
    - Loads joblib IF model and uses -score_samples as anomaly score (higher = more anomalous)
    """
    def __init__(self, model_dir: str, dataset: str):
        self.paths = _artifact_paths(model_dir, dataset)
        self.iforest = joblib.load(self.paths["iforest"])

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(X.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
                          if hasattr(X, "to_numpy") else np.nan_to_num(X, copy=False)).astype(np.float32)
        return -self.iforest.score_samples(X)


def _artifact_paths(model_dir: str, dataset: str) -> Dict[str, str]:
    """
    Resolves standard artifact paths for a dataset.
    Expected files (produced by train.py + ae_trainer):
      - models/<dataset>/scaler.joblib
      - models/<dataset>/autoencoder.pth
      - models/<dataset>/autoencoder_threshold.npy
      - models/<dataset>/autoencoder_config.json
      - models/<dataset>/iforest.joblib (optional)
    """
    base = get_dataset_dir(model_dir, dataset)
    paths = {
        "scaler":   os.path.join(base, "scaler.joblib"),
        "weights":  os.path.join(base, "autoencoder.pth"),
        "thr":      os.path.join(base, "autoencoder_threshold.npy"),
        "cfg":      os.path.join(base, "autoencoder_config.json"),
        "iforest":  os.path.join(base, "iforest.joblib"),
    }
    return paths


# ----------------------------- convenience --------------------------------- #
def load_ae_detector(model_dir: str, dataset: str) -> AEDetector:
    """
    Convenience function: build a detector in one call.
    """
    return AEDetector(model_dir=model_dir, dataset=dataset)

# todo: has the capability to work realtime
# ------------------------------ quick test --------------------------------- #
if __name__ == "__main__":
    # Example manual run (adjust paths/dataset)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
    DATASET = "CIC-IDS2017"

    # Fake example data: load the scaler, sample a few zeros with right shape
    det = load_ae_detector(MODEL_DIR, DATASET)
    print(f"Loaded AE for {DATASET}: hidden={det.hidden_dims}, latent={det.latent_dim}, thr={det.threshold:.4e}")

    # Minimal shape check (N, D)
    dummy = np.zeros((5, det.input_dim), dtype=np.float32)
    s, y = det.predict(dummy)
    print("scores:", s, "labels:", y)

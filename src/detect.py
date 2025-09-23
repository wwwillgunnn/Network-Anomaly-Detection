"""
detect.py
---------
Minimal inference helpers for CIC-IDS2017:
- Loads scaler, IsolationForest (optional), Autoencoder + threshold
- Produces per-flow anomaly scores in [0,1] and boolean anomalies

Public entrypoint:
    predict_scores(df: pd.DataFrame) -> dict with keys:
        "ae": {"scores": np.ndarray, "anomalies": np.ndarray}
        "iforest": {"scores": np.ndarray, "anomalies": np.ndarray}   # only if model present

Notes:
- Keep incoming columns aligned to features.py normalized names (underscored).
- For production API, wrap `predict_scores` in your FastAPI/Flask route.
"""

from __future__ import annotations
import os
from typing import Dict, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import torch
from autoencoder import Autoencoder  # same class used in training
from features import FEAT_COLS  # your canonical list (17 cols)

# --------- Paths & constants ---------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATASET = "CIC-IDS2017"

# Same compact feature slice used in features.py



# --------- Lazy-loaded artifacts ----------------------------------------------
_scaler: Optional[StandardScaler] = None
_iforest: Optional[IsolationForest] = None
_ae: Optional[torch.nn.Module] = None
_ae_thr: Optional[float] = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_scaler() -> StandardScaler:
    global _scaler
    if _scaler is None:
        path = os.path.join(MODEL_DIR, f"{DATASET}_scaler.joblib")
        _scaler = joblib.load(path)
        print(f"🔌 Loaded scaler: {path}")
    return _scaler


def _load_iforest() -> Optional[IsolationForest]:
    global _iforest
    if _iforest is None:
        path = os.path.join(MODEL_DIR, f"{DATASET}_iforest.joblib")
        if os.path.exists(path):
            _iforest = joblib.load(path)
            print(f"🌲 Loaded IsolationForest: {path}")
        else:
            print("ℹ️ IsolationForest model not found (skipping IF scoring).")
            _iforest = None
    return _iforest


def _load_autoencoder() -> tuple[torch.nn.Module, float]:
    global _ae, _ae_thr
    if _ae is None:
        weight_path = os.path.join(MODEL_DIR, f"{DATASET}_autoencoder.pth")
        cfg_path = os.path.join(MODEL_DIR, f"{DATASET}_autoencoder_config.json")

        if os.path.exists(cfg_path):
            import json
            with open(cfg_path) as f:
                cfg = json.load(f)
        else:
            # LAST-RESORT fallback if no config saved: mirror training defaults you actually used
            cfg = dict(input_dim=len(FEAT_COLS), hidden_dims=[512,128], latent_dim=32, dropout=0.0)

        _ae = Autoencoder(**cfg).to(_device)
        _ae.load_state_dict(torch.load(weight_path, map_location=_device), strict=True)
        _ae.eval()
        print(f"🧠 Loaded Autoencoder: {weight_path}")

    if _ae_thr is None:
        thr_path = os.path.join(MODEL_DIR, f"{DATASET}_autoencoder_threshold.npy")
        _ae_thr = float(np.load(thr_path)[0])
        print(f"🎯 Loaded AE threshold: {thr_path} (≈ FPR 1%)")
    return _ae, _ae_thr


# --------- Feature prep for online data ---------------------------------------
def vectorize_for_online(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0                       # add missing as 0
    X = df.reindex(columns=feat_cols).copy()  # exact order; drop extras
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X


# --------- Scoring -------------------------------------------------------------
def _score_ae(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    AE score: reconstruction MSE per row.
    Score normalization to [0,1]: score = clip(MSE / threshold, 0, 1).
    Anomaly decision: score > 1.0  ==> anomaly (equivalent to MSE > threshold).
    """
    ae, thr = _load_autoencoder()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32, device=_device)
        recon = ae(Xt).cpu().numpy()
    mse = np.mean((X - recon) ** 2, axis=1)  # raw scores (higher = more anomalous)

    # Normalize to [0,1] against the tuned threshold
    eps = 1e-12
    norm = np.clip(mse / max(thr, eps), 0.0, 1.0)
    anomalies = (mse > thr).astype(bool)
    return norm, anomalies


def _score_iforest(X: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    IF score: s = -decision_function(X) (higher = more anomalous).
    Normalize to [0,1] using batch 99th percentile to avoid persisting extra state.
    Anomaly decision: model.predict(X) == -1
    """
    iforest = _load_iforest()
    if iforest is None:
        return None
    raw = -iforest.decision_function(X)
    # Batch-relative normalization to [0,1] (simple, robust):
    q99 = np.percentile(raw, 99.0) if raw.size else 1.0
    q99 = q99 if q99 > 0 else 1.0
    norm = np.clip(raw / q99, 0.0, 1.0)
    anomalies = (iforest.predict(X) == -1)
    return norm, anomalies


# --------- Public API ----------------------------------------------------------
def predict_scores(df: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Given a DataFrame of flows, return per-model anomaly scores/anomalies.
    Returns a dict like:
        {
          "ae": {"scores": np.ndarray, "anomalies": np.ndarray},
          "iforest": {"scores": np.ndarray, "anomalies": np.ndarray}   # only if model present
        }
    """
    # Vectorize & scale
    X_df = vectorize_for_online(df, FEAT_COLS)
    scaler = _load_scaler()
    X = scaler.transform(X_df)

    out: Dict[str, Dict[str, np.ndarray]] = {}

    # Autoencoder
    ae_scores, ae_anoms = _score_ae(X)
    out["ae"] = {"scores": ae_scores, "anomalies": ae_anoms}

    # Isolation Forest (optional)
    if_result = _score_iforest(X)
    if if_result is not None:
        if_scores, if_anoms = if_result
        out["iforest"] = {"scores": if_scores, "anomalies": if_anoms}

    return out


# --------- Example usage -------------------------------------------------------
if __name__ == "__main__":
    # Example: run detection on the first N rows of your processed CSV (no labels needed)
    sample_path = os.path.join(PROJECT_ROOT, "data", "processed", f"{DATASET}_clean.csv")
    df = pd.read_csv(sample_path, nrows=1024, low_memory=False)

    results = predict_scores(df)
    for model_name, res in results.items():
        scores = res["scores"]
        anoms = res["anomalies"]
        print(f"{model_name}: mean score={scores.mean():.4f}, anomalies={anoms.sum()}/{len(anoms)}")

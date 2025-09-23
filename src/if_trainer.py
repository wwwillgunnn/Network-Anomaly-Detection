"""
iforest_trainer.py
Tiny wrapper to train IsolationForest and return anomaly scores (higher=worse).
"""
from __future__ import annotations
import os
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score
from utils.io import get_dataset_dir

def train_and_score_iforest(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray | None,
    *,
    model_dir: str,
    dataset: str,
    contamination: float | None = None,
) -> np.ndarray:
    """
    Train IF and return -decision_function(X_test) as scores (higher = more anomalous).
    If contamination is None, estimate from y_test (clipped to [1%, 10%]) else default to 5%.
    """
    # Create dataset-specific subfolder
    out_dir = get_dataset_dir(model_dir, dataset)

    # Contamination heuristic
    if contamination is None:
        contamination = 0.05 if y_test is None else float(np.clip(np.mean(y_test), 0.01, 0.10))

    # Train Isolation Forest
    iforest = IsolationForest(
        n_estimators=413,
        contamination=contamination,
        max_samples=0.30681690586247196,
        max_features=0.8067859336839047,
        random_state=42,
        n_jobs=-1,
        bootstrap=False,
    ).fit(X_train)

    # Scores (higher = more anomalous)
    scores = -iforest.decision_function(X_test)

    # Report metrics if labels provided
    if y_test is not None:
        try:
            roc_auc = roc_auc_score(y_test, scores)
            ap = average_precision_score(y_test, scores)
            print(f"IF — ROC-AUC: {roc_auc:.4f}  AP: {ap:.4f}")
        except Exception:
            pass

    # Save model artifact
    model_path = os.path.join(out_dir, "iforest.joblib")
    joblib.dump(iforest, model_path)
    print(f"💾 Saved IsolationForest → {model_path}")

    return scores

"""
This file is an isolation forest trainer.

What it does:
1) Choose/estimate contamination (attack rate) from y_test if provided (clipped to 1–10%), otherwise default to 5%.
2) Train IsolationForest on X_train with fixed hyperparameters for reproducibility.
3) Score X_test using -decision_function (so higher = worse/anomalous).
4) If y_test is given, print ROC-AUC and AP.
5) Save the fitted model as <model_dir>/<dataset>/iforest.joblib.
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
    # Create (or reuse) a dataset-specific output directory
    out_dir = get_dataset_dir(model_dir, dataset)

    # Contamination heuristic
    if contamination is None:
        contamination = 0.05 if y_test is None else float(np.clip(np.mean(y_test), 0.01, 0.10))

    # Train Isolation Forest, this is based on some optimal features
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

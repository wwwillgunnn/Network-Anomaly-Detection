import os
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import joblib

from features import load_processed, prepare_features
from ae_trainer import train_and_score_autoencoder
from if_trainer import train_and_score_iforest
from lstm_trainer import train_and_score_lstm
from metrics import full_evaluation
from utils.io import get_dataset_dir


# ----------------- Helpers for dashboard artifacts -----------------
def save_eval_artifacts(
    out_dir: Path,
    split: str,                       # "test" or "val"
    scores: np.ndarray,               # anomaly scores (higher = more anomalous)
    labels: np.ndarray,               # 0/1
    timestamps: np.ndarray | None = None,
    roc_auc: float | None = None,
    ap: float | None = None,
    thr_at_1pct_fpr: float | None = None,
    model_key: str | None = None,     # e.g., "iforest", "autoencoder", "lstm"
):
    """Save model-specific arrays so the Streamlit app can load per-model scores."""
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{model_key}" if model_key else ""

    # scores_<model>.npy
    np.save(out_dir / f"{split}_scores{suffix}.npy", np.asarray(scores, dtype=np.float32))

    # labels are shared across models → keep a single file
    labels_path = out_dir / f"{split}_labels.npy"
    if not labels_path.exists():
        np.save(labels_path, np.asarray(labels, dtype=np.int32))

    # optional timestamps (shared)
    if timestamps is not None:
        np.save(out_dir / "timestamps.npy", np.asarray(timestamps))

    # lightweight KPIs in summary.json
    summary = {}
    if roc_auc is not None:
        summary["roc_auc"] = float(roc_auc)
    if ap is not None:
        summary["ap"] = float(ap)
    if thr_at_1pct_fpr is not None:
        summary["thr_fpr_1pct"] = float(thr_at_1pct_fpr)
    if summary:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def threshold_at_target_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 0.01) -> float:
    """Compute the decision threshold that achieves FPR <= target_fpr (choose highest such threshold; else closest)."""
    fpr, tpr, thresholds = roc_curve(y_true.astype(int), scores.astype(float))
    idxs = np.where(fpr <= target_fpr)[0]
    if idxs.size > 0:
        idx = idxs[-1]
    else:
        idx = int(np.argmin(np.abs(fpr - target_fpr)))
    return float(thresholds[idx])


# ----------------- Project constants -----------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def split_and_scale(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    # Guardrails against inf/nan
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train.to_numpy(), y_test.to_numpy(), scaler


def main():
    dataset = "CIC-IDS2017"
    print(f"🔹 Loading processed dataset: {dataset}")
    df = load_processed()
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    # Save scaler in models/<dataset>/
    out_dir_str = get_dataset_dir(MODEL_DIR, dataset)  # may return str
    out_dir = Path(out_dir_str)
    scaler_path = out_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"💾 Saved scaler → {scaler_path}")

    # ----------------- Isolation Forest -----------------
    if_scores = train_and_score_iforest(
        X_train, X_test, y_test, model_dir=MODEL_DIR, dataset=dataset
    )
    print("✅ IsolationForest @1%FPR")
    if_metrics = full_evaluation(y_test, if_scores, dataset, "iforest", MODEL_DIR)

    # KPIs for IF (used for summary.json)
    try:
        roc_if = float(roc_auc_score(y_test.astype(int), if_scores.astype(float)))
        ap_if = float(average_precision_score(y_test.astype(int), if_scores.astype(float)))
        thr_if_1pct = threshold_at_target_fpr(y_test, if_scores, target_fpr=0.01)
    except Exception:
        roc_if = ap_if = thr_if_1pct = None

    save_eval_artifacts(
        out_dir=out_dir,
        split="test",
        scores=if_scores,
        labels=y_test,
        timestamps=None,
        roc_auc=if_metrics.get("roc_auc") if isinstance(if_metrics, dict) else roc_if,
        ap=if_metrics.get("ap") if isinstance(if_metrics, dict) else ap_if,
        thr_at_1pct_fpr=thr_if_1pct,
        model_key="iforest",
    )

    # ----------------- Autoencoder -----------------
    ae_scores, ae_thr = train_and_score_autoencoder(
        X_train, X_test, y_train,
        model_dir=MODEL_DIR, dataset=dataset
    )
    print("✅ Autoencoder @1%FPR")
    ae_metrics = full_evaluation(y_test, ae_scores, dataset, "autoencoder", MODEL_DIR)

    # KPIs for AE (used for summary.json)
    try:
        roc_ae = float(roc_auc_score(y_test.astype(int), ae_scores.astype(float)))
        ap_ae = float(average_precision_score(y_test.astype(int), ae_scores.astype(float)))
        thr_ae_1pct = threshold_at_target_fpr(y_test, ae_scores, target_fpr=0.01)
    except Exception:
        roc_ae = ap_ae = thr_ae_1pct = None

    save_eval_artifacts(
        out_dir=out_dir,
        split="test",
        scores=ae_scores,
        labels=y_test,
        timestamps=None,
        roc_auc=ae_metrics.get("roc_auc") if isinstance(ae_metrics, dict) else roc_ae,
        ap=ae_metrics.get("ap") if isinstance(ae_metrics, dict) else ap_ae,
        thr_at_1pct_fpr=ae_thr if ae_thr is not None else thr_ae_1pct,  # prefer trainer's threshold if it returns one
        model_key="autoencoder",
    )

    # ----------------- LSTM -----------------
    lstm_scores = train_and_score_lstm(
        X_train, X_test, y_train, y_test, model_dir=MODEL_DIR, dataset=dataset, epochs=12, batch_size=512
    )
    print("✅ LSTM @1%FPR")
    lstm_metrics = full_evaluation(y_test, lstm_scores, dataset, "lstm", MODEL_DIR)

    # KPIs for LSTM (used for summary.json)
    try:
        roc_lstm = float(roc_auc_score(y_test.astype(int), lstm_scores.astype(float)))
        ap_lstm = float(average_precision_score(y_test.astype(int), lstm_scores.astype(float)))
        thr_lstm_1pct = threshold_at_target_fpr(y_test, lstm_scores, target_fpr=0.01)
    except Exception:
        roc_lstm = ap_lstm = thr_lstm_1pct = None

    save_eval_artifacts(
        out_dir=out_dir,
        split="test",
        scores=lstm_scores,
        labels=y_test,
        timestamps=None,
        roc_auc=lstm_metrics.get("roc_auc") if isinstance(lstm_metrics, dict) else roc_lstm,
        ap=lstm_metrics.get("ap") if isinstance(lstm_metrics, dict) else ap_lstm,
        thr_at_1pct_fpr=thr_lstm_1pct,
        model_key="lstm",
    )


if __name__ == "__main__":
    main()
 
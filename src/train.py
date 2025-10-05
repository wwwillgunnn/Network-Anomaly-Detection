"""
This file is the main training pipeline for network anomaly detection.

What it does:
1) Loads and prepares the cleaned dataset (via `load_processed` + `prepare_features`).
2) Splits data into training and test sets, standardizes features, and saves the scaler.
3) Trains and evaluates three models:
   - Isolation Forest
   - Autoencoder
   - LSTM
4) Computes evaluation metrics (ROC-AUC, precision, recall, etc.)
   and saves model artifacts under /models/<dataset>/.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from features import load_processed, prepare_features
from ae_trainer import train_and_score_autoencoder
from if_trainer import train_and_score_iforest
from metrics import full_evaluation
from utils.io import get_dataset_dir, project_root
from lstm_trainer import train_and_score_lstm

PROJECT_ROOT = project_root(__file__)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def split_and_scale(X: pd.DataFrame, y: pd.Series):
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    # Make everything finite (guardrails)
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_test  = X_test.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Standardize features
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train.to_numpy(), y_test.to_numpy(), scaler

def main():
    # TODO: Try with other datasets
    dataset = "CIC-IDS2017"
    print(f"🔹 Loading processed dataset: {dataset}")

    # Load and prepare model-ready data
    df = load_processed()
    X, y = prepare_features(df)

    # Split, scale, and dave the scaler
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    out_dir = get_dataset_dir(MODEL_DIR, dataset)
    scaler_path = os.path.join(out_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"💾 Saved scaler → {scaler_path}")

    # --- Isolation Forest ----------------------------------------------------
    if_scores = train_and_score_iforest(X_train, X_test, y_test, model_dir=MODEL_DIR, dataset=dataset)
    print("✅ IsolationForest @1%FPR")
    if_metrics = full_evaluation(y_test, if_scores, dataset, "iforest", MODEL_DIR)

    # --- Autoencoder ---------------------------------------------------------
    ae_scores, ae_thr = train_and_score_autoencoder(
        X_train, X_test, y_train, model_dir=MODEL_DIR, dataset=dataset,
    )
    print(f"✅ Autoencoder @1%FPR")
    ae_metrics = full_evaluation(y_test, ae_scores, dataset, "autoencoder", MODEL_DIR)

    # --- LSTM ---------------------------------------------------------
    lstm_scores = train_and_score_lstm(
        X_train, X_test, y_train, y_test,
        model_dir=MODEL_DIR, dataset=dataset,
        epochs=12, batch_size=512
    )
    print("✅ LSTM @1%FPR")
    lstm_metrics = full_evaluation(y_test, lstm_scores, dataset, "lstm", MODEL_DIR)

    # Todo: Combine and save metrics summary to CSV for comparison


if __name__ == "__main__":
    main()

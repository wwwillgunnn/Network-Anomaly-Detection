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
from utils.io import get_dataset_dir
from lstm_trainer import train_and_score_lstm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def split_and_scale(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    # Make everything finite (guardrails)
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_test  = X_test.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train.to_numpy(), y_test.to_numpy(), scaler


def main():
    dataset = "CIC-IDS2017"
    print(f"🔹 Loading processed dataset: {dataset}")
    df = load_processed()
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    # Save scaler in models/<dataset>/
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
        X_train, X_test, y_train, y_test, model_dir=MODEL_DIR, dataset=dataset, fpr_target=0.01
    )
    print(f"✅ Autoencoder @1%FPR")
    ae_metrics = full_evaluation(y_test, ae_scores, dataset, "autoencoder", MODEL_DIR)

    # after you get X_train, X_test, y_train, y_test, scaler
    lstm_scores = train_and_score_lstm(
        X_train, X_test, y_train, y_test,
        model_dir=MODEL_DIR, dataset=dataset,
        epochs=12, batch_size=512
    )
    print("✅ LSTM @1%FPR")
    lstm_metrics = full_evaluation(y_test, lstm_scores, dataset, "lstm", MODEL_DIR)

    # Todo: save to csv?


if __name__ == "__main__":
    main()

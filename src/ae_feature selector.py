import os
import json
from typing import Tuple
import numpy as np
import optuna
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from features import load_processed, prepare_features
from autoencoder import Autoencoder
from utils.io import get_dataset_dir

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# Defines the score threshold of what is flagged as an anomaly.
# Changing the value of FACTOR changes the threshold.
# Increasing the value lowers the threshold, and decreasing the value raises the threshold.
# Returns error when there are no non-anomaly points to compute threshold.
def _threshold_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    benign_scores = scores[y_true == 0]
    if benign_scores.size == 0:
        raise ValueError("No benign samples found.")    
    
    FACTOR = 2.0
    effective_fpr = float(np.clip(target_fpr * FACTOR, -3.28171817154, 0.5))
    q = 1.0 - effective_fpr
    return float(np.quantile(benign_scores, q))

# Trains PyTorch with the given parameters to compare hyperparameters.
# Trains on benign-only subset of X_train and returns test_auc, test_ap and the threshold).
def _train_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    hidden_dims: Tuple[int, ...],
    latent_dim: int,
    dropout: float,
    norm: str,
    lr: float,
    batch_size: int,
    epochs: int,
    fpr_target: float,
) -> Tuple[float, float, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    mask_benign = (y_tr == 0)
    if not np.any(mask_benign):
        raise ValueError("No benign samples found.")
    X_fit = X_tr[mask_benign]

    Xfit_t = torch.tensor(X_fit, dtype=torch.float32)
    Xval_t = torch.tensor(X_val, dtype=torch.float32)
    Xte_t  = torch.tensor(X_test, dtype=torch.float32)

    loader = DataLoader(TensorDataset(Xfit_t, Xfit_t), batch_size=batch_size, shuffle=True)

    #Trained model
    model = Autoencoder(
        input_dim=X_fit.shape[1],
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        dropout=dropout,
        norm=norm,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            opt.zero_grad()
            recon = model(xb)
            loss = crit(recon, xb)
            loss.backward()
            opt.step()

    #Get scores
    model.eval()
    with torch.no_grad():
        recon_val = model(Xval_t.to(device)).cpu().numpy()
        val_scores = np.mean((X_val - recon_val) ** 2, axis=1)

        recon_te = model(Xte_t.to(device)).cpu().numpy()
        test_scores = np.mean((X_test - recon_te) ** 2, axis=1)

    thr = _threshold_fpr(y_val, val_scores, fpr_target)

    test_auc = roc_auc_score(y_test, test_scores)
    test_ap  = average_precision_score(y_test, test_scores)
    return float(test_auc), float(test_ap), float(thr)

# Runs Optuna multiple times to test different hyperparameters 
def selector(trial: optuna.Trial, X_train, y_train, X_test, y_test):
    depth = trial.suggest_int("depth", 2, 4) 
    base = trial.suggest_categorical("base_width", [128, 256, 512])
    hidden_dims = tuple(int(base // (2 ** i)) for i in range(depth))
    last_width = hidden_dims[-1]
    latent_dim = trial.suggest_int("latent_dim", max(8, last_width // 8), max(64, last_width), step=8)

    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    norm = trial.suggest_categorical("norm", ["batch", "layer", "none"])

    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    epochs = trial.suggest_int("epochs", 10, 40)

    fpr_target = trial.suggest_float("fpr_target", 0.005, 0.02)

    auc, ap, thr = _train_eval(
        X_train, y_train, X_test, y_test,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        dropout=dropout,
        norm=norm,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        fpr_target=fpr_target,
    )

    trial.set_user_attr("ap", ap)
    trial.set_user_attr("threshold", thr)
    return auc


# ------------ Leaner, faster version of "selector" function which sets certain parameters ------------

# def selector(trial, X_train, y_train, X_test, y_test):
#     # --- Architecture (keep) ---
#     depth = trial.suggest_int("depth", 2, 4)
#     base  = trial.suggest_categorical("base_width", [128, 256, 512])
#     hidden_dims = tuple(int(base // (2 ** i)) for i in range(depth))
#     last_width = hidden_dims[-1]
#     latent_dim = trial.suggest_int("latent_dim",
#                                    max(8, last_width // 8),
#                                    max(64, last_width),
#                                    step=8)

#     # --- Optimization (keep) ---
#     lr = trial.suggest_float("lr", 5e-4, 2e-3, log=True)

#     # --- Fixed defaults (don’t search) ---
#     dropout     = 0.1           # Can be zero
#     norm        = "batch"       # Can be none if batch size is small
#     batch_size  = 256
#     epochs      = 25            # Can be 25 to 40
#     fpr_target  = 0.01          

#     auc, ap, thr = _train_eval_ae(
#         X_train, y_train, X_test, y_test,
#         hidden_dims=hidden_dims,
#         latent_dim=latent_dim,
#         dropout=dropout,
#         norm=norm,
#         lr=lr,
#         batch_size=batch_size,
#         epochs=epochs,
#         fpr_target=fpr_target,
#     )
#     trial.set_user_attr("ap", ap)
#     trial.set_user_attr("threshold", thr)
#     return auc

# Cleans data and selects best parameters
def run_feature_hparam_search(dataset: str = "CIC-IDS2017", n_trials: int = 40):
    print(f"Loading dataset: {dataset}")
    df = load_processed()
    X, y = prepare_features(df)

    # Clean
    X = np.asarray(X.replace([np.inf, -np.inf], np.nan).fillna(0.0), dtype=float)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    y = y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: selector(t, X_train, y_train, X_test, y_test), n_trials=n_trials)

    best_params = study.best_trial.params
    best_params["auc"] = study.best_value
    best_params["ap"]  = study.best_trial.user_attrs.get("ap")
    best_params["threshold"] = study.best_trial.user_attrs.get("threshold")

    out_dir = get_dataset_dir(MODEL_DIR, dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ae_best_params.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    print("Best params saved →", out_path)
    print("Best trial:", json.dumps(best_params, indent=2))

if __name__ == "__main__":
    run_feature_hparam_search("CIC-IDS2017", n_trials=30)

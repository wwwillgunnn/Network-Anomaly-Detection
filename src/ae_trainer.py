"""
Train a benign-only autoencoder and return test anomaly scores + threshold.
Rationale: keep train.py thin and make AE reusable by the inference pipeline.
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
    """Pick threshold so that FPR ~= target_fpr using the benign-score quantile (higher=more anomalous)."""
    benign = scores[y_true == 0]
    if benign.size == 0:
        raise ValueError("No benign samples available to compute FPR-based threshold.")
    q = np.clip(1.0 - target_fpr, 0.0, 1.0)
    return float(np.quantile(benign, q))


def train_and_score_autoencoder(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    model_dir: str,
    dataset: str,
    epochs: int = 15,
    batch_size: int = 256,
    fpr_target: float = 0.01,
) -> Tuple[np.ndarray, float]:
    """
    Train AE on BENIGN-only training data. Tune threshold on validation to hit FPR≈fpr_target.
    Returns:
        test_scores: np.ndarray of reconstruction MSE (higher = more anomalous)
        threshold: float threshold chosen on validation
    """
    # Create dataset-specific subfolder
    out_dir = get_dataset_dir(model_dir, dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Split train into (fit, val); use only benign from fit split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    mask_benign = (y_tr == 0)
    if not mask_benign.any():
        raise ValueError("No benign samples in training split.")
    X_fit = X_tr[mask_benign]

    # Tensors + loader
    Xfit_t = torch.tensor(X_fit, dtype=torch.float32)
    Xval_t = torch.tensor(X_val, dtype=torch.float32)
    Xte_t  = torch.tensor(X_test, dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xfit_t, Xfit_t), batch_size=batch_size, shuffle=True)

    # Model
    input_dim = X_fit.shape[1]
    model = Autoencoder(
        input_dim=input_dim,
        hidden_dims=(512, 128),
        latent_dim=32,
        dropout=0.1,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()

    # Train
    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        for (xb, _) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            recon = model(xb)
            loss = crit(recon, xb)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        print(f"AE Epoch {ep+1}/{epochs} — loss: {ep_loss/len(loader):.6f}")

    # Scores (higher = more anomalous)
    model.eval()
    with torch.no_grad():
        recon_val = model(Xval_t.to(device)).cpu().numpy()
        val_scores = np.mean((X_val - recon_val) ** 2, axis=1)

        recon_te = model(Xte_t.to(device)).cpu().numpy()
        test_scores = np.mean((X_test - recon_te) ** 2, axis=1)

    # Threshold tuned on validation
    threshold = _threshold_at_fpr(y_val, val_scores, target_fpr=fpr_target)

    # Optional summary
    try:
        val_auc = roc_auc_score(y_val, val_scores)
        val_ap  = average_precision_score(y_val, val_scores)
        print(f"AE val — ROC-AUC: {val_auc:.4f}  AP: {val_ap:.4f}  thr@FPR≈{fpr_target:.0%}: {threshold:.4e}")
    except Exception:
        pass

    # Save artifacts in models/<dataset>/
    model_path = os.path.join(out_dir, "autoencoder.pth")
    thr_path   = os.path.join(out_dir, "autoencoder_threshold.npy")

    torch.save(model.state_dict(), model_path)
    np.save(thr_path, np.array([threshold]))
    print(f"💾 Saved Autoencoder weights → {model_path}")
    print(f"💾 Saved validation-tuned threshold → {thr_path}")

    return test_scores, threshold


# could be saucy...
"""
Autoencoder trainer: GPU-optimized (AMP, pinned memory, non-blocking copies).
- Trains on BENIGN-only subset of the training split.
- Picks threshold on validation to hit target FPR.
- Returns test anomaly scores + chosen threshold.
"""

# from __future__ import annotations
# import os
# import platform
# from typing import Tuple
#
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score, average_precision_score
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
#
# from autoencoder import Autoencoder  # ensure your module path is correct
#
#
# def _threshold_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 0.01) -> float:
#     """
#     Pick threshold so that FPR ~= target_fpr using the benign-score quantile (higher=more anomalous).
#     """
#     benign = scores[y_true == 0]
#     if benign.size == 0:
#         raise ValueError("No benign samples available to compute FPR-based threshold.")
#     q = float(np.clip(1.0 - target_fpr, 0.0, 1.0))
#     return float(np.quantile(benign, q))
#
#
# def _batched_recon(
#     model: torch.nn.Module,
#     tensor_cpu_float32: torch.Tensor,
#     device: torch.device,
#     *,
#     batch_size: int,
#     use_cuda_amp: bool,
# ) -> np.ndarray:
#     """
#     Run model forward in batches to reconstruct inputs. Accepts a CPU float32 tensor,
#     streams to device with non_blocking copies, returns CPU numpy array.
#     """
#     model.eval()
#     outs = []
#     with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=use_cuda_amp):
#         for i in range(0, tensor_cpu_float32.size(0), batch_size):
#             xb = tensor_cpu_float32[i : i + batch_size].to(device, non_blocking=True)
#             recon = model(xb).float().cpu()  # ensure fp32 on CPU
#             outs.append(recon)
#     return torch.cat(outs, dim=0).numpy()
#
#
# def train_and_score_autoencoder(
#     X_train: np.ndarray,
#     X_test: np.ndarray,
#     y_train: np.ndarray,
#     y_test: np.ndarray,
#     *,
#     model_dir: str,
#     dataset: str,
#     epochs: int = 15,
#     batch_size: int = 1024,     # tune based on VRAM
#     fpr_target: float = 0.01,
# ) -> Tuple[np.ndarray, float]:
#     """
#     Train AE on BENIGN-only training data. Tune threshold on validation to hit FPR≈fpr_target.
#     Returns:
#         test_scores: np.ndarray of reconstruction MSE (higher = more anomalous)
#         threshold: float threshold chosen on validation
#     """
#     os.makedirs(model_dir, exist_ok=True)
#
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda:0" if use_cuda else "cpu")
#     if use_cuda:
#         torch.backends.cudnn.benchmark = True  # heuristic autotuner
#         # On Ampere+ this can slightly speed up matmuls:
#         try:
#             torch.set_float32_matmul_precision("high")
#         except Exception:
#             pass
#
#     # Split train into (fit, val); use only BENIGN from fit split
#     X_tr, X_val, y_tr, y_val = train_test_split(
#         X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
#     )
#     mask_benign = (y_tr == 0)
#     if not mask_benign.any():
#         raise ValueError("No benign samples in training split.")
#     X_fit = X_tr[mask_benign]
#
#     # CPU tensors; DataLoader will pin them for faster H2D copies
#     Xfit_t = torch.from_numpy(X_fit).to(torch.float32)
#     Xval_t = torch.from_numpy(X_val).to(torch.float32)
#     Xte_t  = torch.from_numpy(X_test).to(torch.float32)
#
#     loader = DataLoader(
#         TensorDataset(Xfit_t, Xfit_t),
#         batch_size=batch_size,
#         shuffle=True,
#         pin_memory=use_cuda,
#         num_workers=min(8, os.cpu_count() or 1),
#         persistent_workers=True if use_cuda else False,
#         drop_last=False,
#     )
#
#     # Model
#     input_dim = X_fit.shape[1]
#     model = Autoencoder(
#         input_dim=input_dim,
#         hidden_dims=(512, 128),
#         latent_dim=32,
#         dropout=0.1,
#     ).to(device)
#
#     # Optional extra perf: torch.compile (skip on Windows to avoid 'cl' error)
#     if use_cuda and platform.system() != "Windows":
#         try:
#             model = torch.compile(model)  # backend="inductor" by default
#         except Exception as e:
#             print(f"Skipping torch.compile: {e}")
#
#     opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
#     crit = nn.SmoothL1Loss(beta=1.0)  # Huber loss: robust vs outliers
#     scaler = torch.amp.GradScaler(enabled=use_cuda)
#
#     # ---- Train --------------------------------------------------------------
#     model.train()
#     for ep in range(epochs):
#         ep_loss = 0.0
#         for (xb, _) in loader:
#             xb = xb.to(device, non_blocking=True)
#
#             opt.zero_grad(set_to_none=True)
#             with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
#                 recon = model(xb)
#                 loss = crit(recon, xb)
#
#             scaler.scale(loss).backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             scaler.step(opt)
#             scaler.update()
#
#             ep_loss += loss.item()
#
#         print(f"AE Epoch {ep+1}/{epochs} — loss: {ep_loss/len(loader):.6f}")
#
#     # ---- Score (higher = more anomalous) -----------------------------------
#     recon_val = _batched_recon(model, Xval_t, device, batch_size=batch_size, use_cuda_amp=use_cuda)
#     val_scores = np.mean((X_val - recon_val) ** 2, axis=1)
#
#     recon_te = _batched_recon(model, Xte_t, device, batch_size=batch_size, use_cuda_amp=use_cuda)
#     test_scores = np.mean((X_test - recon_te) ** 2, axis=1)
#
#     # ---- Threshold tuned on validation -------------------------------------
#     threshold = _threshold_at_fpr(y_val, val_scores, target_fpr=fpr_target)
#
#     # ---- Optional summary ---------------------------------------------------
#     try:
#         val_auc = roc_auc_score(y_val, val_scores)
#         val_ap  = average_precision_score(y_val, val_scores)
#         print(f"AE val — ROC-AUC: {val_auc:.4f}  AP: {val_ap:.4f}  thr@FPR≈{fpr_target:.0%}: {threshold:.4e}")
#     except Exception:
#         pass
#
#     # ---- Save artifacts -----------------------------------------------------
#     torch.save(model.state_dict(), os.path.join(model_dir, f"{dataset}_autoencoder.pth"))
#     np.save(os.path.join(model_dir, f"{dataset}_autoencoder_threshold.npy"), np.array([threshold]))
#     print(f"💾 Saved Autoencoder weights and validation-tuned threshold (device={device}).")
#
#     return test_scores, threshold

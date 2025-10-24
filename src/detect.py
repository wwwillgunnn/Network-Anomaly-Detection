# File: src/detect.py
from __future__ import annotations
import os
import json
from typing import Tuple, Dict, List
import numpy as np
import joblib
import torch
import torch.nn as nn

# Resilient import of Autoencoder
try:
    from .autoencoder import Autoencoder  # type: ignore
except Exception:
    from autoencoder import Autoencoder  # type: ignore


def get_dataset_dir(model_dir: str, dataset: str) -> str:
    return os.path.join(model_dir, dataset)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batched_recon(model: nn.Module, X: np.ndarray, *, batch_size: int = 1024) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    out: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(device=device, dtype=torch.float32, non_blocking=True)
            recon = model(xb).float().cpu().numpy()
            out.append(recon)
    return np.concatenate(out, axis=0) if out else np.empty_like(X)


def _artifact_paths(model_dir: str, dataset: str) -> Dict[str, str]:
    base = get_dataset_dir(model_dir, dataset)
    return {
        "scaler":   os.path.join(base, "scaler.joblib"),
        "weights":  os.path.join(base, "autoencoder.pth"),
        "thr":      os.path.join(base, "autoencoder_threshold.npy"),
        "cfg":      os.path.join(base, "autoencoder_config.json"),  # optional
        "iforest":  os.path.join(base, "iforest.joblib"),
    }


def _infer_cfg_from_weights(weights_path: str, input_dim: int) -> Dict[str, object]:
    """
    Infer hidden_dims and latent_dim from a Linear encoder stack in the state_dict.
    Heuristic: follow the chain of 2D weight matrices whose second dimension matches
    the previous layer's output size, starting from input_dim. Mirror/decoder layers
    are ignored automatically because their weight shapes won't chain from input_dim.
    """
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]  # support lightning-style checkpoints

    # collect (name, tensor) for linear weights
    linear = [(k, v) for k, v in state.items() if isinstance(v, torch.Tensor) and v.ndim == 2]
    # Try to keep original order
    def name_key(n: str) -> tuple:
        # extract numeric indices to sort naturally (encoder.0.weight, encoder.2.weight, ...)
        parts = []
        for p in n.replace("[", ".").replace("]", ".").split("."):
            parts.append(int(p) if p.isdigit() else p)
        return tuple(parts)
    linear.sort(key=lambda kv: name_key(kv[0]))

    dims: List[int] = [input_dim]  # track the chain of dimensions [in, h1, h2, ..., latent]
    cur = input_dim
    for name, w in linear:
        in_features = w.shape[1]
        out_features = w.shape[0]
        if in_features == cur:
            dims.append(int(out_features))
            cur = int(out_features)

    if len(dims) < 2:
        # Fallback: try best guess from any first weight that matches input_dim
        candidates = [int(w.shape[0]) for _, w in linear if int(w.shape[1]) == input_dim]
        if candidates:
            dims = [input_dim, candidates[0]]
        else:
            # Final fallback
            dims = [input_dim, max(8, input_dim // 2)]

    hidden_dims = tuple(int(x) for x in dims[1:-1]) if len(dims) > 2 else (int(dims[1]),)
    latent_dim = int(dims[-1])

    # sensible defaults if not stored
    return {
        "input_dim": int(input_dim),
        "hidden_dims": list(hidden_dims),
        "latent_dim": int(latent_dim),
        "dropout": 0.0,
        "norm": "batch",
    }


class AEDetector:
    """
    Encapsulates Autoencoder inference:
    - Loads scaler, AE config (or infers it), weights, and threshold.
    - .score(X) -> anomaly scores; .predict(X) -> (scores, labels)
    """
    def __init__(self, model_dir: str, dataset: str):
        self.dataset = dataset
        self.paths = _artifact_paths(model_dir, dataset)

        # Load scaler (needed for input_dim)
        self.scaler = joblib.load(self.paths["scaler"])
        # Try to get feature names for UI convenience
        self.feature_names = list(getattr(self.scaler, "feature_names_in_", []))
        input_dim_from_scaler = int(getattr(self.scaler, "n_features_in_", len(self.feature_names) or 0))

        # Load config if present; else infer from weights + scaler
        cfg = None
        if os.path.exists(self.paths["cfg"]):
            try:
                with open(self.paths["cfg"], "r") as f:
                    cfg = json.load(f)
            except Exception:
                cfg = None

        if not cfg:
            if input_dim_from_scaler <= 0:
                raise RuntimeError("Cannot infer input_dim: scaler lacks n_features_in_. Refit or save with feature names.")
            cfg = _infer_cfg_from_weights(self.paths["weights"], input_dim_from_scaler)
            # Save for next runs (best-effort)
            try:
                with open(self.paths["cfg"], "w") as f:
                    json.dump(cfg, f, indent=2)
            except Exception:
                pass  # non-fatal

        # Pull fields
        self.hidden_dims = tuple(int(x) for x in cfg.get("hidden_dims", []))
        self.latent_dim = int(cfg.get("latent_dim", 16))
        self.dropout = float(cfg.get("dropout", 0.0))
        self.norm = cfg.get("norm", "batch")
        # Prefer scaler-derived input dim if available
        self.input_dim = int(cfg.get("input_dim", input_dim_from_scaler or 0)) or input_dim_from_scaler

        # Build model and load weights
        self.model = Autoencoder(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
            norm=self.norm,
        ).to(_device())

        # Load threshold
        thr_arr = np.load(self.paths["thr"])
        self.threshold = float(thr_arr.item() if hasattr(thr_arr, "item") else np.asarray(thr_arr).ravel()[0])

        # --- inside AEDetector.__init__ after building self.model ---

        state = torch.load(self.paths["weights"], map_location=_device())
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        # Safe load: only copy params whose names AND shapes match
        current = self.model.state_dict()
        compatible = {}
        skipped = []
        for k, v in state.items():
            if k in current and current[k].shape == v.shape:
                compatible[k] = v
            else:
                skipped.append((k, tuple(v.shape), tuple(current.get(k, torch.empty(0)).shape)))

        # load what we can
        missing, unexpected = self.model.load_state_dict(compatible, strict=False)

        # (Optional) log a concise summary to help you inspect
        if skipped or missing or unexpected:
            print(
                f"[AEDetector] Loaded {len(compatible)} params. "
                f"Skipped {len(skipped)} mismatched, "
                f"Missing {len(missing)}, Unexpected {len(unexpected)}."


            )
            # If you want to see details during dev:
            # for k, shp_ckpt, shp_cur in skipped:
            #     print(f" - skip {k}: ckpt={shp_ckpt} vs cur={shp_cur}")

    def score(self, X, *, batch_size: int = 1024) -> np.ndarray:
        if hasattr(X, "to_numpy"):
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
        X = np.nan_to_num(np.asarray(X), copy=False).astype(np.float32)
        X_scaled = self.scaler.transform(X).astype(np.float32)
        recon = _batched_recon(self.model, X_scaled, batch_size=batch_size)
        errs = np.mean((X_scaled - recon) ** 2, axis=1)
        return errs

    def predict(self, X, *, batch_size: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        scores = self.score(X, batch_size=batch_size)
        labels = (scores >= self.threshold).astype(np.int32)
        return scores, labels


"""Isolation Forest detector (-score_samples for anomaly score)."""
class IFDetector:
    def __init__(self, model_dir: str, dataset: str):
        self.paths = _artifact_paths(model_dir, dataset)
        self.iforest = joblib.load(self.paths["iforest"])
        self.scaler = joblib.load(self.paths["scaler"])

    def score(self, X) -> np.ndarray:
        if hasattr(X, "to_numpy"):
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
        X = np.nan_to_num(np.asarray(X), copy=False).astype(np.float32)
        Xs = self.scaler.transform(X).astype(np.float32)
        return -self.iforest.score_samples(Xs)

def load_ae_detector(model_dir: str, dataset: str) -> AEDetector:
    return AEDetector(model_dir=model_dir, dataset=dataset)


if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
    DATASET = "CIC-IDS2017"

    det = load_ae_detector(MODEL_DIR, DATASET)
    print(f"Loaded AE for {DATASET}: hidden={det.hidden_dims}, latent={det.latent_dim}, thr={det.threshold:.4e}")
    dummy = np.zeros((5, det.input_dim), dtype=np.float32)
    s, y = det.predict(dummy)
    print("scores:", s, "labels:", y)

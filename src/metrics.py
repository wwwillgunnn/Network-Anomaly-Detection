"""
metrics.py
----------
Shared evaluation helpers for anomaly detection.
Conventions:
- y_true: 0 = benign/normal, 1 = attack/anomaly
- scores: higher = more anomalous
"""

from __future__ import annotations
import os
from typing import Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)


# ----------------------------- Thresholding ---------------------------------- #
def threshold_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 0.01) -> float:
    """
    Choose a threshold so that FPR ~= target_fpr using the benign (y=0) score quantile.
    Rationale: simple, stable for highly imbalanced data.
    """
    benign_scores = scores[y_true == 0]
    if benign_scores.size == 0:
        raise ValueError("No benign samples to compute threshold_at_fpr().")
    q = np.clip(1.0 - target_fpr, 0.0, 1.0)
    return float(np.quantile(benign_scores, q))


def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Binary decisions from scores (1 = anomaly if score > threshold)."""
    return (scores > threshold).astype(int)


def fpr_recall_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Return (FPR, Recall) given binary predictions."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / max(fp + tn, 1)
    rec = tp / max(tp + fn, 1)
    return fpr, rec


# ----------------------------- Reporting ------------------------------------- #
def evaluate_at_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    print_report: bool = True,
) -> Dict[str, float]:
    """
    Apply threshold, print a concise report, and return metrics.
    Returns: dict with auc, ap, fpr, recall, precision, f1, support_pos, support_neg.
    """
    y_pred = apply_threshold(scores, threshold)

    # Summary (classification_report includes precision/recall/F1 by class)
    if print_report:
        print(classification_report(y_true, y_pred, digits=4))

    # Scalar metrics from scores
    metrics = {}
    try:
        metrics["auc_roc"] = float(roc_auc_score(y_true, scores))
        metrics["ap"] = float(average_precision_score(y_true, scores))
    except Exception:
        metrics["auc_roc"] = float("nan")
        metrics["ap"] = float("nan")

    # Derived: FPR and Recall for the anomaly class
    fpr, rec = fpr_recall_from_preds(y_true, y_pred)
    metrics["fpr"] = fpr
    metrics["recall"] = rec

    # Class-wise support (handy for sanity checks)
    metrics["support_neg"] = int((y_true == 0).sum())
    metrics["support_pos"] = int((y_true == 1).sum())

    return metrics


# ----------------------------- Plotting -------------------------------------- #
def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str, save_path: str) -> None:
    """Tiny confusion matrix plot (no seaborn)."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["Normal", "Attack"])
    plt.yticks([0, 1], ["Normal", "Attack"])
    for (i, j), v in np.ndenumerate(cm):
        color = "white" if v > cm.max() / 2 else "black"
        plt.text(j, i, str(v), ha="center", va="center", color=color)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Confusion matrix saved → {save_path}")


def plot_roc_pr(y_true: np.ndarray, scores: np.ndarray, roc_path: str, pr_path: str) -> None:
    """
    Save ROC and PR curves. Useful when comparing models (use same y_true/scores convention).
    Note: no custom colors or styles to keep dependencies light.
    """
    # ROC
    try:
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(roc_path)
        plt.close()
        print(f"📈 ROC saved → {roc_path}")
    except Exception:
        print("ROC could not be computed (check labels/scores).")

    # PR
    try:
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        plt.figure()
        plt.plot(recall, precision, label=f"AP={ap:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(pr_path)
        plt.close()
        print(f"📈 PR saved → {pr_path}")
    except Exception:
        print("PR could not be computed (check labels/scores).")


def full_evaluation(
    y_true: np.ndarray,
    scores: np.ndarray,
    dataset: str,
    model_name: str,
    model_dir: str,
    target_fpr: float = 0.01,
) -> Dict[str, float]:
    """
    Run the full evaluation pipeline for anomaly scores:
      - threshold @ target FPR
      - classification report
      - confusion matrix
      - ROC + PR curves

    Returns:
        dict of metrics (AUC, AP, FPR, Recall, supports).
    """
    thr = threshold_at_fpr(y_true, scores, target_fpr=target_fpr)
    y_pred = apply_threshold(scores, thr)

    # Print + return metrics
    metrics = evaluate_at_threshold(y_true, scores, thr)

    # Save plots into models/<dataset>/figs/
    out_dir = os.path.join(model_dir, dataset, "figs")
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, model_name)

    plot_confusion(
        y_true, y_pred,
        f"{dataset} - {model_name} @FPR≈{target_fpr:.0%}",
        f"{prefix}_cm.png"
    )
    plot_roc_pr(
        y_true, scores,
        f"{prefix}_roc.png",
        f"{prefix}_pr.png"
    )

    return metrics

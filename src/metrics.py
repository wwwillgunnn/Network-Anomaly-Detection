"""
This file has helper functions designed for evaluation.

Conventions:
- y_true: 0 = benign/normal, 1 = attack/anomaly
- scores: higher = more anomalous

Key evaluation metrics:
-----------------------
• ROC-AUC (Receiver Operating Characteristic - Area Under Curve):
  Measures how well a model can distinguish between benign and attack samples.
  It plots True Positive Rate (Recall) vs. False Positive Rate at various thresholds.
  A value of 1.0 = perfect separation; 0.5 = random guessing.

• Average Precision (AP) / PR-AUC:
  Measures the area under the Precision–Recall curve.
  More informative than ROC-AUC for highly imbalanced datasets, as it focuses on
  how well the model identifies rare positive (attack) cases.

Main functions:
---------------
• threshold_at_fpr()      → picks score threshold for a target false positive rate. (FPR ~= target_fpr)
• apply_threshold()       → converts anomaly scores into binary predictions. (1 = anomaly if score > threshold)
• fpr_recall_from_preds() → computes FPR and Recall from binary predictions.
• evaluate_at_threshold() → prints classification report + returns key metrics.
• plot_confusion()        → saves a small confusion matrix plot.
• plot_roc_pr()           → saves ROC and Precision–Recall curve plots.
• full_evaluation()       → end-to-end evaluation (threshold, metrics, and plots).
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
    benign_scores = scores[y_true == 0]
    if benign_scores.size == 0:
        raise ValueError("No benign samples to compute threshold_at_fpr().")
    q = np.clip(1.0 - target_fpr, 0.0, 1.0)
    return float(np.quantile(benign_scores, q))

def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores > threshold).astype(int)

def fpr_recall_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
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
    Apply a threshold, print classification report, and return a dictionary of metrics:
    - auc_roc: ROC-AUC score (probability model ranks attacks above benigns)
    - ap: Average Precision (area under Precision–Recall curve)
    - fpr: false positive rate
    - recall: detection rate (true positive rate)
    - support_neg / support_pos: counts of benign vs attack samples
    """
    y_pred = apply_threshold(scores, threshold)

    if print_report:
        print(classification_report(y_true, y_pred, digits=4))

    metrics = {}
    try:
        metrics["auc_roc"] = float(roc_auc_score(y_true, scores))
        metrics["ap"] = float(average_precision_score(y_true, scores))
    except Exception:
        metrics["auc_roc"] = float("nan")
        metrics["ap"] = float("nan")

    fpr, rec = fpr_recall_from_preds(y_true, y_pred)
    metrics["fpr"] = fpr
    metrics["recall"] = rec
    metrics["support_neg"] = int((y_true == 0).sum())
    metrics["support_pos"] = int((y_true == 1).sum())

    return metrics


# ----------------------------- Plotting -------------------------------------- #
def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str, save_path: str) -> None:
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
    # ROC curve (shows trade-off between FPR and TPR)
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

    # Precision–Recall curve (shows precision vs recall — better for imbalanced datasets)
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


# ----------------------------- Full Pipeline --------------------------------- #
def full_evaluation(
    y_true: np.ndarray,
    scores: np.ndarray,
    dataset: str,
    model_name: str,
    model_dir: str,
    target_fpr: float = 0.01,
) -> Dict[str, float]:
    """
    Run the full evaluation pipeline:
      1. Select a threshold at a target false positive rate (using benign-score quantile)
      2. Compute and print metrics (AUC, AP, FPR, Recall)
      3. Save confusion matrix, ROC, and PR plots
    Returns a dictionary with all metrics.
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

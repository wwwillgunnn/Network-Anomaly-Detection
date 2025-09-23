# Model Metrics Documentation

This document explains the evaluation metrics reported in the Network Anomaly Detection project.

---

## Algorithms
- **IF (Isolation Forest):** Unsupervised anomaly detector using random partitioning.
- **AE (Autoencoder):** Neural network trained on benign-only data to reconstruct inputs; anomalies have high reconstruction error.

---

## ROC-AUC (Receiver Operating Characteristic – Area Under Curve)
- Plots **True Positive Rate (TPR/Recall)** vs **False Positive Rate (FPR)** across thresholds.
- AUC = probability a random anomaly is scored higher than a random benign sample.
- Range: 0.5 = random, 1.0 = perfect separation.

---

## PR-AUC / Average Precision (AP)
- Summarizes the **Precision–Recall curve**.
- Better for highly imbalanced datasets (like intrusion detection).
- Higher AP = better balance of precision and recall across thresholds.

---

## Thresholding
- We set thresholds by target FPR (e.g., **1% FPR**).
- Chosen using validation benign data quantile.
- Then we binarize predictions and compute classification metrics.

---

## Classification Report (Precision, Recall, F1)
- **Precision:** Of samples flagged as attack, how many are truly attacks.
- **Recall:** Of all true attacks, how many we caught.
- **F1-score:** Harmonic mean of precision and recall.
- **Support:** Number of samples per class.

---

## Example Output
AE val — ROC-AUC: 0.8468 AP: 0.7024 thr@FPR≈1%: 3.3108e-02

✅ Autoencoder @1%FPR (val-tuned)

precision recall f1-score support

       0     0.9058    0.9900    0.9461    628946

       1     0.9093    0.4935    0.6397    127763


- ROC-AUC and AP show ranking performance.
- Threshold = `0.033` chosen for ~1% FPR.
- Precision (attack): 0.91 → most flagged are real.
- Recall (attack): 0.49 → about half of attacks caught.

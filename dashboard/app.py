from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.express as px
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
)

# ---------- CONFIG ----------
# If you run `streamlit run app.py` from the repo root, this will resolve to ./models
REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = (REPO_ROOT / "models").resolve()

# Expected filenames inside each dataset folder (all optional except the folder itself)
AE_PTH = "autoencoder.pth"
AE_THR = "autoencoder_threshold.npy"
IF_JOBLIB = "iforest.joblib"
SCALER = "scaler.joblib"

# Optional arrays (recommended to save at train time)
VAL_SCORES = "val_scores.npy"
VAL_LABELS = "val_labels.npy"
TEST_SCORES = "test_scores.npy"
TEST_LABELS = "test_labels.npy"

# Optional summary (quick KPIs)
SUMMARY_JSON = "summary.json"

# Figure names under models/<dataset>/figs/
AE_CM = "ae_cm.png"
AE_ROC = "ae_roc.png"
AE_PR = "ae_pr.png"
IF_CM = "if_cm.png"
IF_ROC = "if_roc.png"
IF_PR = "if_pr.png"


# ---------- HELPERS ----------
def list_available_datasets(models_dir: Path) -> list[str]:
    """Return dataset names from subdirectories in models/."""
    if not models_dir.exists():
        return []
    return sorted([p.name for p in models_dir.iterdir() if p.is_dir()])

def dpath(dataset: str, *parts: str) -> Path:
    """Build path under models/<dataset>/..."""
    return MODELS_DIR / dataset / Path(*parts)

def load_optional_np(path: Path):
    try:
        if path.exists():
            return np.load(path)
    except Exception:
        pass
    return None

def load_summary_json(dataset: str) -> dict | None:
    f = dpath(dataset, SUMMARY_JSON)
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            return None
    return None

def fig_path(dataset: str, fname: str) -> Path | None:
    p = dpath(dataset, "figs", fname)
    return p if p.exists() else None

def render_image_if_exists(title: str, path: Path | None):
    st.subheader(title)
    if path and path.exists():
        st.image(Image.open(path), use_column_width=True)
    else:
        st.info(f"No saved figure found for **{title}**")

def compute_metrics_from_scores(y_true: np.ndarray, scores: np.ndarray, thr: float):
    # Higher score => more anomalous
    y_pred = (scores >= thr).astype(int)

    fpr, tpr, _ = roc_curve(y_true, scores)
    prec, rec, _ = precision_recall_curve(y_true, scores)
    roc_auc = float(roc_auc_score(y_true, scores))
    ap = float(average_precision_score(y_true, scores))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(y_true, y_pred, labels=[0, 1], output_dict=True, zero_division=0)

    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    pr_df = pd.DataFrame({"recall": rec, "precision": prec})
    return roc_auc, ap, report, cm, roc_df, pr_df


# ---------- UI ----------
st.set_page_config(page_title="NAD Dashboard (Streamlit)", layout="wide")
st.title("🛰️ Network Anomaly Detection — Dashboard")

datasets = list_available_datasets(MODELS_DIR)
if not datasets:
    st.error(f"No dataset folders found in {MODELS_DIR}.\n\n"
             "Expected structure:\n"
             "models/\n  └─ <DATASET>/\n     ├─ autoencoder.pth\n     ├─ autoencoder_threshold.npy\n     ├─ iforest.joblib\n     └─ figs/ (png plots)\n")
    st.stop()

c1, c2, c3 = st.columns([2, 2, 2])
dataset = c1.selectbox("Dataset", datasets, index=0)
model_choice = c2.selectbox("Model", ["Autoencoder", "IsolationForest"], index=0)

# Threshold default: if AE, try load npy; else a sane default
pref_thr = 0.033
if model_choice == "Autoencoder":
    arr = load_optional_np(dpath(dataset, AE_THR))
    if isinstance(arr, np.ndarray) and arr.size > 0:
        pref_thr = float(arr.ravel()[0])

thr = c3.slider("Decision threshold (for live metrics if score arrays exist)", 0.0, 0.2, float(pref_thr), 0.001)

# Load scores/labels (prefer TEST, fall back to VAL)
scores = load_optional_np(dpath(dataset, TEST_SCORES)) or load_optional_np(dpath(dataset, VAL_SCORES))
labels = load_optional_np(dpath(dataset, TEST_LABELS)) or load_optional_np(dpath(dataset, VAL_LABELS))

# Figures per model
if model_choice == "Autoencoder":
    fig_cm = fig_path(dataset, AE_CM)
    fig_roc = fig_path(dataset, AE_ROC)
    fig_pr = fig_path(dataset, AE_PR)
else:
    fig_cm = fig_path(dataset, IF_CM)
    fig_roc = fig_path(dataset, IF_ROC)
    fig_pr = fig_path(dataset, IF_PR)

summary = load_summary_json(dataset)

st.markdown("### 📊 Model Health")

if (
    scores is not None
    and labels is not None
    and isinstance(scores, np.ndarray)
    and isinstance(labels, np.ndarray)
    and scores.shape[0] == labels.shape[0]
):
    roc_auc, ap, report, cm, roc_df, pr_df = compute_metrics_from_scores(labels, scores, thr)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("AP", f"{ap:.3f}")
    k2.metric("ROC-AUC", f"{roc_auc:.3f}")
    k3.metric("Precision@thr (class 1)", f"{report['1']['precision']:.3f}")
    k4.metric("Recall@thr (class 1)", f"{report['1']['recall']:.3f}")

    # ROC + PR charts
    c_roc, c_pr = st.columns(2)
    with c_roc:
        st.subheader("ROC Curve")
        fig = px.area(roc_df, x="fpr", y="tpr")
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c_pr:
        st.subheader("Precision–Recall Curve")
        fig2 = px.area(pr_df, x="recall", y="precision")
        fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    # Confusion matrix @ thr
    st.subheader(f"Confusion Matrix @ thr={thr:.4f}")
    cm_df = pd.DataFrame(cm, index=["Benign (0)", "Attack (1)"], columns=["Pred 0", "Pred 1"])
    st.dataframe(cm_df.style.format("{:,}"), use_container_width=True)

else:
    # No arrays → show any saved summary + figures
    k1, k2 = st.columns(2)
    ap_val = summary.get("ap") if summary else None
    roc_val = summary.get("roc_auc") if summary else None
    k1.metric("AP (summary)", f"{ap_val:.3f}" if ap_val is not None else "—")
    k2.metric("ROC-AUC (summary)", f"{roc_val:.3f}" if roc_val is not None else "—")

    st.info(
        "Live metrics need score/label arrays inside the dataset folder:\n"
        f"  - {TEST_SCORES} / {TEST_LABELS} (preferred) or\n"
        f"  - {VAL_SCORES} / {VAL_LABELS}\n\n"
        "Falling back to your saved figures."
    )
    cA, cB = st.columns(2)
    with cA:
        render_image_if_exists("ROC Curve (saved)", fig_roc)
    with cB:
        render_image_if_exists("PR Curve (saved)", fig_pr)
    render_image_if_exists("Confusion Matrix (saved)", fig_cm)

st.caption(f"Artifacts dir: {MODELS_DIR} • Dataset: {dataset} • Model: {model_choice}")

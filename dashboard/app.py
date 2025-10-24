from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parent.parent  # <repo>/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.detect import load_ae_detector, IFDetector  # package import
import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.express as px
import streamlit.components.v1 as components
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
)

# ---------- CONFIG ----------
MODELS_DIR = (REPO_ROOT / "models").resolve()
DATA_PROCESSED = (REPO_ROOT / "data" / "processed").resolve()
SHARED_ARRAYS_DIR = MODELS_DIR / "demo"  # silent fallback for arrays only
MODEL_KEY = {"Autoencoder": "autoencoder", "IsolationForest": "iforest", "LSTM": "lstm"}
# Arrays
AE_THR = "autoencoder_threshold.npy"
VAL_SCORES = "val_scores.npy"
VAL_LABELS = "val_labels.npy"
TEST_SCORES = "test_scores.npy"
TEST_LABELS = "test_labels.npy"
SUMMARY_JSON = "summary.json"
TIMESTAMPS = "timestamps.npy"

# Model artifacts (under models/<dataset>/)
IFOREST_JOBLIB = "iforest.joblib"
SCALER_JOBLIB = "scaler.joblib"
AE_PTH = "autoencoder.pth"
LSTM_PT = "lstm.pt"

# Saved figures (under models/<dataset>/figs/)
AE_CM = "autoencoder_cm.png"
AE_ROC = "autoencoder_roc.png"
AE_PR = "autoencoder_pr.png"
IF_CM = "iforest_cm.png"
IF_ROC = "iforest_roc.png"
IF_PR = "iforest_pr.png"
LSTM_CM = "lstm_cm.png"
LSTM_ROC = "lstm_roc.png"
LSTM_PR = "lstm_pr.png"


# ---------- HELPERS ----------
def list_available_datasets(models_dir: Path):
    if not models_dir.exists():
        return []
    return sorted([p.name for p in models_dir.iterdir() if p.is_dir()])

def ecdf_normalize(score: float, ref_scores: np.ndarray) -> float:
    """
    Map a raw anomaly score to [0,1] using the empirical CDF of reference scores.
    Returns the percentile (higher = more anomalous).
    """
    if ref_scores.size == 0:
        return float("nan")
    # Percentile of where `score` sits vs ref distribution
    pct = (ref_scores <= score).mean()
    return float(np.clip(pct, 0.0, 1.0))

def dpath(dataset: str, *parts: str) -> Path:
    return MODELS_DIR / dataset / Path(*parts)


def load_optional_np(path: Path):
    try:
        if path.exists():
            return np.load(path, allow_pickle=False)
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


def render_image_if_exists(title: str, path: Path | None):
    st.subheader(title)
    if path and path.exists():
        st.image(Image.open(path), use_container_width=True)
    else:
        st.info(f"No saved figure found for **{title}**")


def _try_load_label_array(dir_path: Path) -> np.ndarray | None:
    # Labels are shared across models; prefer test, then val
    for name in (TEST_LABELS, VAL_LABELS):
        arr = load_optional_np(dir_path / name)
        if isinstance(arr, np.ndarray):
            return arr
    return None


def _try_load_score_array(dir_path: Path, model_choice: str) -> np.ndarray | None:
    key = MODEL_KEY.get(model_choice, "").lower()
    candidates = [
        f"test_scores_{key}.npy",
        TEST_SCORES,
        VAL_SCORES,
    ]
    for name in candidates:
        arr = load_optional_np(dir_path / name)
        if isinstance(arr, np.ndarray):
            return arr
    return None


def _try_load_timestamps(dir_path: Path) -> np.ndarray | None:
    return load_optional_np(dir_path / TIMESTAMPS)


def load_arrays_with_silent_fallback(dataset: str, model_choice: str):
    ds_dir = dpath(dataset)

    # Primary: dataset folder
    scores = _try_load_score_array(ds_dir, model_choice)
    labels = _try_load_label_array(ds_dir)
    timestamps = _try_load_timestamps(ds_dir)
    if (scores is not None) and (labels is not None):
        return scores, labels, timestamps, "dataset", ds_dir

    # Fallback: models/demo
    if SHARED_ARRAYS_DIR.exists():
        fb_scores = _try_load_score_array(SHARED_ARRAYS_DIR, model_choice)
        fb_labels = _try_load_label_array(SHARED_ARRAYS_DIR)
        fb_timestamps = _try_load_timestamps(SHARED_ARRAYS_DIR)
        if (fb_scores is not None) and (fb_labels is not None):
            return fb_scores, fb_labels, fb_timestamps, "shared", SHARED_ARRAYS_DIR

    return None, None, None, "none", ds_dir


def validate_and_align_arrays(scores, labels):
    if scores is None or labels is None:
        return None, None, "Missing arrays."
    try:
        s = np.asarray(scores).ravel().astype(float)
        y = np.asarray(labels).ravel()
        if y.dtype == bool:
            y = y.astype(int)
        elif not np.issubdtype(y.dtype, np.integer):
            try:
                y = y.astype(int)
            except Exception:
                uniq = np.unique(y)
                if uniq.size == 2:
                    mapping = {uniq[0]: 0, uniq[1]: 1}
                    y = np.vectorize(mapping.get)(y)
                else:
                    return None, None, f"Labels not integer-like (unique={uniq})."
        n = min(s.shape[0], y.shape[0])
        note = ""
        if s.shape[0] != y.shape[0]:
            note = f"Length mismatch: scores={s.shape[0]}, labels={y.shape[0]}. Trimmed to {n}."
        return s[:n], y[:n], (note or "OK")
    except Exception as e:
        return None, None, f"Error aligning arrays: {e}"


def normalize_timestamps(ts: np.ndarray | None, n_expected: int | None):
    if ts is None:
        return None, ""
    try:
        raw = np.asarray(ts).ravel()
        if np.issubdtype(raw.dtype, np.datetime64):
            series = pd.to_datetime(raw)
        elif np.issubdtype(raw.dtype, np.number):
            series = pd.to_datetime(raw, unit="s", utc=True).tz_convert(None)
        else:
            series = pd.to_datetime(raw, utc=True, errors="coerce").tz_convert(None)
        note = ""
        if n_expected is not None and len(series) != n_expected:
            m = min(len(series), n_expected)
            series = series[:m]
            note = f"Timestamps length {len(raw)} != {n_expected}. Trimmed to {m}."
        if series.isna().any():
            before = len(series)
            series = series.dropna()
            note += f" Dropped {before - len(series)} invalid timestamps."
        return series, note.strip()
    except Exception as e:
        return None, f"Timestamp parsing failed: {e}"


def compute_metrics_safe(y_true: np.ndarray, scores: np.ndarray, thr: float):
    # metric computation that handles single-class labels.
    y_true = np.asarray(y_true).ravel().astype(int)
    scores = np.asarray(scores).ravel().astype(float)
    y_pred = (scores >= thr).astype(int)

    pos = int((y_true == 1).sum())
    neg = int((y_true == 0).sum())
    both = (pos > 0) and (neg > 0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(
        y_true, y_pred, labels=[0, 1], output_dict=True, zero_division=0
    )

    roc_auc = None
    ap = None
    roc_df = None
    pr_df = None

    if both:
        fpr, tpr, _ = roc_curve(y_true, scores)
        prec, rec, _ = precision_recall_curve(y_true, scores)
        roc_auc = float(roc_auc_score(y_true, scores))
        ap = float(average_precision_score(y_true, scores))
        roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
        pr_df = pd.DataFrame({"recall": rec, "precision": prec})

    info = {"positives": pos, "negatives": neg, "both_classes": both}
    return roc_auc, ap, report, cm, roc_df, pr_df, info


def threshold_at_target_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 0.01):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    le = np.where(fpr <= target_fpr)[0]
    if le.size > 0:
        idx = le[-1]  # highest threshold that still meets the target FPR
    else:
        idx = int(np.argmin(np.abs(fpr - target_fpr)))
    return float(thresholds[idx]), float(fpr[idx])


def infer_split_label(source_dir: Path | None, model_choice: str) -> str:
    if source_dir is None:
        return "arrays"
    key = MODEL_KEY.get(model_choice, "").lower()
    test_name = f"test_scores_{key}.npy"
    if (source_dir / test_name).exists() and (source_dir / TEST_LABELS).exists():
        return "test"
    return "arrays"


# ---------- FIGURES / ARTIFACTS: SEARCH ACROSS DATASETS ----------
def find_in_dataset_or_others(preferred_dataset: str, rel_path: str) -> Path | None:
    first = dpath(preferred_dataset, rel_path)
    if first.exists():
        return first
    for ds in list_available_datasets(MODELS_DIR):
        if ds == preferred_dataset:
            continue
        p = dpath(ds, rel_path)
        if p.exists():
            return p
    return None


def find_png_with_fallback(dataset: str, fname: str) -> Path | None:
    return find_in_dataset_or_others(dataset, str(Path("figs") / fname))


def find_artifact_with_fallback(dataset: str, fname: str) -> Path | None:
    return find_in_dataset_or_others(dataset, fname)


# ---------- UI ----------
st.set_page_config(page_title="NAD Dashboard (Streamlit)", layout="wide")
st.title("🛰️ Network Anomaly Detection — Dashboard")


def render_manual_packet_scoring(dataset: str, model_choice: str, thr: float, scores, labels):
    st.markdown("### 🔎 Manual Packet Scoring")
    if model_choice not in ("Autoencoder", "IsolationForest"):
        st.info("Manual scoring currently implemented for Autoencoder and IsolationForest.")
        return

    try:
        if model_choice == "Autoencoder":
            det = load_ae_detector(str(MODELS_DIR), dataset)
            # Feature names from scaler if available; else fallback
            feature_names = list(getattr(det.scaler, "feature_names_in_", []))
            input_dim = det.input_dim
            if not feature_names or len(feature_names) != input_dim:
                feature_names = [f"f{i}" for i in range(input_dim)]
        else:
            # Keep IF feature order aligned to AE’s scaler if possible
            try:
                det_ae_for_schema = load_ae_detector(str(MODELS_DIR), dataset)
                feature_names = list(getattr(det_ae_for_schema.scaler, "feature_names_in_", []))
                if not feature_names:
                    feature_names = [f"f{i}" for i in range(det_ae_for_schema.input_dim)]
            except Exception:
                feature_names = []
            det = IFDetector(str(MODELS_DIR), dataset)

        with st.form("manual_packet_form"):
            st.caption("Enter raw feature values in the exact order your model expects.")
            cols = st.columns(4)
            values = {}
            for i, name in enumerate(feature_names):
                with cols[i % 4]:
                    values[name] = st.number_input(name, value=0.0, step=0.1, format="%.6f")
            submitted = st.form_submit_button("Score packet")

        if not submitted:
            return

        # Create a one-row DataFrame with the right column order
        x_df = pd.DataFrame([values])[feature_names]

        # Score
        if model_choice == "Autoencoder":
            s_arr, _ = det.predict(x_df)   # AE returns (scores, labels) but we ignore labels here
            score = float(np.asarray(s_arr).ravel()[0])
        else:
            s_arr = det.score(x_df)        # IF returns scores directly (higher = more anomalous)
            score = float(np.asarray(s_arr).ravel()[0])

        pred = int(score >= float(thr))

        # Confidence (empirical if possible, else margin heuristic)
        if isinstance(scores, np.ndarray) and scores.size > 0:
            pool = scores[(labels == 0)] if (isinstance(labels, np.ndarray) and (labels == 0).any()) else scores
            pool = pool.astype(float)
            pool_sorted = np.sort(pool.astype(float))
            rank = int(np.searchsorted(pool_sorted, score, side="right"))
            pct = rank / max(1, len(pool_sorted))  # P(benign_score ≤ score)
            p_anom = float(np.clip(pct, 0.0, 1.0))  # higher score → higher anomaly probability
            p_norm = 1.0 - p_anom
            confidence = p_anom if pred == 1 else p_norm
        else:
            fake_std = float(np.std(scores)) if isinstance(scores, np.ndarray) and scores.size > 1 else 1e-3
            z = (score - float(thr)) / (fake_std + 1e-9)
            p_anom = 1.0 / (1.0 + np.exp(-z))
            confidence = p_anom if pred == 1 else (1.0 - p_anom)

        # Output
        out_l, out_r, out_r2 = st.columns(3)
        out_l.metric("Prediction", "🚨 Anomaly" if pred == 1 else "✅ Normal")
        out_r.metric("Score", f"{score:.4e}")
        out_r2.metric("Confidence", f"{confidence * 100:.1f}%")

        st.caption(
            "Confidence is estimated from the empirical distribution of evaluation scores "
            "(normal-class ECDF if available). It is **not** a calibrated probability."
        )
    except Exception as e:
        st.error(f"Manual scoring failed: {e}")

# Dataset picker & model choice
c1, c2, c3 = st.columns([2, 2, 2])
datasets = list_available_datasets(MODELS_DIR)
if not datasets:
    st.error(
        f"No dataset folders found in {MODELS_DIR}.\n\n"
        "Expected structure:\n"
        "models/\n  └─ <DATASET>/\n"
        "      ├─ test_scores_<model>.npy\n"
        "      ├─ test_labels.npy\n"
        "      ├─ timestamps.npy\n"
        "      └─ figs/ (png plots)\n"
    )
    st.stop()

dataset = c1.selectbox("Dataset", datasets, index=0)
model_choice = c2.selectbox("Model", ["Autoencoder", "IsolationForest", "LSTM"], index=0)

# Threshold default (AE uses saved threshold if present)
pref_thr = 0.033
if model_choice == "Autoencoder":
    arr_thr = load_optional_np((MODELS_DIR / dataset / AE_THR))
    if isinstance(arr_thr, np.ndarray) and arr_thr.size > 0:
        pref_thr = float(arr_thr.ravel()[0])

SLIDER_MIN, SLIDER_MAX = 0.0, 0.2
thr = c3.slider("Decision threshold", SLIDER_MIN, SLIDER_MAX, float(pref_thr), 0.001)


# Load arrays early (used by manual scoring confidence and later sections)
scores_raw, labels_raw, timestamps_raw, source_kind, source_dir = \
    load_arrays_with_silent_fallback(dataset, model_choice)
scores, labels, align_note = validate_and_align_arrays(scores_raw, labels_raw)

# ---------- MANUAL PACKET SCORING AT TOP ----------
render_manual_packet_scoring(dataset, model_choice, thr, scores, labels)

# Resolve figure paths (with global fallback) for the selected model
if model_choice == "Autoencoder":
    fig_roc = (MODELS_DIR / dataset / "figs" / AE_ROC)
    fig_pr = (MODELS_DIR / dataset / "figs" / AE_PR)
    fig_cm = (MODELS_DIR / dataset / "figs" / AE_CM)
elif model_choice == "IsolationForest":
    fig_roc = (MODELS_DIR / dataset / "figs" / IF_ROC)
    fig_pr = (MODELS_DIR / dataset / "figs" / IF_PR)
    fig_cm = (MODELS_DIR / dataset / "figs" / IF_CM)
else:  # LSTM
    fig_roc = (MODELS_DIR / dataset / "figs" / LSTM_ROC)
    fig_pr = (MODELS_DIR / dataset / "figs" / LSTM_PR)
    fig_cm = (MODELS_DIR / dataset / "figs" / LSTM_CM)

summary = load_summary_json(dataset)

pref_thr = 0.033
if model_choice == "Autoencoder":
    arr_thr = load_optional_np((MODELS_DIR / dataset / AE_THR))
    if isinstance(arr_thr, np.ndarray) and arr_thr.size > 0:
        pref_thr = float(arr_thr.ravel()[0])
else:
    if summary and "thr_fpr_1pct" in summary:
        pref_thr = float(summary["thr_fpr_1pct"])


# ---------- MODEL HEALTH ----------
st.markdown("### 📊 Model Health")

if source_kind == "none":
    st.warning(f"No score/label arrays found in: {source_dir}")
if align_note and align_note != "OK" and source_kind != "none":
    st.caption(f"Array alignment note: {align_note}")

# Metrics
if isinstance(scores, np.ndarray) and isinstance(labels, np.ndarray) and scores.size == labels.size and scores.size > 0:
    roc_auc, ap, report, cm, roc_df, pr_df, info = compute_metrics_safe(labels, scores, thr)

    total = info["positives"] + info["negatives"]
    prevalence = (info["positives"] / total) * 100 if total > 0 else 0.0
    st.caption(f"Label prevalence — positives: {info['positives']} ({prevalence:.2f}%), negatives: {info['negatives']}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("AP", f"{ap:.3f}" if ap is not None else "—")
    k2.metric("ROC-AUC", f"{roc_auc:.3f}" if roc_auc is not None else "—")
    if info["positives"] > 0:
        k3.metric("Precision@thr", f"{report['1']['precision']:.3f}")
        k4.metric("Recall@thr", f"{report['1']['recall']:.3f}")
    else:
        k3.metric("Precision@thr", "—")
        k4.metric("Recall@thr", "—")
        st.warning(
            "No positive labels detected. Load arrays with both classes to enable ROC-AUC/AP and class-1 Precision/Recall.")

    # ----- Summary (thr tuned to ~1% FPR) -----
    if info["both_classes"]:
        thr_star, fpr_star = threshold_at_target_fpr(labels, scores, target_fpr=0.01)

        # Classification @threshold
        y_pred_star = (scores >= thr_star).astype(int)
        rep_dict = classification_report(
            labels, y_pred_star, labels=[0, 1], output_dict=True, zero_division=0
        )

        split = infer_split_label(source_dir, model_choice)
        abbr = {"Autoencoder": "AE", "IsolationForest": "IF", "LSTM": "LSTM"}.get(model_choice, model_choice)

        st.markdown(
            f"**{abbr} {split} — ROC-AUC: {roc_auc:.4f}  AP: {ap:.4f}  "
            f"thr@FPR≈1%: {thr_star:.4e}**"
        )
        st.success(f"✅ {model_choice} @1%FPR ({split}-tuned)")

        # Accuracy @ threshold
        acc_val = rep_dict.get("accuracy", None)
        if acc_val is not None:
            st.metric("Accuracy @ tuned threshold", f"{acc_val:.4f}")

        # DataFrame
        rows = [r for r in ["0", "1", "macro avg", "weighted avg"] if r in rep_dict]
        rep_df = pd.DataFrame(rep_dict).T.loc[rows, ["precision", "recall", "f1-score", "support"]]

        # Friendly row labels
        index_map = {"0": "Normal (0)", "1": "Attack (1)", "macro avg": "Macro avg", "weighted avg": "Weighted avg"}
        rep_df.index = [index_map.get(idx, idx) for idx in rep_df.index]

        rep_df["support"] = rep_df["support"].astype(int)
        st.dataframe(
            rep_df.style.format({
                "precision": "{:.4f}",
                "recall": "{:.4f}",
                "f1-score": "{:.4f}",
                "support": "{:,}",
            }),
            use_container_width=True,
        )
    else:
        st.caption("Example summary skipped (needs both classes in labels).")

    # Confusion matrix
    st.subheader(f"Confusion Matrix @ thr={thr:.4f}")
    cm_df = pd.DataFrame(cm, index=["Normal (0)", "Attack (1)"], columns=["Pred 0", "Pred 1"])
    st.dataframe(cm_df.style.format("{:,}"), use_container_width=True)

    # Live ROC/PR
    if info["both_classes"]:
        c_roc_live, c_pr_live = st.columns(2)
        with c_roc_live:
            st.subheader("ROC Curve (live)")
            fig_live_roc = px.area(roc_df, x="fpr", y="tpr")
            fig_live_roc.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_live_roc, use_container_width=True)
        with c_pr_live:
            st.subheader("Precision–Recall Curve (live)")
            fig_live_pr = px.area(pr_df, x="recall", y="precision")
            fig_live_pr.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_live_pr, use_container_width=True)
    else:
        st.caption("ROC/PR charts require both classes in labels; skipped.")
else:
    # Fallback to summary.json KPIs
    k1, k2 = st.columns(2)
    ap_val = summary.get("ap") if summary else None
    roc_val = summary.get("roc_auc") if summary else None
    k1.metric("AP (summary)", f"{ap_val:.3f}" if ap_val is not None else "—")
    k2.metric("ROC-AUC (summary)", f"{roc_val:.3f}" if roc_val is not None else "—")

# ---------- SAVED FIGURES (PNGs) ----------
c_roc_img, c_pr_img = st.columns(2)
with c_roc_img:
    render_image_if_exists("ROC Curve (saved)", fig_roc)
with c_pr_img:
    render_image_if_exists("Precision–Recall Curve (saved)", fig_pr)

# Confusion Matrix (saved) & Scores over Time
c_cm_img, c_ts = st.columns(2)
with c_cm_img:
    render_image_if_exists("Confusion Matrix (saved)", fig_cm)

with c_ts:
    st.subheader("Scores over Time")
    if isinstance(scores, np.ndarray) and scores.size > 0:
        # Parse timestamps; fall back to simple index if missing/mismatched
        ts_series, ts_note = normalize_timestamps(timestamps_raw, n_expected=scores.shape[0])
        if ts_note:
            st.caption(ts_note)

        if ts_series is None or len(ts_series) != scores.shape[0]:
            ts_series = pd.Series(range(scores.shape[0]), name="index_axis")
            x_col, x_label = "index_axis", "index"
        else:
            x_col, x_label = "timestamp", "timestamp"

        df_time = pd.DataFrame({
            x_col: ts_series,
            "score": scores.astype(float),
            "label": (labels if isinstance(labels, np.ndarray) else np.zeros_like(scores)).astype(int),
            "pred": (scores >= thr).astype(int),
        }).sort_values(x_col, kind="stable")

        line = px.line(df_time, x=x_col, y="score", labels={x_col: x_label, "score": "anomaly score"})
        line.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(line, use_container_width=True)
    else:
        st.info("Scores not available to plot a time series.")

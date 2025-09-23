"""
Exploratory Data Analysis (EDA) for cleaned network-traffic data.

Usage (from repo root):
    python -m src.eda --dataset CIC-IDS2017 --out eda_reports --limit 1000000

This script will:
  • Load the CLEANED dataset via src/ingest.py (expected to return a pandas DataFrame).
  • Compute dataset overview stats (rows, cols, memory, dtypes, missingness).
  • Visualize class imbalance, numeric distributions, correlations, top IPs/ports, and a basic time plot if timestamp exists.
  • Save all plots under the output directory (PNG files) and print a concise summary to stdout.

Assumptions / Conventions:
  • ingest.py exposes a function `load_clean(dataset: str)` or `load_dataset(dataset: str, cleaned: bool=True)`.
    If not found, we fallback to reading CSVs under data/processed/<dataset>/*.csv and concatenating.
  • Binary label column is one of: ["label", "Label", "y", "Attack", "attack", "is_attack", "malicious"].
  • Timestamp column (optional) is one of: ["timestamp", "Timestamp", "flow_start", "time", "Time"].
  • IP/port columns (optional) are: src_ip, dst_ip, src_port, dst_port (case-insensitive variants supported).

Author: EDA utility for ICT: Network Anomaly project.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Matplotlib defaults
# ----------------------------
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "savefig.bbox": "tight",
})

# ----------------------------
# Utils
# ----------------------------
LABEL_CANDIDATES = ["label", "Label", "y", "Attack", "attack", "is_attack", "malicious"]
TIME_CANDIDATES = ["timestamp", "Timestamp", "flow_start", "time", "Time"]
SRC_IP_CAND = ["src_ip", "Src IP", "source", "source_ip"]
DST_IP_CAND = ["dst_ip", "Dst IP", "destination", "destination_ip"]
SRC_PORT_CAND = ["src_port", "Src Port", "sport"]
DST_PORT_CAND = ["dst_port", "Dst Port", "dport"]


def find_first(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_cols:
            return lower_cols[cand.lower()]
    return None


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _savefig(outdir: Path, name: str) -> None:
    outpath = outdir / f"{name}.png"
    plt.savefig(outpath)
    print(f"🖼  Saved: {outpath}")


# ----------------------------
# Data loading
# ----------------------------

def load_clean_dataset(dataset: str, limit: Optional[int] = None) -> pd.DataFrame:
    """Attempt to load cleaned dataset via ingest helpers; fallback to data/processed/<dataset>.
    The function samples (head) `limit` rows if provided.
    """
    # Try to import project ingest helpers
    df: Optional[pd.DataFrame] = None
    try:
        from src import ingest  # type: ignore
        if hasattr(ingest, "load_clean"):
            df = ingest.load_clean(dataset)  # expected: returns a DataFrame
        elif hasattr(ingest, "load_dataset"):
            try:
                df = ingest.load_dataset(dataset, cleaned=True)  # type: ignore[arg-type]
            except TypeError:
                # Some signatures may differ
                df = ingest.load_dataset(dataset)
    except Exception as e:
        print(f"⚠️  Could not import/use src.ingest helpers ({e}). Falling back to CSV glob.")

    if df is None:
        proc_dir = Path("data/processed") / dataset
        csvs = sorted(proc_dir.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(
                f"No CSVs found in {proc_dir}. Ensure cleaned data exists or expose load_clean in ingest.py."
            )
        frames = []
        for p in csvs:
            print(f"📥 Reading: {p}")
            frames.append(pd.read_csv(p))
            if limit is not None and sum(len(f) for f in frames) >= limit:
                break
        df = pd.concat(frames, ignore_index=True)

    if limit is not None and len(df) > limit:
        df = df.head(limit).copy()
    return df


# ----------------------------
# EDA plots
# ----------------------------

def plot_class_balance(df: pd.DataFrame, outdir: Path) -> None:
    label_col = find_first(df, LABEL_CANDIDATES)
    if label_col is None:
        print("ℹ️  No label column found; skipping class balance plot.")
        return
    vc = df[label_col].value_counts(dropna=False).sort_index()
    ax = vc.plot(kind="bar")
    ax.set_title(f"Class Balance — {label_col}")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    for c in ax.containers:
        ax.bar_label(c)
    _savefig(outdir, "class_balance")
    plt.close()


def plot_missingness(df: pd.DataFrame, outdir: Path) -> None:
    miss = df.isna().mean().sort_values(ascending=False)
    ax = miss.head(50).plot(kind="bar")
    ax.set_title("Missingness (Top 50 Columns)")
    ax.set_ylabel("Fraction Missing")
    _savefig(outdir, "missingness_top50")
    plt.close()


def plot_numeric_histograms(df: pd.DataFrame, outdir: Path, max_cols: int = 24, bins: int = 50) -> None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        print("ℹ️  No numeric columns found; skipping histograms.")
        return
    cols = num_cols[:max_cols]
    n = len(cols)
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols + 2, 3 * n_rows + 2))
    axes = np.array(axes).reshape(n_rows, n_cols)
    for i, col in enumerate(cols):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        ax.hist(series, bins=bins)
        ax.set_title(col)
    # Hide leftover axes
    for j in range(n, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r, c].axis("off")
    fig.suptitle("Numeric Distributions (First N Columns)")
    fig.tight_layout()
    _savefig(outdir, "numeric_histograms")
    plt.close(fig)


def plot_correlations(df: pd.DataFrame, outdir: Path, method: str = "pearson", top_k: int = 30) -> None:
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        print("ℹ️  Not enough numeric columns for correlations.")
        return
    # Use top_k numeric columns by variance to avoid gigantic mats
    var = num.var().sort_values(ascending=False)
    cols = var.index[:top_k]
    corr = num[cols].corr(method=method)
    fig, ax = plt.subplots(figsize=(1 + 0.3 * len(cols), 1 + 0.3 * len(cols)))
    cax = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90)
    ax.set_yticklabels(cols)
    ax.set_title(f"Correlation Heatmap ({method}, top {len(cols)})")
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _savefig(outdir, f"corr_{method}")
    plt.close(fig)


def plot_top_ips_ports(df: pd.DataFrame, outdir: Path, top_n: int = 20) -> None:
    src_ip = find_first(df, SRC_IP_CAND)
    dst_ip = find_first(df, DST_IP_CAND)
    src_port = find_first(df, SRC_PORT_CAND)
    dst_port = find_first(df, DST_PORT_CAND)

    if src_ip and src_ip in df:
        ax = df[src_ip].value_counts().head(top_n).plot(kind="bar")
        ax.set_title(f"Top {top_n} Source IPs")
        ax.set_xlabel("src_ip")
        ax.set_ylabel("Count")
        _savefig(outdir, "top_src_ips")
        plt.close()

    if dst_ip and dst_ip in df:
        ax = df[dst_ip].value_counts().head(top_n).plot(kind="bar")
        ax.set_title(f"Top {top_n} Destination IPs")
        ax.set_xlabel("dst_ip")
        ax.set_ylabel("Count")
        _savefig(outdir, "top_dst_ips")
        plt.close()

    if src_port and src_port in df:
        ax = df[src_port].value_counts().head(top_n).plot(kind="bar")
        ax.set_title(f"Top {top_n} Source Ports")
        ax.set_xlabel("src_port")
        ax.set_ylabel("Count")
        _savefig(outdir, "top_src_ports")
        plt.close()

    if dst_port and dst_port in df:
        ax = df[dst_port].value_counts().head(top_n).plot(kind="bar")
        ax.set_title(f"Top {top_n} Destination Ports")
        ax.set_xlabel("dst_port")
        ax.set_ylabel("Count")
        _savefig(outdir, "top_dst_ports")
        plt.close()


def plot_time_series(df: pd.DataFrame, outdir: Path, freq: str = "1min") -> None:
    tcol = find_first(df, TIME_CANDIDATES)
    if tcol is None or tcol not in df:
        print("ℹ️  No timestamp column found; skipping time series plot.")
        return
    s = pd.to_datetime(df[tcol], errors="coerce").dropna().sort_values()
    if s.empty:
        print("ℹ️  Timestamp column exists but could not parse to datetime; skipping.")
        return
    counts = s.dt.floor(freq).value_counts().sort_index()
    ax = counts.plot()
    ax.set_title(f"Flows per {freq}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Count")
    _savefig(outdir, "time_series_counts")
    plt.close()


# ----------------------------
# Main
# ----------------------------

def main(dataset: str, outdir: Path, limit: Optional[int]) -> None:
    _safe_mkdir(outdir)
    print(f"📦 Loading cleaned dataset: {dataset} (limit={limit}) …")
    df = load_clean_dataset(dataset, limit=limit)

    # Overview
    n_rows, n_cols = df.shape
    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"✅ Loaded {n_rows:,} rows × {n_cols:,} cols  |  memory ≈ {mem_mb:,.2f} MB")

    # Dtypes summary
    dtype_counts = df.dtypes.value_counts()
    print("\n📑 dtypes summary:\n", dtype_counts)

    # Basic stats (numeric)
    desc = df.describe(include=[np.number]).T
    desc.to_csv(outdir / "numeric_summary.csv")
    print(f"📄 Saved numeric summary → {outdir / 'numeric_summary.csv'}")

    # Plots
    plot_class_balance(df, outdir)
    plot_missingness(df, outdir)
    plot_numeric_histograms(df, outdir)
    plot_correlations(df, outdir, method="pearson")
    plot_correlations(df, outdir, method="spearman")
    plot_top_ips_ports(df, outdir)
    plot_time_series(df, outdir)

    print("\n🎉 EDA complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EDA on cleaned network dataset")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g., CIC-IDS2017 / CIC-IDS2018 / CIC-IDS2019")
    parser.add_argument("--out", default="eda_reports", help="Output directory for PNGs and CSV")
    parser.add_argument("--limit", type=int, default=None, help="Optional row cap for quick EDA")
    args = parser.parse_args()

    # Ensure repo root is on path for `from src import ingest`
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    main(dataset=args.dataset, outdir=Path(args.out), limit=args.limit)


# old file
# # eda_tailored.py
# import os
# from typing import List, Optional, Dict
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# from ingest import load_dataset  # your repo loader
#
# # Dataset-specific columns aligned with features.py (plus common IP fields)
# COLS: Dict[str, dict] = {
#     "CIC-IDS2017": {
#         "num": [
#             "Flow Duration",
#             "Total Fwd Packets",
#             "Total Backward Packets",
#             "Total Length of Fwd Packets",
#             "Total Length of Bwd Packets",
#         ],
#         "proto": ["Protocol"],           # will map to proto in features.py
#         "label": ["Label"],              # will map to label in features.py
#         "src": ["Source IP", "src_ip", "Src IP"],
#         "dst": ["Destination IP", "dst_ip", "Dst IP"],
#         "dur": ["Flow Duration"],
#         "bytes_like": [
#             "Total Length of Fwd Packets",
#             "Total Length of Bwd Packets",
#             "Bytes", "bytes"
#         ],
#         "ts": ["Timestamp", "timestamp", "Time", "time"]
#     },
#     "UNSW-NB15": {
#         "num": ["dur", "sbytes", "dbytes", "sttl", "dttl"],
#         "proto": ["proto"],
#         "label": ["attack_cat"],
#         "src": ["srcip"],
#         "dst": ["dstip"],
#         "dur": ["dur"],
#         "bytes_like": ["sbytes", "dbytes", "bytes"],
#         "ts": ["stime", "time", "timestamp"]
#     }
# }
#
# def _pick(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
#     for c in cands:
#         if c in df.columns:
#             return c
#     return None
#
# def _ensure_dt(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
#     col = _pick(df, cands)
#     if not col:
#         return None
#     if not np.issubdtype(df[col].dtype, np.datetime64):
#         try:
#             if np.issubdtype(df[col].dtype, np.number):
#                 df[col] = pd.to_datetime(df[col], unit="s", errors="coerce")
#             else:
#                 df[col] = pd.to_datetime(df[col], errors="coerce")
#         except Exception:
#             df[col] = pd.to_datetime(df[col], errors="coerce")
#     return col
#
# def _save(outdir: str, name: str):
#     os.makedirs(outdir, exist_ok=True)
#     plt.savefig(os.path.join(outdir, f"{name}.png"), dpi=150, bbox_inches="tight")
#     plt.show()
#
# def eda_run(dataset: str, outdir: str = "plots", sample: Optional[int] = 200_000):
#     # 1) Load raw (keeps all cols, perfect for EDA)
#     df = load_dataset(dataset)
#
#     # Optional sampling for speed
#     if sample and len(df) > sample:
#         df = df.sample(sample, random_state=42)
#
#     spec = COLS[dataset]
#
#     # 2) Class balance (binarize label: attack=1, benign/normal/background=0)
#     lab_col = _pick(df, spec["label"])
#     if lab_col:
#         labels = df[lab_col].astype(str).str.lower().str.strip()
#         y_bin = (~labels.isin({"benign", "normal", "background"})).astype(int)
#         vc = y_bin.value_counts().reindex([0, 1]).fillna(0)
#         plt.figure()
#         plt.bar(["benign(0)", "attack(1)"], vc.values)
#         plt.title(f"{dataset} – class balance"); plt.ylabel("count")
#         _save(outdir, "class_balance")
#
#     # 3) Missing values (top N)
#     miss = df.isna().mean().sort_values(ascending=False).head(30)
#     plt.figure()
#     plt.bar(miss.index.astype(str), miss.values)
#     plt.xticks(rotation=90); plt.ylabel("fraction missing")
#     plt.title(f"{dataset} – missing values (top 30)")
#     _save(outdir, "missing_values")
#
#     # 4) Numeric histograms for the dataset’s main features
#     for c in spec["num"]:
#         if c in df.columns and np.issubdtype(df[c].dtype, np.number):
#             x = df[c].dropna().values
#             plt.figure()
#             plt.hist(x, bins=60)
#             plt.title(f"{dataset} – {c}"); plt.xlabel(c); plt.ylabel("count")
#             _save(outdir, f"hist_{c.replace(' ','_').lower()}")
#
#     # 5) Protocol distribution (Protocol/proto)
#     proto_col = _pick(df, spec["proto"])
#     if proto_col:
#         vc = df[proto_col].astype(str).value_counts()
#         plt.figure()
#         plt.bar(vc.index.astype(str), vc.values)
#         plt.xticks(rotation=90); plt.ylabel("count")
#         plt.title(f"{dataset} – protocol distribution")
#         _save(outdir, "protocol_distribution")
#
#     # 6) Correlation heatmap over top-variance numeric columns (up to 20)
#     num_cols = [c for c in spec["num"] if c in df.columns and np.issubdtype(df[c].dtype, np.number)]
#     if len(num_cols) >= 2:
#         var_order = df[num_cols].var(numeric_only=True).sort_values(ascending=False)
#         chosen = var_order.head(20).index.tolist()
#         corr = df[chosen].corr().values
#         plt.figure(figsize=(6,5))
#         im = plt.imshow(corr, aspect="auto", interpolation="nearest")
#         plt.colorbar(im, fraction=0.046, pad=0.04)
#         plt.xticks(range(len(chosen)), chosen, rotation=90)
#         plt.yticks(range(len(chosen)), chosen)
#         plt.title(f"{dataset} – correlation (top variance)")
#         _save(outdir, "correlation_heatmap")
#
#     # 7) Bytes vs duration (log–log)
#     dur_col = _pick(df, spec["dur"])
#     bytes_col = _pick(df, spec["bytes_like"])
#     if dur_col and bytes_col and np.issubdtype(df[dur_col].dtype, np.number) and np.issubdtype(df[bytes_col].dtype, np.number):
#         D = df[[dur_col, bytes_col]].dropna()
#         if len(D) > 50_000:
#             D = D.sample(50_000, random_state=42)
#         plt.figure()
#         plt.scatter(D[dur_col].values + 1e-9, D[bytes_col].values + 1e-9, s=6, alpha=0.5)
#         plt.xscale("log"); plt.yscale("log")
#         plt.xlabel(dur_col); plt.ylabel(bytes_col)
#         plt.title(f"{dataset} – bytes vs duration (log–log)")
#         _save(outdir, "bytes_vs_duration")
#
#     # 8) Top IPs & degree-ish view (if IP columns exist)
#     src_col = _pick(df, spec["src"]); dst_col = _pick(df, spec["dst"])
#     if src_col:
#         top_src = df[src_col].astype(str).value_counts().head(20)[::-1]
#         plt.figure(figsize=(6, max(3, len(top_src)*0.3)))
#         plt.barh(top_src.index.astype(str), top_src.values)
#         plt.xlabel("count"); plt.title(f"{dataset} – top source IPs")
#         _save(outdir, "top_src_ips")
#     if dst_col:
#         top_dst = df[dst_col].astype(str).value_counts().head(20)[::-1]
#         plt.figure(figsize=(6, max(3, len(top_dst)*0.3)))
#         plt.barh(top_dst.index.astype(str), top_dst.values)
#         plt.xlabel("count"); plt.title(f"{dataset} – top destination IPs")
#         _save(outdir, "top_dst_ips")
#     if src_col and dst_col:
#         deg = pd.concat([df[src_col], df[dst_col]]).value_counts()
#         ranks = np.arange(1, len(deg)+1)
#         plt.figure()
#         plt.loglog(ranks, np.sort(deg.values)[::-1])
#         plt.xlabel("rank"); plt.ylabel("degree"); plt.title(f"{dataset} – degree rank (log–log)")
#         _save(outdir, "degree_rank")
#
#     # 9) Time series (events/bytes) if a timestamp-like column exists
#     ts_col = _ensure_dt(df, spec["ts"])
#     if ts_col:
#         counts = df.set_index(ts_col).sort_index().resample("5T").size()
#         plt.figure()
#         counts.plot()
#         plt.title(f"{dataset} – events per 5 minutes")
#         plt.xlabel("time"); plt.ylabel("count")
#         _save(outdir, "ts_counts")
#
#         if bytes_col:
#             series = df[[ts_col, bytes_col]].dropna().set_index(ts_col).sort_index()[bytes_col].resample("5T").sum()
#             plt.figure()
#             series.plot()
#             plt.title(f"{dataset} – bytes per 5 minutes")
#             plt.xlabel("time"); plt.ylabel("bytes")
#             _save(outdir, "ts_bytes")
#
#     # 10) PCA (2D) on the numeric feature set with label overlay if present
#     try:
#         from sklearn.decomposition import PCA
#         nums = [c for c in num_cols if c in df.columns]
#         if len(nums) >= 2:
#             D = df[nums].dropna()
#             if len(D) > 20_000:
#                 D = D.sample(20_000, random_state=42)
#             X = (D - D.mean()) / (D.std(ddof=0) + 1e-9)
#             Z = PCA(n_components=2, random_state=42).fit_transform(X.values)
#             plt.figure()
#             if lab_col is not None:
#                 y_local = (~df.loc[D.index, lab_col].astype(str).str.lower().str.strip()
#                            .isin({"benign","normal","background"})).astype(int).values
#                 m0 = y_local == 0; m1 = ~m0
#                 plt.scatter(Z[m0,0], Z[m0,1], s=8, alpha=0.6, label="benign(0)")
#                 plt.scatter(Z[m1,0], Z[m1,1], s=8, alpha=0.6, label="attack(1)")
#                 plt.legend()
#             else:
#                 plt.scatter(Z[:,0], Z[:,1], s=8, alpha=0.6)
#             plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title(f"{dataset} – PCA (2D)")
#             _save(outdir, "pca_2d")
#     except Exception as e:
#         print(f"PCA skipped: {e}")
#
# if __name__ == "__main__":
#     # Examples:
#     # python eda_tailored.py  (edit below to switch dataset)
#     eda_run("CIC-IDS2017", outdir="plots_cic")
#     eda_run("UNSW-NB15", outdir="plots_unsw")

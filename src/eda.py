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
import re

def _read_csvs_from(path: Path, limit: Optional[int] = None) -> pd.DataFrame:
    files = []
    if path.is_dir():
        files = sorted(path.glob("*.csv"))
    elif path.is_file() and path.suffix.lower() == ".csv":
        files = [path]
    if not files:
        raise FileNotFoundError(f"No CSVs found at {path}")

    frames: List[pd.DataFrame] = []
    total = 0
    for p in files:
        print(f"📥 Reading: {p}")
        frames.append(pd.read_csv(p, low_memory=False))
        total += len(frames[-1])
        if limit is not None and total >= limit:
            break
    df = pd.concat(frames, ignore_index=True)
    if limit is not None and len(df) > limit:
        df = df.head(limit).copy()
    return df

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
SRC_IP_CAND = ["src_ip", "Src IP", "source", "source_ip"]
DST_IP_CAND = ["dst_ip", "Dst IP", "destination", "destination_ip"]
SRC_PORT_CAND = ["src_port", "Src Port", "sport"]
DST_PORT_CAND = ["dst_port", "Dst Port", "dport"]


def find_first(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    norm_map = {re.sub(r"[^a-z0-9]+", "", str(c).lower()): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r"[^a-z0-9]+", "", cand.lower())
        if key in norm_map:
            return norm_map[key]
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
        series = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            ax.set_title(f"{col} (no finite values)")
            ax.axis("off")
            continue
        # Optional winsorize to avoid extreme outliers wrecking bins
        try:
            lo, hi = np.nanpercentile(series, [0.5, 99.5])
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                series = series.clip(lo, hi)
        except Exception:
            pass
        ax.hist(series.values, bins=bins)
        ax.set_title(col)

    for j in range(n, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r, c].axis("off")
    fig.suptitle("Numeric Distributions (First N Columns)")
    fig.tight_layout()
    _savefig(outdir, "numeric_histograms")
    plt.close(fig)


def plot_correlations(df: pd.DataFrame, outdir: Path, method: str = "pearson", top_k: int = 30) -> None:
    num = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    num = num.dropna(axis=1, how="all")
    if num.shape[1] < 2:
        print("ℹ️  Not enough numeric columns for correlations.")
        return
    var = num.var(skipna=True).replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
    if var.empty:
        print("ℹ️  No finite-variance columns for correlations.")
        return
    cols = var.index[:top_k]
    corr = num[cols].corr(method=method, min_periods=1)
    fig, ax = plt.subplots(figsize=(1 + 0.3 * len(cols), 1 + 0.3 * len(cols)))
    cax = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90); ax.set_yticklabels(cols)
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

# ----------------------------
# Main
# ----------------------------

def run_eda(df: pd.DataFrame, outdir: Path) -> None:
    _safe_mkdir(outdir)

    # Make all numeric columns finite for EDA
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Clean column names: trim and collapse whitespace (handles " Label", " Timestamp", etc.)
    df.rename(columns=lambda c: re.sub(r"\s+", " ", str(c)).strip(), inplace=True)

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

    print("\n🎉 EDA complete.")

def load_for_stage(
        dataset: str,
        stage: str = "processed",  # 'raw' | 'processed'
        limit: Optional[int] = None,
        prefer_cache: bool = True,
) -> pd.DataFrame:
    """
    - If `dataset` is a PATH:
        raw: read CSVs from that folder/file
        processed: read then clean in-memory (no cache file written)
    - If `dataset` is a NAME:
        raw: use ingest.load_data(NAME, base_path=RAW_DATA_PATH)
        processed: use cached PROCESSED file if present (unless --no-cache);
                   otherwise build from raw via ingest.clean_data and save_preprocessed
    """
    # Import here to avoid hard dependency if user runs EDA outside project
    try:
        from ingest import RAW_DATA_PATH, PROCESSED_DATA_PATH, load_data, clean_data, \
            save_preprocessed  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Could not import ingest helpers: {e}")

    ds_path = Path(dataset)

    if ds_path.exists():  # PATH MODE
        if stage == "raw":
            return _read_csvs_from(ds_path, limit=limit)
        else:
            raw_df = _read_csvs_from(ds_path, limit=limit)
            return clean_data(raw_df)

    # NAME MODE
    name = dataset
    if stage == "raw":
        df = load_data(name, base_path=RAW_DATA_PATH)
        if limit is not None and len(df) > limit:
            df = df.head(limit).copy()
        return df

    # processed
    cache = os.path.join(PROCESSED_DATA_PATH, f"{name}_clean.csv")
    if prefer_cache and os.path.exists(cache):
        print(f"📦 Using cached processed dataset → {cache}")
        df = pd.read_csv(cache, low_memory=False)
    else:
        print("🧹 Building processed dataset from raw CSVs...")
        df = clean_data(load_data(name, base_path=RAW_DATA_PATH))
        # Save a cache for future runs
        try:
            save_preprocessed(df, name, base_path=PROCESSED_DATA_PATH)
        except Exception as e:
            print(f"⚠️ Could not save processed cache: {e}")
    if limit is not None and len(df) > limit:
        df = df.head(limit).copy()
    return df

def _read_csvs_from(path: Path, limit: Optional[int] = None) -> pd.DataFrame:
    files: List[Path] = []
    if path.is_dir():
        files = sorted(path.glob("*.csv"))
    elif path.is_file() and path.suffix.lower() == ".csv":
        files = [path]
    if not files:
        raise FileNotFoundError(f"No CSVs found at {path}")

    frames: List[pd.DataFrame] = []
    total = 0
    for p in files:
        print(f"📥 Reading: {p}")
        frames.append(pd.read_csv(p, low_memory=False))
        total += len(frames[-1])
        if limit is not None and total >= limit:
            break
    df = pd.concat(frames, ignore_index=True)
    if limit is not None and len(df) > limit:
        df = df.head(limit).copy()
    return df


def load_for_stage(
    dataset: str,
    stage: str = "processed",              # 'raw' | 'processed'
    limit: Optional[int] = None,
    prefer_cache: bool = True,
) -> pd.DataFrame:
    """
    Accepts a dataset NAME (e.g., 'CIC-IDS2017') or a PATH to CSVs.
    - stage='raw'      -> raw rows
    - stage='processed'-> cleaned/processed rows if available; else raw as fallback
    """
    # Try both import styles depending on your repo layout
    ingest_mod = None
    try:
        from src import ingest as ingest_mod  # type: ignore
    except Exception:
        try:
            import ingest as ingest_mod  # type: ignore
        except Exception:
            ingest_mod = None

    ds_path = Path(dataset)

    # --- PATH MODE -----------------------------------------------------------
    if ds_path.exists():
        raw_df = _read_csvs_from(ds_path, limit=limit)
        if stage == "raw":
            return raw_df
        # If we have a cleaner in ingest, use it; otherwise just treat as processed
        if ingest_mod is not None and hasattr(ingest_mod, "clean_data"):
            try:
                return ingest_mod.clean_data(raw_df)  # type: ignore[attr-defined]
            except Exception as e:
                print(f"⚠️  clean_data failed; using raw as-is for processed EDA ({e})")
        return raw_df

    # --- NAME MODE -----------------------------------------------------------
    name = dataset

    # 1) RAW
    if stage == "raw":
        # Preferred: ingest helper(s)
        if ingest_mod is not None:
            if hasattr(ingest_mod, "load_dataset"):
                try:
                    df = ingest_mod.load_dataset(name)  # type: ignore[attr-defined]
                    return df if limit is None else df.head(limit).copy()
                except TypeError:
                    pass
            if hasattr(ingest_mod, "load_data"):
                df = ingest_mod.load_data(name)  # type: ignore[attr-defined]
                return df if limit is None else df.head(limit).copy()
        # Fallback: data/raw/<name> or data/samples/<name>
        for base in ["data/raw", "data/samples", "data"]:
            p = Path(base) / name
            if p.exists():
                return _read_csvs_from(p, limit=limit)
        raise FileNotFoundError(f"Could not find raw dataset '{name}'")

    # 2) PROCESSED
    # Prefer cached processed CSVs if present
    if prefer_cache:
        for p in [
            Path("data/processed") / name,
            Path("data/processed") / f"{name}.csv",
            Path("data/processed") / f"{name}_clean.csv",
        ]:
            if p.exists():
                return _read_csvs_from(p, limit=limit)

    # Build processed via ingest if possible
    if ingest_mod is not None:
        if hasattr(ingest_mod, "load_clean"):
            df = ingest_mod.load_clean(name)  # type: ignore[attr-defined]
            return df if limit is None else df.head(limit).copy()
        if hasattr(ingest_mod, "load_dataset"):
            try:
                df = ingest_mod.load_dataset(name, cleaned=True)  # type: ignore[attr-defined]
                return df if limit is None else df.head(limit).copy()
            except TypeError:
                pass
        if hasattr(ingest_mod, "clean_data"):
            # Build from raw using the cleaner
            raw_df = load_for_stage(name, stage="raw", limit=limit, prefer_cache=prefer_cache)
            try:
                df = ingest_mod.clean_data(raw_df)  # type: ignore[attr-defined]
                return df if limit is None else df.head(limit).copy()
            except Exception as e:
                print(f"⚠️  clean_data failed; using raw as processed fallback ({e})")

    # Last fallback: try processed folder anyway
    p = Path("data/processed") / name
    if p.exists():
        return _read_csvs_from(p, limit=limit)

    raise FileNotFoundError(f"Could not load processed dataset '{name}'")

def _get_str_attr(mod, names):
    for n in names:
        if hasattr(mod, n):
            v = getattr(mod, n)
            if isinstance(v, str) and v.strip():
                return v
    return None


def resolve_dataset(arg_ds: Optional[str]) -> str:
    # 1) CLI wins
    if arg_ds:
        return arg_ds

    # 2) Env var
    env = os.getenv("ICT_DATASET")
    if env:
        return env

    # 3) Try defaults exposed by your code
    for mod_name in ("ingest", "src.ingest", "features", "src.features"):
        try:
            mod = __import__(mod_name, fromlist=["*"])
        except Exception:
            continue
        ds = _get_str_attr(mod, ["DATASET_NAME", "DEFAULT_DATASET", "DATASET", "dataset"])
        if ds:
            return ds

    # 4) Auto-detect a single dataset folder with CSVs
    candidates = []
    for base in ("data/processed", "data/raw", "data/samples", "data"):
        bp = Path(base)
        if not bp.exists():
            continue
        for d in bp.iterdir():
            if d.is_dir() and any(d.glob("*.csv")):
                candidates.append(d.name)

    uniq = sorted(set(candidates))
    if len(uniq) == 1:
        return uniq[0]

    msg = "error: Could not determine dataset automatically."
    if uniq:
        msg += "\nDetected datasets:\n  - " + "\n  - ".join(uniq)
    raise SystemExit(msg + "\nPass --dataset, set ICT_DATASET, or expose DATASET_NAME in ingest.py.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EDA on network dataset (raw or processed)")
    parser.add_argument("--dataset", required=False, help="Dataset NAME or PATH (optional if ingest exposes DATASET_NAME)")
    parser.add_argument("--stage", choices=["raw", "processed", "both"], default="both")
    parser.add_argument("--out", default=None, help="Output dir root (default: eda_reports/<dataset>/<stage>)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    # Ensure repo root is on sys.path for 'import ingest'
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    dataset = resolve_dataset(args.dataset)        # <-- no arguments needed in PyCharm now
    ds_label = Path(dataset).name if Path(dataset).exists() else dataset
    stages = ["raw", "processed"] if args.stage == "both" else [args.stage]
    prefer_cache = not args.no_cache

    for st in stages:
        outdir = Path(args.out) if args.out else Path("eda_reports") / ds_label / st
        print(f"\n=== EDA for {ds_label} [{st}] ===")
        df = load_for_stage(dataset, stage=st, limit=args.limit, prefer_cache=prefer_cache)
        run_eda(df, outdir)


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

from __future__ import annotations

import os
from glob import glob
import pandas as pd

def get_dataset_dir(model_dir: str, dataset: str, subdir: str | None = None) -> str:
    if subdir:
        out_dir = os.path.join(model_dir, dataset, subdir)
    else:
        out_dir = os.path.join(model_dir, dataset)

    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def project_root(file_: str) -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(file_)))

def data_paths(root: str) -> tuple[str, str]:
    raw = os.path.join(root, "data", "samples")
    processed = os.path.join(root, "data", "processed")
    os.makedirs(processed, exist_ok=True)
    return raw, processed

def list_csvs(dataset_dir: str) -> list[str]:
    # Supports .csv and simple compressed csvs
    files = sorted(glob(os.path.join(dataset_dir, "*.csv*")))
    if not files:
        raise ValueError(f"No CSV files found in {dataset_dir}")
    return files

def read_csv_safely(path: str, low_memory: bool = False) -> pd.DataFrame:
    # Use pyarrow engine if available for speed; fallback to default
    read_kwargs = {"low_memory": low_memory}
    try:
        return pd.read_csv(path, engine="pyarrow")  # type: ignore[arg-type]
    except Exception:
        return pd.read_csv(path, **read_kwargs)

def save_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

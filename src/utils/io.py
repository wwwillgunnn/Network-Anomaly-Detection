"""
This file has helper functions designed for managing dataset file paths and CSV operations.

- get_dataset_dir(): Builds (and creates if needed) a directory path for a dataset.
- project_root(): Returns the root directory of the project based on the current file.
- data_paths(): Returns paths to the raw and processed data folders.
- list_csvs(): Lists all CSV files in a folder.
- read_csv_safely(): Reads a CSV file using 'pyarrow' (fast, if available) or fallback to pandas’ default reader.
- save_csv(): Saves a DataFrame to a CSV file, creating directories if needed.
"""

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
    os.makedirs(raw, exist_ok=True)
    os.makedirs(processed, exist_ok=True)
    return raw, processed


def list_csvs(dataset_dir: str) -> list[str]:
    files = sorted(glob(os.path.join(dataset_dir, "*.csv*")))
    if not files: raise ValueError(f"No CSV files found in {dataset_dir}")
    return files


def read_csv_safely(path: str, low_memory: bool = False) -> pd.DataFrame:
    read_kwargs = {"low_memory": low_memory}
    try:
        return pd.read_csv(path, engine="pyarrow")
    except Exception:
        return pd.read_csv(path, **read_kwargs)


def save_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

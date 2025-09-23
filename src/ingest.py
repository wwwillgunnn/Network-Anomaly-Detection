from __future__ import annotations
import os
from typing import Literal
import pandas as pd
from utils.io import project_root, data_paths, list_csvs, read_csv_safely, save_csv
from utils.clean import basic_clean

DatasetName = Literal["CIC-IDS2017", "CIC-IDS2018", "CIC-IDS2019"]

# --- Paths ------------------------------------------------------------------
PROJECT_ROOT = project_root(__file__)
RAW_DATA_PATH, PROCESSED_DATA_PATH = data_paths(PROJECT_ROOT)


def load_data(dataset: DatasetName, base_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Concat all CSVs under data/samples/<dataset> into a single DataFrame."""
    dataset_path = os.path.join(base_path, dataset)
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    frames = []
    for file_path in list_csvs(dataset_path):
        print(f"📂 Loading {file_path} ...")
        frames.append(read_csv_safely(file_path, low_memory=False))
    combined = pd.concat(frames, ignore_index=True)
    print(f"✅ Loaded {len(combined):,} rows from {dataset} ({len(frames)} files)")
    return combined


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Conservative, dataset-agnostic cleanup."""
    return basic_clean(df)


def save_preprocessed(df: pd.DataFrame, dataset: DatasetName, base_path: str = PROCESSED_DATA_PATH) -> str:
    """Write /data/processed/<dataset>_clean.csv and return its path."""
    out_path = os.path.join(base_path, f"{dataset}_clean.csv")
    save_csv(df, out_path)
    print(f"💾 Saved processed dataset → {out_path}")
    return out_path

# Backward-compat alias
save_processed = save_preprocessed


if __name__ == "__main__":
    for ds in ["CIC-IDS2017"]:
        raw = load_data(ds)
        clean = clean_data(raw)
        save_preprocessed(clean, ds)
        print(clean.head(), clean.shape)

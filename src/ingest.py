"""
This file takes care of data ingestion and preprocessing

- load_data(): Reads and merges all CSV files under data/samples/<dataset> into one DataFrame.
- clean_data(): Applies basic cleaning (remove junk columns, NaNs, duplicates, etc.).
- save_preprocessed(): Saves the cleaned dataset to data/processed/<dataset>_clean.csv.

When run directly (python ingest.py), it performs the full pipeline:
loads the raw dataset, cleans it, saves the processed version, and prints a preview.
"""

from __future__ import annotations
import os
from typing import Literal
import pandas as pd
from utils.io import project_root, data_paths, list_csvs, read_csv_safely, save_csv
from utils.clean import basic_clean

DatasetName = Literal["CIC-IDS2017", "CIC-IDS2018", "CIC-IDS2019"]

PROJECT_ROOT = project_root(__file__)
RAW_DATA_PATH, PROCESSED_DATA_PATH = data_paths(PROJECT_ROOT)

def load_data(dataset: DatasetName, base_path: str = RAW_DATA_PATH) -> pd.DataFrame:
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
    return basic_clean(df)

def save_preprocessed(df: pd.DataFrame, dataset: DatasetName, base_path: str = PROCESSED_DATA_PATH) -> str:
    out_path = os.path.join(base_path, f"{dataset}_clean.csv")
    save_csv(df, out_path)
    print(f"💾 Saved processed dataset → {out_path}")
    return out_path


if __name__ == "__main__":
    # Example standalone usage: load, clean, and save a dataset.
    for ds in ["CIC-IDS2017"]:
        raw = load_data(ds)
        clean = clean_data(raw)
        save_preprocessed(clean, ds)
        print(clean.head(), clean.shape)

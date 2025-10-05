"""
This file is a dataset preparation pipeline for a dataset.

- load_processed(): Loads a cleaned, preprocessed version of the dataset.
    Uses a cached CSV if available; otherwise, builds it from raw data using ingest and cleaning utilities.

- prepare_features(): Converts the dataset into model-ready features (X) and binary labels (y).
    handles missing columns, numeric conversion, and label binarization ('BENIGN' → 0, others → 1).
"""

from __future__ import annotations
import os
from typing import Tuple
import pandas as pd
from ingest import RAW_DATA_PATH, PROCESSED_DATA_PATH, load_data, clean_data, save_preprocessed
from utils.prep import select_existing, to_numeric_finite, binarize_benign_label

__all__ = ["load_processed", "prepare_features"]

# TODO: try with other datasets
DATASET_NAME = "CIC-IDS2017"

# TODO: optomise feat cols
FEAT_COLS = [
    "Flow_Duration",
    "Total_Fwd_Packets",
    "Total_Backward_Packets",
    "Total_Length_of_Fwd_Packets",
    "Total_Length_of_Bwd_Packets",
    "Packet_Length_Mean",
    "Packet_Length_Std",
    "Bwd_Packet_Length_Std",
    "Flow_Bytes_s",
    "Flow_Packets_s",
    "Fwd_Packets_s",
    "Bwd_Packets_s",
    "Fwd_IAT_Mean",
    "Bwd_IAT_Mean",
    "SYN_Flag_Count",
    "FIN_Flag_Count",
    "ACK_Flag_Count",
]

def load_processed(
    *, prefer_cache: bool = True,
    processed_path: str = PROCESSED_DATA_PATH,
    raw_path: str = RAW_DATA_PATH,
) -> pd.DataFrame:
    cache = os.path.join(processed_path, f"{DATASET_NAME}_clean.csv")
    if prefer_cache and os.path.exists(cache):
        print(f"📦 Using cached processed dataset → {cache}")
        return pd.read_csv(cache, low_memory=False)
    print("🧹 Building processed dataset from raw CSVs...")
    cleaned = clean_data(load_data(DATASET_NAME, base_path=raw_path))
    path = save_preprocessed(cleaned, DATASET_NAME, base_path=processed_path)
    return pd.read_csv(path, low_memory=False)

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    y = binarize_benign_label(df)
    X, missing = select_existing(df, FEAT_COLS)
    if missing:
        print(f"⚠️ Missing expected features (continuing without them): {missing}")
    X = to_numeric_finite(X)
    return X, y

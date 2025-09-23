"""
feature preparation — CIC-IDS2017 → (X, y)
Select a compact, stable numeric slice; coerce to numeric & finite; BENIGN→0, attack→1.
"""
from __future__ import annotations
import os
from typing import Tuple
import pandas as pd

from ingest import RAW_DATA_PATH, PROCESSED_DATA_PATH, load_data, clean_data, save_preprocessed
from utils.prep import select_existing, to_numeric_finite, binarize_benign_label

__all__ = ["load_processed", "prepare_features"]

DATASET_NAME = "CIC-IDS2017"

# Compact, discriminative slice; safe with CIC-IDS2017 normalized column names
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
    """Return cleaned CIC-IDS2017; use cache if present, else build from raw."""
    cache = os.path.join(processed_path, f"{DATASET_NAME}_clean.csv")
    if prefer_cache and os.path.exists(cache):
        print(f"📦 Using cached processed dataset → {cache}")
        return pd.read_csv(cache, low_memory=False)
    print("🧹 Building processed dataset from raw CSVs...")
    cleaned = clean_data(load_data(DATASET_NAME, base_path=raw_path))
    path = save_preprocessed(cleaned, DATASET_NAME, base_path=processed_path)
    return pd.read_csv(path, low_memory=False)

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Produce model-ready (X, y) for CIC-IDS2017."""
    # y first (explicit failure if missing)
    y = binarize_benign_label(df)

    # feature subset (warn if some missing, but proceed with what we have)
    X, missing = select_existing(df, FEAT_COLS)
    if missing:
        print(f"⚠️ Missing expected features (continuing without them): {missing}")

    # numeric + finite
    X = to_numeric_finite(X)
    return X, y

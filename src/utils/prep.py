"""
This file has helper functions designed for selecting, cleaning, and preparing dataset features and labels.

- select_existing(): Selects columns that exist in the DataFrame, return a list for existing and missing columns.
- to_numeric_finite(): Converts all values to numeric, replacing NaN or infinite values with 0.
    Guarantee all columns are numeric if the dataset has categorical or text data
- binarize_benign_label(): Creates a binary label column where 'BENIGN' → 0 and all others like 'ATTACK' → 1.
"""

from __future__ import annotations
from typing import Iterable, Tuple
import numpy as np
import pandas as pd

def select_existing(df: pd.DataFrame, cols: Iterable[str]) -> Tuple[pd.DataFrame, list[str]]:
    cols = list(cols)
    exist = [col for col in cols if col in df.columns]
    missing = [col for col in cols if col not in df.columns]
    if not exist:
        raise ValueError("No expected feature columns found. Check ingest/normalization.")
    return df[exist].copy(), missing

def to_numeric_finite(df: pd.DataFrame) -> pd.DataFrame:
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df

def binarize_benign_label(df: pd.DataFrame, candidates: tuple[str, ...] = ("Label", "label")) -> pd.Series:
    lbl = next((c for c in candidates if c in df.columns), None)
    if lbl is None:
        raise KeyError(f"No label column found (tried {candidates}).")
    y = (~df[lbl].astype(str).str.upper().eq("BENIGN")).astype("int64")
    y.name = "label"
    return y

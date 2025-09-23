from __future__ import annotations
from typing import Iterable, Tuple
import numpy as np
import pandas as pd

def select_existing(df: pd.DataFrame, cols: Iterable[str]) -> Tuple[pd.DataFrame, list[str]]:
    """Return df[existing_cols], warn-list of missing."""
    cols = list(cols)
    exist = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if not exist:
        raise ValueError("No expected feature columns found. Check ingest/normalization.")
    return df[exist].copy(), missing

def to_numeric_finite(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce to numeric, replace NaN/±inf with 0."""
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df

def binarize_benign_label(df: pd.DataFrame, candidates: tuple[str, ...] = ("Label", "label")) -> pd.Series:
    """Return y where BENIGN→0, else→1."""
    lbl = next((c for c in candidates if c in df.columns), None)
    if lbl is None:
        raise KeyError(f"No label column found (tried {candidates}).")
    y = (~df[lbl].astype(str).str.upper().eq("BENIGN")).astype("int64")
    y.name = "label"
    return y

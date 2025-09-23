from __future__ import annotations
import numpy as np
import pandas as pd

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.strip("_")
    )
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # drop "Unnamed" junk + all-NaN columns, dedupe, fill, make finite
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed", case=False)]
    df = df.dropna(axis=1, how="all")
    df = normalize_columns(df)
    df = df.drop_duplicates()
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

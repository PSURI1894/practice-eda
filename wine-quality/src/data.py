"""Load + clean the UCI Wine Quality dataset (red + white combined: 6,497 wines).

11 physicochemical features + an ordinal `quality` score (3-9) + a `wine_type` flag (red/white).
Two things make this dataset distinctive vs the time-series practices:
  * the target is **ordinal and heavily imbalanced** (quality 5-6 ~ 76%; the extremes ~7%),
  * there are **1,177 duplicate rows (18%)** — which cause train/test leakage if split carelessly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_PROC, DATA_RAW

# The 11 physicochemical measurements (all numeric).
NUMERIC = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
           "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"]


def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW / "wine.csv")


def clean(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Light clean: type the wine flag. Duplicates are KEPT here (flagged, not dropped) — whether
    they are genuine repeat measurements or artifacts is a judgement call made explicit in Part 0."""
    df = load_raw() if df is None else df.copy()
    df["wine_type"] = df["wine_type"].astype("category")
    return df


def dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact-duplicate rows — used before modelling so a wine can't sit in both train and test."""
    return df.drop_duplicates().reset_index(drop=True)


def quality_band(q: pd.Series) -> pd.Series:
    """Collapse the ordinal score into 3 coarse, ordered classes — a simpler, imbalance-friendly
    framing of the problem (still imbalanced: mid ~ 77%)."""
    return pd.cut(q, [2, 4, 6, 9], labels=["low (<=4)", "mid (5-6)", "high (>=7)"])


def features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[NUMERIC + ["wine_type"]], df["quality"]


def build_processed() -> pd.DataFrame:
    d = clean()
    d.to_csv(DATA_PROC / "wine_clean.csv", index=False)
    return d


if __name__ == "__main__":
    d = build_processed()
    print(d.shape, "| duplicates:", int(d.duplicated().sum()), "| quality range:", d.quality.min(), "-", d.quality.max())

"""Load + clean the UCI Bike-Sharing hourly dataset (Washington D.C., 2011-2012).

Target is `cnt` = hourly rental count. Three things to get right (Part 0):
  * LEAKAGE — `cnt == casual + registered` exactly, so casual/registered must NOT be predictors.
  * Integer-coded categoricals — season/weathersit/hr/... are stored as ints but are categorical.
  * Normalized weather — temp/atemp/hum/windspeed are scaled to [0,1]; we restore real units.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_PROC, DATA_RAW

SEASON = {1: "spring", 2: "summer", 3: "fall", 4: "winter"}
WEATHER = {1: "clear", 2: "mist", 3: "light_precip", 4: "heavy_precip"}
# Stored as integers but genuinely categorical/ordinal — recast so stats don't treat them as numbers.
CAT_COLS = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
# cnt = casual + registered, so these leak the target. Exclude from any predictor set.
LEAKAGE = ["casual", "registered"]


def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW / "bike_hour.csv")


def clean(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return an analysis-ready frame with a datetime index, real-unit weather, and
    human-readable categoricals (codes kept too). Drops the row index `instant`."""
    df = load_raw() if df is None else df.copy()

    # 1. Build a proper hourly DatetimeIndex from the date + hour columns.
    df["datetime"] = pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
    df = df.set_index("datetime").sort_index()

    # 2. De-normalize weather back to interpretable units (UCI divided by these constants).
    df["temp_C"] = df["temp"] * 41        # Celsius
    df["atemp_C"] = df["atemp"] * 50      # "feels like" Celsius
    df["hum_pct"] = df["hum"] * 100       # %
    df["wind_kmh"] = df["windspeed"] * 67 # km/h

    # 3. Readable labels for the key categoricals + a real year.
    df["season_name"] = df["season"].map(SEASON).astype("category")
    df["weather_name"] = df["weathersit"].map(WEATHER).astype("category")
    df["year"] = df["yr"].map({0: 2011, 1: 2012})

    # 4. Recast the integer-coded categoricals so describe()/corr() don't treat them as continuous.
    for c in CAT_COLS:
        df[c] = df[c].astype("category")

    # 5. Drop the pure row identifier and the now-redundant raw date.
    df = df.drop(columns=["instant", "dteday"])
    return df


def features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Model-ready X (leakage + target removed) and y = cnt."""
    X = df.drop(columns=LEAKAGE + ["cnt"], errors="ignore")
    return X, df["cnt"]


def missing_hours(df: pd.DataFrame) -> int:
    """How many hourly slots are absent from the 2-year grid (the series has gaps)."""
    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    return len(full.difference(df.index))


def hourly_cnt() -> pd.Series:
    """`cnt` on a GAPLESS hourly grid (the 165 absent slots time-interpolated), freq='h'.
    Time-series models assume a regular index — this supplies it."""
    s = clean()["cnt"]
    full = pd.date_range(s.index.min(), s.index.max(), freq="h")
    out = s.reindex(full).interpolate("time")
    out.index.freq = "h"
    return out


def build_processed() -> pd.DataFrame:
    clean_df = clean()
    clean_df.to_csv(DATA_PROC / "bike_clean.csv")
    return clean_df


if __name__ == "__main__":
    d = build_processed()
    print(d.shape, "| missing hourly slots:", missing_hours(d))

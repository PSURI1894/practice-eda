"""Load + clean the Beijing PM2.5 hourly dataset (US Embassy, 2010-2014).

Target is `pm25` (PM2.5 concentration, ug/m3). The timeline is **complete** — 5 years of hourly
readings (43,824 rows, no missing rows) — but the **pm25 values have 2,067 real gaps (4.7%)**, and
they are *structured* (the first 24 hours; runs up to 155 consecutive hours). That real missingness
is the theme this practice is built around.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_PROC, DATA_RAW

# Weather covariates (all complete). DEWP=dew point, PRES=pressure, Iws=cumulative wind speed,
# Is=cumulative snow hours, Ir=cumulative rain hours.
WEATHER = ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
WHO_24H = 15.0   # WHO 2021 24-hour PM2.5 guideline (ug/m3) — Beijing's mean is ~7x this.


def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW / "beijing_pm25.csv")


def clean(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Datetime-indexed frame with readable names. pm25 gaps are KEPT as NaN (the imputation work
    happens later); the weather columns are already complete."""
    df = load_raw() if df is None else df.copy()
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.set_index("datetime").sort_index()
    df = df.rename(columns={"pm2.5": "pm25", "cbwd": "wind_dir"})
    df = df.drop(columns=["No", "year", "month", "day", "hour"])
    df["wind_dir"] = df["wind_dir"].astype("category")   # SE / NW / NE / cv(=calm-variable)
    df.index.freq = "h"
    return df


def gap_runs(s: pd.Series) -> pd.Series:
    """Lengths of each run of consecutive missing values — shows the *shape* of the missingness
    (a few long sensor outages vs many scattered single hours)."""
    isna = s.isna().to_numpy()
    runs, c = [], 0
    for v in isna:
        if v:
            c += 1
        elif c:
            runs.append(c); c = 0
    if c:
        runs.append(c)
    return pd.Series(runs, dtype=int)


def build_processed() -> pd.DataFrame:
    d = clean()
    d.to_csv(DATA_PROC / "beijing_clean.csv")
    return d


if __name__ == "__main__":
    d = build_processed()
    print(d.shape, "| pm25 missing:", int(d.pm25.isna().sum()), "| longest gap:", int(gap_runs(d.pm25).max()))

"""Feature engineering for demand forecasting: Fourier seasonal terms (for harmonic
regression) and the lag/calendar/weather matrix (for tree models).

Key discipline: at a **24-hour-ahead** horizon, any lag of ≥24 hours is already observed when
the forecast is made, so `lag24`/`lag168`/`roll24` are leakage-free *known* regressors — no
recursion needed. (A longer horizon would require recursion; see the notebook.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Columns known for any future hour (calendar is exact; weather assumed known — a caveat).
CALENDAR = ["hr", "dow", "mnth", "yr", "workingday", "holiday", "season", "weathersit"]
WEATHER = ["temp_C", "hum_pct", "wind_kmh"]


def fourier_terms(index: pd.DatetimeIndex, period: int, K: int) -> pd.DataFrame:
    """K sin/cos harmonic pairs encoding a seasonal cycle of length `period`.
    Computed on a contiguous position vector so train and (sliced) test stay phase-aligned."""
    t = np.arange(len(index))
    cols = {}
    for k in range(1, K + 1):
        cols[f"sin{period}_{k}"] = np.sin(2 * np.pi * k * t / period)
        cols[f"cos{period}_{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(cols, index=index)


def add_lags(g: pd.DataFrame, target: str = "cnt") -> pd.DataFrame:
    """Append the autoregressive features used by the tree model (all ≥24h, so 24h-ahead-safe)."""
    g = g.copy()
    g["lag24"] = g[target].shift(24)               # same hour yesterday
    g["lag168"] = g[target].shift(168)             # same hour last week
    g["roll24"] = g[target].shift(1).rolling(24).mean()   # last-24h average (ends at t-1)
    return g


LAG_COLS = ["lag24", "lag168", "roll24"]

"""Reframe a time series as supervised learning, leakage-safe, and forecast recursively.

The whole trick of ML forecasting is turning a 1-D series into an (X, y) table of lagged /
rolling / calendar features. The whole *danger* is leakage: every feature for time t must use
only information available strictly before t. We enforce that with `.shift(1)` on rolling stats
and by only ever using past lags.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_supervised(y: pd.Series, n_lags: int = 12, roll_windows=(3, 6, 12),
                    calendar: bool = True) -> pd.DataFrame:
    """Build the supervised table for series `y`.

    Features (all strictly past-only):
      lag1..lag_n     : y shifted back 1..n steps
      rmean_w / rstd_w: rolling mean/std of the *shifted* series (window ends at t-1)
      month           : calendar month (trees handle the integer fine)
    Target column 'y' is the value at t. Rows with any NaN (the warm-up) are dropped.
    """
    df = pd.DataFrame(index=y.index)
    for L in range(1, n_lags + 1):
        df[f"lag{L}"] = y.shift(L)
    shifted = y.shift(1)                       # <-- the leakage guard: rolling can't see t
    for w in roll_windows:
        df[f"rmean{w}"] = shifted.rolling(w).mean()
        df[f"rstd{w}"] = shifted.rolling(w).std()
    if calendar:
        df["month"] = y.index.month
    df["y"] = y.values
    return df.dropna()


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "y"]


def recursive_forecast(model, history: pd.Series, steps: int, cols: list[str],
                       n_lags: int = 12, roll_windows=(3, 6, 12), calendar: bool = True,
                       freq: str = "MS") -> pd.Series:
    """Multi-step forecast by feeding each prediction back in as the newest lag.

    Mirrors `make_supervised` exactly so the prediction-time feature row matches training.
    """
    hist = history.astype(float).copy()
    future_idx = pd.date_range(hist.index[-1], periods=steps + 1, freq=freq)[1:]
    preds = []
    for t in future_idx:
        feat = {f"lag{L}": hist.iloc[-L] for L in range(1, n_lags + 1)}
        for w in roll_windows:
            window = hist.iloc[-w:]
            feat[f"rmean{w}"] = window.mean()
            feat[f"rstd{w}"] = window.std()
        if calendar:
            feat["month"] = t.month
        row = pd.DataFrame([feat])[cols]
        yhat = float(model.predict(row)[0])
        hist.loc[t] = yhat
        preds.append(yhat)
    return pd.Series(preds, index=future_idx, name="forecast")


def reconstruct_from_diff(diff_preds: pd.Series, last_level: float) -> pd.Series:
    """Turn forecasts of the first difference back into levels: y_t = last_level + cumsum(Δ)."""
    return last_level + diff_preds.cumsum()

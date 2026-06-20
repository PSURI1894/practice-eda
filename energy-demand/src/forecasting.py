"""Forecasting helpers for the PJM load series (Parts 7–9).

Demand has triple seasonality (24 / 168 / 8766 h), so the building blocks here are: Fourier features
for harmonic regression, simple seasonal-naive baselines, and the standard accuracy metrics scaled to
the daily seasonal-naive (MASE < 1 beats "same hour yesterday").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HOUR, DAY, WEEK, YEAR = 1, 24, 168, 8766


def fourier_features(index: pd.DatetimeIndex, periods=((24, 5), (168, 3), (8766, 3))) -> pd.DataFrame:
    """Sin/cos harmonics for each (period_in_hours, n_harmonics) — lets a linear model represent the
    daily, weekly and annual cycles smoothly. t is hours since the series start."""
    t = (index - index[0]) / pd.Timedelta(hours=1)
    out = {}
    for P, K in periods:
        for k in range(1, K + 1):
            out[f"sin_{P}_{k}"] = np.sin(2 * np.pi * k * t / P)
            out[f"cos_{P}_{k}"] = np.cos(2 * np.pi * k * t / P)
    return pd.DataFrame(out, index=index)


def seasonal_naive(train: pd.Series, horizon: int, m: int) -> np.ndarray:
    """Forecast by repeating the last full season of length m (e.g. m=168 → 'same hour last week')."""
    last = train.values[-m:]
    return np.resize(last, horizon)


def mase(y_true, y_pred, train: pd.Series, m: int = 24) -> float:
    """Mean Absolute Scaled Error vs the in-sample seasonal-naive (period m). <1 beats naive."""
    scale = np.mean(np.abs(np.diff(train.values, n=1)[m-1:] if m == 1 else
                           train.values[m:] - train.values[:-m]))
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / scale


def metrics(y_true, y_pred, train: pd.Series, name: str = "") -> pd.Series:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
    return pd.Series({"MAE": np.mean(np.abs(y_true - y_pred)), "RMSE": rmse,
                      "MAPE%": mape, "WAPE%": wape, "MASE": mase(y_true, y_pred, train, m=24)},
                     name=name).round(3)

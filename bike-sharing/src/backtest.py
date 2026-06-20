"""Rigorous evaluation for 24-hour-ahead demand forecasting: a walk-forward backtester,
proper probabilistic metrics (pinball loss, coverage), and Conformalized Quantile Regression.

Leakage rigor: a forecast made at midnight may use only data through the previous day. So the
autoregressive features here are all **>=24h old** (`lag24`, `lag168`, `lag_dayavg`) — unlike a
`roll24` ending at t-1, which would peek at the same morning's demand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STRICT_LAGS = ["lag24", "lag168", "lag_dayavg"]


def add_strict_lags(g: pd.DataFrame, target: str = "cnt") -> pd.DataFrame:
    """Autoregressive features knowable at a 24h-ahead forecast origin (all >=24h back)."""
    g = g.copy()
    g["lag24"] = g[target].shift(24)                       # same hour yesterday
    g["lag168"] = g[target].shift(168)                     # same hour last week
    g["lag_dayavg"] = g[target].shift(24).rolling(24).mean()  # mean of the day ending 24h ago
    return g


def walk_forward_24h(g: pd.DataFrame, y: pd.Series, cols: list[str], make_model,
                     n_test_days: int, refit_every: int = 7, warmup: int = 168) -> pd.Series:
    """Operational backtest: each day, forecast the next 24h with a model trained on everything
    before that day; refit every `refit_every` days. Returns the stitched out-of-sample forecast."""
    n = len(g)
    start = n - n_test_days * 24
    pred = np.full(n_test_days * 24, np.nan)
    model = None
    for d in range(n_test_days):
        ds = start + d * 24
        if d % refit_every == 0:
            model = make_model()
            model.fit(g[cols].iloc[warmup:ds], y.iloc[warmup:ds])
        pred[d * 24:(d + 1) * 24] = model.predict(g[cols].iloc[ds:ds + 24])
    return pd.Series(np.clip(pred, 0, None), index=g.index[start:])


def pinball_loss(y_true, q_pred, alpha: float) -> float:
    """The proper score for a single quantile forecast at level `alpha` (a.k.a. quantile loss)."""
    e = np.asarray(y_true, float) - np.asarray(q_pred, float)
    return float(np.mean(np.maximum(alpha * e, (alpha - 1) * e)))


def coverage(y_true, lo, hi) -> float:
    y = np.asarray(y_true, float)
    return float(np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi))))


def cqr_offset(y_cal, lo_cal, hi_cal, alpha: float = 0.10) -> float:
    """Conformalized Quantile Regression (Romano et al. 2019): from a calibration set, the amount
    to widen [lo, hi] so the interval reaches (1-alpha) coverage — a finite-sample guarantee that
    fixes the chronic under-coverage of raw quantile regression."""
    E = np.maximum(np.asarray(lo_cal) - np.asarray(y_cal), np.asarray(y_cal) - np.asarray(hi_cal))
    return float(np.quantile(E, 1 - alpha))

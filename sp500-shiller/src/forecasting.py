"""Univariate forecasting helpers: time-ordered split, forecast metrics, naive baselines,
and a forecast-vs-actual plot. Keep the bookkeeping here so notebooks read as analysis.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .eda import savefig


# --------------------------------------------------------------- split
def ts_train_test(s: pd.Series, h: int) -> tuple[pd.Series, pd.Series]:
    """Hold out the LAST h observations. Time series are split by time, never shuffled —
    the test set must lie strictly in the future of the training set."""
    return s.iloc[:-h], s.iloc[-h:]


# --------------------------------------------------------------- metrics
def forecast_metrics(y_true, y_pred, y_train, m: int = 1) -> dict:
    """Standard forecast accuracy metrics.

    MAE / RMSE  — scale-dependent (same units as the series; RMSE punishes big misses).
    MAPE        — percentage error; undefined / explosive when y_true ≈ 0 (e.g. returns).
    sMAPE       — symmetric variant, bounded, but still odd near 0.
    MASE        — MAE scaled by the in-sample naive MAE. <1 beats naive, >1 loses to it.
                  This is the metric to trust across series of different scale.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    y_train = np.asarray(y_train, float)
    e = y_true - y_pred

    mae = np.mean(np.abs(e))
    rmse = np.sqrt(np.mean(e ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.mean(np.abs(e / y_true)) * 100
        smape = np.mean(2 * np.abs(e) / (np.abs(y_true) + np.abs(y_pred))) * 100
    wape = np.sum(np.abs(e)) / np.sum(np.abs(y_true)) * 100   # aggregate first -> robust to zeros
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))       # in-sample (seasonal) naive MAE
    mase = mae / scale if scale else np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE%": mape, "sMAPE%": smape, "WAPE%": wape, "MASE": mase}


def compare_models(test: pd.Series, preds: dict[str, np.ndarray], y_train: pd.Series,
                   m: int = 1) -> pd.DataFrame:
    """One row of metrics per model, sorted best-MASE first."""
    rows = {name: forecast_metrics(test.values, p, y_train.values, m) for name, p in preds.items()}
    return pd.DataFrame(rows).T.sort_values("MASE").round(4)


# --------------------------------------------------------------- baselines (the bar to beat)
def baseline_forecasts(train: pd.Series, h: int, m: int = 1) -> dict[str, np.ndarray]:
    """The four classical baselines. Any 'real' model must beat the best of these to earn its keep.

    naive          — repeat the last value (best for a random walk).
    seasonal_naive — repeat the value from m steps ago (best when seasonality dominates).
    drift          — extrapolate the straight line from first to last training point.
    mean           — the training mean (best when the series is flat noise around a level).
    """
    y = train.values
    last, n = y[-1], len(y)
    naive = np.repeat(last, h)
    season = np.array([y[-m + (i % m)] for i in range(h)]) if m > 1 else naive
    slope = (y[-1] - y[0]) / (n - 1)
    drift = last + slope * np.arange(1, h + 1)
    meanf = np.repeat(y.mean(), h)
    return {"naive": naive, "seasonal_naive": season, "drift": drift, "mean": meanf}


# --------------------------------------------------------------- plot
def plot_forecast(train: pd.Series, test: pd.Series, preds: dict[str, np.ndarray],
                  name: str, fname: str | None = None, tail: int | None = None,
                  interval: tuple[np.ndarray, np.ndarray] | None = None):
    """Plot (a tail of) train + the actual test path + one or more forecast paths."""
    fig, ax = plt.subplots(figsize=(12, 5))
    tr = train.iloc[-tail:] if tail else train
    ax.plot(tr.index, tr.values, color="0.4", lw=1, label="train")
    ax.plot(test.index, test.values, color="black", lw=2.2, label="actual (test)")
    for label, p in preds.items():
        ax.plot(test.index, p, lw=1.8, ls="--", label=label)
    if interval is not None:
        ax.fill_between(test.index, interval[0], interval[1], color="tab:blue", alpha=0.15,
                        label="95% interval")
    ax.axvline(test.index[0], color="red", ls=":", lw=1)
    ax.set_title(f"{name}: forecast vs actual"); ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    if fname:
        savefig(fig, fname)
    return fig

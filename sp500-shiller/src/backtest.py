"""Honest evaluation: time-series cross-validation, a model-agnostic walk-forward
backtester, and conformal prediction intervals.

The single train/test split of Parts 3 & 5 measures one window — luck. Real evaluation
slides the split across time and refits at each step, the way the model would actually run.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .eda import savefig


# --------------------------------------------------------------- CV split schemes
def cv_folds(n: int, n_splits: int = 5, horizon: int = 24, mode: str = "expanding"):
    """Yield (train_idx, test_idx) arrays for time-ordered CV.

    expanding : train grows from the start each fold (anchored).
    sliding   : train is a fixed-width window that rolls forward.
    Test blocks of length `horizon` tile the end of the series; train is everything before
    (gap-free). Never shuffles — the test fold is always in the future of its train fold.
    """
    test_starts = [n - (n_splits - k) * horizon for k in range(n_splits)]
    width = test_starts[0]                      # first fold's train length = sliding window width
    for start in test_starts:
        test_idx = np.arange(start, start + horizon)
        train_idx = np.arange(0, start) if mode == "expanding" else np.arange(max(0, start - width), start)
        yield train_idx, test_idx


def plot_cv(n: int, n_splits: int, horizon: int, fname: str | None = None):
    """Visualise expanding vs sliding folds (the classic CV-scheme diagram)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharex=True, sharey=True)
    for ax, mode in zip(axes, ["expanding", "sliding"]):
        for i, (tr, te) in enumerate(cv_folds(n, n_splits, horizon, mode)):
            ax.barh(i, len(tr), left=tr[0], color="tab:blue", height=0.6)
            ax.barh(i, len(te), left=te[0], color="tab:orange", height=0.6)
        ax.set_title(f"{mode} window"); ax.set_xlabel("time index"); ax.set_ylabel("fold")
        ax.invert_yaxis()
    axes[0].legend(["train", "test"], loc="upper left")
    fig.tight_layout()
    if fname:
        savefig(fig, fname)
    return fig


# --------------------------------------------------------------- walk-forward backtest
def walk_forward(y: pd.Series, forecaster, initial: int, horizon: int = 1, step: int = 1) -> pd.Series:
    """Re-fit and forecast across the series, the way it would run live.

    `forecaster(train_series, h)` returns an array of h predictions. With horizon=step=1 this is
    classic one-step-ahead backtesting -> a full out-of-sample prediction series.
    """
    preds = []
    start = initial
    while start + horizon <= len(y):
        train = y.iloc[:start]
        fc = np.asarray(forecaster(train, horizon)).ravel()
        idx = y.index[start:start + horizon]
        preds.append(pd.Series(fc[:len(idx)], index=idx))
        start += step
    return pd.concat(preds)


# --------------------------------------------------------------- conformal intervals
def conformal_q(calib_residuals: np.ndarray, alpha: float = 0.1) -> float:
    """Split-conformal radius: the (1-alpha) quantile of |calibration residuals|.
    point ± q is a distribution-free interval with ~(1-alpha) coverage."""
    r = np.abs(np.asarray(calib_residuals, float))
    return float(np.quantile(r, 1 - alpha))


def coverage(y_true, lower, upper) -> float:
    y_true = np.asarray(y_true, float)
    return float(((y_true >= np.asarray(lower)) & (y_true <= np.asarray(upper))).mean())

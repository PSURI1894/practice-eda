"""Time-series foundations helpers: stationarity tests + the ADF×KPSS decision,
ACF/PACF identification, rolling-statistic and decomposition plots.

These wrap statsmodels so the notebooks read as analysis, not boilerplate.
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss

from .eda import savefig


# --------------------------------------------------------------- stationarity tests
def adf_test(s: pd.Series, regression: str = "c") -> dict:
    """Augmented Dickey-Fuller. H0: a unit root is present (NON-stationary).
    p < 0.05  ->  reject H0  ->  evidence the series IS stationary."""
    s = s.dropna()
    stat, p, lags, nobs, crit, _ = adfuller(s, regression=regression, autolag="AIC")
    return {"test": "ADF", "stat": stat, "p_value": p, "used_lags": lags,
            "crit_5%": crit["5%"], "stationary_5pct": p < 0.05}


def kpss_test(s: pd.Series, regression: str = "c") -> dict:
    """Kwiatkowski-Phillips-Schmidt-Shin. H0: the series IS (trend-)stationary.
    p < 0.05  ->  reject H0  ->  evidence the series is NON-stationary.
    (Opposite null to ADF — that is exactly why they are used together.)
    regression='c' tests level-stationarity, 'ct' tests trend-stationarity."""
    s = s.dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # p-value is interpolated & capped to [0.01, 0.10]
        stat, p, lags, crit = kpss(s, regression=regression, nlags="auto")
    return {"test": "KPSS", "stat": stat, "p_value": p, "used_lags": lags,
            "crit_5%": crit["5%"], "stationary_5pct": p > 0.05}


def stationarity_report(s: pd.Series, regression: str = "c", name: str = "series"):
    """Run ADF + KPSS and resolve them with the canonical 2x2 decision table.

    ADF stationary?  KPSS stationary?   verdict
    -----------------------------------------------------------------
        yes              yes            STATIONARY
        no               no             NON-STATIONARY (unit root) -> difference
        yes              no             DIFFERENCE-STATIONARY       -> difference
        no               yes            TREND-STATIONARY            -> detrend
    """
    adf = adf_test(s, regression="c")
    kp = kpss_test(s, regression=regression)
    a, k = adf["stationary_5pct"], kp["stationary_5pct"]
    if a and k:
        verdict = "STATIONARY"
    elif not a and not k:
        verdict = "NON-STATIONARY (unit root) -> difference"
    elif a and not k:
        verdict = "DIFFERENCE-STATIONARY -> difference"
    else:
        verdict = "TREND-STATIONARY -> detrend"

    table = pd.DataFrame([adf, kp]).set_index("test")[
        ["stat", "p_value", "crit_5%", "stationary_5pct"]
    ].round(4)
    table.attrs["verdict"] = verdict
    table.attrs["name"] = name
    return table, verdict


# --------------------------------------------------------------- visual diagnostics
def rolling_plot(s: pd.Series, window: int, name: str, fname: str | None = None):
    """Series with rolling mean & std. A drifting mean or a changing band = non-stationary."""
    s = s.dropna()
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax[0].plot(s.index, s, lw=0.8, color="steelblue", label=name)
    ax[0].plot(s.index, s.rolling(window).mean(), color="black", lw=1.6, label=f"rolling mean ({window})")
    ax[0].set_title(f"{name}: level & rolling mean"); ax[0].legend(loc="upper left")
    ax[1].plot(s.index, s.rolling(window).std(), color="crimson", lw=1.2)
    ax[1].set_title(f"{name}: rolling std ({window}) — variance over time (volatility clustering)")
    fig.tight_layout()
    if fname:
        savefig(fig, fname)
    return fig


def acf_pacf_plot(s: pd.Series, lags: int, name: str, fname: str | None = None):
    """ACF (total autocorrelation, incl. indirect) vs PACF (direct effect at each lag).
    Identification: AR(p) -> PACF cuts off at p, ACF tails off; MA(q) -> ACF cuts off at q."""
    s = s.dropna()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    plot_acf(s, lags=lags, ax=ax[0], zero=False)
    ax[0].set_title(f"{name}: ACF")
    plot_pacf(s, lags=lags, ax=ax[1], method="ywm", zero=False)
    ax[1].set_title(f"{name}: PACF")
    fig.tight_layout()
    if fname:
        savefig(fig, fname)
    return fig


def decomposition_strengths(result) -> dict:
    """Hyndman-Athanasopoulos strength of trend / seasonality from a decomposition result,
    each in [0, 1]:  F = max(0, 1 - Var(resid) / Var(component + resid)).
    ~1 = that component dominates the remainder; ~0 = it is negligible."""
    r = pd.Series(result.resid).dropna()
    s = pd.Series(result.seasonal).reindex(r.index)
    t = pd.Series(result.trend).reindex(r.index)
    f_trend = max(0.0, 1 - r.var() / (t + r).var())
    f_seas = max(0.0, 1 - r.var() / (s + r).var())
    return {"trend_strength": round(float(f_trend), 3), "seasonal_strength": round(float(f_seas), 3)}


def ljung_box(s: pd.Series, lags: int = 12) -> pd.DataFrame:
    """Ljung-Box white-noise test. H0: no autocorrelation up to `lags`.
    p < 0.05 at some lag -> the series still carries structure (not white noise)."""
    return acorr_ljungbox(s.dropna(), lags=lags, return_df=True).round(4)

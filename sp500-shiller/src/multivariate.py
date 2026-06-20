"""Multivariate time-series helpers: correlation clustering, PCA factors, VAR lag
selection, Granger-causality matrix, and cointegration (Engle-Granger / Johansen) +
the pairs-trading spread.

Stationary inputs (returns) for correlation / PCA / VAR / Granger; price *levels* for
cointegration (which is precisely about levels sharing a common stochastic trend).
"""
from __future__ import annotations

import contextlib
import io
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller, coint, grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from .eda import savefig


# --------------------------------------------------------------- correlation (clustered)
def cluster_order(corr: pd.DataFrame) -> list[str]:
    """Order columns so correlated names sit together (hierarchical clustering on 1-corr)."""
    dist = squareform(1 - corr.values, checks=False)
    link = linkage(dist, method="average")
    return [corr.columns[i] for i in leaves_list(link)]


def corr_heatmap(returns: pd.DataFrame, fname: str | None = None):
    corr = returns.corr()
    order = cluster_order(corr)
    corr = corr.loc[order, order]
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Return correlations (clustered — sectors group together)")
    fig.tight_layout()
    if fname:
        savefig(fig, fname)
    return corr


# --------------------------------------------------------------- PCA factor model
def pca_factors(returns: pd.DataFrame):
    """PCA on standardized returns. PC1 is typically the 'market factor' (all-positive
    loadings). Returns (pca, explained_variance_ratio, loadings_df, scores_df)."""
    z = (returns - returns.mean()) / returns.std()
    pca = PCA().fit(z.values)
    evr = pd.Series(pca.explained_variance_ratio_, index=[f"PC{i+1}" for i in range(pca.n_components_)])
    loadings = pd.DataFrame(pca.components_.T, index=returns.columns,
                            columns=evr.index)
    scores = pd.DataFrame(pca.transform(z.values), index=returns.index, columns=evr.index)
    return pca, evr, loadings, scores


# --------------------------------------------------------------- VAR + Granger
def granger_matrix(returns: pd.DataFrame, maxlag: int = 5) -> pd.DataFrame:
    """Pairwise Granger causality. Entry [row=effect, col=cause] = smallest p-value over
    1..maxlag for H0 'col does NOT Granger-cause row'. Small p => col helps predict row."""
    cols = list(returns.columns)
    out = pd.DataFrame(np.nan, index=cols, columns=cols)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        for effect in cols:
            for cause in cols:
                if effect == cause:
                    continue
                res = grangercausalitytests(returns[[effect, cause]], maxlag=maxlag)
                out.loc[effect, cause] = min(res[l][0]["ssr_ftest"][1] for l in range(1, maxlag + 1))
    return out.astype(float)


# --------------------------------------------------------------- cointegration / pairs
def engle_granger(y: pd.Series, x: pd.Series) -> dict:
    """Engle-Granger two-step. OLS hedge ratio beta, then test the spread for stationarity.
    Returns the coint p-value, beta, and the spread series."""
    pair = pd.concat([y, x], axis=1).dropna()
    yy, xx = pair.iloc[:, 0], pair.iloc[:, 1]
    beta = np.polyfit(xx, yy, 1)[0]                     # slope (hedge ratio)
    spread = yy - beta * xx
    t_stat, p_value, _ = coint(yy, xx)                 # EG cointegration test
    adf_p = adfuller(spread, autolag="AIC")[1]         # ADF on the spread itself
    return {"coint_p": p_value, "beta": beta, "spread_adf_p": adf_p, "spread": spread}


def cointegration_scan(prices: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Run Engle-Granger over candidate pairs, most-cointegrated first."""
    rows = []
    for a, b in pairs:
        r = engle_granger(prices[a], prices[b])
        rows.append({"pair": f"{a}-{b}", "coint_p": r["coint_p"], "beta": r["beta"],
                     "spread_adf_p": r["spread_adf_p"]})
    return pd.DataFrame(rows).sort_values("coint_p").reset_index(drop=True).round(4)


def johansen_summary(prices: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> pd.DataFrame:
    """Johansen trace test. For each rank r, trace statistic vs the 95% critical value.
    The cointegration rank is the first r where trace < crit (fail to reject)."""
    jres = coint_johansen(prices.values, det_order, k_ar_diff)
    n = prices.shape[1]
    return pd.DataFrame({
        "H0_rank<=r": [f"r<={i}" for i in range(n)],
        "trace_stat": jres.lr1.round(3),
        "crit_95%": jres.cvt[:, 1].round(3),
        "cointegrated": jres.lr1 > jres.cvt[:, 1],
    })


def zscore(s: pd.Series, window: int | None = None) -> pd.Series:
    """Standardised spread. Rolling window = a tradeable signal (no look-ahead); None = full-sample."""
    if window:
        return (s - s.rolling(window).mean()) / s.rolling(window).std()
    return (s - s.mean()) / s.std()

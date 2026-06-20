"""Volatility modelling (ARCH/GARCH family): test for ARCH effects, fit conditional-variance
models, build Value-at-Risk, and backtest it.

Returns themselves are ~unforecastable (Parts 2-3), but their *variance* clusters and is highly
predictable. These models forecast that conditional variance — the foundation of risk management.
Inputs are returns in **percent** (×100), as the `arch` library expects for numerical stability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats as sps
from statsmodels.stats.diagnostic import het_arch


def arch_lm(returns: pd.Series, lags: int = 12) -> dict:
    """Engle's ARCH-LM test. H0: no ARCH effect (constant variance).
    Small p -> squared residuals are autocorrelated -> volatility clusters -> use GARCH."""
    r = pd.Series(returns).dropna()
    stat, p, _, _ = het_arch(r - r.mean(), nlags=lags)
    return {"lm_stat": float(stat), "p_value": float(p)}


def fit_garch(returns_pct: pd.Series, p: int = 1, o: int = 0, q: int = 1,
              dist: str = "t", mean: str = "Constant"):
    """Fit a (GJR-)GARCH(p,o,q) on percent returns. `o>0` adds the asymmetry/leverage term;
    `dist='t'` uses Student-t innovations (fat tails). Returns the fitted arch result."""
    am = arch_model(returns_pct.dropna(), mean=mean, vol="GARCH", p=p, o=o, q=q, dist=dist)
    return am.fit(disp="off")


def persistence(res) -> float:
    """alpha + beta (+ gamma/2 for GJR): how slowly shocks to variance decay. ~1 = very persistent."""
    pr = res.params
    a = sum(v for k, v in pr.items() if k.startswith("alpha"))
    b = sum(v for k, v in pr.items() if k.startswith("beta"))
    g = sum(v for k, v in pr.items() if k.startswith("gamma"))
    return float(a + b + g / 2)


def var_series(res, returns_pct: pd.Series, alpha: float = 0.01) -> pd.Series:
    """One-day Value-at-Risk (positive loss, in %) from conditional volatility and the empirical
    alpha-quantile of standardized residuals (so fat tails are respected)."""
    cv = res.conditional_volatility
    mu = float(res.params.get("mu", 0.0))
    std_resid = (returns_pct.reindex(cv.index) - mu) / cv
    q = np.quantile(std_resid.dropna(), alpha)         # left-tail quantile (negative)
    return -(mu + cv * q)                              # positive VaR threshold


def var_backtest(returns_pct: pd.Series, var_pct: pd.Series, alpha: float = 0.01) -> dict:
    """Backtest VaR: count breaches (loss worse than VaR) and run the Kupiec POF LR test.
    H0: breach rate == alpha. A good model breaches ~alpha of the time."""
    r, v = returns_pct.align(var_pct, join="inner")
    viol = r < -v
    n, x = len(r), int(viol.sum())
    pi = min(max(x / n, 1e-9), 1 - 1e-9)
    lr = -2 * ((n - x) * np.log(1 - alpha) + x * np.log(alpha)
               - ((n - x) * np.log(1 - pi) + x * np.log(pi)))
    return {"n": n, "breaches": x, "rate": x / n, "expected": alpha,
            "kupiec_LR": float(lr), "kupiec_p": float(1 - sps.chi2.cdf(lr, 1))}

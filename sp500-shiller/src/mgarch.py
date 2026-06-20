"""Multivariate volatility: Engle's DCC-GARCH for *time-varying* correlations.

Part 4 measured one static correlation matrix; Part 7 made each asset's variance dynamic. DCC
puts them together — a conditional correlation matrix R_t that moves through time. Two steps:

  1. fit a univariate GARCH to each asset -> standardized residuals z (devolatized returns),
  2. let their correlation evolve:  Q_t = (1-a-b)·Q̄ + a·z_{t-1} z_{t-1}' + b·Q_{t-1},
     R_t = normalise(Q_t).   (a, b) are estimated by maximising the DCC log-likelihood.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .volatility import fit_garch


def standardized_residuals(returns_pct: pd.DataFrame, dist: str = "t") -> pd.DataFrame:
    """Step 1: a univariate GARCH(1,1) per column -> standardized residuals z_{i,t}=ε/σ
    (each ~unit variance, but still cross-correlated — that is what DCC models)."""
    z = {}
    for col in returns_pct.columns:
        res = fit_garch(returns_pct[col].dropna(), p=1, q=1, dist=dist)
        z[col] = res.std_resid
    return pd.DataFrame(z).dropna()


def _ewma_avg_corr(z: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """RiskMetrics EWMA average pairwise correlation through time (the simple baseline)."""
    T, N = z.shape
    S = np.cov(z.T)
    out = np.empty(T)
    off = ~np.eye(N, dtype=bool)
    for t in range(T):
        S = lam * S + (1 - lam) * np.outer(z[t], z[t])
        d = np.sqrt(np.diag(S))
        R = S / np.outer(d, d)
        out[t] = R[off].mean()
    return out


def dcc(z_df: pd.DataFrame, pairs: list[tuple[str, str]] | None = None) -> dict:
    """Step 2: estimate DCC(1,1) by MLE and return the conditional-correlation paths."""
    cols = list(z_df.columns)
    z = z_df.to_numpy(float)
    T, N = z.shape
    Qbar = np.corrcoef(z.T)
    off = ~np.eye(N, dtype=bool)

    def neg_loglik(params):
        a, b = params
        if a < 0 or b < 0 or a + b >= 0.999:
            return 1e12
        Q = Qbar.copy()
        ll = 0.0
        for t in range(T):
            d = np.sqrt(np.diag(Q))
            R = Q / np.outer(d, d)
            sign, logdet = np.linalg.slogdet(R)
            zt = z[t]
            ll += -0.5 * (logdet + zt @ np.linalg.solve(R, zt) - zt @ zt)
            Q = (1 - a - b) * Qbar + a * np.outer(zt, zt) + b * Q
        return -ll

    opt = minimize(neg_loglik, [0.02, 0.95], method="L-BFGS-B",
                   bounds=[(1e-4, 0.5), (1e-4, 0.997)])
    a, b = opt.x

    # Replay to extract the correlation paths we want to report.
    Q = Qbar.copy()
    avg = np.empty(T)
    pair_paths = {f"{x}-{y}": np.empty(T) for x, y in (pairs or [])}
    for t in range(T):
        d = np.sqrt(np.diag(Q))
        R = Q / np.outer(d, d)
        avg[t] = R[off].mean()
        for x, y in (pairs or []):
            pair_paths[f"{x}-{y}"][t] = R[cols.index(x), cols.index(y)]
        zt = z[t]
        Q = (1 - a - b) * Qbar + a * np.outer(zt, zt) + b * Q

    idx = z_df.index
    return {
        "a": float(a), "b": float(b), "persistence": float(a + b),
        "avg_corr": pd.Series(avg, index=idx, name="DCC avg corr"),
        "ewma_avg_corr": pd.Series(_ewma_avg_corr(z), index=idx, name="EWMA avg corr"),
        "pair_corr": pd.DataFrame(pair_paths, index=idx) if pairs else None,
        "uncond_avg_corr": float(Qbar[off].mean()),
    }

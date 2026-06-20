"""Missing-data imputation methods + an honest evaluation harness.

You can't measure imputation error on *truly* missing data (you don't know the answer). So we
**mask known values** that mimic the real gap structure, impute, and score the reconstruction —
stratified by gap length, because short and long gaps need different methods.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (registers IterativeImputer)
from sklearn.impute import IterativeImputer, KNNImputer


# --------------------------------------------------------------- gap geometry
def gap_length_per_point(s: pd.Series) -> pd.Series:
    """For each missing point, the length of the consecutive-NaN run it belongs to."""
    isna = s.isna().to_numpy()
    n = len(s); out = np.zeros(n); i = 0
    while i < n:
        if isna[i]:
            j = i
            while j < n and isna[j]:
                j += 1
            out[i:j] = j - i; i = j
        else:
            i += 1
    return pd.Series(out, index=s.index)


def inject_gaps(truth: pd.Series, source_lengths, n_gaps: int, seed: int = 0):
    """Mask *observed* values into artificial gaps whose lengths are drawn from `source_lengths`
    (the real gap-run distribution). Returns (mask, gap_len) arrays for scoring."""
    obs = truth.notna().to_numpy()
    n = len(truth); rng = np.random.default_rng(seed)
    mask = np.zeros(n, bool); glen = np.zeros(n)
    for L in rng.choice(np.asarray(source_lengths), size=n_gaps, replace=True):
        L = int(L)
        for _ in range(30):                      # try to find an all-observed, unused stretch
            st = int(rng.integers(0, n - L))
            sl = slice(st, st + L)
            if obs[sl].all() and not mask[sl].any():
                mask[sl] = True; glen[sl] = L
                break
    return mask, glen


# --------------------------------------------------------------- the methods
def _time_feats(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Cyclical hour/month features so KNN/MICE can use the daily & seasonal pattern."""
    return pd.DataFrame({"hsin": np.sin(2 * np.pi * idx.hour / 24), "hcos": np.cos(2 * np.pi * idx.hour / 24),
                         "msin": np.sin(2 * np.pi * idx.month / 12), "mcos": np.cos(2 * np.pi * idx.month / 12)},
                        index=idx)


def _design(gappy: pd.Series, weather: pd.DataFrame) -> np.ndarray:
    return pd.concat([gappy.rename("pm25"), weather, _time_feats(gappy.index)], axis=1).to_numpy()


def impute_all(gappy: pd.Series, weather: pd.DataFrame, climatology: pd.Series) -> dict[str, pd.Series]:
    """Five named imputers, each returning a fully-filled pm25 series.

    ffill        - carry the last value forward (a step function).
    interp       - draw a straight line between the bracketing values (great for short gaps).
    climatology  - fill with the typical value for that month & hour (the seasonal average).
    KNN          - average the K most-similar hours (by weather + time-of-cycle).
    MICE         - iteratively regress pm25 on weather + time (multivariate).
    """
    out = {"ffill": gappy.ffill().bfill(),
           "interp": gappy.interpolate("time").bfill().ffill()}
    fill = pd.Series([climatology.get((m, h)) for m, h in zip(gappy.index.month, gappy.index.hour)], index=gappy.index)
    out["climatology"] = gappy.fillna(fill)
    X = _design(gappy, weather)
    out["KNN"] = pd.Series(KNNImputer(n_neighbors=8).fit_transform(X)[:, 0], index=gappy.index)
    out["MICE"] = pd.Series(IterativeImputer(random_state=0, max_iter=10).fit_transform(X)[:, 0], index=gappy.index)
    return out


def hybrid(gappy: pd.Series, weather: pd.DataFrame, short_max: int = 48) -> pd.Series:
    """The recommendation: interpolate gaps <= `short_max` hours, MICE for the long ones."""
    interp = gappy.interpolate("time").bfill().ffill()
    mice = pd.Series(IterativeImputer(random_state=0, max_iter=10).fit_transform(_design(gappy, weather))[:, 0],
                     index=gappy.index)
    use_mice = gappy.isna() & (gap_length_per_point(gappy) > short_max)
    out = interp.copy()
    out[use_mice.to_numpy()] = mice[use_mice.to_numpy()]
    return out

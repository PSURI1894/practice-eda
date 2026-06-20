"""Shared modelling helpers for the wine-quality practice (Parts 4+).

Quality is an **ordinal, imbalanced** target. Three decisions recur, so they live here once:
  * **dedup before splitting** — a duplicated wine in both train and test leaks the answer;
  * **stratify** the split / folds — keep the rare grades (quality 3, 4, 8, 9) present everywhere;
  * **score with quadratic weighted kappa (QWK)** — an ordinal-aware metric that penalises a
    prediction by *how far off* it is, not merely whether it is wrong.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                             mean_absolute_error)
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import data

LABELS = np.arange(3, 10)   # the full ordinal range 3..9 (so metrics see every grade)


def prep(df: pd.DataFrame, engineered: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Feature matrix (numeric [+ engineered] + an `is_red` flag) and the ordinal target."""
    d = data.engineer(df) if engineered else df
    cols = data.NUMERIC + (data.ENGINEERED if engineered else [])
    X = d[cols].copy()
    X["is_red"] = (d["wine_type"] == "red").astype(int)
    return X, d["quality"].astype(int)


def split(df: pd.DataFrame, test_size: float = 0.25, seed: int = 0,
          engineered: bool = True, dedup: bool = True):
    """Dedup → stratified train/test split. Returns X_train, X_test, y_train, y_test."""
    d = data.dedup(df) if dedup else df.copy()
    X, y = prep(d, engineered=engineered)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def qwk(y_true, y_pred) -> float:
    """Quadratic Weighted Kappa: chance-corrected agreement weighted by the SQUARED distance
    between predicted and true grade. Regressor outputs are rounded and clipped to 3..9."""
    yt = np.asarray(y_true).astype(int)
    yp = np.clip(np.round(np.asarray(y_pred)), 3, 9).astype(int)
    return cohen_kappa_score(yt, yp, weights="quadratic", labels=LABELS)


def report(y_true, y_pred, name: str = "") -> pd.Series:
    """Four complementary scores — accuracy (lies under imbalance), macro-F1 (rewards rare-class
    skill), MAE (average ordinal distance), QWK (the headline ordinal metric)."""
    yt = np.asarray(y_true).astype(int)
    yp = np.clip(np.round(np.asarray(y_pred)), 3, 9).astype(int)
    return pd.Series({"accuracy": accuracy_score(yt, yp),
                      "macro_F1": f1_score(yt, yp, average="macro"),
                      "MAE": mean_absolute_error(yt, yp),
                      "QWK": qwk(yt, yp)}, name=name).round(3)


def cv_qwk(model, X: pd.DataFrame, y: pd.Series, n: int = 5, seed: int = 0) -> np.ndarray:
    """Stratified k-fold QWK (clones the estimator per fold; handles regressors via rounding)."""
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
    scores = []
    for tr, te in skf.split(X, y):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        scores.append(qwk(y.iloc[te], m.predict(X.iloc[te])))
    return np.array(scores)


class OptimizedRounder:
    """Learn the cut points that map a *continuous* ordinal prediction to grades 3..9 so as to
    MAXIMISE QWK on the training fold — instead of naive round-at-.5. Under class imbalance the best
    cut points are asymmetric (e.g. the 8/9 boundary is pushed high because grade 9 is so rare).
    A standard trick from ordinal-regression / Kaggle practice."""

    def __init__(self, labels=LABELS):
        self.labels = np.asarray(labels)
        self.cuts_ = (self.labels[:-1] + 0.5).astype(float)   # default = naive rounding

    def _apply(self, p, cuts):
        return self.labels[np.digitize(np.asarray(p), np.sort(cuts))]

    def fit(self, p, y):
        loss = lambda c: -qwk(y, self._apply(p, c))
        self.cuts_ = minimize(loss, self.cuts_, method="Nelder-Mead",
                              options={"maxiter": 2000, "xatol": 1e-3, "fatol": 1e-4}).x
        return self

    def predict(self, p):
        return self._apply(p, self.cuts_)

"""Reusable Advanced-EDA helpers: distribution diagnostics, the four-view battery,
normality tests, outlier flags, correlation/MI, VIF, and categorical association.

Notebooks stay thin by calling these; the statistics live here in one tested place.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .config import FIGS


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["axes.titleweight"] = "bold"


def savefig(fig, name: str) -> None:
    """Persist a standalone PNG to reports/figures (notebooks also embed it inline)."""
    fig.savefig(FIGS / name, bbox_inches="tight", dpi=130)


# --------------------------------------------------------------- distribution summary
def moments(s: pd.Series) -> pd.Series:
    """The four moments + practical reads. Kurtosis is *excess* (normal == 0)."""
    s = s.dropna()
    return pd.Series({
        "n": int(s.size),
        "mean": s.mean(),
        "std": s.std(),
        "skew": s.skew(),                 # >0 right tail, <0 left tail
        "excess_kurtosis": s.kurt(),      # pandas .kurt() is already excess (Fisher)
        "min": s.min(),
        "median": s.median(),
        "max": s.max(),
    })


def normality_battery(s: pd.Series) -> pd.DataFrame:
    """Run the standard normality tests. Small p -> reject 'data is normal'.
    Shapiro is only meaningful up to n~5000, so it is sampled when larger."""
    s = s.dropna().to_numpy()
    rows = []

    jb_stat, jb_p = stats.jarque_bera(s)
    rows.append(("Jarque-Bera", jb_stat, jb_p))

    k2_stat, k2_p = stats.normaltest(s)            # D'Agostino-Pearson
    rows.append(("D'Agostino K^2", k2_stat, k2_p))

    s_shap = s if s.size <= 5000 else np.random.default_rng(0).choice(s, 5000, replace=False)
    sh_stat, sh_p = stats.shapiro(s_shap)
    label = "Shapiro-Wilk" + ("" if s.size <= 5000 else " (n=5000 sample)")
    rows.append((label, sh_stat, sh_p))

    out = pd.DataFrame(rows, columns=["test", "statistic", "p_value"])
    out["normal_at_5pct"] = out["p_value"] > 0.05
    return out


def ecdf(s: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(s.dropna().to_numpy())
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def four_view(s: pd.Series, name: str, fname: str | None = None):
    """The 4-panel battery for one numeric variable:
    histogram+KDE (shape) | box (spread/outliers) | ECDF (percentiles) | Q-Q (normality)."""
    s = s.dropna()
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    sns.histplot(s, kde=True, ax=ax[0, 0], color="steelblue")
    ax[0, 0].set_title(f"{name}: histogram + KDE")

    sns.boxplot(x=s, ax=ax[0, 1], color="lightcoral")
    ax[0, 1].set_title(f"{name}: box (IQR & outliers)")

    x, y = ecdf(s)
    ax[1, 0].plot(x, y, lw=1.6, color="seagreen")
    ax[1, 0].set_title(f"{name}: ECDF")
    ax[1, 0].set_ylabel("cumulative proportion")

    stats.probplot(s, dist="norm", plot=ax[1, 1])
    ax[1, 1].set_title(f"{name}: Q-Q vs normal")
    ax[1, 1].get_lines()[0].set_markersize(3)

    m = moments(s)
    fig.suptitle(
        f"{name}  |  skew={m['skew']:.2f}  excess_kurt={m['excess_kurtosis']:.2f}  n={int(m['n'])}",
        y=1.02, fontsize=13,
    )
    fig.tight_layout()
    if fname:
        savefig(fig, fname)
    return fig


# --------------------------------------------------------------- outliers
def outlier_flags(s: pd.Series) -> pd.DataFrame:
    """Flag outliers two robust ways: Tukey 1.5*IQR fences, and modified z (MAD-based,
    |z|>3.5). MAD resists the very outliers that inflate a classic z-score."""
    s = s.dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    iqr_out = (s < lo) | (s > hi)

    med = s.median()
    mad = np.median(np.abs(s - med))
    mod_z = 0.6745 * (s - med) / mad if mad else pd.Series(0.0, index=s.index)
    mad_out = mod_z.abs() > 3.5

    return pd.DataFrame({
        "method": ["IQR (1.5x)", "Modified z (MAD>3.5)"],
        "n_flagged": [int(iqr_out.sum()), int(mad_out.sum())],
        "pct": [100 * iqr_out.mean(), 100 * mad_out.mean()],
    })


# --------------------------------------------------------------- correlation / association
def correlation_trio(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Pearson (linear), Spearman (monotonic rank), Kendall (concordance) side by side.
    Big gaps between them flag non-linear or outlier-driven relationships."""
    pair = df[[a, b]].dropna()
    rows = [(m, *getattr(stats, fn)(pair[a], pair[b]))
            for m, fn in [("pearson", "pearsonr"), ("spearman", "spearmanr"), ("kendall", "kendalltau")]]
    return pd.DataFrame(rows, columns=["method", "coef", "p_value"])


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramér's V: association between two categoricals in [0, 1]."""
    confusion = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion)[0]
    n = confusion.to_numpy().sum()
    phi2 = chi2 / n
    r, k = confusion.shape
    phi2corr = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else np.nan


def cramers_v_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    m = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = cramers_v(df[a], df[b])
            m.loc[a, b] = m.loc[b, a] = v
    return m


def vif_table(df_numeric: pd.DataFrame) -> pd.DataFrame:
    """Variance Inflation Factor per column. VIF>5 = notable, >10 = serious collinearity."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X = df_numeric.dropna().assign(_const=1.0)
    cols = [c for c in X.columns if c != "_const"]
    vifs = [variance_inflation_factor(X.values, X.columns.get_loc(c)) for c in cols]
    return pd.DataFrame({"feature": cols, "VIF": vifs}).sort_values("VIF", ascending=False)

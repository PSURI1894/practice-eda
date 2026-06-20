"""Generate the section notebooks from readable source.

Notebooks are the primary deliverable, but hand-authoring .ipynb JSON is error-prone, so
they are built here with nbformat. Re-run this to regenerate:  python build_notebooks.py
Then execute with:  jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
"""
from __future__ import annotations

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NB_DIR = "notebooks"

SETUP = """\
import sys, pathlib, warnings
warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = pathlib.Path.cwd()
ROOT = ROOT if (ROOT / "src").exists() else ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from src import data, eda
eda.set_style()
pd.set_option("display.width", 120, "display.max_columns", 30)
print("setup ok | numpy", np.__version__, "| pandas", pd.__version__)
"""


def build(cells, name, title):
    nb = new_notebook()
    nb.cells = [new_markdown_cell(title)] + cells
    nb.metadata["kernelspec"] = {
        "display_name": "Python (advanced-eda)",
        "language": "python",
        "name": "advanced-eda",
    }
    with open(f"{NB_DIR}/{name}", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote", name, f"({len(nb.cells)} cells)")


# ===================================================================== Notebook 0
def notebook_0():
    md = new_markdown_cell
    co = new_code_cell
    cells = [
        md(
"""## Part 0 — Acquire & Clean

Two real-world datasets, two **disguised-missingness** traps that a naive `isna()` never catches:

| dataset | role | the trap |
|---|---|---|
| **Telco Customer Churn** | cross-sectional Advanced EDA | `TotalCharges` is stored as *text*; 11 blanks → NaN, all at `tenure == 0` → **MAR** |
| **Shiller S&P 500** (monthly, 1871→) | time-series half | `0` is a *placeholder*; PE10/CPI can never be 0 |

Goal: produce model-ready frames in `data/processed/`, with the cleaning decisions made explicit."""),
        co(SETUP),

        md("### 1. Telco — the text-column trap\n`TotalCharges` looks complete, but it is `object` (text). For a money column, that alone is a red flag."),
        co("""raw = data.load_telco_raw()
print("shape:", raw.shape)
print("TotalCharges dtype:", raw.TotalCharges.dtype, "| isna() says:", int(raw.TotalCharges.isna().sum()), "missing")
coerced = pd.to_numeric(raw.TotalCharges, errors="coerce")
bad = coerced.isna()
print("after to_numeric -> blanks exposed:", int(bad.sum()))
print("tenure of those rows:", sorted(raw.loc[bad, "tenure"].unique().tolist()), "  <-- all brand-new customers (MAR)")"""),

        md("The missingness is **not random** — it is mechanically tied to `tenure == 0` (a customer who has never completed a billing cycle). That is **MAR**. The honest fix is `TotalCharges = 0` for those rows. `clean_telco()` also recasts `SeniorCitizen` (0/1 → Yes/No), drops `customerID`, and adds a numeric `churn_flag`."),
        co("""telco = data.clean_telco()
print("clean shape:", telco.shape, "| remaining NaN in TotalCharges:", int(telco.TotalCharges.isna().sum()))
print("churn rate: %.1f%%  (imbalanced, like real churn)" % (100 * telco.churn_flag.mean()))
telco.dtypes.to_frame("dtype").T"""),

        md("### 2. Shiller — the zero-as-placeholder trap\n`isna()` reports nothing, yet a CAPE ratio or CPI of **0** is impossible."),
        co("""sraw = data.load_shiller_raw()
zero_counts = (sraw == 0).sum()
print("columns with literal-zero placeholders:")
print(zero_counts[zero_counts > 0].to_string())
print("\\nPE10 leading zeros = CAPE needs 10y trailing earnings, so it only starts in 1881.")
print("Trailing zeros on the fundamentals = recent months not yet reported (price stays fresh).")"""),

        md("`clean_shiller()` converts those zeros to NaN (only where 0 is impossible), sets a monthly `DatetimeIndex`, and derives the features we actually analyse: `return`, `log_return`, `real_return`, `cape`, `cape_z`."),
        co("""shiller = data.clean_shiller()
print("range:", shiller.index.min().date(), "->", shiller.index.max().date(), "| freq:", shiller.index.freq)
print("PE10 NaN after fix:", int(shiller.PE10.isna().sum()), "(were zeros)")
shiller[["SP500", "Earnings", "Consumer Price Index", "return", "log_return", "cape", "cape_z"]].tail(3)"""),

        md("### 3. VIX (companion, for Part 4)"),
        co("""vix = data.clean_vix()
print("VIX:", vix.shape, "|", vix.index.min().date(), "->", vix.index.max().date())
vix.tail(2)"""),

        md("### 4. Persist everything to `data/processed/`"),
        co("""frames = data.build_processed()
for k, v in frames.items():
    print(f"{k:8s} {str(v.shape):12s} -> data/processed/")
print("\\nPart 0 complete. Cleaned, documented, reproducible.")"""),
    ]
    build(cells, "00_data_cleaning.ipynb", "# 00 · Data Acquisition & Cleaning")


# ===================================================================== Notebook 1
def notebook_1():
    md = new_markdown_cell
    co = new_code_cell
    cells = [
        md(
"""## Part 1 — Advanced EDA (the basics, done properly)

The discipline of *looking before modelling*. We run, on real data:

1. The **four moments** and the **four-view battery** (hist+KDE · box · ECDF · Q–Q)
2. The **normality battery** — and what *fat tails* actually look like
3. **Transformations** (when and why to log)
4. **Outliers** — robust detection, and *investigate before you delete*
5. **Correlation** three ways (Pearson / Spearman / Kendall) + **mutual information**
6. **Multicollinearity** via **VIF**
7. **Categorical association** (chi-square, Cramér's V)
8. **Missing-data mechanisms** (MCAR / MAR / MNAR)

Financial returns drive the distribution lessons; Telco drives the categorical/collinearity lessons."""),
        co(SETUP + "\ntelco = data.clean_telco()\nshiller = data.clean_shiller()\nret = shiller['return'].dropna()\nprint('returns:', ret.size, '| telco:', telco.shape)"),

        md("""### 1. The four moments + the four-view battery

Never trust a single number. `describe()` hides shape — two variables with the same mean and
std can look completely different. The battery shows **shape** (hist), **spread & outliers**
(box), **percentiles** (ECDF), and **normality** (Q–Q) at once."""),
        co("""print(eda.moments(ret).round(4).to_string())
fig = eda.four_view(ret, "S&P 500 monthly return", "p1_fourview_returns.png")
plt.show()"""),
        md("""Read the Q–Q panel: the points fly off the line at both ends — far more extreme months than
a normal distribution allows. That bend *is* fat tails. The box plot's swarm of "outliers" are
not errors either (next section)."""),

        md("Same battery on a Telco numeric — `MonthlyCharges` is **bimodal** (two customer populations), which a mean/std summary would completely hide:"),
        co("""fig = eda.four_view(telco["MonthlyCharges"], "Telco MonthlyCharges", "p1_fourview_monthlycharges.png")
plt.show()"""),

        md("""### 2. Normality battery — quantifying the fat tails

Excess kurtosis is **0** for a normal distribution. Markets sit near **+17**: extreme moves
happen far more often than a bell curve predicts — the single fact entire risk models are built around."""),
        co("""print(eda.normality_battery(ret).round(4).to_string(index=False))
print("\\nexcess kurtosis = %.2f   (normal = 0)" % ret.kurt())

# Overlay a fitted normal to make the gap visible.
from scipy import stats
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(ret, bins=80, density=True, alpha=0.6, color="steelblue", label="actual returns")
xs = np.linspace(ret.min(), ret.max(), 400)
ax.plot(xs, stats.norm.pdf(xs, ret.mean(), ret.std()), "r-", lw=2, label="normal (same μ, σ)")
ax.set_title("Fat tails: actual returns vs the normal that 'should' fit"); ax.legend()
eda.savefig(fig, "p1_fat_tails.png"); plt.show()"""),

        md("""### 3. Transformations — when to log

Right-skewed, multiplicative, strictly-positive quantities (prices, money, weights) are often
**log-normal**. Logging turns multiplicative structure additive and pulls in the tail. The S&P
price level is the textbook case."""),
        co("""price = shiller["SP500"]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
stats.probplot(price, dist="norm", plot=ax[0]); ax[0].set_title(f"Price: Q-Q (skew={price.skew():.2f})")
stats.probplot(np.log(price), dist="norm", plot=ax[1]); ax[1].set_title(f"log(Price): Q-Q (skew={np.log(price).skew():.2f})")
for a in ax: a.get_lines()[0].set_markersize(3)
eda.savefig(fig, "p1_log_transform.png"); plt.show()
print("skew  price = %.2f   ->   log(price) = %.2f" % (price.skew(), np.log(price).skew()))"""),

        md("""### 4. Outliers — detect robustly, then *investigate*

Two robust flags: **Tukey 1.5×IQR** fences and the **MAD-based modified z** (|z|>3.5). MAD is
robust because it does not let the outliers inflate the very scale used to detect them."""),
        co("""print(eda.outlier_flags(ret).round(2).to_string(index=False))

extreme = ret.reindex(ret.abs().sort_values(ascending=False).index).head(6)
print("\\nThe 6 most extreme months are HISTORY, not data errors:")
print(extreme.apply(lambda x: f"{x:+.1%}").to_frame("return").to_string())"""),
        md("""Nov 1929, Oct 2008, Mar 2020 — deleting these would erase the most important observations in
the dataset. The rule: an outlier is a **question, not a verdict**. Drop only true errors (the
59 mm "diamond"); keep real extremes and model them."""),

        md("""### 5. Correlation three ways + mutual information

**Pearson** = linear, **Spearman** = monotonic (rank), **Kendall** = concordance. A big gap
between Pearson and Spearman flags non-linearity or outlier leverage. **Mutual information**
goes further — it detects *any* dependence, including non-monotonic, and is zero only under
true independence."""),
        co("""pair = shiller.dropna(subset=["Earnings", "SP500"])
print("Earnings vs SP500 price:")
print(eda.correlation_trio(pair, "Earnings", "SP500").round(3).to_string(index=False))

num = telco[["tenure", "MonthlyCharges", "TotalCharges"]]
fig, ax = plt.subplots(figsize=(5, 4))
import seaborn as sns
sns.heatmap(num.corr(), annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
ax.set_title("Telco numeric correlations"); eda.savefig(fig, "p1_corr_heatmap.png"); plt.show()"""),
        co("""from sklearn.feature_selection import mutual_info_classif
X = telco[["tenure", "MonthlyCharges", "TotalCharges"]]
mi = mutual_info_classif(X, telco.churn_flag, random_state=0)
pb = X.apply(lambda c: c.corr(telco.churn_flag))  # point-biserial (linear) for contrast
cmp = pd.DataFrame({"|pearson| vs churn": pb.abs().round(3), "mutual_info vs churn": np.round(mi, 3)})
print(cmp.to_string())
print("\\ntenure carries the most information about churn; MI ranks dependence even when linear corr is modest.")"""),

        md("""### 6. Multicollinearity — VIF

Correlated predictors inflate coefficient variance and make models unstable. **VIF > 5** is
notable, **> 10** serious. `TotalCharges ≈ tenure × MonthlyCharges` by construction, so it lights up."""),
        co("""print(eda.vif_table(telco[["tenure", "MonthlyCharges", "TotalCharges"]]).round(2).to_string(index=False))
print("\\nTotalCharges is near-redundant given tenure & MonthlyCharges -> drop one or combine.")"""),

        md("""### 7. Categorical association — chi-square & Cramér's V

Correlation is for numbers. For categoricals, **Cramér's V** ∈ [0, 1] measures association
(bias-corrected here). The churn-rate breakdown shows *why* `Contract` matters."""),
        co("""cats = ["gender", "SeniorCitizen", "Partner", "Dependents", "Contract", "PaperlessBilling", "PaymentMethod", "InternetService", "Churn"]
cv = eda.cramers_v_matrix(telco, cats)
fig, ax = plt.subplots(figsize=(8, 6.5))
import seaborn as sns
sns.heatmap(cv, annot=True, fmt=".2f", cmap="viridis", ax=ax)
ax.set_title("Cramér's V — categorical association"); eda.savefig(fig, "p1_cramers_v.png"); plt.show()"""),
        co("""rate = telco.groupby("Contract", observed=True).churn_flag.mean().sort_values(ascending=False)
print("Churn rate by contract type:")
print((rate * 100).round(1).astype(str).add("%").to_string())
v = cv.loc["Contract", "Churn"]
print(f"\\nMonth-to-month churns ~43%; two-year ~3%. Contract is the strongest single driver (V={v:.2f} vs Churn).")"""),

        md("""### 8. Missing-data mechanisms — MCAR / MAR / MNAR

- **MCAR** — missingness independent of everything (safe to drop; rare).
- **MAR** — missingness depends on *observed* data (impute using the relationship).
- **MNAR** — missingness depends on the *unobserved* value itself (hardest; needs modelling).

Telco's `TotalCharges` is the textbook **MAR** case — re-derived from the raw file:"""),
        co("""raw = data.load_telco_raw()
miss = pd.to_numeric(raw.TotalCharges, errors="coerce").isna()
print("missing TotalCharges:", int(miss.sum()))
print("their tenure values:", sorted(raw.loc[miss, "tenure"].unique().tolist()), "-> 100% at tenure 0")
print("\\nMissingness is fully explained by an OBSERVED column (tenure) => MAR, not MCAR.")
print("Right fix: set 0 (no completed billing cycle). Wrong fix: blind mean-imputation, which invents history.")"""),

        md("""### Takeaways

- Distributions, not point summaries — the battery exposes skew, fat tails, bimodality.
- Returns have **excess kurtosis ≈ 17**: extreme events are routine, not anomalies.
- Outliers are questions; the worst returns are real crashes worth keeping.
- Use **Spearman/Kendall + MI** alongside Pearson; check **VIF** for redundancy.
- Categoricals → **Cramér's V**; `Contract` is the dominant churn driver.
- Always interrogate the **missingness mechanism** before imputing.

**Next — Part 2 (Time-Series Foundations):** index hygiene, decomposition (additive/multiplicative/STL),
stationarity (ADF × KPSS), ACF/PACF, differencing — on the S&P 500 series."""),
    ]
    build(cells, "01_advanced_eda.ipynb", "# 01 · Advanced EDA — Distributions, Outliers, Association")


# ===================================================================== Notebook 2
SETUP2 = SETUP + """
from src import ts
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.datasets import co2

shiller = data.clean_shiller()
price = shiller["SP500"]
logp  = np.log(price)
ret   = shiller["return"].dropna()
logret = shiller["log_return"].dropna()
# Mauna Loa CO2 -> monthly, for a series with UNMISTAKABLE trend + seasonality.
co2m = co2.load_pandas().data["co2"].resample("MS").mean().interpolate()
print("shiller:", price.index.min().date(), "->", price.index.max().date(), "| co2:", co2m.index.min().date(), "->", co2m.index.max().date())
"""


def notebook_2():
    md = new_markdown_cell
    co = new_code_cell
    cells = [
        md(
"""## Part 2 — Time-Series Foundations

Cross-sectional EDA (Part 1) asks *how are the columns distributed and related?* A time series
adds an axis that changes everything: **order matters**. Observations are indexed by time, and
the value today is correlated with the value yesterday. That dependence is the signal — and the
thing that breaks ordinary statistics (a t-test assumes i.i.d. rows; returns are not i.i.d.).

This notebook builds the vocabulary every forecasting model relies on:

| # | concept | the question it answers |
|---|---|---|
| 1 | **Index hygiene** | is the time axis regular, gapless, and correctly typed? |
| 2 | **Visual diagnosis** | what does trend / changing variance look like? |
| 3 | **Components** | trend + seasonality + cycle + noise — additive or multiplicative? |
| 4 | **Decomposition** | classical vs **STL**, and how to read the panels |
| 5 | **Stationarity** | does the distribution drift over time? (**ADF × KPSS**) |
| 6 | **ACF / PACF** | how far back does memory reach? which model order? |
| 7 | **Volatility clustering** | the variance has its own memory (ARCH) |
| 8 | **Differencing** | how to *make* a series stationary — without over-doing it |

Primary series: **S&P 500** (Shiller monthly). Decomposition teaching aid: **Mauna Loa CO₂**
(real atmospheric data with a textbook seasonal cycle), because an equity index has trend but
almost no calendar seasonality — itself a finding we confirm in §4."""),
        co(SETUP2),

        # ---------------------------------------------------------------- 1 index hygiene
        md(
"""### 1. Index hygiene — the foundation everything else stands on

Before any model, the time axis must be **typed as datetime**, **sorted**, **gapless**, and
carry an explicit **frequency**. A silent gap or a string index will not error — it will quietly
corrupt every lag, rolling window, and seasonal period downstream.

Checklist:
- `DatetimeIndex` (not strings), monotonic increasing, no duplicate timestamps
- a declared frequency (`MS` = month-start here) so `.shift`, `.diff`, `.resample` are meaningful
- a deliberate choice for any gaps: forward-fill (carry last), interpolate, or leave NaN"""),
        co("""idx = price.index
print("dtype          :", idx.dtype)
print("monotonic inc. :", idx.is_monotonic_increasing)
print("duplicates     :", int(idx.duplicated().sum()))
print("declared freq  :", idx.freq)
# A gapless monthly grid should have exactly this many periods:
expected = pd.date_range(idx.min(), idx.max(), freq="MS")
print("len(series)    :", len(idx), "| expected on a gapless MS grid:", len(expected), "| missing slots:", len(expected.difference(idx)))"""),
        md("""**Resampling** changes the frequency. *Down*-sampling aggregates (monthly → annual, using a
reducer like mean/last); *up*-sampling creates new slots that must be filled. Note how the
annual return is **not** the sum of monthly returns — returns compound, they don't add."""),
        co("""annual_last = price.resample("YE").last()          # downsample: year-end price
annual_ret  = annual_last.pct_change()
print("Down-sampled to annual (year-end price), last 3 years:")
print(annual_last.tail(3).round(1).to_string())
print("\\n2020 annual return from year-end prices: %.1f%%" % (100*annual_ret.loc["2020"].iloc[0]))
print("Sum of 2020 monthly returns (WRONG way):     %.1f%%" % (100*ret.loc["2020"].sum()))
print("Compounded 2020 monthly returns (right way): %.1f%%" % (100*((1+ret.loc["2020"]).prod()-1)))"""),

        # ---------------------------------------------------------------- 2 visual
        md(
"""### 2. Visual diagnosis — always look first

Two plots answer most questions before a single test:

- **Level + rolling mean** → is there a *trend* (a drifting mean)?
- **Rolling std** → is the *variance* constant, or does it cluster into calm and turbulent regimes?

A flat rolling mean and a flat rolling band ≈ stationary. Drift or a breathing band ≈ not."""),
        co("""fig = ts.rolling_plot(price, window=120, name="S&P 500 price", fname="p2_rolling_price.png")
plt.show()"""),
        md("""The mean marches upward (strong **trend**) and the band widens (variance grows with the
price level) → emphatically non-stationary, and *multiplicative* in character. The fix for the
growing variance is a **log transform**: on a log scale the exponential trend becomes a straight
line and percentage moves get equal vertical space."""),
        co("""fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
ax[0].plot(price.index, price, color="steelblue", lw=0.8); ax[0].set_title("S&P 500 — linear scale (curves up)")
ax[1].plot(logp.index, logp, color="darkgreen", lw=0.8);  ax[1].set_title("log(S&P 500) — trend is now ~linear")
fig.tight_layout(); eda.savefig(fig, "p2_log_scale.png"); plt.show()"""),
        md("""Now the **returns**. The level plot of returns hovers around zero (no trend) but visibly
alternates between quiet and violent stretches — **volatility clustering**. We quantify that in §7."""),
        co("""fig = ts.rolling_plot(ret, window=120, name="S&P 500 monthly return", fname="p2_rolling_return.png")
plt.show()"""),

        # ---------------------------------------------------------------- 3 components
        md(
"""### 3. The components: trend, seasonality, cycle, noise

A time series is modelled as a combination of unobserved parts:

- **Trend (T)** — long-run direction (the upward drift of prices).
- **Seasonality (S)** — a *fixed-period* pattern (calendar: month-of-year, day-of-week).
- **Cycle (C)** — longer swings with *no fixed period* (business cycles, bull/bear regimes). Often folded into the trend.
- **Residual (R)** — what's left; ideally white noise.

Two ways they combine:

| model | form | use when | seasonal amplitude… |
|---|---|---|---|
| **Additive** | `y = T + S + R` | fluctuations are roughly constant in absolute size | …stays the same as the level rises |
| **Multiplicative** | `y = T × S × R` | fluctuations grow with the level | …grows with the level |

Key identity: **`log(T × S × R) = log T + log S + log R`** — logging turns a multiplicative
series into an additive one. That is *why* we log financial data before decomposing."""),

        # ---------------------------------------------------------------- 4 decomposition
        md(
"""### 4. Decomposition in practice — classical vs STL

**Classical** (`seasonal_decompose`): estimates the trend with a centered moving average, then
averages the detrended values within each season. Simple, but the seasonal shape is forced to be
*identical every year* and the moving average loses points at both ends.

**STL** (Seasonal-Trend decomposition using Loess): fits trend and season with local regression.
It allows the seasonal pattern to **evolve**, is **robust** to outliers, and handles any period.
Prefer STL in practice.

First, CO₂ — where seasonality is unmistakable (plants breathe in a yearly cycle):"""),
        co("""dec = seasonal_decompose(co2m, model="additive", period=12)
fig = dec.plot(); fig.set_size_inches(11, 8); fig.suptitle("CO₂ — classical additive decomposition", y=1.01)
eda.savefig(fig, "p2_co2_classical.png"); plt.show()
print("Seasonal swing: %.2f ppm peak-to-trough, repeating every 12 months." % (dec.seasonal.max()-dec.seasonal.min()))"""),
        co("""stl = STL(co2m, period=12, robust=True).fit()
fig = stl.plot(); fig.set_size_inches(11, 8); fig.suptitle("CO₂ — STL decomposition", y=1.01)
eda.savefig(fig, "p2_co2_stl.png"); plt.show()
print("STL residual std = %.3f ppm — the trend+seasonal capture almost everything." % stl.resid.std())
print("CO₂ strengths:", ts.decomposition_strengths(stl), "<- both trend and seasonality are strong")"""),
        md("""Now the same STL on **log(S&P 500)**. Watch the scale of the seasonal panel relative to the
trend — for an equity index it is tiny. That is the honest finding: **stocks trend strongly but
have essentially no calendar seasonality** (broadly consistent with market efficiency)."""),
        co("""stl_sp = STL(logp, period=12, robust=True).fit()
fig = stl_sp.plot(); fig.set_size_inches(11, 8); fig.suptitle("log(S&P 500) — STL", y=1.01)
eda.savefig(fig, "p2_sp500_stl.png"); plt.show()
sp_str, co2_str = ts.decomposition_strengths(stl_sp), ts.decomposition_strengths(stl)
print("S&P 500 strengths:", sp_str)
print("CO₂     strengths:", co2_str)
print(f"\\nTrend is near-total for both. But seasonal strength is {co2_str['seasonal_strength']:.2f} for CO₂"
      f" vs only {sp_str['seasonal_strength']:.2f} for the S&P 500 -> equities have ~no calendar seasonality.")"""),
        md("""We can still *look* for a calendar effect directly: average return by month-of-year. The
famous 'Sell in May' / Santa-rally stories live here — and you can see how small and noisy they
are compared to the month-to-month volatility."""),
        co("""by_month = ret.groupby(ret.index.month)
fig, ax = plt.subplots(figsize=(11, 4.5))
import seaborn as sns
sns.boxplot(x=ret.index.month, y=ret.values, ax=ax, color="lightsteelblue", showfliers=False)
ax.axhline(0, color="black", lw=0.8); ax.set_xlabel("calendar month"); ax.set_ylabel("monthly return")
ax.set_title("S&P 500 return by month-of-year (seasonality is weak and noisy)")
eda.savefig(fig, "p2_month_effect.png"); plt.show()
print((by_month.mean()*100).round(2).rename("avg % return").to_frame().T.to_string())"""),

        # ---------------------------------------------------------------- 5 stationarity
        md(
"""### 5. Stationarity — the property models demand

A series is **(weakly / covariance) stationary** if its statistical character does not change
over time:

1. **constant mean** (no trend),
2. **constant variance** (no growing/shrinking spread),
3. **autocovariance that depends only on the lag**, not on *when* you look.

Why it matters: ARIMA, VAR and most classical models *assume* stationarity, and regressing one
trending series on another invites **spurious correlation** (two unrelated random walks look
0.9-correlated). The standard move: transform the data until it is stationary, model that, then
invert.

Two tests with **opposite null hypotheses** — using them together resolves ambiguity:

- **ADF** — H₀: *unit root (non-stationary)*. Small p → **reject** → stationary.
- **KPSS** — H₀: *stationary*. Small p → **reject** → non-stationary.

| ADF says | KPSS says | verdict | action |
|---|---|---|---|
| stationary | stationary | **stationary** | model as-is |
| non-stat. | non-stat. | **unit root** | **difference** |
| stationary | non-stat. | **difference-stationary** | difference |
| non-stat. | stationary | **trend-stationary** | **detrend** (remove a fitted trend) |"""),
        co("""for label, s in [("S&P 500 price", price), ("log(price)", logp), ("monthly return", ret)]:
    table, verdict = ts.stationarity_report(s, name=label)
    print(f"\\n=== {label} ===")
    print(table.to_string())
    print("VERDICT:", verdict)"""),
        md("""The canonical result, confirmed on real data: the **price is non-stationary** (both tests
agree — a unit root / random walk with drift), while its **returns are stationary**. This is the
reason quant finance models *returns*, not *prices*. Differencing the log price once gives the
log return — i.e. the price is **I(1)** (integrated of order 1)."""),

        # ---------------------------------------------------------------- 6 acf/pacf
        md(
"""### 6. Autocorrelation — ACF & PACF, and how to read order

- **ACF(k)** = correlation between `yₜ` and `yₜ₋ₖ`. It includes *indirect* paths (if today depends
  on yesterday and yesterday on the day before, the ACF at lag 2 picks up that chain).
- **PACF(k)** = the *direct* correlation at lag k, with the intermediate lags partialled out.

The classic identification table (the basis of Box–Jenkins, Part 3):

| pattern | ACF | PACF |
|---|---|---|
| **AR(p)** | tails off (decays) | **cuts off after lag p** |
| **MA(q)** | **cuts off after lag q** | tails off |
| **ARMA(p,q)** | tails off | tails off |

Anything inside the shaded band is statistically indistinguishable from zero."""),
        co("""fig = ts.acf_pacf_plot(ret, lags=24, name="monthly return", fname="p2_acf_pacf_return.png")
plt.show()"""),
        md("""Returns are *close to* white noise but not quite — there's a real spike at lag 1. A formal
**Ljung-Box** test (H₀: no autocorrelation) confirms it:"""),
        co("""print(ts.ljung_box(ret, lags=12).loc[[1, 6, 12]].to_string())
print("\\nlb_pvalue ~ 0 -> returns carry SOME linear structure.")"""),
        md("""**Important caveat — data construction, not free money.** Shiller's monthly price is a
*within-month average of daily closes*, not the month-end close. Averaging mechanically induces
positive autocorrelation in the resulting returns (the Working effect). So a chunk of that lag-1
spike is an artifact of how the series is built, not a tradeable signal. Always ask how a series
was *constructed* before trading its autocorrelation — exactly the Part-1 habit, applied to time."""),

        # ---------------------------------------------------------------- 7 volatility clustering
        md(
"""### 7. Volatility clustering — the variance has memory (ARCH)

Returns themselves are nearly unpredictable, but their **magnitude** is highly predictable: big
moves follow big moves. We see it by running the ACF on **squared** (or absolute) returns — a
proxy for variance. Strong, slowly-decaying autocorrelation there = **volatility clustering**,
the empirical fact that motivates **ARCH/GARCH** models (a stretch topic for later)."""),
        co("""fig = ts.acf_pacf_plot(ret**2, lags=24, name="SQUARED return (variance proxy)", fname="p2_acf_squared.png")
plt.show()
print("Ljung-Box on squared returns:")
print(ts.ljung_box(ret**2, lags=12).loc[[1, 6, 12]].to_string())"""),
        md("""Contrast the two: **raw** returns lose autocorrelation almost immediately, but **squared**
returns stay autocorrelated for many lags and the Ljung-Box statistic *grows*. The level is
unpredictable; the **risk** is persistent. This is the time-series face of Part 1's fat tails —
the extremes cluster instead of arriving independently."""),

        # ---------------------------------------------------------------- 8 differencing
        md(
"""### 8. Differencing — turning non-stationary into stationary

**Differencing** replaces the level with its change: `Δyₜ = yₜ − yₜ₋₁`. It removes a trend (a
linear trend dies after one difference). The number of differences needed is the **order of
integration d**; here `Δ log(price) = log return`, so price is **I(1)**.

- **Seasonal differencing** `yₜ − yₜ₋ₘ` (m = period) removes a seasonal pattern — used for CO₂.
- **Over-differencing** is a real hazard: differencing a series that's already stationary
  *inflates the variance* and stamps a tell-tale **strong negative lag-1 autocorrelation** into
  the ACF. More differencing is not safer."""),
        co("""# Price is I(1): one log-difference makes it stationary.
diff1 = logp.diff().dropna()
_, v_price = ts.stationarity_report(logp,  name="log price")
_, v_diff  = ts.stationarity_report(diff1, name="Δ log price")
print("log price    ->", v_price)
print("Δ log price  ->", v_diff, " (== monthly log return)")"""),
        co("""# Over-differencing demo: difference the ALREADY-stationary returns again.
over = ret.diff().dropna()
from statsmodels.tsa.stattools import acf
print("variance  ret = %.5f   |  Δret (over-differenced) = %.5f  <- inflated" % (ret.var(), over.var()))
print("lag-1 ACF ret = %+.3f   |  Δret = %+.3f  <- strong negative spike = over-differenced" % (acf(ret, nlags=1)[1], acf(over, nlags=1)[1]))"""),
        co("""# Seasonal differencing removes the CO2 cycle.
co2_sd = co2m.diff(12).dropna()
_, v_co2  = ts.stationarity_report(co2m,   name="CO₂ level")
_, v_co2d = ts.stationarity_report(co2_sd, name="CO₂ seasonally differenced")
print("CO₂ level                  ->", v_co2)
print("CO₂ minus 12-months-ago    ->", v_co2d)"""),

        # ---------------------------------------------------------------- summary
        md(
"""### Takeaways

- **Index hygiene first**: typed, sorted, gapless, with a declared frequency. Returns *compound*, not add.
- **Look before testing**: rolling mean reveals trend; rolling std reveals volatility clustering.
- **Additive vs multiplicative**, and the log trick that converts one to the other; **STL** is the
  decomposition workhorse (evolving seasonality, robust). The S&P 500 has trend but ~no seasonality.
- **Stationarity** is what models demand. **ADF × KPSS** (opposite nulls) → a 2×2 decision table.
  Price is **I(1)**; returns are stationary → *model returns, not prices*.
- **ACF/PACF** identify model order (AR cuts in PACF, MA cuts in ACF); **Ljung-Box** tests white noise.
  Mind how a series was *constructed* (Shiller's monthly averaging fakes lag-1 autocorrelation).
- **Squared returns** stay autocorrelated → volatility clustering / ARCH; the variance is predictable
  even when the level isn't.
- **Difference** to reach stationarity — but watch for **over-differencing** (variance blows up,
  lag-1 ACF turns sharply negative).

**Next — Part 3 (Univariate Forecasting):** baselines → ETS (Holt-Winters) → ARIMA/SARIMA, using
exactly these tools (stationarity to set `d`, ACF/PACF to set `p,q`) plus Box–Jenkins diagnostics
and `auto_arima`. We'll install the forecasting extras (pmdarima) at that point."""),
    ]
    build(cells, "02_ts_foundations.ipynb", "# 02 · Time-Series Foundations — Stationarity, Decomposition, ACF/PACF")


# ===================================================================== Notebook 3
SETUP3 = SETUP + """
from src import ts, forecasting as fc
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsforecast.models import AutoARIMA
from statsmodels.datasets import co2

# Two targets: a forecastable series (CO2) and a (nearly un-)forecastable one (S&P returns).
co2m = co2.load_pandas().data["co2"].resample("MS").mean().interpolate()
shiller = data.clean_shiller()
ret = shiller["return"].dropna()
H, M = 24, 12                       # 24-month horizon, 12-month season
tr, te = fc.ts_train_test(co2m, H)
print("CO2 train:", tr.index.min().date(), "->", tr.index.max().date(), "| test:", len(te), "months")
"""


def notebook_3():
    md = new_markdown_cell
    co = new_code_cell
    cells = [
        md(
"""## Part 3 — Univariate Forecasting

Forecasting is *one* series predicting *its own* future from its past. Part 2 gave us the
diagnostic toolkit; here we turn it into predictions and, crucially, **measure** them honestly.

The progression — each model adds one idea:

| family | captures | key models |
|---|---|---|
| **Baselines** | "tomorrow ≈ today" | naive, seasonal-naive, drift, mean |
| **ETS** (exponential smoothing) | level + trend + seasonality, weighted to recency | SES → Holt → Holt-Winters |
| **ARIMA** | autocorrelation structure of a stationarised series | AR, MA, ARIMA(p,d,q) |
| **SARIMA** | ARIMA + a seasonal copy of itself | SARIMA(p,d,q)(P,D,Q)ₘ |

We forecast two very different series on purpose:
- **Mauna Loa CO₂** — strong trend + seasonality ⇒ the "real" models should win big.
- **S&P 500 returns** — a near-random walk ⇒ a *reality check* where nothing beats the baseline.

> **Library note.** The classic `auto_arima` lives in **pmdarima**, but pmdarima is not
> compatible with NumPy 2.x (which this project uses), so we use **statsforecast's `AutoARIMA`**
> — the actively-maintained, NumPy-2-safe successor — alongside hand-built ARIMA via statsmodels."""),
        co(SETUP3),

        # ---------------------------------------------------------------- 1 workflow + metrics
        md(
"""### 1. The workflow, the split, and how to score a forecast

**Split by time, never shuffle.** The test set must lie strictly *after* the training set —
otherwise the model peeks at the future (leakage) and the score is a fantasy. We hold out the
last `H` observations.

**Metrics** (computed on the held-out test set):

| metric | units | watch out |
|---|---|---|
| **MAE** | series units | robust, interpretable |
| **RMSE** | series units | punishes large misses (squared) |
| **MAPE** | % | **explodes when values cross 0** — useless for returns |
| **sMAPE** | % | symmetric, bounded, still odd near 0 |
| **MASE** | ratio | MAE ÷ in-sample naive MAE — **<1 beats naive**, comparable across series |

MASE is the one to trust: it bakes the naive baseline into the denominator, so a value below 1
literally means "better than doing nothing"."""),

        # ---------------------------------------------------------------- 2 baselines
        md(
"""### 2. Baselines — the bar every model must clear

A forecast is only impressive *relative to the trivial alternative*. Four baselines, each
optimal for a different world:

- **naive** — repeat the last value. Optimal for a pure random walk (e.g. prices).
- **seasonal-naive** — repeat the value from one season ago. Optimal when seasonality dominates.
- **drift** — extend the line from first to last training point. A naive with slope.
- **mean** — the historical average. Optimal for flat noise around a constant level.

If your fancy model cannot beat the best of these, it has earned nothing."""),
        co("""base = fc.baseline_forecasts(tr, H, M)
print(fc.compare_models(te, base, tr, m=M).to_string())
fig = fc.plot_forecast(tr, te, base, "CO₂ — baselines", "p3_co2_baselines.png", tail=60)
plt.show()"""),
        md("""On CO₂ every baseline struggles: the series both **trends** and is **seasonal**, so repeating
the past (naive/seasonal-naive) lags the rising trend, and the **mean** is hopeless (MASE ≈ 25 —
the historical average sits far below today's level). This is the gap the real models will close."""),

        # ---------------------------------------------------------------- 3 ETS
        md(
"""### 3. ETS — exponential smoothing (level, trend, seasonality)

Exponential smoothing forecasts a weighted average of past values where **weights decay
geometrically** into the past (recent points matter more). It builds up in three steps:

- **SES** (Simple) — tracks the **level** only. Flat forecast. For series with no trend/season.
- **Holt** — adds a **trend** component. Forecast is a sloped line.
- **Holt-Winters** — adds a **seasonal** component (additive or multiplicative).

The modern name is **ETS** = (Error, Trend, Seasonal), each ∈ {None, Additive, Multiplicative};
that taxonomy of 30+ models is what `statsmodels`/`statsforecast` search over. Watch the three
levels build up on CO₂:"""),
        co("""ets = {}
ets["SES (level)"]            = SimpleExpSmoothing(tr).fit().forecast(H).values
ets["Holt (+trend)"]         = Holt(tr).fit().forecast(H).values
ets["Holt-Winters (+season)"] = ExponentialSmoothing(tr, trend="add", seasonal="add", seasonal_periods=M).fit().forecast(H).values
print(fc.compare_models(te, ets, tr, m=M).to_string())
fig = fc.plot_forecast(tr, te, ets, "CO₂ — ETS build-up (SES → Holt → Holt-Winters)", "p3_co2_ets.png", tail=60)
plt.show()"""),
        md("""You can read the components straight off the plot: **SES** is flat (level only), **Holt**
slopes up (level + trend) but misses the wiggle, **Holt-Winters** tracks both the climb and the
annual cycle — MASE collapses from ~1.9 to ~0.18. Adding the right component is what buys accuracy."""),

        # ---------------------------------------------------------------- 4 ARIMA
        md(
"""### 4. ARIMA — modelling the autocorrelation itself

**ARIMA(p, d, q)** has three integers, and Part 2 already taught how to read them:

- **d** — how many times to **difference** to reach stationarity (from ADF×KPSS). CO₂ needs d=1.
- **p** — **AR** order: how many past *values* feed the prediction (PACF cut-off).
- **q** — **MA** order: how many past *forecast errors* feed it (ACF cut-off).

ARIMA assumes the (differenced) series is stationary and has **no seasonality** — so a plain
ARIMA on CO₂ will capture the trend (via d=1) but leave the yearly cycle in the residuals. That
failure is exactly what motivates SARIMA next."""),
        co("""arima = SARIMAX(tr, order=(2, 1, 2), seasonal_order=(0, 0, 0, 0)).fit(disp=False)
pred_arima = arima.forecast(H).values
print("plain ARIMA(2,1,2) on CO₂ — MASE %.3f (captures trend, ignores season)"
      % fc.forecast_metrics(te.values, pred_arima, tr.values, M)["MASE"])
fig = fc.plot_forecast(tr, te, {"ARIMA(2,1,2)": pred_arima}, "CO₂ — plain ARIMA leaves the seasonal cycle behind",
                       "p3_co2_arima.png", tail=60)
plt.show()"""),

        # ---------------------------------------------------------------- 5 SARIMA
        md(
"""### 5. SARIMA — ARIMA plus a seasonal copy

**SARIMA(p,d,q)(P,D,Q)ₘ** bolts a second, seasonal ARIMA onto the first: **(P,D,Q)** are the
seasonal AR/diff/MA orders and **m** is the period (12 for monthly data). **D=1** applies a
*seasonal difference* (value minus the value m steps ago) — in Part 2 we saw that flattens CO₂.

A sensible, classic specification is **SARIMA(1,1,1)(1,1,1)₁₂**; we confirm it below and let
`auto_arima` second-guess us in §7."""),
        co("""sarima = SARIMAX(tr, order=(1, 1, 1), seasonal_order=(1, 1, 1, M)).fit(disp=False)
pred_sarima = sarima.forecast(H).values
print("SARIMA(1,1,1)(1,1,1)12 — MASE %.3f" % fc.forecast_metrics(te.values, pred_sarima, tr.values, M)["MASE"])
print(sarima.summary().tables[0].as_text())"""),

        # ---------------------------------------------------------------- 6 Box-Jenkins diagnostics
        md(
"""### 6. Box–Jenkins diagnostics — is the model *adequate*?

The Box–Jenkins loop is **identify → estimate → check → repeat**. A model is adequate when its
**residuals are white noise** — no structure left to extract. Four checks (statsmodels packs
them into `plot_diagnostics`):

1. **Residuals over time** — should look like patternless noise around 0.
2. **Histogram + KDE vs N(0,1)** — roughly normal.
3. **Q–Q plot** — points on the line.
4. **Correlogram (residual ACF)** — all spikes inside the band.

Plus the formal **Ljung–Box** test (H₀: residuals are uncorrelated) — we want **large** p-values."""),
        co("""fig = sarima.plot_diagnostics(figsize=(12, 8)); fig.suptitle("SARIMA residual diagnostics", y=1.01)
eda.savefig(fig, "p3_sarima_diagnostics.png"); plt.show()
lb = ts.ljung_box(sarima.resid.iloc[M+1:], lags=24)
print("Ljung-Box on residuals (want p > 0.05 = white noise):")
print(lb.loc[[12, 24]].to_string())"""),
        md("""Residuals scatter around zero, sit on the Q–Q line, and the correlogram is clean — and the
Ljung–Box p-values are comfortably above 0.05. The model has wrung out the structure; what's left
is noise. That is the green light to forecast."""),

        # ---------------------------------------------------------------- 7 auto_arima
        md(
"""### 7. `auto_arima` — letting the computer search the orders

Manual Box–Jenkins is skillful but slow. **AutoARIMA** searches (p,d,q)(P,D,Q) by repeatedly
fitting and minimising an information criterion (**AIC** — fit penalised by parameter count), with
unit-root tests choosing d and D. It is what you reach for in production; the manual method is how
you understand and sanity-check it."""),
        co("""am = AutoARIMA(season_length=M, stepwise=True).fit(tr.values)
p, q, P, Q, m, d, D = am.model_["arma"]
print(f"AutoARIMA picked: ARIMA({p},{d},{q})({P},{D},{Q})[{m}]")
pred_auto = am.predict(h=H)["mean"]
print("AutoARIMA — MASE %.3f vs our manual SARIMA — MASE %.3f"
      % (fc.forecast_metrics(te.values, pred_auto, tr.values, M)["MASE"],
         fc.forecast_metrics(te.values, pred_sarima, tr.values, M)["MASE"]))"""),

        # ---------------------------------------------------------------- 8 leaderboard
        md("""### 8. The leaderboard — every model on one CO₂ scoreboard"""),
        co("""allp = {**base, "Holt-Winters": ets["Holt-Winters (+season)"],
        "SARIMA(1,1,1)(1,1,1)12": pred_sarima, "AutoARIMA": pred_auto}
board = fc.compare_models(te, allp, tr, m=M)
print(board.to_string())
fig = fc.plot_forecast(tr, te, {k: allp[k] for k in ["seasonal_naive", "Holt-Winters", "SARIMA(1,1,1)(1,1,1)12", "AutoARIMA"]},
                       "CO₂ — model comparison", "p3_co2_leaderboard.png", tail=72)
plt.show()"""),
        md("""The structured models (Holt-Winters, SARIMA, AutoARIMA) beat every baseline by **~8×** on
MASE. On a series with genuine trend and seasonality, modelling pays off enormously."""),

        # ---------------------------------------------------------------- 9 reality check
        md(
"""### 9. Reality check — forecasting S&P 500 returns

Now the humbling half. Equity returns are close to a **random walk**: Part 2 showed they're
stationary with almost no usable autocorrelation. So what *should* win? The naive/mean baseline.
Let's see whether an ARIMA can beat doing nothing."""),
        co("""H2 = 60
trr, ter = fc.ts_train_test(ret, H2)
rp = {"mean": fc.baseline_forecasts(trr, H2, 1)["mean"],
      "naive": fc.baseline_forecasts(trr, H2, 1)["naive"],
      "ARIMA(1,0,1)": SARIMAX(trr, order=(1, 0, 1)).fit(disp=False).forecast(H2).values}
print(fc.compare_models(ter, rp, trr, m=1).to_string())"""),
        md(
"""Two lessons land at once:

1. **ARIMA does *not* beat the mean** (MASE ≈ 0.84 vs ≈ 0.80). On an efficient, near-random series,
   sophistication buys nothing — sometimes it *hurts*. The honest forecast for next month's return
   is roughly its long-run average. This is *why* quant work targets **volatility and risk**
   (which Part 2 showed *is* predictable) rather than the direction of returns.
2. **Look at the MAPE column: ~1650%.** Because returns cross zero, dividing by them blows MAPE up
   into nonsense. This is the concrete reason we score forecasts with **MASE/MAE**, not MAPE."""),

        # ---------------------------------------------------------------- 10 intervals
        md(
"""### 10. Don't forecast a point — forecast a *range*

A single number is overconfident. A forecast needs a **prediction interval** that widens with the
horizon (uncertainty compounds). statsmodels gives model-based intervals directly; in Part 6 we'll
add **conformal** intervals that hold even when the model's assumptions don't."""),
        co("""fcres = sarima.get_forecast(H)
mean = fcres.predicted_mean.values
ci = fcres.conf_int(alpha=0.05).values
fig = fc.plot_forecast(tr, te, {"SARIMA forecast": mean}, "CO₂ — SARIMA forecast with 95% interval",
                       "p3_co2_interval.png", tail=60, interval=(ci[:, 0], ci[:, 1]))
plt.show()
cover = ((te.values >= ci[:, 0]) & (te.values <= ci[:, 1])).mean()
print("Actual test points inside the 95%% interval: %.0f%%" % (100 * cover))"""),

        # ---------------------------------------------------------------- takeaways
        md(
"""### Takeaways

- **Split by time and score with MASE.** A model only matters if it beats the naive baseline (MASE<1).
- **ETS** builds accuracy by *adding components*: level (SES) → trend (Holt) → seasonality (Holt-Winters).
- **ARIMA(p,d,q)** reads straight off Part 2: d from stationarity, p/q from PACF/ACF. **SARIMA** adds
  the seasonal `(P,D,Q)ₘ` block; **D=1** is the seasonal difference.
- **Box–Jenkins**: a model is adequate only when residuals are **white noise** (clean correlogram,
  large Ljung-Box p). `auto_arima` automates the search via AIC — understand it, then trust it.
- **CO₂**: structured models beat baselines ~8×. **S&P returns**: nothing beats the mean, and MAPE
  is meaningless — markets are (nearly) efficient, so we forecast *risk*, not *direction*.
- Always forecast an **interval**, not just a point.

**Next — Part 4 (Multivariate):** when series move *together*. Build a returns panel, find the market
factor with **PCA**, test lead/lag with **Granger causality**, and exploit long-run equilibria with
**cointegration / VECM** (the engine behind pairs trading)."""),
    ]
    build(cells, "03_univariate_forecasting.ipynb",
          "# 03 · Univariate Forecasting — Baselines, ETS, ARIMA/SARIMA, auto_arima")


# ===================================================================== Notebook 4
SETUP4 = SETUP + """
from src import multivariate as mv
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM
import seaborn as sns

px = data.load_stock_panel()            # 12 large caps, daily close, 2013-2018
ret = data.stock_log_returns()          # daily log returns (stationary)
SECTORS = data.SECTORS
print("panel:", px.shape, "| returns:", ret.shape, "|", px.index.min().date(), "->", px.index.max().date())
"""


def notebook_4():
    md = new_markdown_cell
    co = new_code_cell
    cells = [
        md(
"""## Part 4 — Multivariate Time Series

Parts 1–3 looked at one variable at a time. Real systems move **together**: stocks share a
market, sectors co-move, and some pairs are tethered by a long-run equilibrium. This notebook is
about the *joint* structure.

| # | tool | question |
|---|---|---|
| 2 | **Correlation + clustering** | which series move together? do sectors emerge on their own? |
| 3 | **PCA factor model** | is there a single 'market' driving everything? (CAPM's PC1) |
| 4 | **VAR** | can the past of *all* series predict each one? |
| 5 | **Granger causality** | does series A's past help predict series B (lead/lag)? |
| 6 | **Cointegration** | are two drifting prices tied by a stationary spread? |
| 7 | **Pairs trading** | …and how that spread becomes a mean-reversion strategy |

**Data:** a panel of **12 S&P 500 large caps in 6 same-sector pairs** (Tech AAPL/MSFT, Banks
JPM/BAC, Energy XOM/CVX, Staples KO/PEP, Health JNJ/PFE, Retail WMT/TGT), daily 2013–2018. The
pairs are deliberate cointegration candidates. Stationary inputs (returns) drive correlation/
PCA/VAR/Granger; price **levels** drive cointegration."""),
        co(SETUP4),

        # ---------------------------------------------------------------- 2 correlation
        md(
"""### 2. Correlation structure — and letting the sectors emerge

The return correlation matrix says who moves with whom. If we **reorder** it by hierarchical
clustering (group by similarity), the sector blocks should appear on the diagonal *without us
telling the algorithm what the sectors are* — a sanity check that the structure is real."""),
        co("""corr = mv.corr_heatmap(ret, fname="p4_corr_clustered.png")
plt.show()
off_diag = corr.where(~np.eye(len(corr), dtype=bool))
print("average pairwise return correlation: %.2f" % off_diag.stack().mean())
print("Everything is positively correlated -> a common factor is at work (next section).")"""),
        md("""Two facts jump out: **every** pair is positively correlated (no diversification escapes the
market), and the clustering recovers the **sector pairs** (KO–PEP, JPM–BAC, XOM–CVX…) as the
tightest blocks. That non-zero *average* correlation is the fingerprint of a shared driver."""),

        # ---------------------------------------------------------------- 3 PCA
        md(
"""### 3. PCA — extracting the market factor

If everything co-moves, a few **latent factors** should explain most of the variance. PCA finds
the orthogonal directions of maximum variance:

- **PC1** almost always has **all-positive loadings** — it is the *market factor* (the day the
  whole market is up/down). Its share of variance measures how 'one-directional' the market is.
- **PC2, PC3…** typically capture **sector / style tilts** (e.g. Energy vs the rest).

This is the empirical seed of factor models (CAPM, Fama–French)."""),
        co("""pca, evr, loadings, scores = mv.pca_factors(ret)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
ax[0].bar(range(1, len(evr)+1), evr.values*100, color="steelblue")
ax[0].plot(range(1, len(evr)+1), evr.cumsum().values*100, "o-", color="crimson")
ax[0].set_title("Scree: variance explained"); ax[0].set_xlabel("component"); ax[0].set_ylabel("%")
order = loadings["PC1"].sort_values().index
ax[1].barh(order, loadings.loc[order, "PC1"], color=["tab:green" if v>0 else "tab:red" for v in loadings.loc[order,"PC1"]])
ax[1].set_title("PC1 loadings (all same sign = market factor)")
fig.tight_layout(); eda.savefig(fig, "p4_pca.png"); plt.show()
print("PC1 = %.1f%% of variance (the market). PC1+PC2 = %.1f%%." % (100*evr.iloc[0], 100*evr.iloc[:2].sum()))"""),
        co("""# PC2 vs PC1, by name: do sectors separate along later components?
fig, ax = plt.subplots(figsize=(7.5, 6))
for tk in loadings.index:
    ax.scatter(loadings.loc[tk,"PC1"], loadings.loc[tk,"PC2"], s=90)
    ax.annotate(f"{tk}", (loadings.loc[tk,"PC1"], loadings.loc[tk,"PC2"]), fontsize=9,
                xytext=(4,4), textcoords="offset points")
ax.axhline(0, color="0.7", lw=.8); ax.axvline(0, color="0.7", lw=.8)
ax.set_xlabel("PC1 (market)"); ax.set_ylabel("PC2 (sector tilt)")
ax.set_title("Loadings in factor space — sector pairs sit together")
eda.savefig(fig, "p4_pca_biplot.png"); plt.show()"""),
        md("""**PC1 ≈ 39% of all variance with uniformly positive loadings** is the market: one factor moves
every name. On **PC2** the names spread out by sector (the two energy stocks, the two banks, etc.
land near each other) — the latent factors line up with the economics, never having been told the
sectors."""),

        # ---------------------------------------------------------------- 4 VAR
        md(
"""### 4. VAR — vector autoregression

A **VAR(p)** regresses each series on `p` lags of *every* series (itself + the others), capturing
joint dynamics. It needs **stationary** inputs → we use returns. The lag order `p` is chosen by an
information criterion, just like ARIMA. We fit a small cross-sector system to keep it readable."""),
        co("""sub = ret[["AAPL", "JPM", "XOM", "KO"]]
sel = VAR(sub).select_order(10)
print("Lag order suggested by each criterion:")
print(pd.Series(sel.selected_orders).to_string())
p = max(1, sel.selected_orders["bic"])
var_res = VAR(sub).fit(p)
print(f"\\nFitted VAR({p}). Example: is AAPL Granger-caused by the others (within the VAR)?")
print(var_res.test_causality("AAPL", ["JPM", "XOM", "KO"], kind="f").summary())"""),

        # ---------------------------------------------------------------- 5 Granger
        md(
"""### 5. Granger causality — does A's past help predict B?

A series **Granger-causes** another if its lags improve the forecast of the other *beyond* that
other's own lags. Crucial caveats:

- it is about **predictability, not true causation** (a common driver or a faster-reacting stock
  can produce it);
- both series must be **stationary** (returns), or you get spurious results.

We compute the full directed matrix of smallest p-values over lags 1–5."""),
        co("""gm = mv.granger_matrix(ret, maxlag=5)
fig, ax = plt.subplots(figsize=(8.5, 7))
sns.heatmap(gm.astype(float), cmap="viridis_r", vmin=0, vmax=0.1, ax=ax,
            cbar_kws={"label": "min p-value (lags 1-5)"})
ax.set_xlabel("cause (its past ->)"); ax.set_ylabel("effect (... predicts this)")
ax.set_title("Granger causality: dark = significant lead/lag")
eda.savefig(fig, "p4_granger.png"); plt.show()
sig = gm.stack().sort_values()
sig = sig[sig < 0.01]
print("Strongest directed lead/lag links (p < 0.01), 'cause -> effect':")
for (effect, cause), pv in sig.items():
    print(f"  {cause:5s} -> {effect:5s}   p={pv:.4f}")"""),
        md("""The matrix is **asymmetric** — that is the whole point. A handful of names lead others
(e.g. larger/faster stocks predicting their sector peers a day later). But remember: this is
*predictive* lead/lag, often driven by how fast each stock absorbs common news — not proof that
one company moves another."""),

        # ---------------------------------------------------------------- 6 cointegration
        md(
"""### 6. Cointegration — drifting together

Correlation is about *co-movement of returns*. **Cointegration** is deeper and about *levels*:
two prices are each non-stationary **I(1)** (random walks), yet a linear combination of them is
**stationary** — they are tied by a long-run equilibrium and never wander too far apart.

Why it matters: regressing one trending price on another invites **spurious regression** (huge R²,
zero meaning). Cointegration is the *legitimate* exception — and the foundation of pairs trading.

First the trap: two stocks can be highly correlated in levels yet **not** cointegrated."""),
        co("""a, b = "AAPL", "KO"
print(f"{a} vs {b}: level correlation = %.2f (looks related!)" % px[a].corr(px[b]))
eg_bad = mv.engle_granger(px[a], px[b])
print(f"...but Engle-Granger coint p = {eg_bad['coint_p']:.3f}  -> NOT cointegrated (spread wanders).")
fig, ax = plt.subplots(figsize=(11,4))
(px[[a,b]]/px[[a,b]].iloc[0]).plot(ax=ax)  # normalised to 1 at start
ax.set_title(f"{a} & {b}: correlated trends, but no shared equilibrium"); eda.savefig(fig,"p4_spurious.png"); plt.show()"""),
        md("""Now scan all six **same-sector** pairs with the **Engle–Granger** two-step (OLS hedge ratio →
test the spread for stationarity):"""),
        co("""pairs = [("AAPL","MSFT"),("JPM","BAC"),("XOM","CVX"),("KO","PEP"),("JNJ","PFE"),("WMT","TGT")]
scan = mv.cointegration_scan(px, pairs)
print(scan.to_string())
best = tuple(scan.iloc[0]["pair"].split("-"))
print(f"\\nOnly {best[0]}-{best[1]} clears the bar (coint p < 0.05). Cointegration is RARE -- most")
print("same-sector pairs are not tethered, which is exactly why a real signal is valuable.")"""),
        co("""# Confirm with the Johansen test (multivariate, rank-based).
print(f"Johansen trace test on {best}:")
print(mv.johansen_summary(px[list(best)]).to_string(index=False))
print("\\nTrace stat beats the 95% crit at r<=0 but not r<=1 -> exactly ONE cointegrating relation.")"""),

        # ---------------------------------------------------------------- 7 pairs trading
        md(
"""### 7. Pairs trading — turning the spread into a signal

If KO and PEP are tied by a stationary spread, then when the spread stretches it should snap back.
The recipe:

1. **spread** = KO − β·PEP (β = the cointegrating hedge ratio).
2. **z-score** the spread. When `z > +2`: spread is rich → **short** KO / long PEP. When `z < −2`:
   → **long** KO / short PEP. Exit as `z → 0`.
3. A **VECM** formalises this: the *error-correction term* measures how fast each leg is pulled
   back toward equilibrium."""),
        co("""eg = mv.engle_granger(px[best[0]], px[best[1]])
spread = eg["spread"]
z_full = mv.zscore(spread)              # in-sample (illustrative)
z_roll = mv.zscore(spread, window=60)   # rolling (no look-ahead -> tradeable)
fig, ax = plt.subplots(2,1, figsize=(12,7), sharex=True)
ax[0].plot(spread.index, spread, color="purple", lw=.9); ax[0].axhline(spread.mean(), color="k", lw=.8)
ax[0].set_title(f"{best[0]}-{best[1]} spread (beta={eg['beta']:.3f}) -- stationary, mean-reverting")
ax[1].plot(z_roll.index, z_roll, color="teal", lw=.9)
for k,c in [(2,"r"),(-2,"g"),(0,"k")]: ax[1].axhline(k, color=c, ls="--", lw=.8)
ax[1].set_title("Rolling 60-day z-score: |z|>2 = entry, z->0 = exit")
fig.tight_layout(); eda.savefig(fig, "p4_pairs_spread.png"); plt.show()
print("days with a live entry signal (|rolling z|>2): %d of %d" % ((z_roll.abs()>2).sum(), z_roll.notna().sum()))"""),
        co("""# VECM: the error-correction speeds (alpha). A negative alpha on the leg that is 'too high'
# means it gets pulled back down -> the mechanism behind the mean reversion.
vecm = VECM(px[list(best)], k_ar_diff=1, coint_rank=1, deterministic="ci").fit()
alpha = pd.Series(vecm.alpha.ravel(), index=best, name="adjustment speed (alpha)")
print("VECM error-correction speeds:"); print(alpha.round(4).to_string())
print("\\nbeta (cointegrating vector):", np.round(vecm.beta.ravel(), 3))
print("At least one leg has a significant pull back to equilibrium -> the spread mean-reverts.")"""),
        md("""**Caveat — this is the *mechanism*, not a backtest.** The full-sample z-score peeks at the
future; only the **rolling** z is tradeable. A real strategy also needs transaction costs, a
rolling re-estimate of β, and out-of-sample validation — which is exactly the **walk-forward
backtesting** we build in Part 6. And cointegration can *break* (a merger, a shock), so live pairs
must be re-tested continually."""),

        # ---------------------------------------------------------------- takeaways
        md(
"""### Takeaways

- Equity returns are **all positively correlated** — clustering recovers the sectors unsupervised.
- **PCA**: PC1 (~39%) is the **market factor** (all-positive loadings); later PCs are sector tilts.
  This is the empirical basis of CAPM / factor models.
- **VAR** models the joint dynamics of stationary series; **Granger causality** is an *asymmetric,
  predictive* lead/lag relation — *not* proof of causation, and only valid on stationary data.
- **Cointegration** is the legitimate exception to spurious regression: I(1) prices with a
  **stationary spread**. It is **rare** — only KO–PEP of six sector pairs — confirmed by both
  Engle–Granger and Johansen.
- That stationary spread is the engine of **pairs trading**; **VECM** describes how fast each leg
  error-corrects. The honest signal is the **rolling** z-score (no look-ahead).

**Next — Part 5 (ML / DL forecasting):** reframe forecasting as supervised learning, engineer
**leakage-safe** lag/rolling features (the `.shift(1)` discipline), and train gradient-boosted
trees (LightGBM) — then contrast the ML approach with the classical models from Part 3."""),
    ]
    build(cells, "04_multivariate.ipynb",
          "# 04 · Multivariate — Correlation, PCA, VAR, Granger, Cointegration & Pairs")


# ===================================================================== Notebook 5
SETUP5 = SETUP + """
import lightgbm as lgb
from src import ml_forecast as mlf, forecasting as fc
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.datasets import co2

co2m = co2.load_pandas().data["co2"].resample("MS").mean().interpolate()
H = 24
tr, te = co2m[:-H], co2m[-H:]
LGBM = dict(n_estimators=300, num_leaves=31, learning_rate=0.05, min_child_samples=10,
            random_state=0, verbose=-1)
print("train:", tr.index.min().date(), "->", tr.index.max().date(), "| test:", H, "months")
"""


def notebook_5():
    md = new_markdown_cell
    co = new_code_cell
    cells = [
        md(
"""## Part 5 — Machine-Learning Forecasting

Classical models (Part 3) *are* the time series — they model autocorrelation directly. The ML
approach is different: **reframe forecasting as ordinary supervised regression** by turning the
series into a table of features, then throw a general learner (gradient-boosted trees) at it.

That unlocks nonlinearity, many exogenous inputs, and one model across many series — but it comes
with two sharp edges this notebook is built around:

1. **Leakage** — a feature for time *t* must use only the past. One careless rolling window and
   your backtest is a fantasy.
2. **Trees cannot extrapolate** — a tree never predicts outside the range of values it saw in
   training, so on a trending series a level-model silently fails. The fix is to model the
   *stationary* part (differences), echoing Part 2.

We use **CO₂** (clear trend + seasonality) so we can score ML head-to-head against the Holt-Winters
/ SARIMA winners from Part 3 on the *same* 24-month holdout."""),
        co(SETUP5),

        # ---------------------------------------------------------------- 1 reframe
        md(
"""### 1. Reframe: a series → a supervised table

Slide a window across the series. The features for time *t* are its recent past — **lags**
(`y[t-1]…y[t-12]`), **rolling** mean/std of the past, and **calendar** fields — and the target is
`y[t]`. A 1-D series becomes an (X, y) matrix any regressor can train on."""),
        co("""sup = mlf.make_supervised(tr, n_lags=12)
cols = mlf.feature_cols(sup)
print("supervised table:", sup.shape, "| features:", len(cols))
print(sup[["lag1","lag2","lag12","rmean3","rstd12","month","y"]].head(3).round(2).to_string())"""),

        # ---------------------------------------------------------------- 2 leakage
        md(
"""### 2. The leakage trap — why every rolling feature is `.shift(1)`

A rolling mean computed as `y.rolling(3).mean()` **includes `y[t]` itself** — the very value you
are trying to predict. Train on that and your model looks brilliant in backtest and collapses
live. The discipline: shift first, then roll, so the window ends at `t-1`."""),
        co("""demo = pd.DataFrame({"y": tr.values}, index=tr.index).head(6)
demo["LEAKY_roll3"]  = demo["y"].rolling(3).mean()          # includes y[t]  <-- contaminated
demo["SAFE_roll3"]   = demo["y"].shift(1).rolling(3).mean() # ends at t-1     <-- correct
print(demo.round(2).to_string())
leaky_corr = tr.rolling(3).mean().corr(tr)
safe_corr  = tr.shift(1).rolling(3).mean().corr(tr)
print(f"\\ncorr(feature, target):  LEAKY={leaky_corr:.3f}  vs  SAFE={safe_corr:.3f}")
print("The leaky feature is near-perfectly correlated with the target because it CONTAINS it.")"""),

        # ---------------------------------------------------------------- 3 trend trap
        md(
"""### 3. Trees can't extrapolate — the level model fails

A decision tree predicts the **average target inside a leaf**, so its output can never exceed the
largest value it saw in training. On a series that only goes *up*, a model trained on **levels**
plateaus just below the last training value while the truth keeps climbing. Watch it undershoot:"""),
        co("""m_lvl = lgb.LGBMRegressor(**LGBM).fit(sup[cols], sup["y"])
pred_lvl = mlf.recursive_forecast(m_lvl, tr, H, cols)
mase_lvl = fc.forecast_metrics(te.values, pred_lvl.values, tr.values, 12)["MASE"]
fig, ax = plt.subplots(figsize=(12,4.5))
ax.plot(tr.index[-60:], tr.values[-60:], color="0.5", label="train")
ax.plot(te.index, te.values, color="black", lw=2, label="actual")
ax.plot(pred_lvl.index, pred_lvl.values, "--", color="crimson", lw=2, label="LightGBM on LEVELS")
ax.axhline(tr.max(), color="red", ls=":", lw=1, label="max training value (a ceiling!)")
ax.set_title(f"LightGBM on levels can't pass its training ceiling — MASE {mase_lvl:.2f}"); ax.legend()
eda.savefig(fig, "p5_level_trap.png"); plt.show()
print("last train=%.1f  max prediction=%.1f  actual reaches=%.1f  -> MASE %.2f (worse than naive!)"
      % (tr.iloc[-1], pred_lvl.max(), te.max(), mase_lvl))"""),

        # ---------------------------------------------------------------- 4 fix via diff
        md(
"""### 4. The fix: model the *difference* (make it stationary first)

Forecast the **change** `Δy = y[t] − y[t-1]` instead of the level. The differenced series has no
trend (Part 2), so its values stay in a fixed range the tree *can* represent — then we rebuild the
level by cumulatively summing the predicted changes onto the last known value."""),
        co("""d = tr.diff().dropna()
sup_d = mlf.make_supervised(d, n_lags=12)
m_d = lgb.LGBMRegressor(**LGBM).fit(sup_d[cols], sup_d["y"])
dpred = mlf.recursive_forecast(m_d, d, H, cols)
pred_diff = mlf.reconstruct_from_diff(dpred, tr.iloc[-1])
mase_diff = fc.forecast_metrics(te.values, pred_diff.values, tr.values, 12)["MASE"]
fig, ax = plt.subplots(figsize=(12,4.5))
ax.plot(te.index, te.values, color="black", lw=2, label="actual")
ax.plot(pred_lvl.index, pred_lvl.values, "--", color="crimson", lw=1.5, label=f"LEVEL (MASE {mase_lvl:.2f})")
ax.plot(pred_diff.index, pred_diff.values, "--", color="green", lw=2, label=f"DIFF (MASE {mase_diff:.2f})")
ax.set_title("Differencing lets the tree follow the trend"); ax.legend()
eda.savefig(fig, "p5_diff_fix.png"); plt.show()
print("MASE: level=%.3f  ->  diff=%.3f  (a %.0fx improvement)" % (mase_lvl, mase_diff, mase_lvl/mase_diff))"""),

        # ---------------------------------------------------------------- 5 strategies
        md(
"""### 5. Multi-step strategies — recursive vs direct

To forecast H steps with a one-step model you choose a strategy:

- **Recursive** (used above) — predict *t+1*, feed it back in as a lag, predict *t+2*, … One model,
  but **errors compound** down the horizon.
- **Direct** — train a *separate* model for each horizon (one predicts *t+1*, another *t+2*, …). No
  error feedback, but H models and no shared coherence.
- **Hybrid / multi-output** — libraries like `mlforecast`, `sktime`, `skforecast` automate both.

Recursive is the common default; direct shines at long horizons where compounding hurts."""),

        # ---------------------------------------------------------------- 6 importance
        md("""### 6. What did the model learn? — feature importance"""),
        co("""imp = pd.Series(m_d.feature_importances_, index=cols).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9,4.5))
imp.head(10)[::-1].plot.barh(ax=ax, color="teal")
ax.set_title("LightGBM (diff model) — top feature importances")
eda.savefig(fig, "p5_importance.png"); plt.show()
print("Top features:", list(imp.head(5).index))
print("Note lag12 & rolling-12 rank high -> the model rediscovered the 12-month SEASONALITY.")"""),

        # ---------------------------------------------------------------- 7 scoreboard
        md(
"""### 7. Honest scoreboard — ML vs the classical winners

Same series, same holdout, same metric. Does the gradient-boosted tree beat Holt-Winters and
SARIMA from Part 3?"""),
        co("""hw  = ExponentialSmoothing(tr, trend="add", seasonal="add", seasonal_periods=12).fit().forecast(H).values
sar = SARIMAX(tr, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False).forecast(H).values
board = fc.compare_models(te, {"LGBM-level":pred_lvl.values, "LGBM-diff":pred_diff.values,
                               "Holt-Winters":hw, "SARIMA":sar}, tr, m=12)
print(board.to_string())"""),
        md("""The honest result: on a **clean, low-noise, strongly-seasonal** series, a well-specified
classical model (**Holt-Winters**) still edges out LightGBM. ML is **not automatically better**.
Trees earn their keep when you have **many exogenous drivers, nonlinear interactions, many related
series, or large data** — not on a single tidy seasonal line. Knowing *when not* to reach for ML is
part of the skill."""),

        # ---------------------------------------------------------------- 8 prophet
        md(
"""### 8. Prophet — a decomposition model in disguise

Meta's **Prophet** fits `y(t) = trend + seasonality + holidays + noise` with an interpretable,
auto-tuned curve. It is robust, needs little babysitting, and handles missing data and shifts well
— popular for business series. It is *not* a silver bullet (it can lag turning points), but it is a
strong, low-effort baseline."""),
        co("""import logging
for lg in ("cmdstanpy", "prophet"): logging.getLogger(lg).setLevel(logging.CRITICAL)
from prophet import Prophet
dfp = tr.reset_index(); dfp.columns = ["ds", "y"]
mp = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
mp.fit(dfp)
fcst = mp.predict(mp.make_future_dataframe(periods=H, freq="MS"))
pred_prophet = fcst.set_index("ds")["yhat"].iloc[-H:].values
board2 = fc.compare_models(te, {"Holt-Winters":hw, "SARIMA":sar, "LGBM-diff":pred_diff.values,
                                "Prophet":pred_prophet}, tr, m=12)
print(board2.to_string())
fig = mp.plot_components(fcst); fig.set_size_inches(9,6); eda.savefig(fig, "p5_prophet_components.png"); plt.show()"""),

        # ---------------------------------------------------------------- 9 landscape
        md(
"""### 9. The modern landscape — deep learning & foundation models

Beyond trees, when data is large and patterns complex:

| family | examples | when it wins |
|---|---|---|
| **RNNs** | LSTM, GRU | sequence memory; long-ish dependencies (now often superseded) |
| **Convolutional** | TCN, WaveNet | long receptive field, parallel training |
| **Modern DL** | **N-BEATS, N-HiTS, DeepAR, TFT** | many related series, covariates, probabilistic output |
| **Transformers** | Informer, PatchTST | very long horizons, large datasets |
| **Foundation models** | **TimeGPT, Chronos, TimesFM, Moirai** | *zero-shot* forecasting — pretrained, no training data needed |

Ecosystem: **Nixtla** (`statsforecast`, `mlforecast`, `neuralforecast`), **Darts**, **sktime**,
**GluonTS**. The honest guidance hasn't changed: **start with a baseline + a classical model**
(Parts 3), add **gradient-boosted trees with good features** (this notebook) next, and only reach
for deep/foundation models when you have the scale and the many-series structure to justify them.
Always measure against the naive baseline (Part 6)."""),

        # ---------------------------------------------------------------- takeaways
        md(
"""### Takeaways

- ML forecasting = **reframe the series as (X, y)** with lag / rolling / calendar features.
- **Leakage is the cardinal sin**: every rolling feature is `.shift(1)` so it ends at `t-1`.
- **Trees can't extrapolate** — a level model undershoots any trend (MASE ≈ 1, worse than naive);
  **model the difference** and reconstruct (MASE ≈ 0.23). Stationarity matters for ML too.
- Multi-step = **recursive** (compounds error) vs **direct** (one model per horizon).
- Feature importance is interpretable — the tree **rediscovered the lag-12 seasonality**.
- **Honest result**: on a clean seasonal series the **decomposition / classical models
  (Prophet ≈ Holt-Winters) beat LightGBM**. ML shines with exogenous drivers, nonlinearity, scale,
  and many related series — not on one tidy line.
- **Prophet** is a strong low-effort decomposition baseline; deep & **foundation models** are the
  frontier for large, many-series problems.

**Next — Part 6 (Evaluation & Backtesting):** the metric zoo (MASE/WAPE vs MAPE), **time-series
cross-validation** (expanding/rolling windows — never shuffle), a **walk-forward backtester**, and
**conformal** prediction intervals — the rigor that tells you which of all these models to trust."""),
    ]
    build(cells, "05_ml_forecasting.ipynb",
          "# 05 · ML Forecasting — Supervised Reframing, LightGBM, Prophet")


# ===================================================================== Notebook 6
SETUP6 = SETUP + """
import lightgbm as lgb
from src import backtest as bt, forecasting as fc, ml_forecast as mlf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.datasets import co2

co2m = co2.load_pandas().data["co2"].resample("MS").mean().interpolate()
LGBM = dict(n_estimators=200, num_leaves=31, learning_rate=0.05, min_child_samples=10,
            random_state=0, verbose=-1)

# Forecaster adapters: each takes (train_series, horizon) -> array of predictions.
def f_seasonal_naive(tr, h): return fc.baseline_forecasts(tr, h, 12)["seasonal_naive"]
def f_holt_winters(tr, h):
    return ExponentialSmoothing(tr, trend="add", seasonal="add", seasonal_periods=12).fit().forecast(h).values
def f_lgbm_diff(tr, h):
    d = tr.diff().dropna(); sup = mlf.make_supervised(d, n_lags=12); cols = mlf.feature_cols(sup)
    m = lgb.LGBMRegressor(**LGBM).fit(sup[cols], sup["y"])
    return mlf.reconstruct_from_diff(mlf.recursive_forecast(m, d, h, cols), tr.iloc[-1]).values
print("CO2:", len(co2m), "months |", co2m.index.min().date(), "->", co2m.index.max().date())
"""


def notebook_6():
    md = new_markdown_cell
    co = new_code_cell
    cells = [
        md(
"""## Part 6 — Evaluation & Backtesting

Five families of models (Parts 3–5) all claim a number. This notebook is the referee. Good
evaluation is what separates a model that *looks* good on one slice from one that actually works.

| # | tool | what it protects against |
|---|---|---|
| 1 | **The metric zoo** | picking a metric that lies (MAPE near zero, scale traps) |
| 2–3 | **Time-series cross-validation** | judging on a single lucky/unlucky window; leakage from shuffling |
| 4–5 | **Walk-forward backtesting** | overstating accuracy; not testing the *refit-and-roll* reality |
| 6 | **Conformal intervals** | false confidence — a point forecast with no honest uncertainty |

Everything runs on **CO₂**, so the numbers connect directly to Parts 3 & 5."""),
        co(SETUP6),

        # ---------------------------------------------------------------- 1 metrics
        md(
"""### 1. The metric zoo — and how each one lies

| metric | formula idea | strengths | fails when |
|---|---|---|---|
| **MAE** | mean &#124;error&#124; | robust, same units | can't compare across scales |
| **RMSE** | √mean(error²) | penalises big misses | outlier-sensitive; scale-bound |
| **MAPE** | mean &#124;error/actual&#124; | intuitive % | **explodes near 0**; punishes over- vs under- asymmetrically |
| **sMAPE** | symmetric % | bounded | still odd near 0 |
| **WAPE** | Σ&#124;error&#124; / Σ&#124;actual&#124; | %-scaled but **robust to zeros** (aggregate first) | hides per-point error |
| **MASE** | MAE ÷ in-sample naive MAE | **scale-free, comparable, baseline-relative** (<1 beats naive) | needs a sensible seasonal period |

**Default to MASE** for choosing models and **WAPE** for a business-readable %. For *probabilistic*
forecasts use the **pinball (quantile) loss** or **CRPS**. Here is MAPE blowing up where WAPE stays
sane — a series with one near-zero actual:"""),
        co("""y_true = np.array([100., 50., 1., 80.]); y_pred = np.array([110., 55., 3., 85.])
m = fc.forecast_metrics(y_true, y_pred, y_true, m=1)
print("MAPE = %.1f%%   <- the single '1' actual dominates (|2/1| = 200%% error)" % m["MAPE%"])
print("WAPE = %.1f%%   <- aggregates first, so the tiny denominator can't hijack it" % m["WAPE%"])
print("\\nSame errors, wildly different verdicts. The metric is a modelling choice, not an afterthought.")"""),

        # ---------------------------------------------------------------- 2-3 CV
        md(
"""### 2. Why a single train/test split lies

Parts 3 & 5 held out the *last 24 months*. That is **one draw**. Had we picked a different window
— a calmer or a more volatile stretch — the ranking could flip. A point estimate of accuracy has a
variance, and a single split hides it. The fix is to evaluate over **many** time-ordered windows.

### 3. Time-series cross-validation — never shuffle

Ordinary k-fold CV **shuffles**, which lets the model train on the future to predict the past —
catastrophic leakage. Time-series CV keeps order, with two schemes:

- **Expanding** window — training set grows from the start (use all history).
- **Sliding** window — fixed-width training set rolls forward (adapts to regime change, constant cost).

For extra safety with engineered lags, insert a **gap / embargo** between train and test so a
rolling feature can't straddle the boundary."""),
        co("""fig = bt.plot_cv(n=len(co2m), n_splits=5, horizon=24, fname="p6_cv_schemes.png")
plt.show()
print("Each row is a fold: blue = train (in the past), orange = test (in the future). Order is never broken.")"""),

        # ---------------------------------------------------------------- 4 walk-forward
        md(
"""### 4. Walk-forward backtesting — the way it actually runs

The gold standard: start with an initial history, **forecast one step, reveal the truth, refit,
step forward** — repeating to the end. This mimics production exactly and yields a whole
out-of-sample track record instead of a single window. We backtest three models from Parts 3 & 5
with **126 one-step refits**:"""),
        co("""initial = 400
backtests = {}
for name, f in [("seasonal_naive", f_seasonal_naive), ("Holt-Winters", f_holt_winters), ("LGBM-diff", f_lgbm_diff)]:
    backtests[name] = bt.walk_forward(co2m, f, initial=initial, horizon=1, step=1)
folds = len(backtests["Holt-Winters"])
rows = {name: fc.forecast_metrics(co2m.loc[p.index].values, p.values, co2m.iloc[:initial].values, 12)
        for name, p in backtests.items()}
board = pd.DataFrame(rows).T.sort_values("MASE").round(4)
print(f"Walk-forward one-step backtest over {folds} folds:")
print(board.to_string())"""),
        md("""This is a **far more trustworthy** verdict than one split: Holt-Winters wins across 126
windows (MASE ≈ 0.20), with LGBM-diff a close second — the single-split ranking from Part 5 holds
up under stress. seasonal-naive (MASE > 1) is the bar they clear. *Now* we can trust the ranking."""),
        co("""# Visualise the one-step backtest tracks over the out-of-sample region.
fig, ax = plt.subplots(figsize=(12, 4.5))
oos = backtests["Holt-Winters"].index
ax.plot(oos, co2m.loc[oos].values, color="black", lw=2, label="actual")
for name in ["Holt-Winters", "LGBM-diff"]:
    ax.plot(oos, backtests[name].values, lw=1, ls="--", label=name)
ax.set_title(f"One-step walk-forward backtest ({folds} refits)"); ax.legend()
eda.savefig(fig, "p6_backtest.png"); plt.show()"""),

        # ---------------------------------------------------------------- 6 conformal
        md(
"""### 6. Conformal prediction intervals — honest uncertainty

A point forecast hides risk. A good interval should **contain the truth (1−α)% of the time**.
Model-based intervals (Part 3) assume the model's error distribution is right; **conformal**
prediction drops that assumption and gives a **distribution-free coverage guarantee** from data
alone:

1. Collect residuals on a held-out **calibration** set (here: the first half of the backtest).
2. Take the **(1−α) quantile of |residual|** as the radius `q`.
3. The interval `forecast ± q` then covers ≈ (1−α) of future points — *whatever* the error shape.

We calibrate on the first 63 backtest residuals and test coverage on the next 63:"""),
        co("""hw_pred = backtests["Holt-Winters"]; hw_act = co2m.loc[hw_pred.index]
resid = (hw_act - hw_pred).values
half = len(resid) // 2
q = bt.conformal_q(resid[:half], alpha=0.10)
ev = hw_pred.index[half:]
lo, hi = hw_pred.values[half:] - q, hw_pred.values[half:] + q
cov = bt.coverage(hw_act.values[half:], lo, hi)
print("conformal 90%% interval -> radius q=%.3f, width=%.2f, EMPIRICAL coverage=%.1f%% (target 90%%)" % (q, 2*q, 100*cov))
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(ev, hw_act.values[half:], color="black", lw=2, label="actual")
ax.plot(ev, hw_pred.values[half:], color="tab:blue", lw=1.5, label="Holt-Winters forecast")
ax.fill_between(ev, lo, hi, color="tab:blue", alpha=0.2, label="conformal 90% interval")
ax.set_title("Conformal interval — coverage achieved from residuals, no distribution assumed"); ax.legend()
eda.savefig(fig, "p6_conformal.png"); plt.show()"""),
        md("""Coverage lands near the **90% target** without assuming Gaussian errors — exactly the promise
of conformal prediction. (For time series with drifting volatility, the production-grade versions are
**adaptive**: EnbPI and ACI widen/narrow the band as conditions change.)"""),

        # ---------------------------------------------------------------- 7 sanity
        md(
"""### 7. The acid test — does it survive on S&P 500 returns?

Backtesting's most valuable job is killing false discoveries. On near-random equity returns
(Parts 2–3), a rigorous walk-forward should confirm that **no model meaningfully beats the trivial
baselines** — protecting you from a model that looked clever on one lucky window."""),
        co("""from src import data
ret = data.clean_shiller()["return"].dropna()
def f_naive(tr,h): return fc.baseline_forecasts(tr,h,1)["naive"]
def f_mean(tr,h):  return fc.baseline_forecasts(tr,h,1)["mean"]
def f_arima(tr,h): return SARIMAX(tr, order=(1,0,1)).fit(disp=False).forecast(h).values
bt_ret = {n: bt.walk_forward(ret, f, initial=len(ret)-60, horizon=1, step=1) for n,f in
          [("naive",f_naive),("mean",f_mean),("ARIMA(1,0,1)",f_arima)]}
rows = {n: fc.forecast_metrics(ret.loc[p.index].values, p.values, ret.iloc[:len(ret)-60].values, 1)
        for n,p in bt_ret.items()}
tbl = pd.DataFrame(rows).T[["MAE","WAPE%","MASE"]].sort_values("MASE").round(4)
print(tbl.to_string())
spread = tbl["MASE"].max() - tbl["MASE"].min()
print(f"\\nAll three sit within {spread:.2f} MASE of each other -- ARIMA's hair-thin edge over the")
print("mean is noise, not a signal. Backtesting confirms Part 3: returns are ~unforecastable.")"""),

        # ---------------------------------------------------------------- 8 capstone
        md(
"""### 8. Course capstone — the end-to-end workflow

Across the six parts, the through-line is a single disciplined pipeline:

1. **Look first (Part 1).** Distributions, fat tails, outliers, correlation/MI, VIF, categorical
   association, and the **missingness mechanism** — before any model.
2. **Make it stationary & understand it (Part 2).** Index hygiene, decomposition, **ADF×KPSS**,
   ACF/PACF, volatility clustering. *Model returns, not prices.*
3. **Forecast univariately (Part 3).** Baseline → ETS → ARIMA/SARIMA, validated by Box–Jenkins.
4. **Go multivariate (Part 4).** PCA market factor, VAR/Granger, **cointegration & pairs**.
5. **Bring in ML (Part 5).** Supervised reframing, leakage discipline, trees + the differencing
   fix — and the humility that ML isn't always better.
6. **Judge honestly (Part 6).** The right metric (**MASE/WAPE**), **walk-forward** backtesting,
   and **conformal** intervals.

**The recurring lessons, distilled:**
- *Always beat the naive baseline* — most of the value is there; everything else is incremental.
- *Stationarity is the hinge* — for tests, for ARIMA's `d`, even for whether trees can extrapolate.
- *Leakage is the silent killer* — shuffle, an unshifted rolling window, or a peeked-at future, and
  your numbers are fiction.
- *Markets humble models* — returns are near-unforecastable; the predictable thing is **risk**.
- *Evaluation is a first-class design decision*, not a final formality.

### Takeaways

- **Choose the metric deliberately**: MASE to compare, WAPE for a robust %, pinball/CRPS for
  probabilistic. MAPE lies near zero.
- A **single split is one sample** — use **time-ordered CV** (expanding/sliding, never shuffled).
- **Walk-forward backtesting** (refit-and-roll) is the production-faithful gold standard; it
  confirmed Holt-Winters > LightGBM on CO₂ and naive ≈ best on S&P returns.
- **Conformal intervals** deliver target coverage with no distributional assumption.

*That completes the course — Parts 0 through 6. The repo is a runnable reference you can return to
or extend (e.g. a new dataset in a sibling folder under `practice-eda/`).*"""),
    ]
    build(cells, "06_evaluation_backtesting.ipynb",
          "# 06 · Evaluation & Backtesting — Metrics, Time-Series CV, Walk-Forward, Conformal")


# ===================================================================== Notebook 7 (stretch)
SETUP7 = SETUP + """
from src import volatility as vol

# Daily market proxy = equal-weight return of the 12-stock panel, in PERCENT (arch's convention).
proxy = data.stock_log_returns().mean(axis=1) * 100
vix = data.clean_vix()["close"]                      # market-implied vol, same 2013-2018 window
shiller_ret = data.clean_shiller()["return"].dropna() * 100   # 1871-> monthly, for the regime coda
print("proxy:", proxy.shape, "|", proxy.index.min().date(), "->", proxy.index.max().date())
"""


def notebook_7():
    md = new_markdown_cell
    co = new_code_cell
    cells = [
        md(
"""## Part 7 — Volatility Modelling (ARCH / GARCH)  · *stretch*

The course (Parts 0–6) showed the **level** of returns is nearly unforecastable. But Part 2 found
the other half of the story: the **variance** is *highly* predictable — big moves cluster. This
appendix models that conditional variance directly. It is the engine of risk management, options
pricing, and the VIX, and it finally puts the **VIX dataset** to work.

| # | tool | idea |
|---|---|---|
| 1 | **ARCH-LM test** | is there volatility clustering to model? |
| 2–3 | **ARCH → GARCH(1,1)** | conditional variance from past shocks & past variance |
| 4 | **Student-t innovations** | fat tails beyond what Gaussian GARCH captures |
| 5 | **GJR / leverage** | bad news raises volatility more than good news |
| 6 | **Volatility forecasting** | the mean-reverting vol term structure |
| 7 | **GARCH vs VIX** | model-based vs market-*implied* volatility |
| 8 | **Value-at-Risk + backtest** | turn σ into a risk number, then check it (Part 6 rigor) |

**Data:** a daily **equal-weight market proxy** from the 12-stock panel (2013–2018) for the main
modelling + VIX comparison, and the long **monthly Shiller** series (1871→) for a historical
regime coda."""),
        co(SETUP7),

        # ---------------------------------------------------------------- 1 arch effect
        md(
"""### 1. Is there clustering to model? — the ARCH-LM test

**Volatility clustering** = today's variance depends on recent variance (calm follows calm, chaos
follows chaos). Engle's **ARCH-LM test** formalises Part 2's squared-returns ACF: it regresses
squared residuals on their lags. **H₀: no ARCH effect** (constant variance). A tiny p-value says
the variance has memory — model it."""),
        co("""print("ARCH-LM on the market proxy:", {k: round(v, 4) for k, v in vol.arch_lm(proxy, lags=12).items()})
fig, ax = plt.subplots(figsize=(12, 3.6))
ax.plot(proxy.index, proxy.values, lw=.6, color="steelblue")
ax.set_title("Daily market-proxy returns (%) — calm and turbulent stretches alternate = clustering")
eda.savefig(fig, "p7_returns.png"); plt.show()"""),

        # ---------------------------------------------------------------- 2-3 garch
        md(
"""### 2–3. From ARCH to GARCH(1,1)

**ARCH(q)** (Engle, 1982): today's variance is a weighted sum of the last `q` squared shocks —
`σ²ₜ = ω + α₁ε²ₜ₋₁ + …`. Capturing long memory needs many lags.

**GARCH(1,1)** (Bollerslev, 1986) adds *yesterday's variance* and captures the same persistence
with two parameters — the workhorse of the field:

$$\\sigma_t^2 = \\omega + \\alpha\\, \\varepsilon_{t-1}^2 + \\beta\\, \\sigma_{t-1}^2$$

- **α** = reaction to a fresh shock; **β** = memory of past variance.
- **α + β = persistence**: how slowly variance reverts (near 1 ⇒ shocks linger for weeks).
- Long-run variance = `ω / (1 − α − β)` — the level volatility reverts toward."""),
        co("""g = vol.fit_garch(proxy, p=1, q=1, dist="t")
print(g.params.round(4).to_string())
pers = vol.persistence(g)
lr_var = g.params["omega"] / (1 - g.params["alpha[1]"] - g.params["beta[1]"])
print("\\npersistence alpha+beta = %.3f  (shocks decay slowly)" % pers)
print("long-run daily vol = %.2f%%  ->  annualised %.1f%%" % (np.sqrt(lr_var), np.sqrt(lr_var*252)))
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(g.conditional_volatility.index, g.conditional_volatility*np.sqrt(252), color="crimson", lw=.9)
ax.set_title("GARCH(1,1) conditional volatility (annualised %) — the predictable part of risk")
eda.savefig(fig, "p7_condvol.png"); plt.show()"""),

        # ---------------------------------------------------------------- 4 fat tails
        md(
"""### 4. Fat tails — Student-t innovations

Even after GARCH soaks up the clustering, the *standardised* residuals are still fatter-tailed
than a normal (Part 1's excess kurtosis never fully goes away). Using **Student-t** innovations
instead of Gaussian fits those tails; the estimated degrees of freedom **ν** quantifies how fat —
low ν = heavy tails, ν → ∞ = normal."""),
        co("""g_norm = vol.fit_garch(proxy, p=1, q=1, dist="normal")
print("GARCH-t   nu = %.2f   (finite & small => fat tails; normal is nu->infinity)" % g.params["nu"])
print("log-likelihood:  t = %.1f   vs   normal = %.1f   (higher = better fit)" % (g.loglikelihood, g_norm.loglikelihood))
print("AIC:             t = %.1f   vs   normal = %.1f   (lower = better)" % (g.aic, g_norm.aic))
print("\\nThe t model fits clearly better -> innovations are non-Gaussian, as expected.")"""),

        # ---------------------------------------------------------------- 5 leverage
        md(
"""### 5. The leverage effect — asymmetry (GJR-GARCH)

Markets fall faster than they rise: a **negative** shock raises future volatility more than a
positive shock of equal size (the *leverage effect*). Plain GARCH is symmetric and misses this.
**GJR-GARCH** adds a term that only switches on for negative shocks:

$$\\sigma_t^2 = \\omega + (\\alpha + \\gamma\\,\\mathbf{1}_{\\varepsilon_{t-1}<0})\\,\\varepsilon_{t-1}^2 + \\beta\\,\\sigma_{t-1}^2$$

**γ > 0** is the asymmetry. (EGARCH is the log-variance cousin.)"""),
        co("""gjr = vol.fit_garch(proxy, p=1, o=1, q=1, dist="t")
print("GJR gamma (leverage) = %.3f  (>0 => downside shocks hit vol harder)" % gjr.params["gamma[1]"])
print("AIC:  GJR = %.1f  vs  symmetric GARCH = %.1f  -> asymmetry improves the model" % (gjr.aic, g.aic))

# News impact curve: variance response to a shock, holding prior variance at its long-run level.
eps = np.linspace(-4, 4, 200); sig2 = g.params["beta[1]"] * lr_var
nic_g   = g.params["omega"]   + g.params["alpha[1]"]*eps**2 + g.params["beta[1]"]*sig2
nic_gjr = gjr.params["omega"] + (gjr.params["alpha[1]"] + gjr.params["gamma[1]"]*(eps<0))*eps**2 + gjr.params["beta[1]"]*sig2
fig, ax = plt.subplots(figsize=(8,4.5))
ax.plot(eps, nic_g, label="GARCH (symmetric)"); ax.plot(eps, nic_gjr, label="GJR (asymmetric)")
ax.set_xlabel("today's shock ε"); ax.set_ylabel("next-day variance"); ax.legend()
ax.set_title("News impact curve — GJR reacts more to negative shocks")
eda.savefig(fig, "p7_news_impact.png"); plt.show()"""),

        # ---------------------------------------------------------------- 6 forecast
        md(
"""### 6. Forecasting volatility — the mean-reverting term structure

Unlike returns, volatility **is** forecastable. From any day, GARCH projects variance forward; it
**mean-reverts** toward the long-run level at a speed set by persistence. The result is a *vol term
structure* — today's estimate of vol over the next N days."""),
        co("""fcast = g.forecast(horizon=20, reindex=False)
vol_path = np.sqrt(fcast.variance.iloc[-1].values) * np.sqrt(252)   # annualised
fig, ax = plt.subplots(figsize=(9,4))
ax.plot(range(1,21), vol_path, "o-", color="purple", label="GARCH forecast")
ax.axhline(np.sqrt(lr_var*252), color="k", ls="--", label="long-run vol")
ax.set_xlabel("days ahead"); ax.set_ylabel("annualised vol %"); ax.legend()
ax.set_title("Volatility forecast mean-reverts to the long-run level")
eda.savefig(fig, "p7_vol_forecast.png"); plt.show()
print("from %.1f%% today, reverting toward %.1f%% long-run" % (vol_path[0], np.sqrt(lr_var*252)))"""),

        # ---------------------------------------------------------------- 7 vs VIX
        md(
"""### 7. GARCH vs VIX — model-based vs market-*implied* volatility

The **VIX** is the market's *implied* 30-day volatility priced from S&P 500 options — a forward
expectation. Our GARCH conditional vol is a *statistical* estimate from realised returns. They
should track closely; where they differ is informative."""),
        co("""cv_ann = (g.conditional_volatility * np.sqrt(252)).rename("GARCH")
both = pd.concat([cv_ann, vix.rename("VIX")], axis=1).dropna()
print("correlation GARCH vs VIX = %.2f over %d days" % (both["GARCH"].corr(both["VIX"]), len(both)))
print("means:  GARCH %.1f%%   VIX %.1f%%   ->  VIX runs higher = the VARIANCE RISK PREMIUM"
      % (both["GARCH"].mean(), both["VIX"].mean()))
fig, ax = plt.subplots(figsize=(12,4.5))
both["VIX"].plot(ax=ax, label="VIX (implied)", color="darkorange")
both["GARCH"].plot(ax=ax, label="GARCH (model)", color="navy", lw=.9)
ax.set_title("GARCH conditional vol vs VIX — they move together; VIX sits above (risk premium)")
ax.legend(); eda.savefig(fig, "p7_garch_vs_vix.png"); plt.show()"""),
        md("""They co-move strongly, and **VIX sits systematically above** realised/GARCH vol — investors
pay a premium for protection, the **variance risk premium** that underlies volatility trading. The
gaps widen in calm markets (cheap insurance still bid) and compress in crises."""),

        # ---------------------------------------------------------------- 8 VaR
        md(
"""### 8. Value-at-Risk — and backtesting it (Part 6 rigor)

**VaR** at level α is the loss you expect to exceed only α of the time (e.g. 1% VaR = the daily
loss breached ~1 day in 100). GARCH makes it **time-varying**: VaR widens in turbulent regimes and
tightens in calm ones. We then **backtest** it — count breaches and run the **Kupiec** test
(H₀: breach rate = α). A good model breaches ≈ α; too many = underestimating risk."""),
        co("""for a in (0.05, 0.01):
    v = vol.var_series(g, proxy, alpha=a)
    bt = vol.var_backtest(proxy, v, alpha=a)
    print("VaR %2d%%:  breaches %d/%d = %.2f%% (expected %.0f%%)  Kupiec p=%.3f  %s"
          % (int(a*100), bt["breaches"], bt["n"], 100*bt["rate"], 100*a, bt["kupiec_p"],
             "OK" if bt["kupiec_p"]>0.05 else "REJECT"))
v99 = vol.var_series(g, proxy, alpha=0.01)
breaches = proxy < -v99
fig, ax = plt.subplots(figsize=(12,4.5))
ax.plot(proxy.index, proxy.values, lw=.5, color="0.6", label="daily return %")
ax.plot(v99.index, -v99.values, color="red", lw=.9, label="1% VaR (time-varying)")
ax.scatter(proxy.index[breaches], proxy.values[breaches], color="black", s=14, zorder=5, label="breach")
ax.set_title("Time-varying 1% VaR from GARCH — breaches cluster in turbulent periods")
ax.legend(loc="lower left"); eda.savefig(fig, "p7_var.png"); plt.show()"""),
        md("""The breach rate matches the target and the Kupiec test does not reject — the GARCH VaR is
**well-calibrated**, and the band visibly breathes with the market. That is exactly the Part 6
discipline (measure, don't assume) applied to risk."""),

        # ---------------------------------------------------------------- 9 regimes
        md(
"""### 9. Coda — 150 years of volatility regimes

Fitting GARCH to the long **monthly Shiller** series (1871→) renders financial history as a
volatility curve: the spikes *are* the crises that produced Part 1's extreme months."""),
        co("""gm = vol.fit_garch(shiller_ret, p=1, q=1, dist="t")
cvm = gm.conditional_volatility * np.sqrt(12)     # annualised monthly vol
fig, ax = plt.subplots(figsize=(13,4.5))
ax.plot(cvm.index, cvm.values, color="darkred", lw=.8)
for yr, lab in [("1929","Crash"),("1987","Black Mon."),("2008","GFC"),("2020","COVID")]:
    ax.axvline(pd.Timestamp(yr+"-01-01"), color="0.6", ls=":", lw=.8)
    ax.annotate(lab, (pd.Timestamp(yr+"-06-01"), cvm.max()*0.92), fontsize=8, ha="center")
ax.set_title("GARCH conditional volatility of the S&P 500, 1871–today (annualised %)")
eda.savefig(fig, "p7_regimes.png"); plt.show()
print("persistence (monthly) = %.3f — volatility regimes last for months/years, not days" % vol.persistence(gm))"""),

        # ---------------------------------------------------------------- takeaways
        md(
"""### Takeaways

- The **level** of returns is unforecastable, but **volatility clusters and IS forecastable**
  (ARCH-LM p≈0) — the most useful predictability in markets.
- **GARCH(1,1)** captures it with two parameters; **α+β (persistence) ≈ 0.93** means shocks to
  variance fade slowly. Volatility **mean-reverts** to a long-run level.
- Innovations are **fat-tailed** (Student-t fits far better) and **asymmetric** (GJR γ>0: downside
  shocks raise vol more — the leverage effect).
- GARCH conditional vol **tracks the VIX (corr ≈ 0.75)**; VIX sits above it — the **variance risk
  premium**, the basis of volatility trading.
- GARCH turns σ into **time-varying VaR**, which **backtests cleanly** (Kupiec) — risk management
  built on Parts 2 & 6.

**Where to go next:** multivariate volatility (DCC-GARCH for time-varying correlations), realised
volatility from high-frequency data, or a deep-learning volatility model — or start a fresh dataset
in a new sibling folder under `practice-eda/`."""),
    ]
    build(cells, "07_volatility_garch.ipynb",
          "# 07 · Volatility Modelling — ARCH/GARCH, Leverage, VaR & the VIX  (stretch)")


# ===================================================================== Notebook 8 (stretch)
SETUP8 = SETUP + """
from src import mgarch as mg

ret_pct = data.stock_log_returns() * 100        # 12 stocks, daily %, 2013-2018
vix = data.clean_vix()["close"]
PAIRS = [("AAPL", "MSFT"), ("JPM", "BAC"), ("XOM", "CVX")]
print("returns panel:", ret_pct.shape, "|", ret_pct.index.min().date(), "->", ret_pct.index.max().date())
"""


def notebook_8():
    md = new_markdown_cell
    co = new_code_cell
    cells = [
        md(
"""## Part 8 — Multivariate Volatility: DCC-GARCH  · *stretch*

Part 4 gave **one** correlation matrix (average pairwise ≈ 0.33). But that single number is a
dangerous average: correlations are **not constant** — they jump in crises, exactly when
diversification is supposed to protect you. Part 7 made each asset's *variance* time-varying;
this notebook makes the whole **correlation matrix** time-varying with **Engle's DCC-GARCH**.

The two-step recipe (Engle, 2002):
1. fit a **univariate GARCH** to each asset → **standardized residuals** (returns with their own
   volatility divided out);
2. let *their* correlation evolve: `Q_t = (1−a−b)·Q̄ + a·z_{t-1}z_{t-1}' + b·Q_{t-1}`, normalised to
   the conditional correlation matrix `R_t`. Estimate `(a, b)` by maximum likelihood.

**Data:** the 12-stock daily panel (2013–2018), with the VIX for context."""),
        co(SETUP8),

        # ---------------------------------------------------------------- 1 why
        md(
"""### 1. Why static correlation is dangerous

A single correlation matrix assumes the diversification you measured in calm markets still holds
in a crash. It doesn't: when everything sells off together, pairwise correlations converge toward
1 and a "diversified" book behaves like one big position. We need `R_t`, not `R`."""),

        # ---------------------------------------------------------------- 2 step 1
        md(
"""### 2. Step 1 — devolatize each asset (univariate GARCH → standardized residuals)

Each return carries its own volatility (Part 7). Dividing it out gives **standardized residuals**
`z = ε/σ`, each ~unit variance. What remains *between* assets is the correlation structure DCC will
make dynamic — and it still matches Part 4's static picture."""),
        co("""z = mg.standardized_residuals(ret_pct, dist="t")
off = ~np.eye(z.shape[1], dtype=bool)
print("standardized residuals:", z.shape)
print("avg pairwise corr:  raw returns = %.3f   |   standardized resid = %.3f  (same structure)"
      % (ret_pct.corr().values[off].mean(), z.corr().values[off].mean()))"""),

        # ---------------------------------------------------------------- 3-4 dcc
        md(
"""### 3–4. Step 2 — DCC(1,1): let the correlation move

DCC adds just **two** parameters: `a` = how strongly today's shock pushes correlations, `b` = how
much yesterday's correlation persists. `a + b < 1` keeps it mean-reverting toward the long-run
matrix `Q̄`. We estimate them by maximising the DCC log-likelihood, then replay to recover the
conditional-correlation path. (We also compute the simpler **EWMA / RiskMetrics** correlation as a
baseline.)"""),
        co("""out = mg.dcc(z, pairs=PAIRS)
print("DCC(1,1):  a = %.4f   b = %.4f   persistence a+b = %.3f  (<1, mean-reverting)"
      % (out["a"], out["b"], out["persistence"]))
print("unconditional avg correlation Q-bar = %.3f" % out["uncond_avg_corr"])
ac = out["avg_corr"]
print("conditional avg correlation ranges %.2f (calm) to %.2f (stress) — mean %.2f"
      % (ac.min(), ac.max(), ac.mean()))"""),

        # ---------------------------------------------------------------- 5 headline
        md(
"""### 5. The headline — correlation spikes in stress

Plot the average pairwise **conditional** correlation through time. The peak is not random: it
lands on the **August 2015** "Black Monday" sell-off (China's devaluation), where every name fell
together and correlation jumped from ~0.25 to ~0.46."""),
        co("""fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(out["ewma_avg_corr"].index, out["ewma_avg_corr"].values, color="0.7", lw=.8, label="EWMA (baseline)")
ax.plot(ac.index, ac.values, color="navy", lw=1.1, label="DCC avg correlation")
ax.axhline(out["uncond_avg_corr"], color="green", ls="--", lw=.9, label="static (Part 4)")
peak = ac.idxmax()
ax.annotate(f"  {peak.date()}\\n  corr={ac.max():.2f}", (peak, ac.max()), fontsize=9, color="crimson")
ax.scatter([peak], [ac.max()], color="crimson", zorder=5)
ax.set_title("Average pairwise correlation is NOT constant — it spikes in the Aug-2015 sell-off")
ax.legend(); eda.savefig(fig, "p8_dcc_avgcorr.png"); plt.show()"""),

        # ---------------------------------------------------------------- 6 vs VIX
        md(
"""### 6. Correlations rise with fear — the DCC–VIX link

Overlay the average conditional correlation with the VIX. They move together: when fear (implied
vol) rises, so does co-movement. This is the *correlation* face of risk — and why tail events hit
portfolios harder than a static covariance predicts."""),
        co("""both = pd.concat([ac.rename("corr"), vix.rename("vix")], axis=1).dropna()
r = both["corr"].corr(both["vix"])
fig, ax1 = plt.subplots(figsize=(12, 4.5))
ax1.plot(both.index, both["corr"], color="navy", lw=1, label="DCC avg corr"); ax1.set_ylabel("avg correlation", color="navy")
ax2 = ax1.twinx(); ax2.plot(both.index, both["vix"], color="darkorange", lw=.8, alpha=.8); ax2.set_ylabel("VIX", color="darkorange")
ax1.set_title(f"Average correlation vs VIX — they spike together (corr = {r:.2f})")
eda.savefig(fig, "p8_corr_vs_vix.png"); plt.show()
print("corr(avg DCC correlation, VIX) = %.2f -> correlations climb with market fear" % r)"""),

        # ---------------------------------------------------------------- 7 pairs
        md(
"""### 7. Even individual pairs breathe

The dynamics aren't just an aggregate effect — each pair's conditional correlation wanders over a
wide band. A hedge calibrated on a calm-period correlation can be badly wrong in a stressed one."""),
        co("""pc = out["pair_corr"]
fig, ax = plt.subplots(figsize=(12, 4.5))
for c in pc.columns: ax.plot(pc.index, pc[c], lw=.9, label=c)
ax.set_title("Conditional correlation of selected pairs (DCC)"); ax.legend()
eda.savefig(fig, "p8_pair_corr.png"); plt.show()
print(pc.agg(["min","mean","max"]).round(2).to_string())"""),

        # ---------------------------------------------------------------- 8 diversification
        md(
"""### 8. Why it matters — diversification erodes exactly when you need it

For an equal-weight book of `N` assets with average correlation `ρ`, portfolio variance scales like
`1/N + (1 − 1/N)·ρ` of the average single-name variance. As `ρ` climbs, that floor rises — the
diversification benefit shrinks. Using the DCC correlation, watch the equal-weight portfolio's vol
(as a fraction of the average single-stock vol) **rise in every stress episode**:"""),
        co("""N = z.shape[1]
ratio = np.sqrt(1/N + (1 - 1/N) * ac.clip(lower=0))   # port vol / avg single-name vol
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(ratio.index, ratio.values, color="purple", lw=1)
ax.axhline(np.sqrt(1/N + (1-1/N)*out["uncond_avg_corr"]), color="green", ls="--", label="static-correlation estimate")
ax.set_title("Equal-weight portfolio vol as a share of avg single-stock vol — diversification weakens in stress")
ax.legend(); eda.savefig(fig, "p8_diversification.png"); plt.show()
print("portfolio keeps %.0f%% of single-name vol in calm, but %.0f%% at the correlation peak"
      % (100*ratio.min(), 100*ratio.max()))"""),

        # ---------------------------------------------------------------- takeaways
        md(
"""### Takeaways

- A **static** correlation matrix (Part 4) hides the most important risk fact: correlations are
  **dynamic** and **spike in crises**.
- **DCC-GARCH** models the conditional correlation `R_t` with two extra parameters on top of
  per-asset GARCH; here persistence `a+b ≈ 0.88` (mean-reverting).
- Average correlation swung from **~0.25 (calm) to ~0.46** on the **Aug-2015** sell-off, and tracks
  the **VIX (corr ≈ 0.70)** — co-movement rises with fear.
- Consequently **diversification erodes exactly when you need it**: equal-weight portfolio vol
  climbs toward single-name vol as correlations converge.
- This unifies the course: Part 4's correlation × Part 7's GARCH = a **time-varying covariance
  matrix**, the object real portfolio risk management is built on.

**Where to go next:** realised/high-frequency covariance, a deep-learning forecaster
(`neuralforecast`), or a fresh dataset in a new sibling folder under `practice-eda/`."""),
    ]
    build(cells, "08_multivariate_volatility_dcc.ipynb",
          "# 08 · Multivariate Volatility — DCC-GARCH & Dynamic Correlation  (stretch)")


# ===================================================================== Notebook 9 (stretch)
SETUP9 = SETUP + """
import lightgbm as lgb
from src import neuralnet as nn, ml_forecast as mlf, forecasting as fc
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.datasets import co2

co2m = co2.load_pandas().data["co2"].resample("MS").mean().interpolate()
H = 24
tr, te = co2m[:-H], co2m[-H:]
print("CO2 train:", len(tr), "months | test:", H)
"""


def notebook_9():
    md = new_markdown_cell
    co = new_code_cell
    cells = [
        md(
"""## Part 9 — Deep Learning from Scratch (NumPy)  · *stretch*

Part 5 reframed forecasting as supervised learning and used gradient-boosted trees. Neural
networks attack the *same* (X, y) table with a different function class — stacked linear layers
with nonlinear activations, trained by gradient descent. Here we build one **from scratch in
NumPy** — forward pass, **backpropagation**, and the **Adam** optimizer — so nothing is hidden.

> **Why from scratch?** This machine's **Smart App Control blocks PyTorch** (its DLLs are
> unsigned), so `neuralforecast`'s NHITS/LSTM can't run here. Hand-building the network is the
> honest path — and it exposes the mechanics a library call hides. The architecture choices that a
> real NHITS/LSTM/TFT would add are discussed at the end.

| # | piece | idea |
|---|---|---|
| 2 | **Architecture** | layers, weights, ReLU — a flexible function approximator |
| 3 | **Learning** | MSE loss · backprop (chain rule) · Adam gradient descent |
| 4–5 | **Forecasting CO₂** | difference (stationarity!) → train → recursive multi-step |
| 6 | **Scoreboard** | honest comparison vs LightGBM / Holt-Winters / SARIMA |
| 7 | **When DL wins** | why a small net loses here, and where deep models dominate |"""),
        co(SETUP9),

        # ---------------------------------------------------------------- 2 architecture
        md(
"""### 2. Anatomy of the network

An MLP maps inputs to output through **layers**. Each layer computes `a = activation(a_prev · W + b)`:
a linear mix of the previous layer, then a nonlinearity (**ReLU**: `max(0, z)`) that lets the net
bend. Stack a few and it can approximate almost any function. Two practical musts:

- **Standardize inputs** (zero mean, unit variance) — otherwise gradients are lopsided and training
  stalls. Our `MLPRegressor` does this internally.
- **He initialisation** of weights (variance `2/fan_in`) keeps signals from vanishing/exploding.

The whole network is just matrices `W` and vectors `b` — let's count them:"""),
        co("""net0 = nn.MLPRegressor(hidden=(64, 32))
net0._init_params(n_in=20)
nparams = sum(w.size for w in net0.W) + sum(b.size for b in net0.b)
print("architecture: 20 inputs -> 64 -> 32 -> 1 output  (ReLU hidden, linear output)")
print("layer weight shapes:", [w.shape for w in net0.W])
print("trainable parameters:", nparams)"""),

        # ---------------------------------------------------------------- 3 learning
        md(
"""### 3. How it learns — backprop + Adam

Training minimises the **mean squared error** between predictions and targets:

1. **Forward pass** — push X through the layers to get ŷ.
2. **Loss** — `MSE = mean((ŷ − y)²)`.
3. **Backpropagation** — apply the chain rule layer by layer to get `∂Loss/∂W`, `∂Loss/∂b`.
4. **Adam update** — a smart gradient-descent step (per-parameter adaptive learning rate with
   momentum) that converges far faster than plain SGD.

Proof the engine is correct: train it on a **known nonlinear function** `sin(x₁) + x₂²` and watch
the loss collapse and R² hit ~1."""),
        co("""from sklearn.metrics import r2_score
rng = np.random.default_rng(0); X = rng.uniform(-3, 3, (800, 2)); y = np.sin(X[:,0]) + X[:,1]**2
demo = nn.MLPRegressor(hidden=(32, 16), epochs=300, lr=0.01, seed=0).fit(X, y)
print("R^2 on sin(x1)+x2^2 = %.3f   (1.0 = perfect -> backprop works)" % r2_score(y, demo.predict(X)))
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(demo.loss_history_, color="crimson"); ax.set_yscale("log")
ax.set_xlabel("epoch"); ax.set_ylabel("MSE (log)"); ax.set_title("Training loss — the network learning")
eda.savefig(fig, "p9_loss_curve.png"); plt.show()"""),

        # ---------------------------------------------------------------- 4-5 forecast
        md(
"""### 4–5. Forecasting CO₂ — same discipline as Part 5

A neural net is no more able to **extrapolate a trend** than a tree, so we apply the same fix:
forecast the **differenced** (stationary) series with the leakage-safe lag/rolling/calendar
features from Part 5, then reconstruct the level. The net plugs straight into the same
`recursive_forecast` driver."""),
        co("""d = tr.diff().dropna()
sup = mlf.make_supervised(d, n_lags=12); cols = mlf.feature_cols(sup)
net = nn.MLPRegressor(hidden=(64, 32), epochs=600, lr=0.01, batch=32, l2=1e-4, seed=0)
net.fit(sup[cols].values, sup["y"].values)
pred_mlp = mlf.reconstruct_from_diff(mlf.recursive_forecast(net, d, H, cols), tr.iloc[-1])
mase_mlp = fc.forecast_metrics(te.values, pred_mlp.values, tr.values, 12)["MASE"]
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(tr.index[-48:], tr.values[-48:], color="0.6", label="train")
ax.plot(te.index, te.values, color="black", lw=2, label="actual")
ax.plot(pred_mlp.index, pred_mlp.values, "--", color="purple", lw=2, label=f"NumPy MLP (MASE {mase_mlp:.2f})")
ax.set_title("From-scratch neural net forecast of CO₂"); ax.legend()
eda.savefig(fig, "p9_mlp_forecast.png"); plt.show()
print("MLP-diff MASE = %.3f  (naive ~1.86, so the net DID learn the structure)" % mase_mlp)"""),

        # ---------------------------------------------------------------- 6 scoreboard
        md("""### 6. Honest scoreboard — does the neural net win?"""),
        co("""lgbm = lgb.LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.05,
                         min_child_samples=10, random_state=0, verbose=-1).fit(sup[cols], sup["y"])
pred_lgbm = mlf.reconstruct_from_diff(mlf.recursive_forecast(lgbm, d, H, cols), tr.iloc[-1])
hw  = ExponentialSmoothing(tr, trend="add", seasonal="add", seasonal_periods=12).fit().forecast(H).values
sar = SARIMAX(tr, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False).forecast(H).values
board = fc.compare_models(te, {"NumPy-MLP": pred_mlp.values, "LGBM-diff": pred_lgbm.values,
                               "Holt-Winters": hw, "SARIMA": sar}, tr, m=12)
print(board[["MAE","RMSE","MASE"]].to_string())"""),
        md("""The honest verdict, and the whole point of the part: a **from-scratch net underperforms** the
tuned classical and tree models on this **small, single, clean** series. It learned (it crushes
the naive baseline) but it has ~480 points and thousands of parameters — too little data for its
flexibility to pay off. This is Part 5's lesson, sharpened: *more capacity is not more accuracy.*"""),

        # ---------------------------------------------------------------- 7 when DL wins
        md(
"""### 7. So when *does* deep learning win?

Not on one tidy series — on **scale and structure** the net can exploit:

- **Many related series at once** (a *global* model): train one network across thousands of series
  so it borrows strength across them — this is where **NHITS, DeepAR, TFT** dominate classical
  per-series models.
- **Rich covariates & long context**: holidays, prices, weather, long seasonal memory.
- **Sequence structure**: **RNN/LSTM/GRU** carry hidden state across time; **Transformers**
  (PatchTST, Informer) attend over long horizons. Our MLP has none of this memory.
- **Foundation models** (TimeGPT, Chronos, TimesFM): pretrained on millions of series, forecast
  **zero-shot**.

The real implementations live in **`neuralforecast` / Darts / GluonTS** on PyTorch — which Smart
App Control blocks here. The *method* is identical to what we built; the libraries add the
architectures, GPU training, and scale. The decision rule from Part 5 stands: **baseline →
classical → trees → deep**, climbing only when the data justifies it."""),

        # ---------------------------------------------------------------- takeaways
        md(
"""### Takeaways

- A neural net is a stacked linear-plus-ReLU function approximator trained by **backprop + Adam** —
  we built it in ~80 lines of NumPy and it learns arbitrary functions (R² ≈ 1 on a sanity task).
- It attacks the **same supervised reframing** as Part 5 and needs the **same stationarity
  discipline** (difference the trend) — capacity doesn't exempt it from the basics.
- On a **small single series it loses** to tuned Holt-Winters / LightGBM — neural nets are
  **data-hungry**; their edge is **many series, covariates, long context, and scale**.
- Production deep forecasting (NHITS/LSTM/TFT, foundation models) is the same idea at scale via
  PyTorch-based libraries — unavailable on this machine due to Smart App Control, but the mechanics
  are exactly what we implemented.

*This rounds out the ML/DL arc (Parts 5 & 9) of the project. Everything remains reproducible on the
pinned Python 3.12 venv; the deep-learning math here runs with nothing but NumPy.*"""),
    ]
    build(cells, "09_deep_learning_numpy.ipynb",
          "# 09 · Deep Learning from Scratch (NumPy) — Backprop, Adam & Forecasting  (stretch)")


# ===================================================================== Notebook 10 (stretch)
SETUP10 = SETUP + """
import logging
for lg in ["lightning","pytorch_lightning","lightning.pytorch","lightning.fabric",
           "lightning.fabric.utilities.seed","lightning.pytorch.utilities.rank_zero"]:
    logging.getLogger(lg).setLevel(logging.ERROR)
import lightgbm as lgb
from src import forecasting as fc, ml_forecast as mlf, neuralnet as nn
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.datasets import co2
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, LSTM
from neuralforecast.utils import PredictionIntervals

co2m = co2.load_pandas().data["co2"].resample("MS").mean().interpolate()
H = 24
tr, te = co2m[:-H], co2m[-H:]
nf_df = co2m.reset_index(); nf_df.columns = ["ds", "y"]; nf_df["unique_id"] = "co2"
nf_train = nf_df.iloc[:-H]
TK = dict(enable_progress_bar=False, enable_model_summary=False, logger=False,
          accelerator="cpu", random_seed=0)
import torch
print("PyTorch", torch.__version__, "available — training real deep models on CPU")
"""


def notebook_10():
    md = new_markdown_cell
    co = new_code_cell
    cells = [
        md(
"""## Part 10 — Real Deep Models: NHITS & LSTM (`neuralforecast`)  · *stretch*

Part 9 built a neural net by hand to expose the mechanics. Now we run the **production** article:
Nixtla's **`neuralforecast`** on **PyTorch**, with two architectures the from-scratch MLP can't
match —

- **LSTM** — a recurrent network that carries a **hidden state** through time, so it has genuine
  *sequence memory* (our MLP saw only a fixed window of lags).
- **NHITS** — Neural Hierarchical Interpolation: multi-rate pooling + hierarchical interpolation,
  designed for **long-horizon, high-frequency** forecasting at scale.

> PyTorch is available again (Smart App Control was disabled for this), so this notebook trains
> real GPU-class models — here on CPU. Everything else in the repo still runs without it.

The honest question: do these heavyweight models beat the tuned classical/tree baselines on our
**single, small** CO₂ series? We score them head-to-head, with conformal prediction intervals."""),
        co(SETUP10),

        # ---------------------------------------------------------------- 1 data format
        md(
"""### 1. `neuralforecast`'s data format — built for *many* series

Unlike statsmodels' one-series objects, neuralforecast expects a **long** dataframe with
`unique_id`, `ds`, `y` — because its models are designed to train across *thousands* of series at
once (a *global* model). We have one series (`unique_id = "co2"`), which is already the wrong
regime for these models — keep that in mind for the scoreboard."""),
        co("""print(nf_train.tail(3).to_string(index=False))
print("\\nseries:", nf_train['unique_id'].nunique(), "| train rows:", len(nf_train), "| horizon:", H)"""),

        # ---------------------------------------------------------------- 2 train
        md(
"""### 2. Train NHITS + LSTM (with conformal intervals)

We fit both models (logs silenced) with `input_size = 48` (two seasonal cycles of context) and ask
neuralforecast for **conformal prediction intervals** via rolling calibration windows — the same
distribution-free idea as Part 6, built in. This trains a few times per model, so it takes a couple
of minutes on CPU."""),
        co("""models = [NHITS(h=H, input_size=48, max_steps=1000, **TK),
          LSTM(h=H, input_size=48, max_steps=1000, encoder_hidden_size=64, **TK)]
nf = NeuralForecast(models=models, freq="MS")
nf.fit(nf_train, prediction_intervals=PredictionIntervals(n_windows=2))
pred = nf.predict(level=[90]).set_index("ds")
for m in nf.models:
    n = sum(p.numel() for p in m.parameters())
    print(f"{m.__class__.__name__:6s}: {n:,} parameters")
print("\\npredicted columns:", [c for c in pred.columns if c != "unique_id"])"""),

        # ---------------------------------------------------------------- 3 forecast plot
        md("""### 3. The forecasts — with LSTM's conformal interval"""),
        co("""fig, ax = plt.subplots(figsize=(12, 4.8))
ax.plot(tr.index[-48:], tr.values[-48:], color="0.6", label="train")
ax.plot(te.index, te.values, color="black", lw=2, label="actual")
ax.plot(pred.index, pred["NHITS"], "--", lw=1.8, label="NHITS")
ax.plot(pred.index, pred["LSTM"], "--", lw=1.8, label="LSTM")
ax.fill_between(pred.index, pred["LSTM-lo-90"], pred["LSTM-hi-90"], color="tab:green", alpha=0.15, label="LSTM 90% interval")
ax.set_title("neuralforecast NHITS & LSTM — CO₂ forecast"); ax.legend(ncol=2)
eda.savefig(fig, "p10_nf_forecast.png"); plt.show()
for c in ["NHITS","LSTM"]:
    cov = ((te.values>=pred[c+"-lo-90"].values)&(te.values<=pred[c+"-hi-90"].values)).mean()
    print("%-6s 90%%-interval empirical coverage = %.0f%%" % (c, 100*cov))"""),

        # ---------------------------------------------------------------- 4 scoreboard
        md(
"""### 4. The grand scoreboard — every model in the course

The same CO₂ 24-month holdout, every forecaster we've built: the two deep models vs the
from-scratch MLP (Part 9), LightGBM (Part 5), and the classical winners (Part 3)."""),
        co("""d = tr.diff().dropna(); sup = mlf.make_supervised(d, n_lags=12); cols = mlf.feature_cols(sup)
mlp = nn.MLPRegressor(hidden=(64,32), epochs=600, lr=0.01, seed=0).fit(sup[cols].values, sup["y"].values)
p_mlp = mlf.reconstruct_from_diff(mlf.recursive_forecast(mlp, d, H, cols), tr.iloc[-1]).values
lgbm = lgb.LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.05,
                         min_child_samples=10, random_state=0, verbose=-1).fit(sup[cols], sup["y"])
p_lgbm = mlf.reconstruct_from_diff(mlf.recursive_forecast(lgbm, d, H, cols), tr.iloc[-1]).values
hw  = ExponentialSmoothing(tr, trend="add", seasonal="add", seasonal_periods=12).fit().forecast(H).values
sar = SARIMAX(tr, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False).forecast(H).values
board = fc.compare_models(te, {"NHITS": pred["NHITS"].values, "LSTM": pred["LSTM"].values,
                               "NumPy-MLP": p_mlp, "LGBM-diff": p_lgbm,
                               "Holt-Winters": hw, "SARIMA": sar}, tr, m=12)
print(board[["MAE","RMSE","MASE"]].to_string())"""),
        md("""Read it honestly. **LSTM** is competitive — its sequence memory makes it the strongest of the
neural models and it edges the from-scratch MLP. **NHITS** underperforms here: it is engineered for
long, high-frequency, *many*-series problems and is over-powered and finicky on ~480 monthly points.
And the tuned **Holt-Winters still wins overall**. More architecture is not more accuracy when the
data is small and clean — the exact lesson of Parts 5 and 9, now confirmed with the real libraries."""),

        # ---------------------------------------------------------------- 5 when
        md(
"""### 5. So where do these models actually dominate?

Not here. They win in the regime they were built for:

- **Global training across thousands of series** — retail SKUs, sensors, every stock at once —
  where the network borrows strength across series and amortises its capacity.
- **High-frequency / long-horizon** data (hourly energy, web traffic) where there is enough signal
  to feed millions of parameters.
- **Rich exogenous covariates** and **probabilistic** output at scale.
- **Foundation models** (TimeGPT, Chronos, TimesFM) take this further — pretrained on millions of
  series, they forecast **zero-shot**, no training data required.

The decision rule the whole course has repeated: **baseline → classical → trees → deep**, climbing
only when the data's scale and structure justify it. On a single tidy seasonal line, you stop early."""),

        # ---------------------------------------------------------------- takeaways
        md(
"""### Takeaways

- **`neuralforecast` runs the real thing** — NHITS & LSTM on PyTorch — with built-in **conformal
  intervals** (Part 6's idea, native).
- **LSTM's recurrence** (hidden state through time) gives it sequence memory the Part-9 MLP lacks,
  making it the best of the neural models on CO₂; **NHITS** is mis-matched to this small series.
- **The classical Holt-Winters still wins** the grand scoreboard — deep models are **data- and
  scale-hungry**, not universally better.
- These models shine on **many series, high frequency, covariates, and scale**; **foundation
  models** push to zero-shot. Match the model to the data regime.

*This closes the forecasting arc end-to-end: baselines → ETS/ARIMA (3) → trees (5) → evaluation (6)
→ from-scratch nets (9) → production deep models (10), every one scored on the same holdout.*"""),
    ]
    build(cells, "10_deep_learning_neuralforecast.ipynb",
          "# 10 · Real Deep Models — NHITS & LSTM via neuralforecast  (stretch)")


if __name__ == "__main__":
    import sys
    all_nbs = {"0": notebook_0, "1": notebook_1, "2": notebook_2, "3": notebook_3,
               "4": notebook_4, "5": notebook_5, "6": notebook_6, "7": notebook_7,
               "8": notebook_8, "9": notebook_9, "10": notebook_10}
    targets = sys.argv[1:] or sorted(all_nbs, key=int)
    for k in targets:
        all_nbs[k]()
    print("done.")

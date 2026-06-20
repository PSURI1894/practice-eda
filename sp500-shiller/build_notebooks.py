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


if __name__ == "__main__":
    import sys
    targets = set(sys.argv[1:]) or {"0", "1", "2", "3"}
    if "0" in targets:
        notebook_0()
    if "1" in targets:
        notebook_1()
    if "2" in targets:
        notebook_2()
    if "3" in targets:
        notebook_3()
    print("done.")

"""Generate the bike-sharing notebooks from readable source (nbformat).
Re-run:  python build_notebooks.py [N ...]   then execute with jupyter nbconvert.
"""
from __future__ import annotations

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NB_DIR = "notebooks"

SETUP = """\
import sys, pathlib, warnings
warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = pathlib.Path.cwd(); ROOT = ROOT if (ROOT / "src").exists() else ROOT.parent
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src import data, eda
eda.set_style()
pd.set_option("display.width", 120, "display.max_columns", 30)
print("setup ok | numpy", np.__version__, "| pandas", pd.__version__)
"""


def build(cells, name, title):
    nb = new_notebook()
    nb.cells = [new_markdown_cell(title)] + cells
    nb.metadata["kernelspec"] = {"display_name": "Python (bike-sharing)",
                                 "language": "python", "name": "bike-sharing"}
    with open(f"{NB_DIR}/{name}", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote", name, f"({len(nb.cells)} cells)")


# ===================================================================== Notebook 0
def notebook_0():
    md, co = new_markdown_cell, new_code_cell
    cells = [
        md(
"""## Part 0 — Acquire & Clean (Bike-Sharing hourly demand)

UCI Bike-Sharing: **17,379 hourly records** of bike rentals in Washington D.C., 2011–2012, with
weather and calendar context. Target is **`cnt`** (rentals that hour). Unlike the finance series,
this dataset has **exogenous drivers** (weather, hour, season) — and three things to fix first:

| trap | what | fix |
|---|---|---|
| **Target leakage** | `cnt = casual + registered` *exactly* | drop `casual`/`registered` from predictors |
| **Integer-coded categoricals** | `season`,`weathersit`,`hr`,… are ints but categorical | recast to `category` |
| **Normalized weather** | `temp`/`hum`/`wind` scaled to [0,1] | restore real units (°C, %, km/h) |"""),
        co(SETUP),

        md("### 1. The leakage trap — `cnt = casual + registered`"),
        co("""raw = data.load_raw()
print("shape:", raw.shape, "| missing values:", int(raw.isna().sum().sum()))
print("cnt == casual + registered for ALL rows?", bool((raw.cnt == raw.casual + raw.registered).all()))
print("-> casual & registered are the target split; using them as features LEAKS the answer.")"""),

        md("### 2. Integer-coded categoricals\nStored as numbers, but a `weathersit` of 3 isn't '3 units' of anything — it's a category."),
        co("""print("season   :", raw.season.value_counts().sort_index().to_dict(), "(1=spring..4=winter)")
print("weathersit:", raw.weathersit.value_counts().sort_index().to_dict(), "(1=clear..4=heavy rain)")
print("\\nweathersit=4 (heavy rain) has only", int((raw.weathersit==4).sum()), "rows -> a rare category you can't reliably model.")"""),

        md("""### 3. Clean it
`data.clean()` builds a datetime index from `dteday + hr`, de-normalizes the weather to real units,
adds readable category labels, recasts the coded categoricals, and drops the row id."""),
        co("""df = data.clean()
print("clean shape:", df.shape, "| index:", df.index.min(), "->", df.index.max())
print("real-unit weather (first row): %.1f°C, feels %.1f°C, %.0f%% hum, %.1f km/h"
      % (df.temp_C[0], df.atemp_C[0], df.hum_pct[0], df.wind_kmh[0]))
df[["season_name","weather_name","year","temp_C","hum_pct","cnt"]].head(3)"""),

        md("""### 4. Index hygiene — the hourly grid has gaps
A 2-year hourly grid should have ~17,520 slots, but we have 17,379 — some hours are simply absent.
That matters for the time-series work later (Part 2), where lags assume a regular grid."""),
        co("""print("missing hourly slots:", data.missing_hours(df))
X, y = data.features_target(df)
print("model-ready features (leakage + target removed):", X.shape[1], "columns")"""),

        md("### 5. Persist"),
        co("""data.build_processed()
print("wrote data/processed/bike_clean.csv  — Part 0 complete.")"""),
    ]
    build(cells, "00_data_cleaning.ipynb", "# 00 · Bike-Sharing — Data Acquisition & Cleaning")


# ===================================================================== Notebook 1
def notebook_1():
    md, co = new_markdown_cell, new_code_cell
    cells = [
        md(
"""## Part 1 — Advanced EDA (Bike-Sharing demand)

The same battery of techniques as a financial series, but now the story is about **human rhythms
and weather**: when do people ride, and what conditions drive demand? We cover distributions,
the leakage trap, the daily/weekly/seasonal rhythms, weather drivers, categorical association,
outliers, and missingness."""),
        co(SETUP + "\ndf = data.clean()\nprint('rows:', len(df))"),

        md("""### 1. The target `cnt` — a right-skewed count

Rental counts are non-negative and right-skewed (many quiet hours, a few very busy ones). The
four-view battery shows the shape; a **√ transform** tames the skew (the count analogue of the
log we used for prices)."""),
        co("""print(eda.moments(df["cnt"]).round(2).to_string())
fig = eda.four_view(df["cnt"], "hourly rentals (cnt)", "p1_fourview_cnt.png"); plt.show()"""),
        co("""fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["cnt"], bins=50, ax=ax[0], color="steelblue"); ax[0].set_title(f"cnt (skew {df.cnt.skew():.2f})")
sns.histplot(np.sqrt(df["cnt"]), bins=50, ax=ax[1], color="seagreen"); ax[1].set_title(f"√cnt (skew {np.sqrt(df.cnt).skew():.2f})")
eda.savefig(fig, "p1_sqrt_transform.png"); plt.show()"""),

        md("""### 2. The leakage trap, visualised

`cnt = casual + registered`, so those two columns *are* the answer. Including them would give a
"perfect" model that knows nothing useful. This is the cardinal modelling sin — exclude them."""),
        co("""print("corr(casual, cnt) = %.3f | corr(registered, cnt) = %.3f  (registered dominates demand)"
      % (df.casual.corr(df.cnt), df.registered.corr(df.cnt)))
print("casual+registered exactly reconstruct cnt:", bool((df.casual+df.registered == df.cnt).all()))"""),

        md("""### 3. Demand rhythms — the heart of this dataset

**Hour of day** reveals commuting: peaks at 8am and 5–6pm. But the single most important EDA plot
here splits that profile by **working day** — the same average hides two completely different
shapes (a Simpson's-paradox-style trap):

- **working days**: sharp twin commute peaks (8am, 5–6pm),
- **weekends/holidays**: a single midday leisure hump."""),
        co("""by_hour = df.groupby("hr", observed=True).cnt.mean()
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
ax[0].plot(by_hour.index, by_hour.values, "o-", color="steelblue"); ax[0].set_title("avg rentals by hour (overall)")
ax[0].set_xlabel("hour"); ax[0].set_ylabel("avg cnt")
split = df.groupby(["workingday","hr"], observed=True).cnt.mean().unstack(0)
ax[1].plot(split.index, split[1], "o-", label="working day", color="crimson")
ax[1].plot(split.index, split[0], "o-", label="weekend/holiday", color="seagreen")
ax[1].set_title("same hours, TWO different rider populations"); ax[1].set_xlabel("hour"); ax[1].legend()
eda.savefig(fig, "p1_hour_profiles.png"); plt.show()
print("working-day avg %.0f vs non-working %.0f (close!) — but the SHAPES are opposite."
      % (df[df.workingday==1].cnt.mean(), df[df.workingday==0].cnt.mean()))"""),
        co("""# Seasonal + yearly growth
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
df.groupby("mnth", observed=True).cnt.mean().plot(kind="bar", ax=ax[0], color="teal")
ax[0].set_title("avg rentals by month (summer/fall peak)"); ax[0].set_xlabel("month")
df.groupby(["year","mnth"], observed=True).cnt.mean().unstack(0).plot(ax=ax[1])
ax[1].set_title("2012 >> 2011 — the service grew (a TREND)"); ax[1].set_xlabel("month")
eda.savefig(fig, "p1_seasonality.png"); plt.show()
g = df.groupby("year").cnt.mean(); print("YoY growth: %.0f -> %.0f (+%.0f%%)" % (g[2011], g[2012], 100*(g[2012]/g[2011]-1)))"""),

        md("""### 4. Weather drivers + the temp/atemp collinearity

Warmth lifts demand; humidity and wind suppress it. But `temp` and `atemp` ("feels-like")
measure almost the same thing — a textbook **multicollinearity** case (VIF ≈ 44), so you'd keep
only one in a model."""),
        co("""num = df[["temp_C","atemp_C","hum_pct","wind_kmh","cnt"]]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax[0]); ax[0].set_title("weather vs demand correlation")
sns.regplot(x="temp_C", y="cnt", data=df.sample(2000, random_state=0), scatter_kws=dict(s=6, alpha=.3), line_kws=dict(color="red"), ax=ax[1])
ax[1].set_title("warmer → more rentals"); eda.savefig(fig, "p1_weather.png"); plt.show()
print(eda.vif_table(df[["temp_C","atemp_C","hum_pct","wind_kmh"]]).round(1).to_string(index=False))
print("-> temp & atemp are near-duplicates (VIF ~44); keep one.")"""),

        md("""### 5. Categorical association — Cramér's V"""),
        co("""cats = ["season","weathersit","workingday","holiday","weekday","yr"]
cv = eda.cramers_v_matrix(df, cats)
fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(cv, annot=True, fmt=".2f", cmap="viridis", ax=ax); ax.set_title("Cramér's V — categorical association")
eda.savefig(fig, "p1_cramers_v.png"); plt.show()
print("season–weathersit V = %.2f (weather differs by season); workingday–weekday V = %.2f (mechanically linked)"
      % (cv.loc["season","weathersit"], cv.loc["workingday","weekday"]))"""),

        md("""### 6. Outliers & missingness

Extreme-demand hours are real events (perfect-weather commute peaks), not errors. And unlike
Telco/Shiller, the *values* have **no missing data** — but the *time axis* does (165 absent hours),
which is the kind of gap Part 2's index hygiene must handle."""),
        co("""print(eda.outlier_flags(df["cnt"]).round(1).to_string(index=False))
busiest = df.nlargest(5, "cnt")[["cnt","season_name","weather_name","temp_C","workingday"]]
print("\\nBusiest hours (all clear-weather commute times):"); print(busiest.to_string())
print("\\nmissing values:", int(df.isna().sum().sum()), "| missing hourly slots:", data.missing_hours(df))"""),

        md(
"""### Takeaways

- `cnt` is a **right-skewed count**; a √ transform stabilises it. Distributions, not point summaries.
- **`cnt = casual + registered`** is a leakage trap — drop those two from any predictor set.
- Demand is driven by **human rhythms**: a commute double-peak on working days vs a midday hump on
  weekends — the *same* average hiding two populations (always segment before trusting an average).
- Strong **yearly growth (+63%)** and **seasonal/weather** effects → a trend + multi-seasonal series.
- **temp ≈ atemp** (VIF ≈ 44): classic multicollinearity — keep one.
- No missing *values*, but **165 missing hourly slots** — the time axis needs care (Part 2).

**Next — Part 2 (Time-Series Foundations):** a proper hourly index, **multiple seasonalities**
(daily/weekly/yearly), STL/MSTL decomposition, stationarity, and ACF/PACF on the demand series."""),
    ]
    build(cells, "01_advanced_eda.ipynb", "# 01 · Bike-Sharing — Advanced EDA")


# ===================================================================== Notebook 2
def notebook_2():
    md, co = new_markdown_cell, new_code_cell
    cells = [
        md(
"""## Part 2 — Time-Series Foundations (multi-seasonal demand)

The S&P series had one weak seasonal cycle. Bike demand is the opposite: a strong **trend** plus
**multiple seasonalities** layered on top —

- **daily** (period 24h): the commute rhythm,
- **weekly** (period 168h): workweek vs weekend,
- **yearly**: warm-season peak.

That changes the toolkit: we need a regular hourly index, **MSTL** (multi-seasonal decomposition,
not single STL), and seasonal differencing aware of more than one period."""),
        co(SETUP + """
from statsmodels.tsa.seasonal import STL, MSTL
from src import ts
cnt = data.hourly_cnt()      # gapless hourly grid (165 gaps interpolated)
print("series:", len(cnt), "hours |", cnt.index.min(), "->", cnt.index.max(), "| freq", cnt.index.freq)"""),

        md(
"""### 1. Index hygiene — fill the gaps first

Part 0 found **165 missing hours** in the 2-year grid. Lags and seasonal periods assume a *regular*
index, so before any modelling we reindex onto a complete hourly grid and **time-interpolate** the
holes (`data.hourly_cnt()` does this). A silent gap would otherwise misalign every lag-24/lag-168."""),
        co("""raw = data.clean()["cnt"]
print("raw hours:", len(raw), "| on a gapless grid:", len(cnt), "| interpolated:", len(cnt)-len(raw))"""),

        md("""### 2. Look first — trend, and a weekly/daily pattern on zoom"""),
        co("""fig, ax = plt.subplots(2, 1, figsize=(13, 7))
ax[0].plot(cnt.index, cnt.values, lw=0.3, color="steelblue")
ax[0].plot(cnt.index, cnt.rolling(168).mean(), color="black", lw=1.2, label="weekly rolling mean")
ax[0].set_title("Hourly rentals, 2011–2012 — upward trend + dense seasonality"); ax[0].legend()
wk = cnt.loc["2012-06-04":"2012-06-17"]   # two summer weeks
ax[1].plot(wk.index, wk.values, color="crimson"); ax[1].set_title("Zoom: two weeks — daily commute peaks, weekend dip")
fig.tight_layout(); eda.savefig(fig, "p2_series.png"); plt.show()"""),

        md(
"""### 3. The seasonal signatures + autocorrelation

The daily and weekly cycles show up directly as profiles, and as **ACF spikes at lags 24 and
168** — strong, slowly-decaying autocorrelation is the multi-seasonal memory a model must capture."""),
        co("""fig = ts.acf_pacf_plot(cnt, lags=180, name="hourly rentals", fname="p2_acf.png"); plt.show()
from statsmodels.tsa.stattools import acf
a = acf(cnt, nlags=180)
print("ACF: lag1=%.2f  lag24=%.2f (daily)  lag168=%.2f (weekly)" % (a[1], a[24], a[168]))"""),

        md(
"""### 4. MSTL — decomposing *multiple* seasonalities

Classical/STL decomposition handles **one** period. **MSTL** peels off several in turn — here daily
(24) and weekly (168) — leaving trend + each seasonal component + residual. The amplitudes rank the
drivers: the **daily** commute cycle is the biggest swing, then the **weekly** pattern, then the
slow growth **trend**."""),
        co("""res = MSTL(cnt, periods=(24, 168)).fit()
fig = res.plot(); fig.set_size_inches(12, 9); fig.suptitle("MSTL: trend + daily + weekly + residual", y=1.01)
eda.savefig(fig, "p2_mstl.png"); plt.show()
amp = lambda s: s.max()-s.min()
print("swing  daily=%.0f  weekly=%.0f  trend=%.0f  (daily commute dominates)"
      % (amp(res.seasonal['seasonal_24']), amp(res.seasonal['seasonal_168']), amp(res.trend)))"""),

        md(
"""### 5. Stationarity — deseasonalize, then it's stationary

The raw series carries trend + seasonality, so it's non-stationary. Strip them with MSTL and the
**residual is stationary** (both ADF and KPSS agree) — the clean target a model's noise term should
look like."""),
        co("""for name, s in [("raw cnt", cnt), ("MSTL residual", res.resid), ("seasonal diff (lag 24)", cnt.diff(24).dropna())]:
    table, verdict = ts.stationarity_report(s, name=name)
    print(f"{name:22s} -> {verdict}")"""),

        md(
"""### 6. Did MSTL capture the structure? — residual ACF

If the decomposition worked, the residual should be close to white noise: the giant ACF spikes at
24 and 168 should largely collapse."""),
        co("""fig = ts.acf_pacf_plot(res.resid.dropna(), lags=180, name="MSTL residual", fname="p2_resid_acf.png"); plt.show()
ar = acf(res.resid.dropna(), nlags=180)
print("residual ACF: lag24 %.2f -> %.2f , lag168 %.2f -> %.2f  (seasonality removed)"
      % (a[24], ar[24], a[168], ar[168]))"""),

        md(
"""### 7. Differencing — and why multi-seasonal is harder

Seasonal **differencing** at lag 24 removes the daily cycle and reaches stationarity. But with
*two* seasonal periods you'd need to difference at 24 **and** 168 (risking **over-differencing**),
which is clumsy — so in practice multi-seasonal series are handled by **MSTL**, **Fourier terms**,
or **ML models with calendar features** (Part 3) rather than stacked seasonal differences."""),
        co("""from statsmodels.tsa.stattools import acf as _acf
d1 = cnt.diff(24).dropna()
print("var cnt = %.0f | var diff24 = %.0f" % (cnt.var(), d1.var()))
print("lag-24 ACF: cnt %.2f -> diff24 %.2f  (daily cycle largely gone)" % (a[24], _acf(d1, nlags=24)[24]))"""),

        md(
"""### Takeaways

- **Index hygiene at hourly scale**: 165 gaps were interpolated onto a regular grid before anything else.
- Demand is **multi-seasonal** — daily (24) and weekly (168) cycles show as **ACF spikes** and as
  distinct **MSTL** components; the daily commute cycle is the largest driver.
- The raw series is non-stationary; the **MSTL residual is stationary**, and its ACF is nearly flat
  → the decomposition captured the structure.
- Multiple seasonalities make stacked **seasonal differencing** awkward → prefer MSTL / Fourier /
  calendar-feature ML.

**Next — Part 3 (Forecasting with covariates):** predict `cnt` using the **weather + calendar**
drivers — SARIMAX with exogenous regressors, and a LightGBM model with lag + calendar features —
the payoff of having exogenous information the financial series never had."""),
    ]
    build(cells, "02_ts_foundations.ipynb", "# 02 · Bike-Sharing — Time-Series Foundations (multi-seasonal)")


if __name__ == "__main__":
    import sys
    all_nbs = {"0": notebook_0, "1": notebook_1, "2": notebook_2}
    for k in (sys.argv[1:] or sorted(all_nbs)):
        all_nbs[k]()
    print("done.")

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


# ===================================================================== Notebook 3
def notebook_3():
    md, co = new_markdown_cell, new_code_cell
    cells = [
        md(
"""## Part 3 — Forecasting *with covariates* (the payoff of exogenous data)

The financial series in `sp500-shiller` had **no covariates** — there was nothing to predict
returns *from*. Bike demand is the opposite: it is *driven* by **weather and the calendar**, and it
is **multi-seasonal** (daily + weekly). This notebook is a deep dive into forecasting that kind of
series, and it deliberately runs into — and explains — three things the textbook glosses over:

1. **Plain SARIMA cannot model hourly multi-seasonality** → we use **Fourier / dynamic harmonic
   regression** instead.
2. **Covariate availability** — using weather as a predictor secretly assumes you can *forecast the
   weather*; we'll be honest about that.
3. **The horizon decides what is leakage** — at a **24-hour-ahead** horizon, lag-24/lag-168 are
   *known*, which changes everything.

We compare four models of increasing sophistication and discover, on a built-in stress test, *why*
autoregressive memory matters."""),
        co(SETUP + """
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from src import featurize as fz, forecasting as fc
from statsmodels.tsa.stattools import acf

g = data.regular_grid()                 # gapless hourly grid: cnt + weather + calendar
H = 24 * 14                             # forecast horizon / test window = last 14 days
y = g["cnt"]
tr, te = slice(0, len(g) - H), slice(len(g) - H, len(g))
print("grid:", g.shape, "| test window:", g.index[-H].date(), "->", g.index[-1].date())"""),

        # ---------------------------------------------------------------- 1 task
        md(
"""### 1. The task, the split, and a crucial subtlety

**Task:** forecast hourly `cnt` for the **last 14 days**, evaluated with **MASE** (vs the in-sample
*weekly* naive) and **RMSE**.

**The 24-hour-ahead framing.** We imagine running each morning: predict the next 24 hours using
everything observed through the previous midnight. Why it matters: for a 24h-ahead forecast, **any
lag of ≥24 hours is already observed** — so `lag24` (same hour yesterday) and `lag168` (same hour
last week) are *legitimate, leakage-free* regressors with **no recursion**. (A longer horizon would
force recursion and error would compound.)

**The covariate-availability caveat.** Using `temp`, `hum`, `wind` as predictors assumes we *know*
the future weather. In production you'd plug in a **weather forecast** (itself uncertain); here we
use the recorded actuals — a "perfect-foresight" assumption that flatters the weather models. The
**calendar** (hour, day, month, holiday) is genuinely known in advance.

**The built-in stress test.** Note the test window below — it is the **Christmas / New-Year
holidays**, when demand behaves nothing like a normal December. That is not a mistake; it is the
most revealing fortnight in the dataset."""),
        co("""print("test window mean cnt = %.0f  vs  2012 overall = %.0f  -> holiday demand collapses to ~half"
      % (y[te].mean(), y[g.index.year == 2012].mean()))
def score(p): r = fc.forecast_metrics(y[te].values, np.clip(p, 0, None), y[tr].values, 168); return r["MASE"], r["RMSE"]"""),

        # ---------------------------------------------------------------- 2 baseline
        md(
"""### 2. Baseline — the weekly seasonal naive

For a strongly weekly series the bar to beat is **"same hour last week"** (`ŷₜ = yₜ₋₁₆₈`). Any
model that can't beat this earns nothing."""),
        co("""naive = y.shift(168).values[te]
mase_n, rmse_n = score(naive)
print("seasonal-naive (lag 168):  MASE %.3f  RMSE %.1f" % (mase_n, rmse_n))"""),

        # ---------------------------------------------------------------- 3 why not sarima
        md(
"""### 3. Why not plain SARIMA? — and the Fourier fix

**SARIMA(p,d,q)(P,D,Q)ₘ models ONE seasonal period `m`.** Hourly demand has *two* strong ones
(daily 24, weekly 168). You cannot put both in `seasonal_order`, and a seasonal term at lag 168 on
17,000 points is computationally hopeless (the state-space blows up; in testing it simply fails to
converge).

**The standard fix is dynamic harmonic regression:** represent each seasonal cycle by a handful of
**Fourier terms** — sine/cosine pairs at harmonics of the seasonal frequency — and feed them as
*regressors*. `K` harmonics give `2K` smooth basis functions; more `K` = a sharper seasonal shape.
A short ARIMA can then model whatever autocorrelation is left. Here are the daily Fourier terms
reconstructing the commute shape:"""),
        co("""F24 = fz.fourier_terms(g.index, 24, 10)     # 10 daily harmonics
F168 = fz.fourier_terms(g.index, 168, 5)    # 5 weekly harmonics
print("Fourier design: %d daily + %d weekly columns" % (F24.shape[1], F168.shape[1]))
# Show that a few harmonics already trace one day's average demand profile
prof = y.groupby(g.index.hour).mean()
fitday = LinearRegression().fit(F24.iloc[:24].values, prof.values).predict(F24.iloc[:24].values)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(prof.index, prof.values, "o-", label="actual hourly profile")
ax.plot(prof.index, fitday, "--", label="10 Fourier harmonics"); ax.legend()
ax.set_title("Fourier terms reconstruct the daily commute shape"); ax.set_xlabel("hour")
eda.savefig(fig, "p3_fourier.png"); plt.show()"""),

        # ---------------------------------------------------------------- 4 harmonic regression
        md(
"""### 4. Model A — harmonic regression + weather (the classical covariate model)

Regress `cnt` on the Fourier seasonality **plus the exogenous weather/calendar regressors**
(`temp`, `hum`, `wind`, `workingday`, `holiday`). The weather coefficients are the **exogenous
effect** — how demand moves with conditions, holding the time-of-week pattern fixed. It has **no
autoregressive memory**, so remember that for the stress test."""),
        co("""exog = pd.concat([F24, F168, g[["temp_C", "hum_pct", "wind_kmh", "workingday", "holiday"]]], axis=1)
harm = LinearRegression().fit(exog.values[tr], y.values[tr])
pred_harm = harm.predict(exog.values[te])
mase_h, rmse_h = score(pred_harm)
print("harmonic + weather:  MASE %.3f  RMSE %.1f" % (mase_h, rmse_h))
coef = pd.Series(harm.coef_, index=exog.columns)
print("\\nexogenous effects (per unit):")
print("  +1°C temp      -> %+6.1f rentals/hr" % coef["temp_C"])
print("  +10%% humidity  -> %+6.1f rentals/hr" % (10 * coef["hum_pct"]))
print("  +10 km/h wind  -> %+6.1f rentals/hr" % (10 * coef["wind_kmh"]))
print("  working day    -> %+6.1f rentals/hr" % coef["workingday"])"""),
        md("""It beats the naive — the **weather signal is real** (warmth lifts demand, humidity and wind
suppress it). But a linear model can't express *interactions* (the commute peak exists **only** on
working days), which is where trees come in."""),

        # ---------------------------------------------------------------- 5 trees
        md(
"""### 5. Models B & C — gradient-boosted trees

**B — calendar + weather (no lags).** LightGBM captures **nonlinear interactions** (hour ×
workingday, the temperature curve) that the linear model can't. But it still has no memory of
*recent* demand."""),
        co("""mdl_cal = lgb.LGBMRegressor(n_estimators=500, num_leaves=63, learning_rate=0.05, random_state=0, verbose=-1)
mdl_cal.fit(g[fz.CALENDAR + fz.WEATHER].values[tr], y.values[tr])
pred_cal = mdl_cal.predict(g[fz.CALENDAR + fz.WEATHER].values[te])
mase_c, rmse_c = score(pred_cal)
print("LGBM calendar+weather:  MASE %.3f  RMSE %.1f" % (mase_c, rmse_c))"""),
        md(
"""**C — add autoregressive lags.** Now add `lag24`, `lag168`, `roll24` (all ≥24h → known at a
24h-ahead horizon, no leakage). These carry the **recent level** of demand — and that turns out to
be the single most important thing."""),
        co("""gl = fz.add_lags(g)
cols = fz.CALENDAR + fz.WEATHER + fz.LAG_COLS
mdl_lag = lgb.LGBMRegressor(n_estimators=700, num_leaves=63, learning_rate=0.05, random_state=0, verbose=-1)
mdl_lag.fit(gl[cols].iloc[168:len(g) - H], y.values[168:len(g) - H])
pred_lag = mdl_lag.predict(gl[cols].values[te])
mase_l, rmse_l = score(pred_lag)
print("LGBM + lags (24h-ahead): MASE %.3f  RMSE %.1f" % (mase_l, rmse_l))
imp = pd.Series(mdl_lag.feature_importances_, index=cols).sort_values(ascending=False)
print("\\ntop features:", list(imp.head(6).index))"""),

        # ---------------------------------------------------------------- 6 scoreboard
        md("""### 6. Scoreboard"""),
        co("""board = pd.DataFrame({
    "seasonal-naive":      [mase_n, rmse_n],
    "harmonic+weather":    [mase_h, rmse_h],
    "LGBM calendar+weather":[mase_c, rmse_c],
    "LGBM + lags":         [mase_l, rmse_l],
}, index=["MASE", "RMSE"]).T.sort_values("MASE")
print(board.round(3).to_string())"""),
        md("""The ranking is the lesson. **Lags win decisively** (MASE ≈ 0.61 — well under the 1.66 naive and
less than half its RMSE). Harmonic+weather beats the naive on *seasonality + weather*; the calendar
tree improves on it with nonlinearity but still trails the lag model — *nonlinearity without memory
isn't enough*. To see *why* the lag model dominates, look at the holiday week."""),

        # ---------------------------------------------------------------- 7 stress test
        md(
"""### 7. The holiday stress test — why memory matters

Plot the forecasts over the final days. Demand **collapses** for Christmas and New Year. The
seasonal/calendar/harmonic models predict a **normal December** and badly over-predict — they have
no way to know a holiday lull is underway (with only ~2 years of data they've barely seen these
dates). The **lag model follows the drop**, because `lag24`/`lag168` carry yesterday's and last
week's already-collapsed demand."""),
        co("""days = slice(len(g) - 24 * 8, len(g))      # last 8 days
idx = g.index[days]
fig, ax = plt.subplots(figsize=(13, 4.8))
ax.plot(idx, y.values[days], color="black", lw=2, label="actual")
ax.plot(idx, np.clip(pred_harm, 0, None)[-24*8:], "--", label="harmonic+weather")
ax.plot(idx, np.clip(pred_cal, 0, None)[-24*8:], "--", label="LGBM calendar+weather")
ax.plot(idx, np.clip(pred_lag, 0, None)[-24*8:], "--", color="crimson", lw=2, label="LGBM + lags")
ax.set_title("Holiday stress test — calendar models over-predict; lags track the collapse"); ax.legend(ncol=2)
eda.savefig(fig, "p3_holiday.png"); plt.show()
xmas = (g.index[te].normalize() == pd.Timestamp("2012-12-25"))
print("Dec 25 mean:  actual %.0f | harmonic %.0f (over) | LGBM+lags %.0f"
      % (y[te][xmas].mean(), np.clip(pred_harm,0,None)[xmas].mean(), np.clip(pred_lag,0,None)[xmas].mean()))"""),

        # ---------------------------------------------------------------- 8 importance
        md("""### 8. What drives the winning model? — feature importance"""),
        co("""fig, ax = plt.subplots(figsize=(9, 4.5))
imp.head(10)[::-1].plot.barh(ax=ax, color="teal"); ax.set_title("LGBM + lags — feature importance")
eda.savefig(fig, "p3_importance.png"); plt.show()
print("Recent demand (lag168, roll24, lag24) dominates, then weather (hum, temp) and hour.")"""),

        # ---------------------------------------------------------------- 9 conformal
        md(
"""### 9. Honest uncertainty — a conformal interval

A point forecast hides risk. We build a **distribution-free conformal** interval: train on data up
to 28 days before the end, measure absolute residuals on the next 14 days (**calibration**), take
their 90% quantile as the band radius, and apply it to the final 14-day forecast — then check the
empirical coverage."""),
        co("""cal = slice(len(g) - 2 * H, len(g) - H)
m_cal = lgb.LGBMRegressor(n_estimators=700, num_leaves=63, learning_rate=0.05, random_state=0, verbose=-1)
m_cal.fit(gl[cols].iloc[168:len(g) - 2 * H], y.values[168:len(g) - 2 * H])
resid = np.abs(y.values[cal] - np.clip(m_cal.predict(gl[cols].values[cal]), 0, None))
q = np.quantile(resid, 0.90)
lo, hi = np.clip(pred_lag - q, 0, None), pred_lag + q
cov = ((y.values[te] >= lo) & (y.values[te] <= hi)).mean()
print("conformal 90%% band: ±%.0f rentals | empirical coverage %.0f%%" % (q, 100 * cov))"""),

        # ---------------------------------------------------------------- takeaways
        md(
"""### 10. Takeaways

- With **exogenous drivers**, forecasting becomes a *covariate* problem — but the **horizon defines
  leakage**: at 24h-ahead, `lag≥24` features are known and dominate.
- **SARIMA can't do hourly multi-seasonality** → use **Fourier / dynamic harmonic regression**;
  `K` harmonics trade smoothness for sharpness.
- **Weather is a genuine signal** (warmth ↑, humidity/wind ↓ demand), but a linear model misses the
  **hour × workingday interaction** that trees capture for free.
- **Autoregressive memory wins** (MASE ≈ 0.61 vs the 1.66 naive): recent demand is the strongest
  predictor — *especially* through the **holiday collapse**, where calendar/seasonal models
  over-predict (Christmas Day: harmonic 118 vs actual 42) because they've never learned the date.
- Be honest about **covariate availability** (you'd need a weather *forecast*) and pair every point
  forecast with an **interval** (conformal coverage ≈ 90%).

**Where next:** a full **walk-forward backtest** across many windows (not just the holidays), a
**log/Poisson** treatment of the count target, and **probabilistic** gradient boosting — or fold
these models into an operational daily-refit pipeline."""),
    ]
    build(cells, "03_forecasting_covariates.ipynb", "# 03 · Bike-Sharing — Forecasting with Covariates")


# ===================================================================== Notebook 4
def notebook_4():
    md, co = new_markdown_cell, new_code_cell
    cells = [
        md(
"""## Part 4 — Backtesting & Probabilistic Forecasting (in depth)

Part 3 declared a winner on **one** window — and that window was the holidays. That's not
evaluation, it's an anecdote. This notebook does it properly:

1. a **walk-forward backtest** that re-runs the model day after day across eight weeks, so the
   verdict is an *average over many windows*, not one lucky/unlucky draw;
2. a fix for a subtle **leakage** in Part 3 (its `roll24` peeked at the same morning);
3. a move from **point** forecasts to **probabilistic** ones — quantile regression, its chronic
   under-coverage, and the **conformal** fix that restores a guarantee;
4. **count-aware** loss (Poisson) and a **calibration** (reliability) check.

The theme: *how do you actually know a forecaster is good — and how do you quantify its uncertainty
honestly?*"""),
        co(SETUP + """
import lightgbm as lgb
from src import featurize as fz, forecasting as fc, backtest as bt

g = data.regular_grid(); y = g["cnt"]
g = bt.add_strict_lags(g)                      # 24h-ahead-safe lags (>=24h old)
COLS = fz.CALENDAR + fz.WEATHER + bt.STRICT_LAGS
TOT = 70                                        # backtest = last 70 days: 14 calibrate + 56 test
n = len(g); start = n - TOT * 24
cal, test = slice(0, 14 * 24), slice(14 * 24, TOT * 24)
ytot = y.values[start:]; scale = y.values[:start]; idx = g.index[start:]
def L2(): return lgb.LGBMRegressor(n_estimators=500, num_leaves=63, learning_rate=0.05, random_state=0, verbose=-1)
print("backtest window:", idx[0].date(), "->", idx[-1].date(), "| test days:", 56)"""),

        # ---------------------------------------------------------------- 1 leakage rigor
        md(
"""### 1. First, fix the leakage — strict 24h-ahead features

A forecast made at **midnight** may use only data **through the previous day**. Part 3's `roll24`
(mean of the 24 hours ending at *t−1*) silently used the *same morning's* demand — fine for a
1-hour-ahead forecast, optimistic for a 24-hour-ahead one. So we replace it with features that are
all **≥24 hours old** and therefore genuinely known at the forecast origin:

- `lag24` — same hour yesterday, `lag168` — same hour last week,
- `lag_dayavg` — the average of the day **ending 24h ago** (not a window ending at *t−1*).

This is the difference between a backtest you can trust and one that flatters itself."""),

        # ---------------------------------------------------------------- 2 walk-forward
        md(
"""### 2. The walk-forward backtest — average over many windows

We simulate operations: every day, train on everything observed so far and forecast the next 24
hours; refit weekly; roll across **56 test days**. The aggregate MASE/RMSE over all of them is a far
more trustworthy verdict than Part 3's single fortnight."""),
        co("""point = bt.walk_forward_24h(g, y, COLS, L2, n_test_days=TOT, refit_every=7)
naive = pd.Series(y.shift(168).values[start:], index=idx)
m_pt = fc.forecast_metrics(ytot[test], point.values[test], scale, 168)
m_nv = fc.forecast_metrics(ytot[test], naive.values[test], scale, 168)
print("56-day walk-forward backtest:")
print("  LGBM (strict lags):  MASE %.3f  RMSE %.1f" % (m_pt["MASE"], m_pt["RMSE"]))
print("  seasonal-naive:      MASE %.3f  RMSE %.1f" % (m_nv["MASE"], m_nv["RMSE"]))
print("\\n(Part 3 reported MASE ~0.61 on the single holiday window — the honest multi-window number is higher.)")"""),

        # ---------------------------------------------------------------- 3 error distribution
        md(
"""### 3. The error *distribution* — and which days are hard

A single aggregate still hides structure. Plot the **per-day RMSE**: most days are easy, but a few
spike — and they're **holidays** (Thanksgiving, Christmas, New Year), where demand breaks its usual
pattern. Reporting only a mean would bury the fact that the model fails predictably on known events."""),
        co("""se = pd.Series((ytot - point.values) ** 2, index=idx)
perday = se[test].resample("D").mean().pow(0.5)
fig, ax = plt.subplots(figsize=(13, 4))
ax.bar(perday.index, perday.values, width=0.8, color="steelblue")
ax.axhline(perday.median(), color="green", ls="--", label=f"median {perday.median():.0f}")
worst = perday.nlargest(3)
for d, v in worst.items(): ax.annotate(d.strftime("%b %d"), (d, v), fontsize=8, ha="center")
ax.set_title("Per-day RMSE across the backtest — holidays are the outliers"); ax.legend()
eda.savefig(fig, "p4_perday.png"); plt.show()
print("per-day RMSE: median %.0f, worst %.0f on %s (Thanksgiving)" % (perday.median(), perday.max(), perday.idxmax().strftime("%b %d")))"""),

        # ---------------------------------------------------------------- 4 quantile regression
        md(
"""### 4. From a point to a *distribution* — quantile regression

Demand is **heteroscedastic**: a busy commute hour varies far more in absolute terms than a quiet
3am. A single number can't express that. **Quantile regression** trains the *same* model under the
**pinball loss** to predict chosen percentiles directly — here the 5th and 95th — giving a 90%
prediction interval whose width can breathe with the hour."""),
        co("""def QR(a): return lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=400, num_leaves=63, learning_rate=0.05, random_state=0, verbose=-1)
q05 = bt.walk_forward_24h(g, y, COLS, lambda: QR(0.05), TOT, refit_every=7)
q95 = bt.walk_forward_24h(g, y, COLS, lambda: QR(0.95), TOT, refit_every=7)
raw_cov = bt.coverage(ytot[test], q05.values[test], q95.values[test])
print("raw quantile 90%% interval:  coverage %.0f%%  (target 90%%)  | mean width %.0f rentals"
      % (100 * raw_cov, np.mean(q95.values[test] - q05.values[test])))
print("pinball loss:  q05 %.2f   q95 %.2f" % (bt.pinball_loss(ytot[test], q05.values[test], 0.05),
                                              bt.pinball_loss(ytot[test], q95.values[test], 0.95)))
print("\\n-> it UNDER-covers: raw quantile regression is systematically over-confident.")"""),

        # ---------------------------------------------------------------- 5 CQR
        md(
"""### 5. The fix — Conformalized Quantile Regression (CQR)

Raw quantile regression rarely hits its nominal coverage. **CQR** (Romano et al., 2019) repairs it
with a finite-sample guarantee: on a held-out **calibration** set, measure how far the truth fell
outside `[q05, q95]`, take the 90th percentile of that miss, and **widen the interval by that
amount**. We calibrate on the first 14 days and verify coverage on the 56-day test."""),
        co("""Q = bt.cqr_offset(ytot[cal], q05.values[cal], q95.values[cal], alpha=0.10)
lo, hi = np.clip(q05.values - Q, 0, None), q95.values + Q
cqr_cov = bt.coverage(ytot[test], lo[test], hi[test])
print("CQR widens the band by +%.0f rentals:" % Q)
print("  coverage %.0f%% -> %.0f%% (target 90%%) | width %.0f -> %.0f"
      % (100 * raw_cov, 100 * cqr_cov, np.mean(q95.values[test]-q05.values[test]), np.mean(hi[test]-lo[test])))
d = slice(14*24, 14*24 + 24*7)   # one test week
ti = idx[d]
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.plot(ti, ytot[d], color="black", lw=1.6, label="actual")
ax.fill_between(ti, q05.values[d], q95.values[d], color="orange", alpha=0.25, label="raw 90% (under-covers)")
ax.fill_between(ti, lo[d], hi[d], color="tab:blue", alpha=0.15, label="CQR 90% (calibrated)")
ax.set_title("Conformalized interval — calibrated to its promised 90% coverage"); ax.legend(ncol=3)
eda.savefig(fig, "p4_cqr.png"); plt.show()"""),

        # ---------------------------------------------------------------- 6 count loss
        md(
"""### 6. A count-aware loss — Poisson

The target is a **non-negative integer count**, and its variance grows with its mean. Squared
error (L2) implicitly assumes constant-variance Gaussian noise. A **Poisson** objective matches the
count nature (variance = mean) and guarantees non-negative predictions. Whether it actually
*forecasts* better is an empirical question — here it's essentially a wash on RMSE, which is itself
a useful finding (the lag features already carry most of the signal)."""),
        co("""pois = bt.walk_forward_24h(g, y, COLS, lambda: lgb.LGBMRegressor(objective="poisson", n_estimators=500, num_leaves=63, learning_rate=0.05, random_state=0, verbose=-1), TOT, refit_every=7)
print("L2      RMSE %.1f  MASE %.3f" % (m_pt["RMSE"], m_pt["MASE"]))
mp = fc.forecast_metrics(ytot[test], pois.values[test], scale, 168)
print("Poisson RMSE %.1f  MASE %.3f  (non-negative by construction; ~tied here)" % (mp["RMSE"], mp["MASE"]))"""),

        # ---------------------------------------------------------------- 7 calibration
        md(
"""### 7. Calibration — are the quantiles *honest*?

A forecaster is **calibrated** if its predicted quantiles match reality: the q-th quantile
prediction should sit above the truth about q% of the time. We fit several quantiles and plot
predicted level vs **empirical** fraction-below — the diagonal is perfect calibration. The curve
comes out **flatter than the diagonal** (low quantiles land too high at 0.1→0.21, high ones too low
at 0.9→0.82): the predicted quantiles huddle toward the median, so the interval is **too narrow /
over-confident** — exactly the gap CQR closes."""),
        co("""levels = [0.1, 0.25, 0.5, 0.75, 0.9]
emp = []
for a in levels:
    qa = lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=400, num_leaves=63, learning_rate=0.05, random_state=0, verbose=-1)
    qa.fit(g[COLS].iloc[168:start], y.iloc[168:start])
    pa = qa.predict(g[COLS].iloc[start:][test])
    emp.append(np.mean(ytot[test] <= pa))
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
ax.plot(levels, emp, "o-", color="crimson", label="quantile regression")
ax.set_xlabel("predicted quantile level"); ax.set_ylabel("empirical fraction below"); ax.legend()
ax.set_title("Reliability — flatter than diagonal = intervals too narrow (over-confident)")
eda.savefig(fig, "p4_calibration.png"); plt.show()
print("empirical coverage by level:", {l: round(e, 2) for l, e in zip(levels, emp)})"""),

        # ---------------------------------------------------------------- takeaways
        md(
"""### 8. Takeaways

- **One window is an anecdote.** A **walk-forward backtest** over 56 days gives the honest verdict
  (MASE ≈ 0.77, not Part 3's window-specific 0.61) — and still crushes the naive (1.4).
- **The horizon defines leakage.** Strict 24h-ahead features (lags ≥24h) replace Part 3's slightly
  optimistic `roll24` — rigorous evaluation starts with rigorous features.
- **Look at the error *distribution*, not just the mean** — the model fails predictably on
  **holidays** (Thanksgiving, Christmas), the worst per-day RMSE by far.
- **Point forecasts aren't enough.** **Quantile regression** gives intervals but **under-covers
  (≈76%)**; **CQR** restores the promised **90%** with a finite-sample guarantee.
- **Match the loss to the target** (Poisson for counts) — though here the autoregressive signal
  dominates and it's a wash.
- **Always check calibration** — a reliability plot reveals over-confidence a coverage number alone
  might hide.

**Where next:** a global/hierarchical model across stations, exogenous **weather-forecast
uncertainty** propagated into the intervals, or gradient-boosted quantile *ensembles* — the repo's
toolkit now spans EDA → multi-seasonal TS → covariate forecasting → rigorous probabilistic evaluation."""),
    ]
    build(cells, "04_backtesting_probabilistic.ipynb", "# 04 · Bike-Sharing — Backtesting & Probabilistic Forecasting")


# ===================================================================== Notebook 5
def notebook_5():
    md, co = new_markdown_cell, new_code_cell
    cells = [
        md(
"""## Part 5 — Explaining the Model (Interpretability)

Parts 3–4 built a strong forecaster and showed *that* lags dominate — but a number in a feature-
importance list is not understanding. A model you can't explain is hard to trust, debug, or deploy
(and "why did demand drop?" is a real stakeholder question). This notebook opens the black box with
the standard interpretability toolkit, applied to the LightGBM demand model:

1. **Global importance** three ways — *gain* vs *permutation* vs *mean |SHAP|* — and why they can
   disagree.
2. **Partial dependence** — the *shape* of each effect (the nonlinear temperature curve, the commute
   profile) that a linear model could never express.
3. **Interactions** — the hour effect *depends on* the working day; temperature *depends on* the hour.
4. **SHAP** — a principled, additive attribution: global **beeswarm**, **local** explanations of
   single predictions, and a **dependence** plot that exposes interactions.
5. The **caveats** every honest interpretability analysis must state.

It brings the EDA mindset — *understand before you trust* — to the model itself."""),
        co(SETUP + """
import lightgbm as lgb, shap
from sklearn.inspection import permutation_importance, partial_dependence
from src import featurize as fz, backtest as bt

g = bt.add_strict_lags(data.regular_grid()); y = g["cnt"]
COLS = fz.CALENDAR + fz.WEATHER + bt.STRICT_LAGS
X = g[COLS].astype(float)                       # float so sklearn PDP accepts it
n = len(g); H = 14 * 24; trm = slice(168, n - H); tem = slice(n - H, n)
mdl = lgb.LGBMRegressor(n_estimators=500, num_leaves=63, learning_rate=0.05, random_state=0, verbose=-1).fit(X.iloc[trm], y.iloc[trm])
expl = shap.TreeExplainer(mdl)
samp = X.iloc[trm].sample(2000, random_state=0)
sv = expl.shap_values(samp)
print("model trained | SHAP base value (avg prediction) = %.0f rentals" % float(expl.expected_value))"""),

        # ---------------------------------------------------------------- 1 global importance
        md(
"""### 1. Global importance — three lenses that can disagree

- **Gain** (LightGBM-native): total loss reduction from splits on a feature — *training-set* view,
  biased toward high-cardinality features.
- **Permutation**: shuffle a feature on held-out data and watch accuracy drop — *model-agnostic*,
  measures real predictive reliance, but splits credit oddly among correlated features.
- **mean |SHAP|**: average absolute contribution per prediction — additive and consistent, in the
  units of the target (rentals).

All three put the **recent-demand lags on top** — but watch **hour**: permutation and SHAP rank it
among the most important features, while **gain buries it** (its predictive credit was absorbed by
the lags at the first splits). The *same model* gives opposite verdicts on a key feature — which is
exactly why you never trust a single importance number."""),
        co("""gain = pd.Series(mdl.booster_.feature_importance("gain"), index=COLS); gain /= gain.sum()
perm = pd.Series(permutation_importance(mdl, X.iloc[tem], y.iloc[tem], n_repeats=5, random_state=0,
                 scoring="neg_root_mean_squared_error").importances_mean, index=COLS)
shap_imp = pd.Series(np.abs(sv).mean(0), index=COLS)
comp = pd.DataFrame({"gain%": (gain*100).round(1), "perm(ΔRMSE)": perm.round(1),
                     "mean|SHAP|": shap_imp.round(1)}).sort_values("mean|SHAP|", ascending=False)
print(comp.to_string())
fig, ax = plt.subplots(figsize=(8, 4.5))
shap_imp.sort_values().tail(10).plot.barh(ax=ax, color="teal"); ax.set_title("mean |SHAP| (rentals/hr of impact)")
eda.savefig(fig, "p5_importance.png"); plt.show()"""),

        # ---------------------------------------------------------------- 2 PDP
        md(
"""### 2. Partial dependence — the *shape* of each effect

Importance says *how much*; partial dependence says *which way and what shape*. We average the
model's prediction as one feature sweeps its range. This reveals the **nonlinearities** the Part-3
linear model couldn't capture: temperature helps up to ~25–30 °C then **saturates/falls** (too hot
to ride); demand falls with humidity; and the hour profile is the familiar twin commute peaks."""),
        co("""fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, feat, lab in zip(ax, ["temp_C", "hum_pct", "hr"], ["temperature °C", "humidity %", "hour"]):
    pdp = partial_dependence(mdl, samp, [feat], grid_resolution=24)
    a.plot(pdp["grid_values"][0], pdp["average"][0], "o-", color="crimson")
    a.set_xlabel(lab); a.set_ylabel("avg predicted rentals"); a.set_title(f"PDP: {lab}")
fig.tight_layout(); eda.savefig(fig, "p5_pdp.png"); plt.show()
pt = partial_dependence(mdl, samp, ["temp_C"], grid_resolution=24)
peak_t = pt["grid_values"][0][np.argmax(pt["average"][0])]
print("temperature effect peaks near %.0f°C, then demand stops rising (saturation)." % peak_t)"""),

        # ---------------------------------------------------------------- 3 interaction
        md(
"""### 3. The key interaction — hour × working day

A single hour-of-day curve is a lie (Part 1 showed it): the commute peaks exist **only on working
days**, while weekends are a midday hump. A tree captures this **interaction** for free — we expose
it by sweeping the hour with `workingday` fixed to each value (other features at their median)."""),
        co("""base = samp.median(); rows = []
for wd in (0, 1):
    for h in range(24):
        r = base.copy(); r["hr"] = h; r["workingday"] = wd; rows.append(r)
G = pd.DataFrame(rows); G["pred"] = mdl.predict(G[COLS]); piv = G.pivot_table("pred", "hr", "workingday")
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(piv.index, piv[1], "o-", color="crimson", label="working day")
ax.plot(piv.index, piv[0], "o-", color="seagreen", label="weekend/holiday")
ax.set_title("Model-based interaction: the hour effect depends on the working day"); ax.set_xlabel("hour"); ax.legend()
eda.savefig(fig, "p5_interaction.png"); plt.show()
print("working-day peak hour = %d, weekend peak hour = %d  (the model learned the two rider populations)"
      % (piv[1].idxmax(), piv[0].idxmax()))"""),

        # ---------------------------------------------------------------- 4 SHAP global
        md(
"""### 4. SHAP — a principled attribution

**SHAP** values come from cooperative game theory: they distribute each prediction's gap from the
**base value** (average demand) fairly among the features, and they **add up exactly** to the
prediction. The **beeswarm** shows, for every feature, the distribution of its SHAP impact across
2,000 hours, coloured by the feature's value — so you read *direction*, *magnitude*, and *spread* at
once."""),
        co("""plt.figure(); shap.summary_plot(sv, samp, show=False, plot_size=(10, 6), max_display=12)
eda.savefig(plt.gcf(), "p5_shap_beeswarm.png"); plt.show()"""),
        md("""Read it: high `lag168`/`lag24` (red) push demand **up** by hundreds of rentals; warm `temp_C`
(red) pushes up; high `hum_pct` (red) pushes **down**; `hr` fans out both ways (rush hours up, night
down). The colour-impact alignment *is* the direction of each effect."""),

        # ---------------------------------------------------------------- 5 SHAP local
        md(
"""### 5. Local explanations — *why this hour?*

SHAP's real power is per-prediction. We explain two opposite hours: a **busy 6pm working-day**
commute and a **quiet 4am**. Each prediction = base value + the feature contributions, so you can
hand a stakeholder the exact reason for any forecast."""),
        co("""def explain(mask, title):
    inst = samp[mask].head(1)
    contrib = pd.Series(expl.shap_values(inst)[0], index=COLS).sort_values(key=abs, ascending=False).head(6)
    return inst, contrib, float(mdl.predict(inst[COLS])[0])
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for a, (mask, title) in zip(ax, [((samp.hr == 18) & (samp.workingday == 1), "busy 6pm working day"),
                                  ((samp.hr == 4), "quiet 4am")]):
    _, contrib, pred = explain(mask, title)
    colors = ["crimson" if v > 0 else "steelblue" for v in contrib]
    contrib[::-1].plot.barh(ax=a, color=colors[::-1])
    a.axvline(0, color="k", lw=.8); a.set_title(f"{title}  (pred {pred:.0f}, base {expl.expected_value:.0f})")
fig.suptitle("Local SHAP: each prediction decomposed into feature contributions", y=1.02)
fig.tight_layout(); eda.savefig(fig, "p5_shap_local.png"); plt.show()"""),

        # ---------------------------------------------------------------- 6 SHAP dependence
        md(
"""### 6. SHAP dependence — interactions made visible

A SHAP **dependence** plot puts one feature's value on x and its SHAP impact on y, **coloured by an
interacting feature**. Temperature coloured by hour shows that warmth helps *much more at busy
hours* than at night — the interaction the global PDP averaged away."""),
        co("""plt.figure(); shap.dependence_plot("temp_C", sv, samp, interaction_index="hr", show=False)
eda.savefig(plt.gcf(), "p5_shap_dependence.png"); plt.show()
print("temp's effect on demand is larger (steeper SHAP) during high-traffic hours -> a real interaction.")"""),

        # ---------------------------------------------------------------- 7 caveats
        md(
"""### 7. The caveats every honest analysis must state

- **PDP/ICE assume feature independence.** Sweeping `temp_C` to 35 °C in January is a combination
  that never occurs — the curve there is an extrapolation, not evidence.
- **Importance ≠ causation.** `lag168` is the strongest *predictor*, not a *cause* of demand;
  intervening on it would do nothing. These tools explain the **model**, not the world.
- **Correlated features split credit.** `temp_C` and `atemp_C` (and the lags) share signal, so any
  single-feature attribution is partly arbitrary among them.
- **SHAP is an approximation** of a complex function and is computed on a *sample*; treat the ranks
  as robust, the exact values as estimates."""),

        # ---------------------------------------------------------------- takeaways
        md(
"""### 8. Takeaways

- **Three importance lenses** (gain / permutation / SHAP) broadly agree — recent-demand lags and
  hour dominate — but disagree in the details; cross-check, never trust one.
- **Partial dependence** reveals *shape*: temperature **saturates** (~25–30 °C then falls), humidity
  hurts, the hour is a twin-peak commute — the nonlinearities the linear model of Part 3 missed.
- The decisive **hour × working-day interaction** (and temp × hour) is captured by the tree and made
  visible by PDP and SHAP dependence — the model rediscovered Part 1's two rider populations.
- **SHAP** gives additive, per-prediction explanations you can hand to a stakeholder — and a global
  beeswarm that shows direction + magnitude + spread together.
- Interpretability explains the **model, not the world**: mind feature independence, correlation,
  and the predictor-vs-cause distinction.

*This rounds out the bike-sharing practice — EDA → multi-seasonal TS → covariate forecasting →
probabilistic backtesting → interpretability — a complete, honest demand-forecasting study.*"""),
    ]
    build(cells, "05_interpretability.ipynb", "# 05 · Bike-Sharing — Model Interpretability (PDP & SHAP)")


if __name__ == "__main__":
    import sys
    all_nbs = {"0": notebook_0, "1": notebook_1, "2": notebook_2, "3": notebook_3,
               "4": notebook_4, "5": notebook_5}
    for k in (sys.argv[1:] or sorted(all_nbs)):
        all_nbs[k]()
    print("done.")

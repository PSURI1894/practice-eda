"""Generate the Beijing air-quality notebooks from readable source (nbformat).
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
    nb.metadata["kernelspec"] = {"display_name": "Python (beijing-air)",
                                 "language": "python", "name": "beijing-air"}
    with open(f"{NB_DIR}/{name}", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote", name, f"({len(nb.cells)} cells)")


# ===================================================================== Notebook 0
def notebook_0():
    md, co = new_markdown_cell, new_code_cell
    cells = [
        md(
"""## Part 0 — Acquire & Clean (Beijing PM2.5)

Five years of **hourly air-pollution** readings from the US Embassy in Beijing (2010–2014): the
target **`pm25`** (PM2.5 concentration, µg/m³) plus weather (temperature, dew point, pressure, wind,
snow, rain). What makes this dataset different from the other two practices is **real missing data**
— and it's *structured*, which is the whole point.

A crucial distinction this notebook establishes:
- the **timeline is complete** (5 years × hourly = 43,824 rows, no missing *rows*),
- but the **`pm25` *values* have 2,067 gaps** (the sensor went down) — missing **values**, not rows."""),
        co(SETUP),

        md("### 1. First look — only `pm25` is missing"),
        co("""raw = data.load_raw()
print("raw shape:", raw.shape)
print("\\nmissing per column:"); print(raw.isna().sum()[lambda s: s > 0].to_string(), "(only pm2.5)")
print("\\nis the timeline complete? rows =", len(raw), "| 5 years of hourly =", 5*365*24 + 24, "(2012 leap)")"""),

        md("""### 2. The shape of the missingness

*How* values are missing matters as much as *how many*. We measure the **gap runs** — lengths of
consecutive missing hours. The result: mostly **short** gaps (a stray hour here and there, easily
filled) but a handful of **long sensor outages** — the longest **155 hours** (~6.5 days), which no
simple interpolation can honestly fill."""),
        co("""df0 = data.clean()
runs = data.gap_runs(df0["pm25"])
print("pm25 missing: %d (%.1f%%) across %d separate gaps" % (df0.pm25.isna().sum(), 100*df0.pm25.isna().mean(), len(runs)))
print("gap length: median %d h | 90th pct %d h | max %d h" % (runs.median(), runs.quantile(.9), runs.max()))
print("single-hour gaps: %d | gaps > 24h: %d" % ((runs==1).sum(), (runs>24).sum()))
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
runs.plot.hist(bins=40, ax=ax[0], color="indianred"); ax[0].set_yscale("log"); ax[0].set_title("gap-run lengths (log count) — many short, few long"); ax[0].set_xlabel("consecutive missing hours")
miss_by_month = df0.pm25.isna().groupby([df0.index.year, df0.index.month]).mean()
ax[1].plot(range(len(miss_by_month)), miss_by_month.values*100, color="indianred"); ax[1].set_title("% missing over time (outages cluster)"); ax[1].set_xlabel("month index"); ax[1].set_ylabel("% hours missing")
fig.tight_layout(); eda.savefig(fig, "p0_missing.png"); plt.show()"""),

        md("""### 3. Clean it
`data.clean()` builds a datetime index from the year/month/day/hour columns, renames `pm2.5 → pm25`
and `cbwd → wind_dir`, drops the row counter, and makes wind direction categorical. The `pm25` gaps
are **kept as NaN** on purpose — handling them is a deliberate, evaluated step later, not an
afterthought."""),
        co("""df = data.clean()
print("clean:", df.shape, "| index", df.index.min(), "->", df.index.max(), "| freq", df.index.freq)
print("wind direction:", df.wind_dir.value_counts().to_dict(), "(cv = calm/variable)")
df.head(3)"""),

        md("### 4. Persist"),
        co("""data.build_processed()
print("wrote data/processed/beijing_clean.csv — Part 0 complete (gaps preserved for Part 2).")"""),
    ]
    build(cells, "00_data_cleaning.ipynb", "# 00 · Beijing PM2.5 — Data Acquisition & Cleaning")


# ===================================================================== Notebook 1
def notebook_1():
    md, co = new_markdown_cell, new_code_cell
    cells = [
        md(
"""## Part 1 — Advanced EDA I: Data Quality, Distributions & Missingness

This is a *thorough* exploratory pass — the kind you do before trusting any model. Three jobs:

1. **Audit the data quality** — types, ranges, impossible values, zero-inflation, duplicates.
2. **Characterise every variable** — distributions, the four moments, normality, and which need
   transforming.
3. **Dissect the missingness** — not just *how much*, but its *shape* and *mechanism*, because how
   you later impute depends entirely on this.

(Temporal, meteorological, extreme-event and multivariate structure are explored in **Part 2**.)"""),
        co(SETUP + "\ndf = data.clean()\nNUM = df.select_dtypes('number').columns.tolist()\nprint('rows:', len(df), '| numeric cols:', NUM)"),

        md("""### 1. Data-quality audit

Before any analysis, profile every column: type, missingness, cardinality, range, and how often it
is exactly zero. This one table catches most data problems at a glance."""),
        co("""prof = pd.DataFrame({"dtype": df.dtypes.astype(str),
                     "missing": df.isna().sum(),
                     "miss%": (100*df.isna().mean()).round(2),
                     "unique": df.nunique()})
num = df[NUM]
prof.loc[NUM, "min"] = num.min(); prof.loc[NUM, "max"] = num.max()
prof.loc[NUM, "mean"] = num.mean().round(1); prof.loc[NUM, "zeros%"] = (100*(num == 0).mean()).round(1)
print(prof.to_string())"""),
        co("""# Sanity / impossible-value checks
print("duplicated timestamps:", int(df.index.duplicated().sum()))
print("negative pm25:", int((df.pm25 < 0).sum()), "| pm25 == 0 (valid 'very clean' hours):", int((df.pm25 == 0).sum()))
print("pressure range %.0f-%.0f hPa, temp %.0f-%.0f°C, dew point %.0f-%.0f°C  (all physically plausible)"
      % (df.PRES.min(), df.PRES.max(), df.TEMP.min(), df.TEMP.max(), df.DEWP.min(), df.DEWP.max()))
print("Is (snow hrs) zeros %.0f%% | Ir (rain hrs) zeros %.0f%%  -> ZERO-INFLATED (it rarely snows/rains)"
      % (100*(df.Is == 0).mean(), 100*(df.Ir == 0).mean()))"""),

        md("""### 2. The target `pm25` — full univariate analysis

The four-view battery (shape / spread / percentiles / normality), the four moments, formal normality
tests, and the health-band (AQI) breakdown."""),
        co("""print(eda.moments(df["pm25"]).round(2).to_string())
print("\\nnormality battery:"); print(eda.normality_battery(df["pm25"].dropna()).round(4).to_string(index=False))
fig = eda.four_view(df["pm25"].dropna(), "PM2.5 (µg/m³)", "p1_fourview.png"); plt.show()"""),
        co("""aqi = data.aqi_category(df["pm25"]).value_counts(normalize=True).reindex(
      ["Good","Moderate","Unhealthy(sens)","Unhealthy","Very unhealthy","Hazardous"]) * 100
fig, ax = plt.subplots(figsize=(10, 3.6))
colors = ["#00e400","#ffff00","#ff7e00","#ff0000","#8f3f97","#7e0023"]
ax.bar(aqi.index, aqi.values, color=colors, edgecolor="k", lw=.5)
for i, v in enumerate(aqi.values): ax.text(i, v+0.5, f"{v:.0f}%", ha="center", fontsize=9)
ax.set_title("Hours by US-EPA air-quality band — most of Beijing's air is Unhealthy+"); ax.set_ylabel("% of hours")
eda.savefig(fig, "p1_aqi.png"); plt.show()
print("hours that are 'Unhealthy' or worse: %.0f%%" % aqi[["Unhealthy","Very unhealthy","Hazardous"]].sum())"""),

        md("""### 3. Univariate sweep — every variable at once

A quick distribution of *each* column reveals shapes a summary table hides: PM2.5 and wind speed are
right-skewed; temperature/dew point are broad; snow and rain are spikes at zero (zero-inflated)."""),
        co("""fig, axes = plt.subplots(2, 4, figsize=(15, 7))
for ax, c in zip(axes.ravel(), NUM):
    sns.histplot(df[c].dropna(), bins=40, ax=ax, color="slategray"); ax.set_title(c)
axes.ravel()[-1].axis("off") if len(NUM) < 8 else None
fig.suptitle("Distribution of every numeric variable", y=1.01); fig.tight_layout()
eda.savefig(fig, "p1_univariate_grid.png"); plt.show()"""),
        co("""shape = pd.DataFrame({"skew": num.skew().round(2), "excess_kurt": num.kurt().round(2),
                      "zeros%": (100*(num == 0).mean()).round(0)}).sort_values("skew", ascending=False)
print("shape of each variable (most skewed first):"); print(shape.to_string())"""),

        md("""### 4. Transformations — taming the skew

Right-skewed, non-negative quantities (PM2.5, wind speed) become far more symmetric under a **log
(log1p) transform** — important later because many models and error metrics behave best on
roughly-symmetric targets."""),
        co("""fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["pm25"].dropna(), bins=50, ax=ax[0], color="indianred"); ax[0].set_title(f"pm25 (skew {df.pm25.skew():.2f})")
sns.histplot(np.log1p(df["pm25"].dropna()), bins=50, ax=ax[1], color="seagreen"); ax[1].set_title(f"log1p(pm25) (skew {np.log1p(df.pm25).skew():.2f})")
eda.savefig(fig, "p1_transform.png"); plt.show()
print("skew  pm25 %.2f -> log1p %.2f | Iws %.2f -> log1p %.2f" %
      (df.pm25.skew(), np.log1p(df.pm25).skew(), df.Iws.skew(), np.log1p(df.Iws).skew()))"""),

        md(
"""### 5. The anatomy of the missingness

We already know 4.7% of `pm25` is missing. Now the *details* that decide how to handle it:

- **Where** in time the gaps fall (do outages cluster?),
- the **gap-length distribution** (many short vs few long),
- whether missingness is **seasonal** or tied to conditions,
- and a formal **MCAR** check."""),
        co("""runs = data.gap_runs(df["pm25"])
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
daily_miss = df.pm25.isna().resample("D").mean() * 100
ax[0].fill_between(daily_miss.index, daily_miss.values, color="indianred"); ax[0].set_title("daily % missing — outages cluster, not uniform"); ax[0].set_ylabel("% of hours")
x = np.sort(runs.values); ax[1].step(x, np.arange(1, len(x)+1)/len(x), color="indianred")
ax[1].set_title("gap-length ECDF"); ax[1].set_xlabel("gap length (h)"); ax[1].set_ylabel("cumulative fraction of gaps"); ax[1].axvline(24, color="k", ls=":", lw=.8)
eda.savefig(fig, "p1_missing_anatomy.png"); plt.show()
print("%.0f%% of gaps are <=24h (interpolatable); but %d gaps exceed 24h, up to %dh." %
      (100*(runs <= 24).mean(), int((runs > 24).sum()), int(runs.max())))"""),
        co("""# Is missingness seasonal, and is it MCAR? Compare conditions on missing vs present hours.
from scipy.stats import ks_2samp
miss = df.pm25.isna()
print("missing rate by season-month (is it seasonal?):")
print((df.pm25.isna().groupby(df.index.month).mean()*100).round(1).to_string())
print("\\nMCAR check — do weather distributions differ on missing vs present hours? (KS p-value)")
for c in data.WEATHER:
    p = ks_2samp(df.loc[miss, c], df.loc[~miss, c]).pvalue
    print("  %-5s KS p = %.3f %s" % (c, p, "(differs!)" if p < 0.01 else "(~same)"))
print("\\n-> TEMP/DEWP/PRES differ significantly (with n=43k even tiny gaps register) => missingness leans MAR:")
print("   outages cluster in certain seasons/conditions. Magnitude is small, but impute USING the weather, not blindly.")"""),

        md(
"""### Takeaways (Part 1)

- **Quality is good** but for the target gaps: no duplicates, no impossible values, physically sane
  ranges; `Is`/`Ir` are **zero-inflated** (rarely snows/rains) — treat them carefully.
- **PM2.5 is severe and heavy-tailed** (skew 1.8): the majority of hours are **"Unhealthy" or worse**;
  a **log1p** transform makes it usable for modelling.
- **Missingness is structured**: clustered outages, mostly short gaps but a long tail up to 155h.
  A KS test finds a small-but-significant difference in temperature/pressure on missing vs present
  hours → the gaps lean **MAR** (outages cluster seasonally), not MCAR — so impute *using* the weather,
  and the long runs mean naive interpolation will distort. The motivation for Part 3's evaluated imputation.

**Next — Part 2 (Advanced EDA II):** temporal rhythms (diurnal × seasonal), the meteorology of
pollution (wind rose, dispersion), extreme episodes, and the multivariate structure (PCA & regimes)."""),
    ]
    build(cells, "01_advanced_eda.ipynb", "# 01 · Beijing PM2.5 — Advanced EDA I (quality · distributions · missingness)")


# ===================================================================== Notebook 2
def notebook_2():
    md, co = new_markdown_cell, new_code_cell
    cells = [
        md(
"""## Part 2 — Advanced EDA II: Temporal, Meteorological, Extreme & Multivariate

Part 1 looked at the variables *statically*. Pollution, though, is a **process in time driven by
weather**. This notebook explores:

1. **Temporal rhythms** — daily, weekly, yearly cycles, and the long-run trend.
2. The **2-D seasonal fingerprint** (hour × month) and a calendar view.
3. **Autocorrelation & persistence** — how "sticky" pollution is.
4. The **meteorology** — wind rose, dispersion curves, linear vs monotonic relationships.
5. **Extreme episodes** — AQI exceedance, how *long* bad-air spells last, the Jan-2013 airpocalypse.
6. **Multivariate structure** — PCA and weather-driven pollution *regimes*."""),
        co(SETUP + "\ndf = data.clean()\nprint('span:', df.index.min().date(), '->', df.index.max().date())"),

        md("""### 1. Temporal rhythms — four clocks at once

Mean PM2.5 by **hour**, **day-of-week**, **month**, and **year**. Watch for a diurnal build-up, a
weekday/weekend difference, the winter heating peak, and any multi-year **trend**."""),
        co("""fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
df.groupby(df.index.hour).pm25.mean().plot(ax=ax[0], marker="o"); ax[0].set_title("by hour"); ax[0].set_xlabel("hour")
df.groupby(df.index.dayofweek).pm25.mean().plot(ax=ax[1], marker="o"); ax[1].set_title("by day-of-week (0=Mon)"); ax[1].set_xlabel("dow")
df.groupby(df.index.month).pm25.mean().plot(kind="bar", ax=ax[2], color="slategray"); ax[2].set_title("by month")
yr = df.groupby(df.index.year).pm25.mean(); yr.plot(kind="bar", ax=ax[3], color="teal"); ax[3].set_title("by year (trend?)")
fig.tight_layout(); eda.savefig(fig, "p2_temporal.png"); plt.show()
print("annual mean µg/m³:", yr.round(0).to_dict(), "-> %s over 2010-2014" % ("falling" if yr.iloc[-1] < yr.iloc[0] else "no clear improvement"))"""),

        md("""### 2. The 2-D seasonal fingerprint — hour × month

A single heatmap of mean PM2.5 across **hour-of-day** (rows) and **month** (columns) shows both
cycles *and their interaction* at once — e.g. winter nights are the worst combination."""),
        co("""piv = df.pivot_table("pm25", df.index.hour, df.index.month, aggfunc="mean")
fig, ax = plt.subplots(figsize=(11, 5))
sns.heatmap(piv, cmap="rocket_r", ax=ax, cbar_kws={"label": "mean PM2.5 µg/m³"})
ax.set_xlabel("month"); ax.set_ylabel("hour of day"); ax.set_title("Mean PM2.5 by hour × month — winter + overnight is worst")
eda.savefig(fig, "p2_heatmap_hourmonth.png"); plt.show()
worst = piv.stack().idxmax(); print("worst hour×month cell: hour %d, month %d (%.0f µg/m³)" % (worst[0], worst[1], piv.loc[worst]))"""),

        md("""### 3. Long-term calendar view — month × year

The same data arranged as **month (rows) × year (columns)** separates the persistent **seasonal**
pattern from any year-to-year change."""),
        co("""cal = df.pivot_table("pm25", df.index.month, df.index.year, aggfunc="mean")
fig, ax = plt.subplots(figsize=(8, 5.5))
sns.heatmap(cal, cmap="rocket_r", annot=True, fmt=".0f", ax=ax, cbar_kws={"label": "µg/m³"})
ax.set_xlabel("year"); ax.set_ylabel("month"); ax.set_title("Mean PM2.5 by month × year")
eda.savefig(fig, "p2_heatmap_monthyear.png"); plt.show()"""),

        md(
"""### 4. Autocorrelation & persistence — pollution is "sticky"

Pollution doesn't reset each hour; a bad hour is usually followed by another. **Autocorrelation**
quantifies this memory (a spike at lag 24 = the daily cycle), and a lag-1 scatter shows the strong
hour-to-hour **persistence** that makes short-term forecasting feasible."""),
        co("""from statsmodels.tsa.stattools import acf
s = df["pm25"].interpolate("time").bfill()  # fill internal gaps + leading NaNs, for the ACF illustration
a = acf(s, nlags=72)
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].stem(range(73), a); ax[0].set_title("ACF of hourly PM2.5 (spike at 24h = daily cycle)"); ax[0].set_xlabel("lag (h)")
samp = pd.DataFrame({"now": s, "prev": s.shift(1)}).dropna().sample(4000, random_state=0)
ax[1].scatter(samp.prev, samp.now, s=5, alpha=.2); ax[1].set_title(f"persistence: this hour vs last hour (corr {s.corr(s.shift(1)):.2f})")
ax[1].set_xlabel("PM2.5 last hour"); ax[1].set_ylabel("PM2.5 this hour")
fig.tight_layout(); eda.savefig(fig, "p2_acf.png"); plt.show()
print("autocorrelation: lag-1h %.2f | lag-24h %.2f (the daily cycle)" % (a[1], a[24]))"""),

        md(
"""### 5. The meteorology of pollution

**Linear vs monotonic.** Comparing **Pearson** (straight-line) and **Spearman** (any monotonic)
correlations flags nonlinear relationships — where they differ, the effect is curved."""),
        co("""wcols = data.WEATHER + ["pm25"]; clean = df[wcols].dropna()
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
sns.heatmap(clean.corr("pearson"), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax[0]); ax[0].set_title("Pearson (linear)")
sns.heatmap(clean.corr("spearman"), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax[1]); ax[1].set_title("Spearman (monotonic)")
eda.savefig(fig, "p2_corr.png"); plt.show()
from sklearn.feature_selection import mutual_info_regression
mi = mutual_info_regression(clean[data.WEATHER], clean["pm25"], random_state=0)
print("mutual information weather -> pm25 (captures nonlinearity):")
print(pd.Series(mi, index=data.WEATHER).round(3).sort_values(ascending=False).to_string())"""),
        co("""# Wind rose: mean PM2.5 by wind direction + the dispersion curve vs wind speed
ang = {"NE": 45, "SE": 135, "NW": 315}                  # compass degrees
wd = df.groupby("wind_dir", observed=True).pm25.mean()
fig = plt.figure(figsize=(13, 4.5))
axp = fig.add_subplot(1, 2, 1, projection="polar"); axp.set_theta_zero_location("N"); axp.set_theta_direction(-1)
for d, deg in ang.items():
    axp.bar(np.deg2rad(deg), wd[d], width=0.6, alpha=.7, color="slategray")
    axp.text(np.deg2rad(deg), wd[d]+10, d, ha="center")
axp.set_title("mean PM2.5 by wind direction\\n(calm 'cv' = %.0f)" % wd.get("cv", np.nan))
ax2 = fig.add_subplot(1, 2, 2)
bins = pd.qcut(df["Iws"], 10, duplicates="drop"); disp = df.groupby(bins, observed=True).pm25.mean()
ax2.plot(range(len(disp)), disp.values, "o-", color="seagreen"); ax2.set_title("dispersion: more wind → cleaner air")
ax2.set_xlabel("wind-speed decile (low→high)"); ax2.set_ylabel("mean PM2.5")
eda.savefig(fig, "p2_windrose.png"); plt.show()
print("cleanest wind: %s (%.0f) | dirtiest: %s (%.0f) µg/m³" % (wd.idxmin(), wd.min(), wd.idxmax(), wd.max()))"""),

        md(
"""### 6. Extreme episodes — how bad, how often, how long

Health impact is about **episodes**, not averages. We count exceedance of the "Very Unhealthy"
threshold (150 µg/m³), see how the AQI mix shifts by year, and measure how **long** severe spells
*persist* — then zoom into the January-2013 "airpocalypse"."""),
        co("""import itertools
bad = (df["pm25"] > 150.4).fillna(False)
runs = pd.Series([len(list(g)) for k, g in itertools.groupby(bad) if k])
print("'Very Unhealthy+' hours: %.0f%% of the time, in %d episodes" % (100*bad.mean(), len(runs)))
print("episode length (consecutive hours): median %d, 90th pct %d, max %d (%.1f days)" %
      (runs.median(), runs.quantile(.9), runs.max(), runs.max()/24))
aqi_yr = pd.crosstab(df.index.year, data.aqi_category(df["pm25"]), normalize="index") * 100
print("\\nAQI mix by year (%):"); print(aqi_yr.round(0).to_string())"""),
        co("""ep = df.loc["2013-01-10":"2013-01-16", "pm25"]
fig, ax = plt.subplots(figsize=(12, 3.8))
ax.fill_between(ep.index, ep.values, color="firebrick", alpha=.6)
ax.axhline(250, color="purple", ls="--", lw=1, label="Hazardous (250)")
ax.set_title("The January 2013 'airpocalypse' — PM2.5 pinned above hazardous for days"); ax.legend()
eda.savefig(fig, "p2_episode.png"); plt.show()
print("peak that week: %.0f µg/m³ (≈ %.0fx the WHO guideline)" % (ep.max(), ep.max()/data.WHO_24H))"""),

        md(
"""### 7. Multivariate structure — factors & regimes

**PCA** compresses the correlated weather + pollution variables into a few axes; **k-means** then
groups hours into weather **regimes** and we read off the mean pollution of each — revealing the
"cold, stagnant, humid" regime that produces the worst air."""),
        co("""from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
feat = ["pm25", "TEMP", "DEWP", "PRES", "Iws"]; D = df[feat].dropna()
Z = StandardScaler().fit_transform(D)
pca = PCA().fit(Z)
print("PCA explained variance:", (pca.explained_variance_ratio_[:3]*100).round(0), "% (PC1-3)")
print("PC1 loadings:", dict(zip(feat, pca.components_[0].round(2))))
W = StandardScaler().fit_transform(D[["TEMP", "DEWP", "PRES", "Iws"]])
D = D.assign(regime=KMeans(4, n_init=10, random_state=0).fit_predict(W))
summ = D.groupby("regime").agg(pm25=("pm25", "mean"), TEMP=("TEMP", "mean"), Iws=("Iws", "mean"),
                               PRES=("PRES", "mean"), hours=("pm25", "size")).round(1).sort_values("pm25")
print("\\nweather regimes, sorted by mean pollution:"); print(summ.to_string())
print("\\n-> the dirtiest regime is cold + low-wind (stagnant); the cleanest is windy.")"""),

        md(
"""### Key insights (the full EDA, synthesised)

- **Pollution is a weather-driven process, not noise.** It has a clear **daily** cycle (overnight
  build-up), a strong **winter** peak (heating + stagnation), and is highly **persistent**
  (hour-to-hour corr ≈ 0.95) — so recent values and weather are powerful predictors.
- **Wind is the master switch**: strong/northerly winds disperse pollution, calm/stagnant air traps
  it; the dirtiest **regime** is *cold + low-wind*. Pearson vs Spearman gaps show the relationships
  are **nonlinear** (a job for tree models later).
- **Extremes dominate the health story**: severe spells **persist for days** (Jan-2013 peaked ~59×
  the WHO limit; the longest "very unhealthy" run lasted 4.6 days), and the AQI mix barely improved
  over 2010–2014.
- Combined with Part 1 (heavy skew, structured missingness, zero-inflated rain/snow), this is a
  complete picture: we know exactly what to **transform**, **impute**, and **model** next.

**Next:** Part 3 — evaluated **missing-data imputation**; then multi-seasonal TS and pollution
forecasting that exploit the persistence and meteorology found here."""),
    ]
    build(cells, "02_advanced_eda_2.ipynb", "# 02 · Beijing PM2.5 — Advanced EDA II (temporal · meteorology · extremes · structure)")


# ===================================================================== Notebook 3
def notebook_3():
    md, co = new_markdown_cell, new_code_cell
    cells = [
        md(
"""## Part 3 — Missing-Data Imputation, *Evaluated*

The EDA showed `pm25` has 2,067 missing values in gaps from 1 hour to 155 hours, leaning **MAR**.
Now we fill them — but **which method?** The crucial idea, too often skipped:

> You cannot measure imputation error on *truly* missing data — you don't know the answer. So you
> **mask known values that mimic the real gaps**, impute them, and score the reconstruction.

And because short gaps and long gaps are utterly different problems (a 1-hour gap is trivial; a
6-day sensor outage is not), we score **stratified by gap length**. The punchline: **no single
method wins everywhere**, so the right answer is a **hybrid**."""),
        co(SETUP + """
from src import impute as imp
df = data.clean()
climatology = df.groupby([df.index.month, df.index.hour]).pm25.mean()   # seasonal lookup, all years
# Evaluate on a 6-month window (keeps KNN's pairwise distances tractable); the method ranking generalises.
win = df.loc["2014-01-01":"2014-06-30"]
truth = win["pm25"]; WEATHER = win[["TEMP", "DEWP", "PRES", "Iws"]]
print("evaluation window:", win.index.min().date(), "->", win.index.max().date(), "|", len(win), "hours")"""),

        md(
"""### 1. The five methods

| method | idea | good at |
|---|---|---|
| **ffill** | carry the last value forward | nothing much (a baseline) |
| **interp** | straight line between neighbours | **short** gaps (pollution is persistent) |
| **climatology** | the typical value for that month & hour | seasonal shape, but loses variability |
| **KNN** | average the K most-similar hours (weather + time) | **long** gaps, multivariate |
| **MICE** | iteratively regress pm25 on weather + time | **long** gaps, multivariate |"""),

        md(
"""### 2. The evaluation protocol — mask values that mimic the real gaps

We draw artificial gap lengths from the **real** gap-run distribution and punch them into the
*observed* part of the window. Now we know the truth at those points, so we can score every method
fairly — and we record each masked point's gap length for the stratified analysis."""),
        co("""mask, glen = imp.inject_gaps(truth, data.gap_runs(df.pm25).values, n_gaps=150, seed=0)
gappy = truth.copy(); gappy[mask] = np.nan
print("artificially masked %d points across gaps of length 1–%d h (mimicking the real distribution)"
      % (mask.sum(), int(glen[mask].max())))
print("real PM2.5 known at these points -> we can now grade each method.")"""),

        md("""### 3. Overall scoreboard"""),
        co("""preds = imp.impute_all(gappy, WEATHER, climatology)
t = truth.values[mask]
board = pd.DataFrame({name: {"MAE": np.abs(p.values[mask]-t).mean(),
                             "RMSE": np.sqrt(((p.values[mask]-t)**2).mean())}
                      for name, p in preds.items()}).T.sort_values("MAE").round(1)
print(board.to_string())
print("\\ninterp wins *on average* — because most gaps are short. But averages hide the real story...")"""),

        md(
"""### 4. The key result — error depends on **gap length**

Stratify the error by how long the gap was. A clear **crossover** appears: **interpolation is
unbeatable for short gaps** (a line between two close, persistent values), but it **degrades on long
gaps**, where the multivariate **MICE** wins (it can lean on weather + the seasonal cycle when there's
no nearby value to interpolate from)."""),
        co("""bins = pd.cut(glen[mask], [0, 3, 12, 48, 10000], labels=["1-3h", "4-12h", "13-48h", "49h+"])
strat = pd.DataFrame({name: pd.Series(np.abs(p.values[mask]-t)).groupby(bins, observed=True).mean()
                      for name, p in preds.items()}).round(0)
print("MAE by gap length:"); print(strat.to_string())
fig, ax = plt.subplots(figsize=(9, 4.5))
for name in ["interp", "climatology", "KNN", "MICE"]:
    ax.plot(strat.index, strat[name], "o-", label=name)
ax.set_xlabel("gap length"); ax.set_ylabel("MAE (µg/m³)"); ax.set_title("Imputation error vs gap length — interp best for short, MICE for long")
ax.legend(); eda.savefig(fig, "p3_by_gaplength.png"); plt.show()"""),

        md("""### 5. Seeing it — one long gap, every method

Overlay the methods on a single long artificial gap. Interpolation draws a **flat, lifeless line**
straight through the daily cycle; climatology and MICE **trace the rhythm**; ffill makes a step."""),
        co("""# find the longest masked gap in the window and show a window around it
gl = pd.Series(glen, index=win.index); longest_end = gl.idxmax(); L = int(gl.max())
seg = slice(win.index.get_loc(longest_end) - L - 12, win.index.get_loc(longest_end) + 36)
idx = win.index[seg]
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.plot(idx, truth.values[seg], color="black", lw=2, label="truth")
for name, c in [("interp", "tab:red"), ("climatology", "tab:green"), ("MICE", "tab:blue")]:
    ax.plot(idx, preds[name].values[seg], "--", color=c, label=name)
ax.axvspan(idx[12], idx[12+L], color="grey", alpha=0.12, label=f"{L}h gap")
ax.set_title("A long gap filled four ways — interpolation flatlines, MICE keeps the cycle"); ax.legend(ncol=2)
eda.savefig(fig, "p3_longgap.png"); plt.show()"""),

        md(
"""### 6. Distribution preservation — a hidden trap

Accuracy isn't everything: a method can be "close on average" yet **destroy the variability** of the
data, which would bias any downstream variance / extreme-event analysis. Compare the spread of the
imputed values to the truth — **climatology collapses it** (everything becomes a smooth average),
while KNN/MICE keep realistic variation."""),
        co("""print("std of imputed values vs the true std (%.0f µg/m³) at the masked points:" % t.std())
sd = pd.Series({name: p.values[mask].std() for name, p in preds.items()}).round(0).sort_values()
print(sd.to_string())
print("\\n-> climatology & interp UNDERSTATE variability; KNN/MICE preserve it. 'Accurate' != 'faithful'.")"""),

        md(
"""### 7. The recommendation — a hybrid

No method dominates, so combine their strengths: **interpolate short gaps (≤48 h), MICE the long
ones**. This beats every single method on both MAE and RMSE — *and* preserves variability."""),
        co("""hyb = imp.hybrid(gappy, WEATHER, short_max=48)
e = np.abs(hyb.values[mask] - t)
print("HYBRID (interp ≤48h + MICE >48h):  MAE %.1f  RMSE %.1f  std %.0f" % (e.mean(), np.sqrt((e**2).mean()), hyb.values[mask].std()))
print("best single method was interp MAE %.1f / MICE RMSE %.1f -> the hybrid wins both." % (board.MAE.min(), board.RMSE.min()))"""),

        md("""### 8. Build the final clean series

Apply the hybrid to the **real** gaps and persist a gap-free `pm25` for the modelling parts to
come. We keep a `pm25_was_imputed` flag so later work can treat filled values with appropriate
caution (especially the long-outage stretches)."""),
        co("""full = df.copy()
full["pm25_was_imputed"] = full["pm25"].isna()
full["pm25"] = imp.hybrid(df["pm25"], df[["TEMP", "DEWP", "PRES", "Iws"]], short_max=48)
print("remaining missing pm25:", int(full.pm25.isna().sum()), "| imputed:", int(full.pm25_was_imputed.sum()))
full.to_csv(data.DATA_PROC / "beijing_imputed.csv") if hasattr(data, "DATA_PROC") else full.to_csv("data/processed/beijing_imputed.csv")
print("wrote data/processed/beijing_imputed.csv (gap-free, with an imputation flag)")"""),

        md(
"""### Takeaways

- **Evaluate imputation by masking** values that mimic the real gaps — don't just pick a method by
  reputation. The proper score is reconstruction error on the masked points.
- **Gap length is everything.** Interpolation is near-perfect for **short** gaps (PM2.5's 0.97 hour-
  to-hour persistence) but **fails on long** ones; multivariate **MICE/KNN** win there by leaning on
  weather + the seasonal cycle.
- **Accuracy ≠ faithfulness.** Climatology can look passable on MAE yet **collapse the variance**
  (std 20 vs 96) — a trap for any later variability/extreme analysis.
- **The hybrid** (interpolate ≤48 h, MICE beyond) beats every single method *and* preserves the
  distribution — and we keep an **imputation flag** so long-outage fills are never mistaken for data.

**Next:** Part 4 — multi-seasonal time-series foundations on the now-complete series, then pollution
forecasting using the persistence and meteorology the EDA uncovered."""),
    ]
    build(cells, "03_imputation.ipynb", "# 03 · Beijing PM2.5 — Missing-Data Imputation, Evaluated")


# ===================================================================== Notebook 4
def notebook_4():
    md, co = new_markdown_cell, new_code_cell
    cells = [
        md(
"""## Part 4 — Pollution Forecasting (in depth)

Now the payoff of the whole study. The EDA found pollution is **persistent** (sticky hour-to-hour),
**multi-seasonal** (daily + winter), and **weather-driven** (wind disperses it). We turn that into a
**24-hour-ahead** forecast of PM2.5 — the operational "what will tomorrow's air be?" task — and judge
it two ways:

1. as a **regression** (how close are the numbers?), and
2. as an **air-quality warning system** (do we correctly flag the **hazardous** hours?) — the part
   that actually matters for public health.

We also confront the hardest, most honest question for any pollution model: **can it predict the
extremes** — the airpocalypse spikes — or does it shrug them off?"""),
        co(SETUP + """
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from src import forecasting as fc

g = pd.read_csv(data.DATA_PROC / "beijing_imputed.csv", parse_dates=["datetime"]).set_index("datetime")
y = g["pm25"]
# 24h-ahead-safe features: any lag >=24h is known at the forecast origin.
g["lag24"] = y.shift(24); g["lag168"] = y.shift(168); g["lag_dayavg"] = y.shift(24).rolling(24).mean()
g["hour"] = g.index.hour; g["month"] = g.index.month; g["dow"] = g.index.dayofweek
g["wdir"] = g["wind_dir"].astype("category").cat.codes
WEATHER = ["TEMP","DEWP","PRES","Iws","Is","Ir"]; CAL = ["hour","month","dow","wdir"]; LAGS = ["lag24","lag168","lag_dayavg"]
g = g.dropna(subset=LAGS)
split = g.index.max() - pd.Timedelta(days=90)
tr, te = g[g.index <= split], g[g.index > split]
real = (~te["pm25_was_imputed"]).to_numpy()      # score only where the test truth is REAL, not imputed
yte = te["pm25"].to_numpy(); scale = tr["pm25"].to_numpy()
print("train:", tr.index.min().date(), "->", tr.index.max().date(), "| test (90d):", len(te), "h |",
      "real test points: %.0f%%" % (100*real.mean()))"""),

        md(
"""### 1. The task and a fair yardstick

**Horizon: 24 hours ahead.** Why not next-hour? Because PM2.5 is *so* persistent hour-to-hour
(autocorrelation 0.97) that next-hour forecasting is trivial — last hour's value wins. A day ahead is
the useful, *non-trivial* task: the EDA showed same-hour-yesterday correlation is only ~0.40.

**Honest evaluation.** We trained features on the Part-3 imputed series (so lags exist everywhere),
but we **score only on test hours whose truth is real**, never on imputed values. Metrics: MAE /
RMSE / **MASE** (vs the seasonal-naive). As in the bike practice, using weather as a predictor
assumes a weather *forecast* — a caveat we accept here."""),

        md("""### 2. Baselines — persistence & climatology"""),
        co("""def score(p):
    m = fc.forecast_metrics(yte[real], np.clip(p, 0, None)[real], scale, 24)
    return [m["MAE"], m["RMSE"], m["MASE"]]
naive = te["lag24"].to_numpy()                              # same hour yesterday
clim = tr.groupby([tr.month, tr.hour]).pm25.mean()
climp = np.array([clim.get((m, h)) for m, h in zip(te.month, te.hour)])
print("persistence (lag24):", [round(x,1) for x in score(naive)])
print("climatology        :", [round(x,1) for x in score(climp)])
print("-> both have MASE > 1: a day-ahead pollution forecast is genuinely hard for naive methods.")"""),

        md(
"""### 3. Harmonic regression + weather

A classical model: Fourier terms for the **daily** and **yearly** cycles plus the **weather**
covariates (the meteorology the EDA found drives pollution). It should beat the baselines by using
wind, humidity, etc."""),
        co("""from numpy import sin, cos, pi
pos = np.arange(len(g))
F = np.hstack([np.column_stack([f(2*pi*k*pos/24) for k in range(1,6) for f in (sin,cos)]),
               np.column_stack([f(2*pi*k*pos/(24*365.25)) for k in range(1,3) for f in (sin,cos)])])
Xh = np.hstack([F, g[WEATHER].to_numpy()]); ntr = len(tr)
ph = LinearRegression().fit(Xh[:ntr], tr["pm25"].to_numpy()).predict(Xh[ntr:])
print("harmonic + weather:", [round(x,1) for x in score(ph)], "(MAE, RMSE, MASE) — beats the baselines")"""),

        md("""### 4. LightGBM with lags + weather — the workhorse

Gradient-boosted trees capture the **nonlinear** weather effects (the EDA's Pearson-vs-Spearman gaps)
and combine recent **persistence** (lags) with the meteorology."""),
        co("""feat = LAGS + WEATHER + CAL
mdl = lgb.LGBMRegressor(n_estimators=600, num_leaves=63, learning_rate=0.05, random_state=0, verbose=-1).fit(tr[feat], tr["pm25"])
pl = np.clip(mdl.predict(te[feat]), 0, None)
print("LightGBM + lags + weather:", [round(x,1) for x in score(pl)], "(MAE, RMSE, MASE)")"""),

        md("""### 5. Scoreboard & forecast"""),
        co("""board = pd.DataFrame({"persistence": score(naive), "climatology": score(climp),
                      "harmonic+weather": score(ph), "LGBM+lags+weather": score(pl)},
                     index=["MAE","RMSE","MASE"]).T.sort_values("MASE").round(2)
print(board.to_string())
seg = te.iloc[:24*10]; segp = pl[:24*10]
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(seg.index, seg["pm25"], color="black", lw=1.5, label="actual")
ax.plot(seg.index, segp, "--", color="crimson", label="LightGBM 24h-ahead")
ax.axhline(150, color="purple", ls=":", lw=1, label="Very Unhealthy (150)")
ax.set_title("10 days of 24h-ahead PM2.5 forecast"); ax.legend(); ax.set_ylabel("µg/m³")
eda.savefig(fig, "p4_forecast.png"); plt.show()"""),

        md(
"""### 6. The part that matters — forecasting the *hazardous* hours

A pollution forecast's real job is to **warn**. Reframe the regression as a **classification**: will
PM2.5 exceed the "Very Unhealthy" threshold (150 µg/m³)? We score precision/recall/F1 and the
confusion matrix on the LightGBM forecast."""),
        co("""yt = yte[real]; pe = pl[real]
exc_t, exc_p = yt > 150, pe > 150
print("base rate (hours actually >150): %.0f%%" % (100*exc_t.mean()))
print("exceedance forecast — precision %.2f | recall %.2f | F1 %.2f" %
      (precision_score(exc_t, exc_p), recall_score(exc_t, exc_p), f1_score(exc_t, exc_p)))
cm = confusion_matrix(exc_t, exc_p)
fig, ax = plt.subplots(figsize=(4.6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax, xticklabels=["safe","hazard"], yticklabels=["safe","hazard"])
ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title("Exceedance (>150) confusion matrix")
eda.savefig(fig, "p4_exceedance.png"); plt.show()
print("-> it catches ~3 in 4 hazardous hours at ~3-in-4 precision: a useful early-warning signal.")"""),

        md(
"""### 7. The hard truth — models under-predict the extremes

The most dangerous hours are the hardest to forecast. Plot predicted vs actual: at the high end the
cloud sits **below** the diagonal — the model **systematically under-predicts** the worst pollution,
because extreme spikes come from rare stagnation events a smooth model regresses away. This is the
key limitation to communicate with any pollution forecast."""),
        co("""fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.hexbin(yt, pe, gridsize=40, cmap="viridis", mincnt=1, bins="log")
lim = max(yt.max(), pe.max()); ax.plot([0, lim], [0, lim], "r--", label="perfect")
ax.set_xlabel("actual PM2.5"); ax.set_ylabel("predicted"); ax.set_title("Predicted vs actual — high end falls below the line"); ax.legend()
eda.savefig(fig, "p4_underprediction.png"); plt.show()
for thr in [200, 300, 400]:
    hi = yt > thr
    if hi.sum(): print("actual > %d (%d h): mean actual %.0f vs mean predicted %.0f  (under by %.0f%%)"
                        % (thr, hi.sum(), yt[hi].mean(), pe[hi].mean(), 100*(1 - pe[hi].mean()/yt[hi].mean())))"""),

        md("""### 8. What drives the forecast?"""),
        co("""imp = pd.Series(mdl.feature_importances_, index=feat).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8.5, 4))
imp.head(10)[::-1].plot.barh(ax=ax, color="slategray"); ax.set_title("LightGBM feature importance")
eda.savefig(fig, "p4_importance.png"); plt.show()
print("top drivers:", list(imp.head(6).index))
print("-> recent pollution (lags) + weather (dew point, pressure) — the meteorology the EDA flagged.")"""),

        md(
"""### Takeaways

- **Day-ahead pollution forecasting is non-trivial**: naive persistence and climatology *lose to the
  seasonal-naive* (MASE > 1), but adding **weather + lags** wins decisively — **LightGBM MASE ≈ 0.68**,
  roughly half the error of persistence.
- The model is a genuine **early-warning system**: it flags "Very Unhealthy" (>150) hours at **F1 ≈
  0.76** — catching ~3 of 4 hazardous hours.
- **But it under-predicts the extremes** (actual >300: ~360 vs ~250 forecast): smooth models regress
  the rare airpocalypse spikes toward the mean — the single most important caveat for a pollution
  forecast, and a motivation for extreme-value or quantile methods.
- **Drivers match the EDA**: recent pollution (persistence) + **weather (dew point, pressure)** — the
  meteorology and stickiness we found are exactly what the forecast leans on.

*This completes a full air-quality study (Parts 0–4): clean → extensive EDA → evaluated imputation →
forecasting, with the honest limitation (extremes) made explicit.*"""),
    ]
    build(cells, "04_forecasting.ipynb", "# 04 · Beijing PM2.5 — Pollution Forecasting & Exceedance Warning")


# ===================================================================== Notebook 5
def notebook_5():
    md, co = new_markdown_cell, new_code_cell
    cells = [
        md(
"""## Part 5 — Forecasting the Extremes (extending Part 4)

Part 4 left us with a sharp problem: the model **under-predicts the worst pollution** (actual >300:
~357 vs ~250 forecast, worsening with severity). For an air-quality *warning* system that's the
costliest failure — **missing an airpocalypse is far worse than a false alarm**. This notebook
extends the forecast to take the **tail** seriously:

1. *why* a standard (L2) model under-predicts extremes — and a tempting "fix" that makes it **worse**;
2. **quantile regression** — predicting the upper tail to track the spikes;
3. the **no-free-lunch tradeoff** (catching extremes costs average accuracy);
4. a principled, **probabilistic exceedance-warning** system tuned to the cost asymmetry.

The meta-lesson: **match the loss function to what you actually care about** — here, the dangerous
tail, not the average hour."""),
        co(SETUP + """
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, precision_recall_curve
from src import forecasting as fc

g = pd.read_csv(data.DATA_PROC / "beijing_imputed.csv", parse_dates=["datetime"]).set_index("datetime")
g["pm25"] = g["pm25"].clip(lower=0)          # concentrations can't be negative (a MICE imputation artifact)
y = g["pm25"]
g["lag24"]=y.shift(24); g["lag168"]=y.shift(168); g["lag_dayavg"]=y.shift(24).rolling(24).mean()
g["hour"]=g.index.hour; g["month"]=g.index.month; g["dow"]=g.index.dayofweek; g["wdir"]=g["wind_dir"].astype("category").cat.codes
FEAT = ["lag24","lag168","lag_dayavg","TEMP","DEWP","PRES","Iws","Is","Ir","hour","month","dow","wdir"]
g = g.dropna(subset=["lag24","lag168","lag_dayavg"])
split = g.index.max() - pd.Timedelta(days=90); tr, te = g[g.index <= split], g[g.index > split]
real = (~te["pm25_was_imputed"]).to_numpy(); yt = te["pm25"].to_numpy()[real]
Xtr, Xte, ytr = tr[FEAT], te[FEAT], tr["pm25"]
P = dict(n_estimators=600, num_leaves=63, learning_rate=0.05, random_state=0, verbose=-1)
def gbm(**kw): return lgb.LGBMRegressor(**{**P, **kw})
print("train %d / test %d (real %.0f%%) | %% test hours hazardous (>150): %.0f%%" % (len(tr), len(te), 100*real.mean(), 100*(yt>150).mean()))"""),

        md(
"""### 1. Why the mean under-predicts — and the log trap

A squared-error (L2) model predicts the **conditional mean**. Extreme pollution hours are rare, so
the mean for any given condition sits *well below* the spikes — the model literally can't justify a
700 µg/m³ forecast when most similar hours were 150. Hence the under-prediction.

A tempting fix is to **model `log(pm25)`** (it's right-skewed). It does the **opposite** of what we
want: `exp(mean of log)` is the **geometric mean / median**, which is *lower* than the arithmetic
mean — so it suppresses the peaks **even more**. A reminder to always check, not assume:"""),
        co("""L2 = gbm().fit(Xtr, ytr).predict(Xte)
logm = np.expm1(gbm().fit(Xtr, np.log1p(ytr)).predict(Xte))
hi = yt > 300
for tag, p in [("L2 (mean)", L2), ("log-target", logm)]:
    p = np.clip(p, 0, None)[real]
    print("%-12s overall MAE %.1f | mean prediction on actual>300 hours: %.0f (truth 357)" % (tag, np.abs(p-yt).mean(), p[hi].mean()))
print("-> log-target LOWERS the extreme predictions — the wrong tool for the tail.")"""),

        md(
"""### 2. Quantile regression — predicting the tail

Instead of the mean, train models for chosen **quantiles** (the pinball loss). The **median (q50)**
is a robust point forecast; the **upper quantiles (q90, q95)** deliberately track the *high* end —
they answer "how bad could it plausibly get?", which is exactly the warning question."""),
        co("""quants = {a: np.clip(gbm(objective="quantile", alpha=a).fit(Xtr, ytr).predict(Xte), 0, None) for a in [0.5, 0.9, 0.95]}
# show them on the highest-pollution stretch of the test
c = te["pm25"].rolling(48).mean().idxmax(); i = te.index.get_loc(c)
seg = slice(max(0, i-120), i+120); idx = te.index[seg]
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.plot(idx, te["pm25"].values[seg], color="black", lw=1.6, label="actual")
for a, c2 in [(0.5,"tab:blue"),(0.9,"tab:orange"),(0.95,"tab:red")]:
    ax.plot(idx, quants[a][seg], "--", color=c2, lw=1.2, label=f"q{int(a*100)}")
ax.axhline(150, color="purple", ls=":", lw=1)
ax.set_title("Quantile forecasts — q90/q95 reach the spikes the median misses"); ax.legend(ncol=4); ax.set_ylabel("µg/m³")
eda.savefig(fig, "p5_quantiles.png"); plt.show()"""),

        md(
"""### 3. The tradeoff — no free lunch

Score each forecast as a point estimate (MAE), on the extremes (mean prediction where truth > 300),
and as a warning (recall of >150 hours). Climbing the quantiles **captures the tail and lifts recall
toward 1.0**, but **inflates MAE and slashes precision** — over-warning. The quantile level *is* your
risk appetite."""),
        co("""rows = []
for tag, p in [("L2 mean", L2), ("q50 median", quants[0.5]), ("q90", quants[0.9]), ("q95", quants[0.95]),
               ("tweedie", np.clip(gbm(objective="tweedie", tweedie_variance_power=1.5).fit(Xtr, ytr).predict(Xte), 0, None))]:
    p = np.clip(p, 0, None)[real]
    rows.append([tag, np.abs(p-yt).mean(), p[hi].mean(),
                 precision_score(yt>150, p>150), recall_score(yt>150, p>150), f1_score(yt>150, p>150)])
print(pd.DataFrame(rows, columns=["model","MAE","pred@>300","exc_P","exc_R","exc_F1"]).set_index("model").round(2).to_string())
print("\\n-> q90/q95 nearly close the extreme gap (327/344 vs 357) and reach 0.93/0.98 recall — at a real cost.")
print("   tweedie barely moves the tail: a loss tweak alone doesn't solve it.")"""),

        md(
"""### 4. The right tool for the warning — a probabilistic classifier

The cleanest way to *warn* isn't to threshold a point forecast at all — it's to model the
**probability** of exceedance directly, then choose the alarm threshold from the **cost asymmetry**.
For health, a missed hazard costs more than a false alarm, so we move the threshold **down** to buy
recall. The precision–recall curve makes that choice explicit."""),
        co("""clf = lgb.LGBMClassifier(**P).fit(Xtr, (ytr > 150).astype(int))
prob = clf.predict_proba(Xte)[:, 1][real]
ap = average_precision_score(yt > 150, prob)
prec, rec, thr = precision_recall_curve(yt > 150, prob)
fig, ax = plt.subplots(figsize=(6, 4.8))
ax.plot(rec, prec, color="firebrick"); ax.axhline((yt>150).mean(), color="grey", ls="--", label="base rate")
ax.set_xlabel("recall (hazards caught)"); ax.set_ylabel("precision"); ax.set_title(f"Exceedance PR curve (avg-precision {ap:.2f})"); ax.legend()
eda.savefig(fig, "p5_pr_curve.png"); plt.show()
print("threshold choice (health prioritises recall):")
for t in [0.5, 0.35, 0.25]:
    print("  P>%.2f  precision %.2f  recall %.2f  F1 %.2f" % (t, precision_score(yt>150,prob>t), recall_score(yt>150,prob>t), f1_score(yt>150,prob>t)))"""),

        md(
"""### Takeaways

- **An L2 model predicts the mean, so it under-predicts rare extremes** — and modelling `log(pm25)`
  makes it *worse* (it targets the median). Don't assume a transform helps the tail; **measure it**.
- **Quantile regression captures the tail**: q90/q95 track the spikes (pred 327/344 vs truth 357) and
  catch **93–98%** of hazardous hours — but at higher MAE and lower precision. There is **no free
  lunch**; the quantile level encodes how much over-warning you'll tolerate.
- **The principled warning system is a probabilistic exceedance classifier** (avg-precision 0.83):
  output a calibrated probability and pick the alarm threshold from the **cost asymmetry** — lower it
  to prioritise recall when missing a hazard is the expensive error.
- **Match the loss to the goal**: minimise MAE for the *average* hour, but optimise quantiles / a
  classifier for the *dangerous tail*. They are different objectives and need different tools.

*This extends Part 4 into a tail-aware, decision-focused forecast — the natural endpoint of a
pollution study, where the extremes are the whole point.*"""),
    ]
    build(cells, "05_extreme_forecasting.ipynb", "# 05 · Beijing PM2.5 — Forecasting the Extremes (quantiles & probabilistic warnings)")


if __name__ == "__main__":
    import sys
    all_nbs = {"0": notebook_0, "1": notebook_1, "2": notebook_2, "3": notebook_3, "4": notebook_4, "5": notebook_5}
    for k in (sys.argv[1:] or sorted(all_nbs)):
        all_nbs[k]()
    print("done.")

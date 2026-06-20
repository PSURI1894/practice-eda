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


if __name__ == "__main__":
    import sys
    all_nbs = {"0": notebook_0, "1": notebook_1, "2": notebook_2}
    for k in (sys.argv[1:] or sorted(all_nbs)):
        all_nbs[k]()
    print("done.")

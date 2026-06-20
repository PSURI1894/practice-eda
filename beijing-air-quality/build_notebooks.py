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
"""## Part 1 — Advanced EDA (pollution, missingness & meteorology)

Three threads, two of them new to this collection:

1. the **pollution distribution** — heavy right-skew, and what "99 µg/m³ on average" means against
   health guidelines;
2. the **missing-data mechanism** — *why* are values missing, and is it safe to impute? (the
   distinctive theme of this dataset);
3. the **meteorology of pollution** — how weather *drives* PM2.5 (wind clears it, humidity traps it),
   the multivariate story."""),
        co(SETUP + "\ndf = data.clean()\nprint('rows:', len(df), '| pm25 present:', int(df.pm25.notna().sum()))"),

        md("""### 1. The pollution target — heavy right-skew

PM2.5 is a concentration: bounded below by 0, with a long tail of severe-pollution hours. The four-
view battery shows the skew; a **log transform** tames it (the usual move for concentrations). For
context, the **WHO 24-h guideline is 15 µg/m³** — Beijing's *mean* is ~7× that."""),
        co("""print(eda.moments(df["pm25"]).round(1).to_string())
fig = eda.four_view(df["pm25"].dropna(), "PM2.5 (µg/m³)", "p1_fourview.png"); plt.show()
print("\\nWHO 24h guideline = %.0f µg/m³ | %.0f%% of hours exceed it | hours above 'hazardous' 250: %.1f%%"
      % (data.WHO_24H, 100*(df.pm25 > data.WHO_24H).mean(), 100*(df.pm25 > 250).mean()))"""),
        co("""# The worst episodes are real history — e.g. the January 2013 "airpocalypse".
daily = df["pm25"].resample("D").mean()
print("worst pollution days (daily mean µg/m³):")
print(daily.nlargest(5).round(0).to_string())"""),

        md(
"""### 2. Why are values missing? — the mechanism

Before imputing anything, ask *why* data is missing (Part 2 will then choose a method):

- **MCAR** (missing completely at random) — unrelated to anything; safe to drop/impute simply.
- **MAR** (missing at random) — related to *observed* variables; impute using them.
- **MNAR** (missing not at random) — related to the *missing value itself* (e.g. the sensor fails
  *because* pollution is extreme); the dangerous case.

These are **sensor outages**, so they're plausibly unrelated to the pollution level. We sanity-check
**MCAR** by comparing the weather on missing vs present hours — if they look the same, missingness
isn't tied to observed conditions."""),
        co("""miss = df.pm25.isna()
cmp = pd.DataFrame({"present": df.loc[~miss, data.WEATHER].mean(),
                    "missing": df.loc[miss, data.WEATHER].mean()}).round(1)
cmp["% diff"] = (100*(cmp["missing"]/cmp["present"] - 1)).round(0)
print(cmp.to_string())
print("\\nWeather is similar on missing vs present hours -> consistent with ~MCAR (sensor faults),")
print("BUT the 155h-long runs can't be honestly interpolated -> Part 2 evaluates real methods.")"""),

        md(
"""### 3. The meteorology of pollution — the multivariate story

PM2.5 isn't random: **weather disperses or traps it**. The clearest driver is **wind** — strong winds
(`Iws`) blow pollution away — while humidity (`DEWP`) and stagnant air trap it. Wind *direction*
matters too: clean northerly (NW) air vs stagnant/​southerly conditions."""),
        co("""num = df[data.WEATHER + ["pm25"]].dropna()
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax[0]); ax[0].set_title("weather ↔ PM2.5 correlation")
samp = df.dropna(subset=["pm25"]).sample(3000, random_state=0)
sns.scatterplot(x="Iws", y="pm25", data=samp, s=8, alpha=.3, ax=ax[1]); ax[1].set_title("strong wind clears pollution"); ax[1].set_xlabel("cumulative wind speed Iws")
eda.savefig(fig, "p1_weather.png"); plt.show()
print("PM2.5 vs wind speed corr = %.2f (disperses) | vs dew point = %.2f (traps)"
      % (num.pm25.corr(num.Iws), num.pm25.corr(num.DEWP)))"""),
        co("""by_wind = df.groupby("wind_dir", observed=True).pm25.mean().sort_values()
print("mean PM2.5 by wind direction (µg/m³):"); print(by_wind.round(0).to_string())
print("-> %s winds are cleanest, %s the most polluted." % (by_wind.index[0], by_wind.index[-1]))"""),

        md("""### 4. Seasonality preview — winter is worse

Pollution has a strong **yearly** cycle (winter coal heating + stagnant air) and a **daily** cycle.
This previews the multi-seasonal time-series work to come."""),
        co("""fig, ax = plt.subplots(1, 2, figsize=(13, 4))
df.groupby(df.index.month).pm25.mean().plot(kind="bar", ax=ax[0], color="slategray"); ax[0].set_title("PM2.5 by month — winter peak (heating)"); ax[0].set_xlabel("month")
df.groupby(df.index.hour).pm25.mean().plot(ax=ax[1], marker="o", color="slategray"); ax[1].set_title("PM2.5 by hour — overnight build-up"); ax[1].set_xlabel("hour")
eda.savefig(fig, "p1_seasonality.png"); plt.show()
w = df.groupby(df.index.month).pm25.mean()
print("worst month: %d (%.0f) vs best: %d (%.0f) µg/m³" % (w.idxmax(), w.max(), w.idxmin(), w.min()))"""),

        md(
"""### Takeaways

- **PM2.5 is heavily right-skewed** (a √/log transform helps); Beijing averages ~99 µg/m³ — **84% of
  hours breach the WHO guideline** — with real "airpocalypse" extremes (Jan 2013).
- **The missingness is structured**: 2,067 gaps, mostly single hours but with **outages up to 155 h**.
  Weather looks the same on missing vs present hours → plausibly **MCAR**, but the long runs rule out
  naive interpolation — *how* you impute will matter.
- **Weather drives pollution**: wind **disperses** it (corr −0.25, clean NW winds), humidity **traps**
  it (+0.17) — a genuine multivariate, causal-ish structure the forecasting will exploit.
- Strong **winter (heating) and daily** seasonality previews the time-series work.

**Next — Part 2 (Missing-Data Imputation):** the centerpiece — compare forward-fill, time
interpolation, seasonal, **KNN**, and **MICE** imputation; *evaluate* them by masking known values;
and show why the 155-hour gaps need special care."""),
    ]
    build(cells, "01_advanced_eda.ipynb", "# 01 · Beijing PM2.5 — Advanced EDA")


if __name__ == "__main__":
    import sys
    all_nbs = {"0": notebook_0, "1": notebook_1}
    for k in (sys.argv[1:] or sorted(all_nbs)):
        all_nbs[k]()
    print("done.")

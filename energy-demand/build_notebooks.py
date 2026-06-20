"""Generate the energy-demand notebooks from readable source (nbformat).
Re-run:  python build_notebooks.py [N ...]   then execute with jupyter nbconvert.

A comprehensive, professional study of the PJM hourly electricity-demand panel:
Advanced EDA + the full time-series toolkit (Parts 0–11).
"""
from __future__ import annotations

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NB_DIR = "notebooks"

SETUP = """\
import sys, pathlib, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path.cwd(); ROOT = ROOT if (ROOT / "src").exists() else ROOT.parent
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src import data, eda
eda.set_style()
pd.set_option("display.width", 130, "display.max_columns", 40)
print("setup ok | numpy", np.__version__, "| pandas", pd.__version__)
"""


def build(cells, name, title):
    nb = new_notebook()
    nb.cells = [new_markdown_cell(title)] + cells
    nb.metadata["kernelspec"] = {"display_name": "Python (energy-demand)",
                                 "language": "python", "name": "energy-demand"}
    with open(f"{NB_DIR}/{name}", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote", name, f"({len(nb.cells)} cells)")


# ===================================================================== Notebook 0
def notebook_0():
    md, co = new_markdown_cell, new_code_cell
    cells = [
        md(
"""## Part 0 — Acquisition & Cleaning

**The dataset.** Sixteen years (2002–2018) of **hourly electricity demand** for the PJM
Interconnection — the grid operator coordinating power across 13 US states from the mid-Atlantic to
the Midwest. Eleven columns are regional zones; demand is in **megawatts (MW)**. We model **PJME**
(PJM East), the longest, cleanest zone.

**Why this dataset is a time-series showcase.** Electricity demand is the textbook example of *rich*
temporal structure, all at once:
- **Triple seasonality** — a *daily* cycle (people wake, work, sleep), a *weekly* cycle (weekday vs
  weekend), and an *annual* cycle.
- **Bimodal annual** — unlike most series, demand peaks **twice** a year: winter (heating) and summer
  (air-conditioning), with mild spring/autumn troughs.
- **Weather-driven extremes** — heat waves and cold snaps drive the peaks that grids are sized for.
- **Local-time artifacts** — the data is in wall-clock time, so **daylight-saving** transitions create
  duplicate and missing hours that must be cleaned.

This part assembles the panel, audits and fixes the data-quality issues, and produces the clean,
gap-free series the rest of the study uses."""),
        co(SETUP),

        md(
"""### 1. The regional panel

The raw data is one CSV per zone; `data.load_panel()` returns them aligned on a single hourly index.
The zones cover **different spans** (PJME runs the full 16 years; smaller zones start later), so the
wide panel is intentionally *ragged* — full where a zone reported, `NaN` before it existed."""),
        co("""panel = data.load_panel()
print("panel shape:", panel.shape, "| span:", panel.index.min(), "->", panel.index.max())
print("\\ncoverage per zone (non-null hours):")
print(panel.notna().sum().sort_values(ascending=False).to_string())"""),

        md(
"""### 2. Data-quality audit — the daylight-saving trap

Demand is recorded in **local wall-clock time**. Twice a year that bites:
- **Autumn "fall back"** repeats the 1–2 am hour → **duplicate timestamps** (removed during assembly,
  keeping the first).
- **Spring "spring forward"** skips an hour → **missing timestamps**.

Building the *complete* hourly grid exposes the gaps. They're few (≈30 hours over 16 years) but must be
filled, or every seasonal calculation silently misaligns."""),
        co("""s_raw = panel[data.PRIMARY]
full_grid = pd.date_range(s_raw.index.min(), s_raw.index.max(), freq="h")
missing = full_grid.difference(s_raw.index)
print("PJME reported hours:", s_raw.notna().sum())
print("complete hourly grid:", len(full_grid))
print("MISSING hours (DST spring-forward + gaps):", len(missing))
print("examples:", [str(t) for t in missing[:4]])"""),

        md(
"""### 3. The clean primary series

`data.primary()` drops any duplicate timestamps, reindexes onto the complete hourly grid, and fills
the few missing hours by **time interpolation** (linear in time — appropriate for a smooth, strongly
autocorrelated load curve). The result is a continuous hourly MW series with no gaps."""),
        co("""s = data.primary()
print("clean series:", s.shape, "| any missing:", int(s.isna().sum()), "| freq:", s.index.freq)
print("MW — mean %.0f | median %.0f | min %.0f | max %.0f" % (s.mean(), s.median(), s.min(), s.max()))"""),

        md(
"""### 4. First look — two weeks of demand

A fortnight makes the **daily** and **weekly** rhythms visible at once: a smooth sinusoid each day
(trough ~4 am, peak in the evening) and a clear drop across the two weekends. This is the structure
the whole study will dissect."""),
        co("""win = s.loc["2017-01-09":"2017-01-22"]
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(win.index, win.values, color="tab:blue", lw=1)
for d0 in pd.date_range("2017-01-14", "2017-01-22", freq="D"):
    if d0.dayofweek == 5: ax.axvspan(d0, d0 + pd.Timedelta(days=2), color="orange", alpha=.12)
ax.set_ylabel("load (MW)"); ax.set_title("PJME demand — two weeks (daily cycle + weekend dips shaded)")
eda.savefig(fig, "p0_fortnight.png"); plt.show()"""),

        md(
"""### 5. Calendar features & persistence

Electricity demand is *driven by the calendar* (clock, weekday, season, holidays), so we attach those
features now and persist the modelling-ready table. `is_holiday` uses the US federal calendar — demand
on holidays behaves like an extra weekend."""),
        co("""d = data.build_processed()
print("saved data/processed/pjme_clean.csv:", d.shape)
print("columns:", list(d.columns))
print("holidays flagged:", int(d.is_holiday.sum()), "hours |", int(d.is_holiday.sum()/24), "days over 16 years")"""),

        md(
"""### Takeaways

- **16 years × hourly** PJME demand, plus 10 other zones (ragged coverage) for later multivariate work.
- The **local-time / DST trap** is real: duplicate timestamps (dropped) and ~30 missing hours
  (interpolated). Handling it is a prerequisite for any seasonal analysis.
- The clean series ranges **14,544 → 62,009 MW**; calendar features (hour, day-of-week, month, holiday,
  season) are attached.
- Two weeks already reveal the **daily + weekly** cycles — Part 1 quantifies the full seasonal
  structure.

**Next — Part 1 (Advanced EDA I):** the demand distribution and the complete calendar anatomy — the
triple seasonality, the summer-vs-winter daily shapes, weekend/holiday effects, and the load-duration
curve."""),
    ]
    build(cells, "00_data_cleaning.ipynb", "# 00 · PJM Energy Demand — Acquisition & Cleaning")


# ===================================================================== Notebook 1
def notebook_1():
    md, co = new_markdown_cell, new_code_cell
    cells = [
        md(
"""## Part 1 — Advanced EDA I: Distributions & Calendar Structure

A thorough cross-section of *what demand looks like* and *how the calendar shapes it*: the
distribution and its moments, the **triple seasonality** (daily × weekly × annual), the way the daily
shape **changes with season**, the weekend and holiday effects, the utility's **load-duration curve**,
and the surprising long-run trend."""),
        co(SETUP + """
d = data.clean_primary(); s = d.load_mw
print("hours:", len(s), "| span:", s.index.min().date(), "->", s.index.max().date())"""),

        md(
"""### 1. The demand distribution

Load is strictly positive and **mildly right-skewed** (skew ≈ 0.74): most hours sit near the ~32 GW
median, with a thinner tail of extreme high-demand hours (heat-wave afternoons). The four-view battery
shows the shape, spread, percentiles and departure from normality at once."""),
        co("""eda.four_view(s, "PJME load (MW)", "p1_dist.png")
print(eda.moments(s).round(2).to_string())"""),

        md(
"""### 2. The triple seasonality

Three calendar rhythms stacked on top of each other:
- **Daily** — trough around 04:00, rising to an evening peak.
- **Weekly** — weekdays higher than weekends (commerce & industry).
- **Annual** — **bimodal**: high in summer (A/C) *and* winter (heating), low in spring/autumn.

The annual bimodality is the signature of this dataset — most seasonal series have a single peak."""),
        co("""fig, ax = plt.subplots(1, 3, figsize=(16, 4))
s.groupby(d.hour).mean().plot(ax=ax[0], marker="o", color="tab:blue"); ax[0].set_title("daily — by hour"); ax[0].set_xlabel("hour")
s.groupby(d.dow).mean().plot(ax=ax[1], marker="o", color="tab:green"); ax[1].set_title("weekly — by day-of-week (0=Mon)"); ax[1].set_xlabel("day of week")
s.groupby(d.month).mean().plot(ax=ax[2], marker="o", color="tab:red"); ax[2].set_title("annual — by month (BIMODAL)"); ax[2].set_xlabel("month")
for a in ax: a.set_ylabel("mean MW")
fig.tight_layout(); eda.savefig(fig, "p1_triple_season.png"); plt.show()
print("season means:", s.groupby(d.season).mean().round(0).to_dict())"""),

        md(
"""### 3. Hour × day-of-week heatmap

Combining two cycles: weekdays show a strong daytime business peak; weekends are lower and flatter and
shift slightly later (no morning commute). This 2-D calendar fingerprint is what a good forecaster must
reproduce."""),
        co("""piv = s.groupby([d.dow, d.hour]).mean().unstack()
fig, ax = plt.subplots(figsize=(14, 4))
sns.heatmap(piv, cmap="YlOrRd", ax=ax, cbar_kws={"label": "mean MW"},
            yticklabels=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
ax.set_xlabel("hour of day"); ax.set_title("Mean demand by day-of-week × hour")
eda.savefig(fig, "p1_heatmap.png"); plt.show()"""),

        md(
"""### 4. The daily shape *changes with season*

A subtlety pure averages hide: **summer and winter have different daily shapes**. Summer is a single
sharp **afternoon** peak (~17:00) as air-conditioning load tracks the day's heat. Winter is **double-
peaked** — a morning bump (people wake, heat homes) and a larger evening peak (return home + lights).
A model with one fixed daily profile will systematically miss this."""),
        co("""prof = s.groupby([d.season, d.hour]).mean().unstack(0)
fig, ax = plt.subplots(figsize=(10, 5))
for seas, c in [("summer","tab:red"), ("winter","tab:blue"), ("spring","tab:green"), ("fall","tab:orange")]:
    ax.plot(prof.index, prof[seas], marker="o", ms=3, label=seas, color=c)
ax.set_xlabel("hour"); ax.set_ylabel("mean MW"); ax.set_title("Daily load shape by season — summer single afternoon peak vs winter double peak"); ax.legend()
eda.savefig(fig, "p1_seasonal_shape.png"); plt.show()
print("summer peak hour %d | winter peak hours %s" % (prof["summer"].idxmax(), list(prof["winter"].nlargest(2).index)))"""),

        md(
"""### 5. Weekend & holiday effects

Demand falls **~10% on weekends** and **~6% on holidays** (vs working weekdays) — holidays behave like
an extra weekend dropped into the week. The Thanksgiving-week zoom shows the holiday carving a
weekend-shaped dip into a Thursday."""),
        co("""wk = 100*(1 - d[d.is_weekend==1].load_mw.mean()/d[d.is_weekend==0].load_mw.mean())
hol = 100*(1 - d[d.is_holiday==1].load_mw.mean()/d[(d.is_holiday==0)&(d.is_weekend==0)].load_mw.mean())
print("weekend reduction %.1f%% | holiday reduction %.1f%% (vs weekdays)" % (wk, hol))
zoom = s.loc["2017-11-20":"2017-11-27"]
fig, ax = plt.subplots(figsize=(13, 4)); ax.plot(zoom.index, zoom.values, color="tab:purple", lw=1.2)
ax.axvspan("2017-11-23", "2017-11-24", color="red", alpha=.15, label="Thanksgiving")
ax.set_ylabel("MW"); ax.set_title("Thanksgiving week 2017 — holiday cuts a weekend-shaped dip"); ax.legend()
eda.savefig(fig, "p1_holiday_zoom.png"); plt.show()"""),

        md(
"""### 6. The load-duration curve

A staple of power-systems analysis: sort every hour's demand from highest to lowest. The steep left
edge is the few **peak hours** the grid must be built to serve; the long flat body is the everyday
base load. Here **peak demand is 4.3× the minimum** — most generating capacity exists for a handful of
extreme hours a year, the central economic problem of electricity systems."""),
        co("""sd = np.sort(s.values)[::-1]; pct = np.arange(len(sd))/len(sd)*100
fig, ax = plt.subplots(figsize=(10, 5)); ax.plot(pct, sd, color="black")
ax.fill_between(pct, sd, sd.min(), where=(pct<1), color="red", alpha=.3, label="top 1% of hours")
ax.set_xlabel("% of hours exceeding"); ax.set_ylabel("MW"); ax.set_title("Load-duration curve"); ax.legend()
eda.savefig(fig, "p1_load_duration.png"); plt.show()
print("top-1%% hours exceed %.0f MW | base (bottom 1%%) below %.0f | peak/base ratio %.2f" % (sd[int(.01*len(sd))], sd[int(.99*len(sd))], s.max()/s.min()))"""),

        md(
"""### 7. The long-run trend — demand is *not* growing

Counterintuitively, annual-average demand **fell ~3% from 2003 to 2017** despite population and
economic growth. Efficiency (LED lighting, better appliances, building codes) and behind-the-meter
rooftop solar have outpaced demand growth — a real, much-discussed feature of mature US grids, and a
caution against assuming an upward trend."""),
        co("""ann = s[(d.year>=2003)&(d.year<=2017)].groupby(d.year).mean()
fig, ax = plt.subplots(figsize=(10, 4)); ann.plot(ax=ax, marker="o", color="firebrick")
ax.set_ylabel("annual mean MW"); ax.set_title("Annual mean demand 2003–2017 — gently declining")
eda.savefig(fig, "p1_yoy.png"); plt.show()
print("2003 %.0f MW -> 2017 %.0f MW (%.1f%%)" % (ann.loc[2003], ann.loc[2017], 100*(ann.loc[2017]/ann.loc[2003]-1)))"""),

        md(
"""### Takeaways

- Demand is **mildly right-skewed** (skew 0.74) around a ~32 GW median, with a heat-wave tail.
- **Triple seasonality** (daily × weekly × annual) with a distinctive **bimodal** annual cycle
  (summer & winter peaks).
- The **daily shape changes with season** — summer single afternoon peak vs winter double peak — a
  trap for fixed-profile models.
- **Weekends −10%, holidays −6%**; the **load-duration curve** shows a peak/base ratio of **4.3**.
- The long-run trend is **flat-to-declining** (−3% over 2003–17) — efficiency beating growth.

**Next — Part 2 (Advanced EDA II):** the temporal deep-dive — ramp rates, intra-day volatility,
year-over-year shape changes, and the autocorrelation that makes load so forecastable."""),
    ]
    build(cells, "01_advanced_eda_distributions.ipynb", "# 01 · PJM Energy Demand — Advanced EDA I: Distributions & Calendar")


# ===================================================================== Notebook 2
def notebook_2():
    md, co = new_markdown_cell, new_code_cell
    cells = [
        md(
"""## Part 2 — Advanced EDA II: The Temporal Deep-Dive

Part 1 mapped the *average* calendar structure. Here we examine the **dynamics** that matter to anyone
operating or forecasting a grid: how fast demand **ramps**, how **volatile** the day is, *when* the
daily peak lands, how the seasonal balance has **shifted over 16 years**, and the **persistence** that
makes load forecastable at all."""),
        co(SETUP + """
d = data.clean_primary(); s = d.load_mw
print("hours:", len(s))"""),

        md(
"""### 1. Ramp rates — the operator's headache

A grid must *follow* demand minute to minute, so the **hour-to-hour change** (the ramp) is as important
as the level. The mean **morning ramp-up peaks at 07:00 (~+2,450 MW/h)** as the system wakes; the
overnight ramp-down mirrors it. Extreme single-hour ramps reach **±6–7 GW/h** — enough swing to start
or stop a dozen power plants in an hour."""),
        co("""ramp = s.diff()
rh = ramp.groupby(d.hour).mean()
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
rh.plot.bar(ax=ax[0], color=["tab:green" if v>0 else "tab:red" for v in rh]); ax[0].set_title("mean hourly ramp by hour of day"); ax[0].set_ylabel("MW/h"); ax[0].axhline(0, color="k", lw=.6)
ax[1].hist(ramp.dropna(), bins=80, color="slateblue"); ax[1].set_title(f"ramp distribution (extremes ±{max(ramp.max(),-ramp.min()):.0f} MW/h)"); ax[1].set_xlabel("MW/h")
fig.tight_layout(); eda.savefig(fig, "p2_ramps.png"); plt.show()
print("steepest mean up-ramp: hr %d (+%.0f) | down-ramp: hr %d (%.0f) | extremes +%.0f / %.0f" %
      (rh.idxmax(), rh.max(), rh.idxmin(), rh.min(), ramp.max(), ramp.min()))"""),

        md(
"""### 2. Intra-day volatility — summer is the hard season

The **daily swing** (that day's max − min) measures how much work the grid does within a day. **Summer
swings ~17 GW vs ~10 GW in winter** — air-conditioning turns hot afternoons into sharp spikes, so even
though winter's *average* demand is high, **summer is the operationally volatile season**."""),
        co("""daily = s.resample("D").agg(["min","max"]); daily["swing"] = daily["max"] - daily["min"]
daily["season"] = pd.Series(daily.index.month % 12 // 3, index=daily.index).map({0:"winter",1:"spring",2:"summer",3:"fall"})
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(x="season", y="swing", data=daily.reset_index(), order=["winter","spring","summer","fall"],
            palette=["tab:blue","tab:green","tab:red","tab:orange"], showfliers=False, ax=ax)
ax.set_ylabel("daily peak−trough (MW)"); ax.set_title("Intra-day demand swing by season — summer is largest")
eda.savefig(fig, "p2_swing.png"); plt.show()
print("mean daily swing:", daily.groupby("season").swing.mean().round(0).to_dict())"""),

        md(
"""### 3. When does the daily peak land?

The timing of the daily maximum shifts with season — **summer peaks ~17:00** (afternoon A/C), while
cooler seasons peak in the **evening (19–21:00)** (lighting + cooking after dark). A forecaster that
assumes a fixed peak hour will be wrong half the year."""),
        co("""peak_hr = s.groupby(s.index.normalize()).apply(lambda x: x.idxmax().hour)
ph = pd.DataFrame({"hr": peak_hr.values})
ph["season"] = pd.Series(peak_hr.index.month % 12 // 3, index=peak_hr.index).map({0:"winter",1:"spring",2:"summer",3:"fall"}).values
fig, ax = plt.subplots(figsize=(10, 4))
for seas, c in [("summer","tab:red"),("winter","tab:blue")]:
    ax.hist(ph[ph.season==seas].hr, bins=range(0,25), alpha=.55, label=seas, color=c)
ax.set_xlabel("hour of daily peak"); ax.set_ylabel("# days"); ax.set_title("Daily-peak hour — summer afternoon vs winter evening"); ax.legend()
eda.savefig(fig, "p2_peakhour.png"); plt.show()
print("modal peak hour by season:", ph.groupby("season").hr.agg(lambda x: int(x.mode().iloc[0])).to_dict())"""),

        md(
"""### 4. A 16-year shift — toward a summer-peaking system

Tracking each year's **summer vs winter peak** shows the balance tilting: the summer/winter peak ratio
rose from ~1.1 in the mid-2000s toward ~1.3 — consistent with rising air-conditioning saturation and
warming summers. Long histories let you see the *system itself changing*, not just noise."""),
        co("""yr = s.groupby([d.year, d.season]).max().unstack()
yr = yr.loc[2003:2017]
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(yr.index, yr["summer"], "o-", color="tab:red", label="summer peak")
ax.plot(yr.index, yr["winter"], "o-", color="tab:blue", label="winter peak")
ax.set_ylabel("annual peak MW"); ax.set_title("Summer vs winter annual peak, 2003–2017"); ax.legend()
eda.savefig(fig, "p2_peakshift.png"); plt.show()
print("summer/winter peak ratio: 2004 %.2f -> 2012 %.2f" % ((yr.loc[2004,"summer"]/yr.loc[2004,"winter"]), (yr.loc[2012,"summer"]/yr.loc[2012,"winter"])))"""),

        md(
"""### 5. Persistence — why load is so forecastable

Demand is **enormously autocorrelated**: this hour looks like last hour (lag-1 ≈ 0.97), like yesterday
(lag-24 ≈ 0.89) and like last week (lag-168 ≈ 0.78). That persistence is the forecaster's friend —
simple seasonal-naive methods are already strong, and it's why lag features dominate the ML models in
Part 8. Part 3 formalises this with the ACF/PACF and decomposition."""),
        co("""lags = {"1h":1, "24h (day)":24, "168h (week)":168, "8766h (year)":8766}
ac = {k: s.autocorr(v) for k, v in lags.items()}
fig, ax = plt.subplots(figsize=(8, 4))
pd.Series(ac).plot.bar(ax=ax, color="teal", rot=0); ax.set_ylabel("autocorrelation"); ax.set_title("Persistence at key lags"); ax.set_ylim(0,1)
for i, v in enumerate(ac.values()): ax.text(i, v+.02, f"{v:.2f}", ha="center")
eda.savefig(fig, "p2_persistence.png"); plt.show()"""),

        md(
"""### Takeaways

- **Ramps** peak at the 07:00 morning wake-up (~+2,450 MW/h mean); extremes hit ±6–7 GW/h — the grid's
  follow-the-load challenge.
- **Summer is the volatile season**: daily swings ~17 GW vs ~10 GW in winter (air-conditioning).
- The **daily peak time shifts** (summer ~17:00, winter ~19–21:00); the **summer/winter peak ratio is
  rising** over 16 years.
- Load is **extremely persistent** (lag-1 0.97, lag-24 0.89, lag-168 0.78) → very forecastable.

**Next — Part 3 (TS foundations):** stationarity tests, the ACF/PACF in full, and **MSTL**
decomposition that separates the trend, the daily/weekly/annual seasonals, and the remainder."""),
    ]
    build(cells, "02_advanced_eda_temporal.ipynb", "# 02 · PJM Energy Demand — Advanced EDA II: Temporal Deep-Dive")


# ===================================================================== Notebook 3
def notebook_3():
    md, co = new_markdown_cell, new_code_cell
    cells = [
        md(
"""## Part 3 — Time-Series Foundations

The formal TS toolkit applied to the load series: **stationarity** testing (and the trap that
seasonality sets for it), the **ACF/PACF** correlation structure, and a full **MSTL** decomposition
that separates the trend, the daily and weekly seasonals, and the remainder. These are the diagnostics
every forecasting choice in Parts 7–9 rests on."""),
        co(SETUP + """
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import MSTL, STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
d = data.clean_primary(); s = d.load_mw
print("series:", len(s), "hourly points")"""),

        md(
"""### 1. Stationarity — and why the two tests disagree

A **stationary** series has constant mean/variance over time — many models assume it. We run two tests
with *opposite* null hypotheses:
- **ADF** (null = has a unit root / non-stationary),
- **KPSS** (null = stationary).

On the raw series they **conflict**: ADF rejects its null (→ "stationary") while KPSS *also* rejects
its null (→ "non-stationary"). This classic disagreement is the fingerprint of **strong seasonality +
mild trend**: the series is mean-reverting (no unit root) but its mean *moves with the season*, which
KPSS detects. The fix is to remove the seasonality."""),
        co("""def report(x, label):
    a = adfuller(x, maxlag=48, autolag=None); k = kpss(x, nlags=48)
    print("%-22s ADF p=%6.3g (%s) | KPSS p=%5.3g (%s)" %
          (label, a[1], "stationary" if a[1]<.05 else "non-stat",
                  k[1], "non-stat" if k[1]<.05 else "stationary"))
report(s.values, "raw level")
report(s.diff(24).dropna().values, "seasonal diff (lag 24)")
print("\\n-> raw: tests CONFLICT (seasonality fools KPSS). After seasonal differencing they AGREE: stationary.")"""),

        md(
"""### 2. ACF & PACF — the correlation skeleton

The **autocorrelation function** shows demand correlating with its past at every lag: huge spikes at
**multiples of 24** (daily) with a slow weekly envelope. The **partial ACF** isolates the direct
effect of each lag — a strong lag-1 and lag-24 dominate, which is why AR-type and lag-feature models
work so well."""),
        co("""fig, ax = plt.subplots(2, 1, figsize=(13, 7))
plot_acf(s.values, lags=200, ax=ax[0]); ax[0].set_title("ACF (200 lags) — spikes every 24h, weekly envelope")
plot_pacf(s.values, lags=60, ax=ax[1], method="ywm"); ax[1].set_title("PACF (60 lags) — direct effects at lag 1 and 24")
fig.tight_layout(); eda.savefig(fig, "p3_acf_pacf.png"); plt.show()"""),

        md(
"""### 3. MSTL — decomposing *multiple* seasonalities

Ordinary STL handles one seasonal cycle; **MSTL** peels off several. We extract the **daily (24h)** and
**weekly (168h)** seasonals simultaneously, leaving a smooth **trend** and a **remainder**. Shown over
three weeks so the components are legible. The **seasonal strength** (share of variance each explains)
confirms the daily cycle dominates."""),
        co("""res = MSTL(s, periods=(24, 168)).fit()
win = slice("2016-07-01", "2016-07-21")
comp = pd.DataFrame({"observed": s, "trend": res.trend,
                     "daily(24)": res.seasonal["seasonal_24"], "weekly(168)": res.seasonal["seasonal_168"],
                     "remainder": res.resid})
fig, axes = plt.subplots(5, 1, figsize=(13, 10), sharex=True)
for ax, c, col in zip(axes, comp.columns, ["black","firebrick","tab:blue","tab:green","grey"]):
    ax.plot(comp.loc[win].index, comp.loc[win, c], color=col, lw=1); ax.set_ylabel(c, fontsize=9)
axes[0].set_title("MSTL decomposition (3-week window): observed = trend + daily + weekly + remainder")
fig.tight_layout(); eda.savefig(fig, "p3_mstl.png"); plt.show()
def strength(seas): return max(0, 1 - res.resid.var()/(res.resid + seas).var())
print("seasonal strength — daily %.2f | weekly %.2f | remainder share of var %.1f%%" %
      (strength(res.seasonal["seasonal_24"]), strength(res.seasonal["seasonal_168"]), 100*res.resid.var()/s.var()))"""),

        md(
"""### 4. The annual cycle

MSTL on hourly data captures the sub-weekly seasonals; the **annual** cycle is clearest on the
**daily-mean** series. An STL with a 365-day period extracts the **bimodal** yearly seasonal (the
summer and winter humps) on top of the slowly declining trend."""),
        co("""daily = s.resample("D").mean()
stl = STL(daily, period=365, robust=True).fit()
fig, ax = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
ax[0].plot(daily.index, stl.trend, color="firebrick"); ax[0].set_ylabel("trend")
ax[1].plot(daily.index, stl.seasonal, color="tab:purple"); ax[1].set_ylabel("annual seasonal")
ax[2].plot(daily.index, stl.resid, color="grey", lw=.5); ax[2].set_ylabel("remainder")
ax[0].set_title("Annual decomposition (daily mean, STL period=365) — bimodal seasonal, declining trend")
fig.tight_layout(); eda.savefig(fig, "p3_annual_stl.png"); plt.show()
print("annual seasonal swing: %.0f MW peak-to-peak" % (stl.seasonal.max() - stl.seasonal.min()))"""),

        md(
"""### Takeaways

- **Stationarity tests conflict** on the raw series (ADF stationary vs KPSS non-stationary) — the
  signature of strong seasonality; **seasonal differencing** (lag 24) makes them agree.
- The **ACF** shows clean 24-hour spikes under a weekly envelope; the **PACF** points to lag-1 and
  lag-24 as the direct drivers (AR / lag features will be strong).
- **MSTL** cleanly separates the **daily (dominant) and weekly** seasonals from a smooth trend; the
  remainder is a small share of variance.
- The **annual** cycle (via daily-mean STL) is **bimodal** atop a gently declining trend.

**Next — Part 4 (Spectral diagnostics):** confirm these cycles in the *frequency* domain with the
periodogram/FFT, watch them with a spectrogram, and probe long-memory."""),
    ]
    build(cells, "03_ts_foundations.ipynb", "# 03 · PJM Energy Demand — Time-Series Foundations (stationarity, ACF/PACF, MSTL)")


# ===================================================================== Notebook 4
def notebook_4():
    md, co = new_markdown_cell, new_code_cell
    cells = [
        md(
"""## Part 4 — Spectral & Advanced Diagnostics

The time-domain analysis (Parts 2–3) found cycles by eye and by ACF. Here we confirm them in the
**frequency domain** — where periodicity becomes sharp, quantifiable peaks — and add two advanced
probes: a **spectrogram** (how the cycles wax and wane through the year) and a **long-memory** estimate
(Hurst). The frequency view even reveals a non-obvious consequence of the *bimodal* annual cycle."""),
        co(SETUP + """
from scipy import signal
d = data.clean_primary(); s = d.load_mw
x = s.values - s.mean()
print("analysing", len(x), "hourly points in the frequency domain")"""),

        md(
"""### 1. The periodogram — cycles as spectral peaks

The **periodogram** decomposes the series into sine waves and shows the **power** at each frequency.
Converting frequency → period (hours), sharp peaks appear exactly where Part 3 predicted: **24 h**
(daily, dominant), its **12 h harmonic**, **168 h** (weekly), and a low-frequency annual cluster."""),
        co("""f, P = signal.periodogram(x, fs=1.0)            # fs = 1 sample/hour
period = 1/f[1:]; power = P[1:]
fig, ax = plt.subplots(figsize=(12, 5))
ax.semilogx(period, power, color="navy", lw=.8)
for p, lab in [(24,"24h daily"),(12,"12h"),(168,"weekly"),(4380,"~½ year"),(8766,"annual")]:
    ax.axvline(p, color="red", ls=":", lw=.8); ax.text(p, power.max()*.6, lab, rotation=90, fontsize=8, color="red")
ax.set_xlabel("period (hours, log scale)"); ax.set_ylabel("power"); ax.set_title("Periodogram — demand's cycles as spectral peaks")
eda.savefig(fig, "p4_periodogram.png"); plt.show()
top = np.argsort(power)[::-1][:6]
print("top peaks (period):", [f"{period[i]:.1f}h ({period[i]/24:.0f}d)" for i in sorted(top, key=lambda j:-power[j])])"""),

        md(
"""### 2. The bimodal cycle hides at *half* a year

A subtle, elegant result: the strongest *low-frequency* peak is near **183 days (½ year)**, not 365.
That's the spectral fingerprint of a **double-peaked** annual cycle — because demand rises in *both*
summer and winter, the dominant yearly rhythm repeats every ~6 months. The frequency domain *sees the
bimodality* that a single annual peak would hide."""),
        co("""lowf = period > 60*24                              # periods longer than ~2 months
lp, lpow = period[lowf], power[lowf]
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(lp/24, lpow, color="purple"); ax.axvline(183, color="red", ls=":"); ax.axvline(365, color="green", ls=":")
ax.text(183, lpow.max()*.7, "½-year peak\\n(= bimodal!)", color="red", fontsize=9)
ax.set_xlabel("period (days)"); ax.set_ylabel("power"); ax.set_title("Low-frequency band — the dominant annual energy sits at ~183 days")
eda.savefig(fig, "p4_bimodal_spectral.png"); plt.show()
print("strongest low-freq period: %.0f days (the bimodal summer+winter signature)" % (lp[np.argmax(lpow)]/24))"""),

        md(
"""### 3. Spectrogram — the cycles change through the year

A single periodogram averages over all 16 years. A **spectrogram** slides a window along the series and
computes a local spectrum, revealing how the **daily-cycle strength varies seasonally** — the 24-hour
band glows brightest in summer (air-conditioning amplifies the day/night contrast) and dims in the mild
shoulder seasons. The structure is *non-stationary in power*, even if mean-stationary."""),
        co("""freqs, times, Sxx = signal.spectrogram(x, fs=1.0, nperseg=24*30, noverlap=24*15)
band = (freqs > 1/26) & (freqs < 1/22)             # ~24h band
daily_power = Sxx[band].mean(0)
t_dates = s.index[(times.astype(int)).clip(max=len(s)-1)]
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(t_dates, daily_power, color="darkorange"); ax.set_yscale("log")
ax.set_ylabel("power in 24h band (log)"); ax.set_title("Strength of the daily cycle over time — peaks each summer")
eda.savefig(fig, "p4_spectrogram.png"); plt.show()
print("daily-cycle power varies ~%.0fx between its seasonal high and low" % (daily_power.max()/daily_power.min()))"""),

        md(
"""### 4. Long-memory — the Hurst exponent

Beyond the cycles, is there **long-range dependence** — do shocks persist? The **Hurst exponent**
(via rescaled-range, R/S) answers: 0.5 = a memoryless random walk, >0.5 = persistent. Demand scores
**~0.84**, strong long-memory — high-demand spells tend to be followed by more high demand (heat waves
last days). Combined with the low **spectral entropy** (energy concentrated in a few cycles), this
series is about as **forecastable** as time series get."""),
        co("""def hurst_rs(x, nmax=16):
    x = np.asarray(x, float); N = len(x)
    ns = np.unique(np.logspace(1, np.log10(N//4), nmax).astype(int)); RS = []
    for n in ns:
        rs = []
        for i in range(N//n):
            seg = x[i*n:(i+1)*n]; Z = np.cumsum(seg - seg.mean()); S = seg.std()
            if S > 0: rs.append((Z.max()-Z.min())/S)
        RS.append(np.mean(rs))
    return np.polyfit(np.log(ns), np.log(RS), 1)[0]
H = hurst_rs(s.values)
Pn = power/power.sum(); spec_ent = -np.sum(Pn*np.log(Pn))/np.log(len(Pn))
print("Hurst exponent       : %.3f  (>0.5 = persistent long-memory)" % H)
print("spectral entropy     : %.3f  (0 = pure cycle, 1 = white noise -> 'highly structured')" % spec_ent)"""),

        md(
"""### Takeaways

- The **periodogram** confirms the 24 h (dominant), 12 h harmonic, 168 h weekly and annual cycles as
  sharp spectral peaks.
- The dominant annual energy sits at **~183 days (½ year)** — the frequency-domain signature of the
  **bimodal** summer+winter cycle.
- A **spectrogram** shows the daily cycle is **non-stationary in power**, peaking each summer
  (air-conditioning).
- **Hurst ≈ 0.84** (strong long-memory) and **low spectral entropy** mark demand as highly structured
  and forecastable — motivating the forecasting parts.

**Next — Part 5 (Multivariate):** bring in the other regional zones — cross-correlation, a **PCA
common factor** (one weather-driven signal moves the whole grid), and lead-lag between regions."""),
    ]
    build(cells, "04_spectral.ipynb", "# 04 · PJM Energy Demand — Spectral & Advanced Diagnostics (FFT, spectrogram, Hurst)")


# ===================================================================== Notebook 5
def notebook_5():
    md, co = new_markdown_cell, new_code_cell
    cells = [
        md(
"""## Part 5 — Multivariate Analysis: The Regional Panel

So far, one zone (PJME). PJM actually reports many regional zones, and the natural question is **how
together do they move?** We take the six zones with full overlapping history (2005–2018) and ask:
how correlated are they, is there a single **common factor**, and does any region **lead** another (a
forecasting edge)? The answer is a textbook illustration of a system driven by one shared force —
**weather** — across a whole continent-scale grid."""),
        co(SETUP + """
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
po = data.panel_overlap()
print("overlap panel:", po.shape, "| zones:", list(po.columns), "|", po.index.min().date(), "->", po.index.max().date())"""),

        md(
"""### 1. How correlated are the zones?

Every pair of zones correlates **0.83–0.94** — extraordinarily tight for separate geographic regions.
They rise and fall almost in lockstep, because the same heat waves and cold snaps sweep the whole
interconnection at once."""),
        co("""fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(po.corr(), annot=True, fmt=".2f", cmap="YlGnBu", vmin=.8, vmax=1, ax=ax)
ax.set_title("Cross-zone demand correlation")
eda.savefig(fig, "p5_corr.png"); plt.show()
print("mean off-diagonal correlation: %.2f" % po.corr().values[np.triu_indices(len(po.columns),1)].mean())"""),

        md(
"""### 2. PCA — one factor runs the grid

Standardise and decompose. **The first principal component explains ~90% of all variance**, with
**equal loadings** on every zone — i.e. PC1 is essentially *total system demand*. It correlates **0.99**
with the simple sum of the zones. One latent, weather-driven signal moves the entire grid up and down;
the regions are near-copies scaled to size."""),
        co("""X = StandardScaler().fit_transform(po)
pca = PCA().fit(X); evr = pca.explained_variance_ratio_
pc1 = pca.transform(X)[:, 0]; total = po.sum(1).values
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
ax[0].bar(range(1, len(evr)+1), evr*100, color="steelblue"); ax[0].set_xlabel("component"); ax[0].set_ylabel("% variance"); ax[0].set_title(f"PC1 = {evr[0]*100:.0f}% (the common factor)")
pd.Series(pca.components_[0], index=po.columns).plot.bar(ax=ax[1], color="teal", rot=0); ax[1].set_title("PC1 loadings — equal across zones (= total demand)")
fig.tight_layout(); eda.savefig(fig, "p5_pca.png"); plt.show()
print("PC1 %.1f%% | PC2 %.1f%% | corr(PC1, total demand) = %.3f" % (evr[0]*100, evr[1]*100, abs(np.corrcoef(pc1, total)[0,1])))"""),

        md(
"""### 3. What the *other* components capture

If PC1 is the shared level, PC2/PC3 are the **regional idiosyncrasies** — the ways zones differ. Their
loadings split the western zones (AEP/DAYTON) from the eastern (PJME/DOM), i.e. a small east–west
contrast riding on top of the dominant common factor. Useful to know which regions are *not*
interchangeable."""),
        co("""load = pd.DataFrame(pca.components_[:3].T, index=po.columns, columns=["PC1","PC2","PC3"])
print(load.round(2).to_string())
print("\\n-> PC2/PC3 (≈%.0f%% combined) encode east-west regional differences atop the shared level." % ((evr[1]+evr[2])*100))"""),

        md(
"""### 4. Lead–lag — do regions move *simultaneously*?

Could one zone's demand *predict* another's an hour ahead? We cross-correlate two zones at hourly
shifts. The peak sits squarely at **lag 0** — the regions move **together**, not in sequence, because
weather is the common driver and it arrives everywhere at once. So there's *no* cross-region
forecasting shortcut; each zone's own past is what helps (Part 8)."""),
        co("""a = (po["PJME"] - po["PJME"].mean()).values; b = (po["AEP"] - po["AEP"].mean()).values
def ccf_lag(a, b, lag):
    if lag > 0: return np.corrcoef(a[lag:], b[:-lag])[0,1]
    if lag < 0: return np.corrcoef(a[:lag], b[-lag:])[0,1]
    return np.corrcoef(a, b)[0,1]
lags = range(-6, 7); cc = [ccf_lag(a, b, l) for l in lags]
fig, ax = plt.subplots(figsize=(9, 4)); ax.stem(list(lags), cc)
ax.axvline(0, color="red", ls=":"); ax.set_xlabel("lag (hours): PJME vs AEP"); ax.set_ylabel("correlation"); ax.set_title("Cross-correlation peaks at lag 0 — regions move simultaneously")
eda.savefig(fig, "p5_leadlag.png"); plt.show()
print("peak cross-correlation at lag %d hours (0 = simultaneous)" % list(lags)[int(np.argmax(cc))])"""),

        md(
"""### Takeaways

- The six zones are **extremely correlated** (0.83–0.94) — a continent-scale grid breathing in unison.
- **PCA finds one dominant factor (~90%)** with equal loadings = *total system demand*, weather-driven
  (corr 0.99 with the sum). PC2/PC3 are small **east–west** regional contrasts.
- **No lead–lag**: zones move **simultaneously** (cross-correlation peaks at lag 0) — weather hits
  everywhere at once, so there's no cross-region forecasting edge.
- Practical upshot: model the **aggregate** (or each zone from its *own* history); the regions carry
  little independent information.

**Next — Part 6 (Anomaly & event detection):** find the unusual hours and days — holidays, heat-wave
peaks, and structural breaks — using STL-residual scoring and change-point detection."""),
    ]
    build(cells, "05_multivariate.ipynb", "# 05 · PJM Energy Demand — Multivariate: Regional Panel, PCA Common Factor, Lead-Lag")


# ===================================================================== Notebook 6
def notebook_6():
    md, co = new_markdown_cell, new_code_cell
    cells = [
        md(
"""## Part 6 — Anomaly & Event Detection

Real grids care intensely about the *unusual*: the heat-wave hour that sets a record, the holiday that
empties the offices, the slow structural drift that reshapes planning. This part finds all three:
**point anomalies** (via a calendar-climatology residual), **systematic events** (holidays), and
**structural breaks** (via change-point detection). A nice sanity check falls out — the anomaly
detector *rediscovers the public-holiday calendar on its own*."""),
        co(SETUP + """
import ruptures as rpt
from statsmodels.tsa.seasonal import STL
d = data.clean_primary(); s = d.load_mw"""),

        md(
"""### 1. A calendar-climatology baseline

To spot the *unexpected*, first model the *expected*. We compute the average demand for each
**(month, weekend?, hour)** cell — a simple climatology that captures the seasonal + weekly + daily
structure — and take the **residual** (actual − expected). Standardised, the residual is an anomaly
score: how many σ a given hour deviates from its calendar norm."""),
        co("""expected = s.groupby([d.month, d.is_weekend, d.hour]).transform("mean")
resid = s - expected
z = resid / resid.std()
print("residual std: %.0f MW | anomalies |z|>4: %d hours (%.2f%% of all hours)" % (resid.std(), (z.abs()>4).sum(), 100*(z.abs()>4).mean()))
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(s.index, z, color="grey", lw=.4); ax.axhline(4, color="red", ls=":"); ax.axhline(-4, color="red", ls=":")
ax.scatter(z[z.abs()>4].index, z[z.abs()>4], s=8, color="red", zorder=3)
ax.set_ylabel("anomaly score (σ)"); ax.set_title("Demand anomalies vs the calendar norm")
eda.savefig(fig, "p6_anomalies.png"); plt.show()"""),

        md(
"""### 2. What the extremes are — heat waves & holidays

The largest **positive** anomalies are **off-season heat waves** — hot spells the calendar doesn't
expect (a 53 GW afternoon on **31-May-2011**, weeks before summer). The largest **negative** anomalies
land on **public holidays** — and since our climatology *doesn't know about holidays*, the detector
flags **4-July-2014** purely from the data. Anomaly detection independently recovered the holiday
calendar — a satisfying validation."""),
        co("""print("TOP positive anomalies (unexpected HIGH — heat waves):")
for t, v in z.nlargest(4).items(): print("   %s  %+.1f σ   %.0f MW" % (t, v, s[t]))
print("\\nTOP negative anomalies (unexpected LOW — holidays):")
for t, v in z.nsmallest(4).items(): print("   %s  %+.1f σ   %.0f MW   (%s)" % (t, v, s[t], t.strftime("%a %d-%b")))
print("\\nholiday mean anomaly: %+.2f σ  vs non-holiday %+.2f σ  -> holidays run systematically low" %
      (z[d.is_holiday==1].mean(), z[d.is_holiday==0].mean()))"""),

        md(
"""### 3. When do anomalies cluster?

Counting extreme hours by month shows anomalies are **summer-heavy** — the season of volatile,
weather-driven demand (heat waves), consistent with Part 2's finding that summer is the operationally
hard season. Winter contributes a secondary cluster (cold snaps + holiday season)."""),
        co("""anom_months = z[z.abs()>4].index.month
fig, ax = plt.subplots(figsize=(9, 4))
pd.Series(anom_months).value_counts().reindex(range(1,13), fill_value=0).plot.bar(ax=ax, color="indianred", rot=0)
ax.set_xlabel("month"); ax.set_ylabel("# anomaly hours (|z|>4)"); ax.set_title("Anomalies cluster in summer (heat waves)")
eda.savefig(fig, "p6_anom_month.png"); plt.show()"""),

        md(
"""### 4. Structural breaks — change-point detection

Point anomalies are single hours; a **change-point** is a lasting shift in the *level*. We extract the
slow **trend** (STL on the daily mean) and ask change-point detection (`ruptures`, optimal
segmentation) for the few dates where the trend steps to a new regime. The breaks (~2004, 2011, 2015)
mark a rise to a mid-2000s plateau, then **two step-downs** into the post-recession **efficiency era** —
the structural story behind Part 1's declining trend."""),
        co("""daily = s.resample("D").mean()
trend = STL(daily, period=365, robust=True).fit().trend
bkps = rpt.Dynp(model="l2", min_size=180).fit(trend.values).predict(n_bkps=3)
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(trend.index, trend, color="firebrick", lw=1.5)
segs = [0] + bkps
for i in range(len(segs)-1):
    seg = trend.iloc[segs[i]:segs[i+1]]
    ax.hlines(seg.mean(), seg.index[0], seg.index[-1], color="navy", lw=2, ls="--")
    if i < len(segs)-2: ax.axvline(seg.index[-1], color="black", ls=":")
ax.set_ylabel("trend MW"); ax.set_title("Structural breaks in the demand trend (dashed = segment means)")
eda.savefig(fig, "p6_changepoints.png"); plt.show()
print("breaks:", [str(trend.index[b-1].date()) for b in bkps[:-1]])
print("segment means:", [round(trend.iloc[segs[i]:segs[i+1]].mean()) for i in range(len(segs)-1)], "MW")"""),

        md(
"""### Takeaways

- A **calendar-climatology residual** is a cheap, interpretable anomaly detector: ~0.2% of hours
  exceed 4σ.
- The extremes decompose cleanly: **positive = off-season heat waves**, **negative = holidays** — and
  the detector **rediscovered the holiday calendar** unaided (July 4th as the top negative anomaly).
- Anomalies **cluster in summer** (weather-driven volatility).
- **Change-point detection** on the trend finds ~2004/2011/2015 breaks — a rise to a plateau then two
  **step-downs into the efficiency era**, the structure behind the long-run decline.

**Next — Part 7 (Univariate forecasting):** put the structure to work — baselines, MSTL-ETS, harmonic
regression and AutoARIMA for day-ahead demand, scored honestly against seasonal-naive."""),
    ]
    build(cells, "06_anomaly_events.ipynb", "# 06 · PJM Energy Demand — Anomaly & Event Detection")


# ===================================================================== Notebook 7
def notebook_7():
    md, co = new_markdown_cell, new_code_cell
    cells = [
        md(
"""## Part 7 — Univariate Forecasting

Now we *forecast*. The operational question for a grid is **day-ahead demand**; to stress the models we
forecast a **14-day horizon** from a fixed origin (the last fortnight of the data) using only the
series' own history. Four approaches of rising sophistication:

1. **Seasonal-naive** — repeat yesterday / last week (the bar to beat),
2. **Harmonic (Fourier) regression** — deterministic daily+weekly+annual cycles,
3. **MSTL + AutoARIMA** — decompose the multiple seasonalities and model the remainder.

The comparison teaches *why* pure calendar models are not enough, and how much the autocorrelated
remainder is worth. (Part 9 does proper rolling day-ahead backtesting; this is the model bake-off.)"""),
        co(SETUP + """
from src import forecasting as F
from sklearn.linear_model import LinearRegression
d = data.clean_primary(); s = d.load_mw
H = 24*14; train, test = s.iloc[:-H], s.iloc[-H:]
print("train %d hours (-> %s) | test %d hours (%s -> %s)" % (len(train), train.index[-1].date(), H, test.index[0].date(), test.index[-1].date()))"""),

        md(
"""### 1. Seasonal-naive baselines

The honest bar: forecast each hour as the **same hour yesterday** (lag 24) or **last week** (lag 168).
For load, "same hour yesterday" is a genuinely strong baseline — any model that can't beat it adds
nothing."""),
        co("""results = {}
for m, nm in [(24, "snaive-24h"), (168, "snaive-168h")]:
    results[nm] = F.seasonal_naive(train, H, m)
print(pd.DataFrame({nm: F.metrics(test.values, yp, train, nm) for nm, yp in results.items()}).T.to_string())"""),

        md(
"""### 2. Harmonic (Fourier) regression

Represent the three cycles as smooth sine/cosine waves and fit a linear trend + harmonics. This
captures the *deterministic* seasonal shape beautifully — but it has **no memory of recent demand**, so
when a heat wave lifts the whole week above the seasonal norm, it can't follow. Expect a good *shape*,
a wrong *level*."""),
        co("""Xtr = F.fourier_features(train.index); Xte = F.fourier_features(test.index)
Xtr["t"] = (train.index - train.index[0]) / pd.Timedelta(hours=1)
Xte["t"] = (test.index - train.index[0]) / pd.Timedelta(hours=1)
lr = LinearRegression().fit(Xtr, train.values)
results["harmonic-OLS"] = lr.predict(Xte)
print(F.metrics(test.values, results["harmonic-OLS"], train, "harmonic-OLS").to_string())"""),

        md(
"""### 3. MSTL + AutoARIMA

The proper multi-seasonal model: **MSTL** strips out the daily (24) and weekly (168) seasonals, an
**AutoARIMA** forecasts the deseasonalised remainder (capturing the recent level the harmonic model
missed), and the seasonals are added back. This is the state of the art for classical load forecasting
— and the only model here that **beats the seasonal-naive baseline**. (Fitting takes a couple of
minutes.)"""),
        co("""from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoARIMA
sdf = pd.DataFrame({"unique_id": "pjme", "ds": train.index, "y": train.values})
sf = StatsForecast(models=[MSTL(season_length=[24, 168], trend_forecaster=AutoARIMA())], freq="h", n_jobs=1)
sf.fit(sdf); fc = sf.predict(h=H)
results["MSTL-AutoARIMA"] = fc["MSTL"].values
print(F.metrics(test.values, results["MSTL-AutoARIMA"], train, "MSTL-AutoARIMA").to_string())"""),

        md(
"""### 4. The bake-off

All four on one scoreboard and one chart. **MSTL+AutoARIMA wins** (MAPE ~7%, beating seasonal-naive by
~18%); the **harmonic model is worst** — vivid proof that for load you need *both* the seasonal shape
*and* a model of recent deviations. The chart shows harmonic tracking the rhythm but sitting at the
wrong level, while MSTL hugs the actual."""),
        co("""board = pd.DataFrame({nm: F.metrics(test.values, yp, train, nm) for nm, yp in results.items()}).T
print(board.to_string())
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(test.index, test.values, color="black", lw=2, label="actual")
for nm, c in [("snaive-24h","tab:gray"), ("harmonic-OLS","tab:orange"), ("MSTL-AutoARIMA","tab:red")]:
    ax.plot(test.index, results[nm], lw=1.2, alpha=.8, label=nm, color=c)
ax.set_ylabel("MW"); ax.set_title("14-day-ahead forecasts vs actual"); ax.legend(ncol=4)
eda.savefig(fig, "p7_forecasts.png"); plt.show()
print("\\nbest: %s (MAPE %.1f%%, %.0f%% better than snaive-24h)" %
      (board.MAE.idxmin(), board.loc[board.MAE.idxmin(),"MAPE%"], 100*(1-board.MAE.min()/board.loc["snaive-24h","MAE"])))"""),

        md(
"""### Takeaways

- **Seasonal-naive (same hour yesterday)** is a strong baseline (~8.8% MAPE) — the bar.
- **Harmonic regression** nails the *shape* but misses the *level* (no memory) → worst MAPE; pure
  deterministic seasonality is not enough for weather-driven load.
- **MSTL + AutoARIMA** is the classical winner (~7.2% MAPE, ~18% better than naive) — decompose the
  seasonalities, model the remainder.
- The gap between harmonic and MSTL = the value of modelling the **autocorrelated remainder** (recent
  weather).

**Next — Part 8 (ML forecasting):** give a gradient-boosted model **lag features + calendar + Fourier
terms** and see it exploit the persistence (Part 2) to push day-ahead accuracy further."""),
    ]
    build(cells, "07_forecasting_univariate.ipynb", "# 07 · PJM Energy Demand — Univariate Forecasting (naive, harmonic, MSTL-AutoARIMA)")


if __name__ == "__main__":
    import sys
    all_nbs = {"0": notebook_0, "1": notebook_1, "2": notebook_2, "3": notebook_3, "4": notebook_4,
               "5": notebook_5, "6": notebook_6, "7": notebook_7}
    for k in (sys.argv[1:] or sorted(all_nbs, key=int)):
        all_nbs[k]()
    print("done.")

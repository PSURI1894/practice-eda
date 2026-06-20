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


if __name__ == "__main__":
    import sys
    all_nbs = {"0": notebook_0, "1": notebook_1, "2": notebook_2}
    for k in (sys.argv[1:] or sorted(all_nbs, key=int)):
        all_nbs[k]()
    print("done.")

# Deep Dive — Beijing PM2.5, Explained from Scratch

The in-depth, **beginner-friendly** companion to the `beijing-air-quality` practice. No prior
knowledge assumed: every idea gets a plain **definition**, a small **example**, and a note on
**why it appears here**. Read it next to the notebooks (`notebooks/00…06`).

Two files make up the documentation:

1. **This file** — a *concept glossary* (in roughly the order the story uses them) + a *part-by-part
   map* of the seven notebooks.
2. **[CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md)** — every function in `src/`, line by line.

> **The story in one sentence.** We take five years of hourly Beijing air-pollution readings, learn
> to (a) *understand* them, (b) *fill the gaps the sensor left*, and (c) *forecast and warn* about
> bad-air days — checking honestly at each step, and paying special attention to the dangerous extremes.

---

## Table of contents

- [1. The basics](#1-the-basics)
- [2. Looking at one variable (distributions)](#2-looking-at-one-variable-distributions)
- [3. Relationships & structure](#3-relationships--structure)
- [4. Missing data — the centerpiece](#4-missing-data--the-centerpiece)
- [5. Time-series structure](#5-time-series-structure)
- [6. Forecasting](#6-forecasting)
- [7. Forecasting the tail (extremes & warnings)](#7-forecasting-the-tail-extremes--warnings)
- [8. Air-quality domain terms](#8-air-quality-domain-terms)
- [9. Part-by-part map](#9-part-by-part-map)

---

## 1. The basics

**Dataset / rows / columns** — A table where each **row** is one observation and each **column** is
one measured thing. *Example:* here each row is **one hour** (e.g. "3pm, 12 Jan 2013") and columns
are the pollution level, temperature, wind, etc. *Why here:* 43,824 hourly rows × 13 columns.

**Target** — The thing we want to understand and predict: **`pm25`**, the PM2.5 concentration in
µg/m³ (micrograms of fine particles per cubic metre of air). *Why here:* it's the pollution we model.

**Feature (covariate / predictor)** — An input used to explain or predict the target. *Example:*
temperature, wind speed, hour-of-day. *Why here:* the weather and calendar columns.

**Concentration** — A non-negative amount-per-volume. It's bounded below by 0 and usually
**right-skewed** (a long tail of high values). *Example:* most hours moderate, a few catastrophic.
*Why here:* PM2.5 is a concentration, which shapes how we transform and model it.

**Missing values vs missing rows** — Two different kinds of "missing". A **missing row** is an absent
*time* (a gap in the timeline); a **missing value** is a present row with a blank *cell*. *Example:*
here the **timeline is complete** (every hour 2010–2014 exists) but **2,067 `pm25` cells are blank**
(the sensor went down). *Why here:* the central distinction of the whole practice.

---

## 2. Looking at one variable (distributions)

**Distribution** — The full picture of what values a variable takes and how often. *Why here:* we
always inspect shape before modelling.

**Skew (skewness)** — How lopsided a distribution is; **right-skew** = a long tail of large values.
*Example:* PM2.5 has skew 1.8 — many normal hours, a few extreme ones. *Why here:* it drives the log
transform and the extreme-forecasting work.

**Kurtosis** — Tail heaviness; high kurtosis = more extreme outliers than a bell curve. *Why here:*
PM2.5's heavy tail (the airpocalypse spikes).

**Log / log1p transform** — Replacing `x` with `log(x)` (or `log(1+x)`) to pull in a long right tail
and make a skewed variable roughly symmetric. *Example:* PM2.5 skew 1.8 → −0.3 after `log1p`. *Why
here:* the standard move for skewed concentrations — but, as Part 5 shows, *not* a fix for predicting
extremes.

**The four-view battery** — Four plots of one variable at once: **histogram** (shape), **box plot**
(spread + outliers), **ECDF** (percentiles), **Q–Q plot** (is it bell-shaped?). *Why here:*
`eda.four_view` profiles PM2.5 from every angle.

**Zero-inflation** — When a variable is exactly **0** an unusually large share of the time. *Example:*
snow hours (`Is`) and rain hours (`Ir`) are 0 ~99% of the time (it rarely precipitates). *Why here:*
flagged in the quality audit; zero-inflated features need care.

**Outlier** — A value far from the rest; may be an error *or* a real extreme. *Example:* a 900 µg/m³
hour is real (the 2013 airpocalypse), not a typo. *Why here:* the extremes are the health story, so
we keep and study them.

---

## 3. Relationships & structure

**Correlation** — A number in [−1, 1] for how two numeric variables move together. *Example:* wind
speed vs pollution ≈ −0.25 (more wind, less pollution). *Why here:* the meteorology.

**Pearson vs Spearman** — **Pearson** measures a *straight-line* relationship; **Spearman** measures
any *monotonic* (consistently up-or-down) one by correlating ranks. When they differ, the link is
**nonlinear**. *Example:* wind's effect on pollution curves (diminishing returns), so Spearman >
|Pearson|. *Why here:* Part 2 compares them to reveal nonlinearity (a hint that tree models will help).

**Multicollinearity / VIF** — When features carry nearly the *same* information, confusing a model.
The **Variance Inflation Factor** scores each one (>5 notable). *Why here:* used in the EDA toolkit
(temperature and dew point are related).

**Cramér's V** — A 0–1 "correlation" for two **categorical** variables. *Why here:* for the wind-
direction category.

**PCA (Principal Component Analysis)** — Compresses many correlated variables into a few new axes
(components) ordered by how much variation they capture. *Example:* squeezing pollution + weather into
2–3 factors. *Why here:* Part 2 summarises the weather/pollution structure (PC1 ≈ 54% of variance).

**Clustering / k-means** — Groups rows into a chosen number of clusters by similarity. *Example:*
group hours into weather **regimes**, then read off each regime's average pollution. *Why here:* Part
2 finds the dirtiest regime is **cold + low-wind (stagnant)**.

---

## 4. Missing data — the centerpiece

**Missing-data mechanism** — *Why* values are absent, which decides how safely you can fill them:
- **MCAR** (Missing Completely At Random) — absence unrelated to anything; simple fills are safe.
- **MAR** (Missing At Random) — absence depends on *observed* variables; fill *using* them.
- **MNAR** (Missing Not At Random) — absence depends on the *missing value itself* (e.g. the sensor
  fails *because* pollution is off-scale); the dangerous case.
*Why here:* Part 1 diagnoses the mechanism before Part 3 chooses a method.

**KS test (Kolmogorov–Smirnov)** — A test of whether two samples come from the same distribution; a
small p-value means they differ. *Example:* comparing the weather on *missing* vs *present* hours. *Why
here:* it found temperature/pressure differ slightly on missing hours → the gaps lean **MAR**, not
MCAR. (A lesson too: with 43k rows even tiny, unimportant differences become "statistically significant".)

**Gap run / gap structure** — A run of *consecutive* missing values. The *shape* of the missingness
matters as much as the amount. *Example:* here 214 gaps — mostly single hours, but a few **sensor
outages up to 155 hours** (~6.5 days). *Why here:* short gaps are easy to fill, long ones aren't — the
key to Part 3.

**Imputation** — Filling missing values with estimates. The methods compared here:
- **forward-fill (ffill)** — carry the last value forward (a step function).
- **interpolation** — draw a straight line between the bracketing known values.
- **climatology** — fill with the typical value for that *month & hour* (the seasonal average).
- **KNN** — average the K most-similar hours (by weather + time).
- **MICE** — iteratively *regress* the missing variable on the others (multivariate).
*Why here:* Part 3 evaluates all five.

**Mask-and-score evaluation** — You can't grade imputation on *truly* missing data (no answer to
check). So you **hide known values** that mimic the real gaps, impute them, and measure the error
against the truth you hid. *Why here:* the rigorous core of Part 3.

**Stratified evaluation** — Reporting the score *separately by condition*, not just overall. *Example:*
imputation error split by **gap length** — interpolation is great for short gaps, terrible for long
ones. *Why here:* the headline finding that *no single method wins everywhere*.

**Distribution preservation** — A good fill should keep the data's *variability*, not just be close on
average. *Example:* climatology is "accurate-ish" but **collapses the variance** (std 20 vs the true
96), which would bias any later extreme analysis. *Why here:* "accurate ≠ faithful".

**Hybrid imputation** — Combining methods by their strengths: **interpolate short gaps, MICE the long
ones**. *Why here:* it beats every single method and is the recommended fill.

---

## 5. Time-series structure

**Autocorrelation / persistence** — How much a series resembles its own recent past; "persistence" is
the everyday word. *Example:* this hour's pollution ≈ last hour's (correlation **0.97**) — pollution
is "sticky". *Why here:* it's what makes short-term forecasting feasible.

**ACF (Autocorrelation Function)** — Autocorrelation at every lag, as a chart; spikes reveal cycles
(e.g. at lag 24 = daily). *Why here:* Part 2's persistence/seasonality diagnostic.

**Seasonality & multiple seasonality** — Patterns that repeat on a *fixed* period. Pollution has a
**daily** cycle (24h) and an **annual** one (winter heating). *Why here:* the temporal rhythms in
Parts 2 & 6.

**Stationarity** — A series whose behaviour (average, spread) doesn't drift over time. *Why here:*
referenced as the property models prefer; the deseasonalised pollution is roughly stationary.

**Spectral analysis / periodogram** — Re-expressing a series as a sum of sine waves and measuring how
much **power** sits at each frequency; sharp peaks pinpoint the dominant **periods**. *Example:* peaks
at 24 hours and 365 days. *Why here:* Part 6 *quantifies* the cycles the ACF only suggested.

**Spectral leakage** — A periodogram artifact: with a finite, non-whole number of cycles, a true peak
"smears" into nearby fake peaks. *Why here:* the 456-day "peak" is leakage, not a real cycle — a
reminder not to trust every bump.

**Cross-correlation (CCF) / lead-lag** — Correlating one series with a *time-shifted* copy of another,
to see whether one *leads* the other. *Example:* does a wind change *precede* a pollution change? *Why
here:* Part 6 finds weather acts **near-instantly** (peak at lag 0) — so forecasting needs a weather
*forecast*, not just past weather.

**Change-point detection** — Finding times where a series *structurally* shifts to a new level/regime
(here via the **PELT** algorithm). *Example:* did a policy change the pollution baseline? *Why here:*
Part 6 finds **no** structural change 2010–2014 — the air didn't improve in this window.

---

## 6. Forecasting

**Forecast horizon** — How far ahead you predict. *Example:* **24 hours** ahead — the operational
"what's tomorrow's air?" question (and non-trivial, since same-hour-yesterday correlation is only ~0.4).
*Why here:* the task in Parts 4–5.

**Train/test split (time-ordered)** — Fit on earlier data, test on later data the model never saw;
**never shuffle** a time series. *Why here:* the basis of honest scores; we also test only on *real*
(non-imputed) hours.

**Baseline** — A trivially simple forecast a real model must beat: **persistence** ("same as 24h ago")
and **climatology** ("the seasonal average"). *Why here:* the bar — both *lose* to the seasonal-naive
here (day-ahead pollution is genuinely hard).

**MASE / RMSE** — Accuracy metrics. **RMSE** = typical error size (punishes big misses); **MASE** =
error ÷ the naive baseline's error, so **<1 beats naive**. *Example:* the best model reaches MASE 0.68.
*Why here:* the model scoreboard.

**Lag feature** — A past value used as an input. *Example:* `lag24` = pollution the same hour
yesterday. At a 24h-ahead horizon, lags ≥24h are *known*, so they're safe. *Why here:* the strongest
predictor (pollution's persistence).

**Exogenous regressor** — An *outside* driver fed to the model. *Example:* wind, temperature, pressure.
*Why here:* the meteorology genuinely drives pollution.

**Fourier terms / harmonic regression** — Smooth sine/cosine waves that let an ordinary regression
represent a seasonal cycle. *Why here:* the classical model in Part 4 (daily + annual waves + weather).

**Gradient boosting / LightGBM** — A model of many small decision trees, each fixing the last's
mistakes; great with tabular features and nonlinear effects. *Why here:* the best forecaster (lags +
weather), MASE 0.68.

**Leakage** — Letting information into the features that wouldn't be available at forecast time. *Why
here:* avoided by using only ≥24h-old lags for a 24h-ahead forecast.

---

## 7. Forecasting the tail (extremes & warnings)

**Why the mean under-predicts extremes** — A standard (squared-error / L2) model predicts the
**conditional mean**. Extreme hours are rare, so the mean sits well below them — the model can't
justify forecasting a spike. *Example:* on actual >300 hours it predicts ~250. *Why here:* the key
limitation Part 5 attacks.

**The log-target trap** — Modelling `log(pm25)` and exponentiating gives the **median/geometric
mean**, which is *lower* than the mean — so it predicts the peaks even *worse*. *Why here:* a tempting
"fix" that backfires; always measure, don't assume.

**Quantile regression** — Training a model to predict a chosen **percentile** (e.g. the 90th or 95th)
instead of the average, via the **pinball loss**. The upper quantiles *track the spikes*. *Why here:*
the right tool for the tail — q95 reaches the extremes and catches 98% of hazardous hours.

**Pinball (quantile) loss** — The scoring rule that makes a model aim at a specific percentile,
penalising the wrong side asymmetrically. *Why here:* the objective behind quantile regression.

**No-free-lunch tradeoff** — Capturing the tail costs average accuracy: higher quantiles improve
extreme-capture and recall but **inflate MAE and cut precision** (over-warning). *Why here:* the
central, honest message of Part 5 — the quantile level encodes your risk appetite.

**Tweedie loss** — A loss for skewed, non-negative targets. *Why here:* tried in Part 5; it barely
helps — a loss tweak alone doesn't solve the tail.

**Classification metrics (precision / recall / F1)** — For a yes/no prediction (here: "will pollution
exceed 150?"). **Recall** = fraction of real hazards caught; **precision** = fraction of alarms that
were real; **F1** = their balance. *Why here:* the air-quality *warning* framing of Parts 4–5.

**Confusion matrix** — A 2×2 table of predicted vs actual yes/no, showing hits, misses, and false
alarms. *Why here:* visualises the exceedance forecast.

**PR curve (precision–recall) & threshold tuning** — As you lower the alarm threshold you catch more
hazards (↑recall) but raise false alarms (↓precision); the curve shows every tradeoff. *Why here:* for
health, a **missed hazard costs more than a false alarm**, so you lower the threshold to buy recall.

**Probabilistic classifier** — A model outputting the **probability** of exceedance (not just yes/no),
which you then threshold by the cost of each error type. *Why here:* the principled warning system
(average-precision 0.83).

---

## 8. Air-quality domain terms

**PM2.5** — Fine particulate matter ≤2.5 µm — small enough to enter the lungs and bloodstream; the
key health-relevant pollutant. *Why here:* the target.

**WHO guideline** — The World Health Organization's safe-exposure level (24-h guideline ≈ 15 µg/m³).
*Example:* Beijing's mean (99) is ~7× this. *Why here:* the health yardstick.

**AQI bands** — Air-Quality-Index categories (Good / Moderate / Unhealthy / … / Hazardous) mapping a
concentration to a health message. *Why here:* `data.aqi_category` turns numbers into health bands; 59%
of hours are "Unhealthy or worse".

**Dispersion** — Wind blowing pollution away. *Example:* strong winds drop PM2.5 sharply. *Why here:*
the dominant clearing mechanism.

**Source attribution** — Working out *where* pollution comes from. *Example:* SE winds stay dirty even
at speed because they *import* pollution from the industrial plains. *Why here:* the wind direction×speed
map in Part 6.

---

## 9. Part-by-part map

Each notebook is generated by `build_notebooks.py`. For each: the **goal**, the `src` it uses, the
**concepts**, and the **headline finding**.

| Part | Notebook | Goal | Key `src` | Headline finding |
|---|---|---|---|---|
| 0 | `00_data_cleaning` | Load + clean | `data.py` | Timeline complete, but `pm25` has 2,067 gaps (runs up to 155 h) |
| 1 | `01_advanced_eda` | Quality, distributions, missingness | `eda.py`, `data.py` | Heavy skew; 59% "Unhealthy+"; missingness leans **MAR** (KS test) |
| 2 | `02_advanced_eda_2` | Temporal, meteorology, structure | `ts.py`, `data.py` | Persistence 0.97; wind disperses; dirtiest regime = cold + stagnant |
| 3 | `03_imputation` | Fill the gaps, evaluated | `impute.py` | No single method wins; **hybrid** (interp short + MICE long) best |
| 4 | `04_forecasting` | 24h-ahead PM2.5 + warning | `forecasting.py` | LightGBM MASE 0.68; flags hazards F1 0.76; under-predicts extremes |
| 5 | `05_extreme_forecasting` | Capture the tail | `forecasting.py` | Quantiles catch 98% of hazards; classifier + tuned threshold for warnings |
| 6 | `06_spectral_diagnostics` | Frequency / lead-lag / structure | — | Daily+annual spectral peaks; weather near-instant; no regime shift |

**The through-line:** *understand → fill honestly → forecast → respect the extremes → diagnose*. The
recurring lesson is that **pollution is a sticky, weather-driven process whose extremes are the whole
point** — and that *how* you handle missing data, *which* loss you optimise, and *what* you measure all
change the conclusion.

> Next: open **[CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md)** for the line-by-line code justification.

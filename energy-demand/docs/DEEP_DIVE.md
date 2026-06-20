# Deep Dive — PJM Energy Demand, Explained from Scratch

The in-depth, **beginner-friendly** companion to the `energy-demand` practice. No prior knowledge
assumed: every idea gets a plain **definition**, a small **example**, and a note on **why it appears
here**. Read it next to the notebooks (`notebooks/00…11`).

Two files make up the documentation:

1. **This file** — a *concept glossary* (in the order the story uses them) + a *part-by-part map* of
   the twelve notebooks.
2. **[CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md)** — every function in `src/`, line by line.

> **The story in one sentence.** We take 16 years of hourly electricity demand for the PJM grid and
> learn to (a) *clean* its local-time quirks, (b) *understand* its rich triple seasonality and
> structure with the full time-series toolkit, and (c) *forecast* it — point and probabilistic — while
> respecting that it's driven by weather and the calendar.

This is the **most time-series-intensive** practice in the collection — the place where the whole
Advanced-EDA + TS curriculum is applied end to end on one exceptionally structured series.

---

## Table of contents

- [1. The basics](#1-the-basics)
- [2. Seasonality & calendar structure](#2-seasonality--calendar-structure)
- [3. Temporal dynamics](#3-temporal-dynamics)
- [4. Time-series foundations](#4-time-series-foundations)
- [5. The frequency domain](#5-the-frequency-domain)
- [6. Multivariate structure](#6-multivariate-structure)
- [7. Anomalies & change-points](#7-anomalies--change-points)
- [8. Forecasting](#8-forecasting)
- [9. Probabilistic forecasting](#9-probabilistic-forecasting)
- [10. Unsupervised profiling](#10-unsupervised-profiling)
- [11. Power-systems terms](#11-power-systems-terms)
- [12. Part-by-part map](#12-part-by-part-map)

---

## 1. The basics

**Time series** — Data indexed by time, where *order matters*. *Example:* electricity demand measured
every hour. *Why here:* the whole practice; unlike a cross-section (e.g. the wine practice), each point
relates to its neighbours.

**Hourly resolution / frequency** — How often the series is sampled. *Example:* one reading per hour →
8,760 points a year. *Why here:* hourly is fine enough to see the daily cycle yet long enough (16 yr)
to see the annual one.

**Megawatt (MW)** — A unit of electrical power (demand). *Example:* PJM East averages ~32,000 MW. *Why
here:* the target variable is demand in MW.

**Local time & daylight saving (DST)** — Wall-clock time shifts by an hour twice a year. *Example:*
the autumn "fall back" makes 1–2 am happen **twice** (duplicate timestamps); spring "spring forward"
**skips** an hour (missing timestamp). *Why here:* the data is in local time, so these create the
data-quality issues Part 0 fixes — ignore them and every seasonal calculation misaligns.

**Interpolation** — Filling a gap with a value between its neighbours. *Example:* a missing 2 am demand
is estimated from 1 am and 3 am. *Why here:* fills the ~30 DST/missing hours so the series is gap-free.

---

## 2. Seasonality & calendar structure

**Seasonality** — A pattern that repeats on a fixed period. *Why here:* demand has *three* at once.

**Triple / multiple seasonality** — Several seasonal cycles stacked together. *Example:* **daily**
(24 h), **weekly** (168 h), and **annual** (8,766 h). *Why here:* the defining feature of load — and
why ordinary single-season tools aren't enough (we use MSTL, Part 3).

**Bimodal seasonality** — A cycle with **two** peaks per period. *Example:* annual demand peaks in
*both* summer (air-conditioning) and winter (heating), with spring/autumn troughs — an "M-shape". *Why
here:* unusual and important; it even shows up in the frequency domain at *half* a year (Part 4).

**Diurnal (daily) cycle** — The within-day rhythm. *Example:* trough ~4 am, peak in the evening. *Why
here:* the strongest, most regular cycle.

**Load shape** — The 24-hour profile of a day. *Example:* summer is a single afternoon peak; winter is
double-peaked (morning + evening). *Why here:* the shape *changes with season* — a trap for fixed
models, and the basis of the clustering in Part 10.

**Calendar effects** — Demand's dependence on weekday/holiday. *Example:* weekends run ~10% below
weekdays; holidays ~6% below (offices and industry idle). *Why here:* big, predictable effects every
forecaster must encode.

**Load-duration curve** — Every hour's demand sorted from highest to lowest. *Example:* the steep left
edge = the rare peak hours the grid must serve. *Why here:* shows the **peak/base ratio (4.3)** — most
capacity exists for a few extreme hours.

---

## 3. Temporal dynamics

**Ramp rate** — The change in demand from one hour to the next. *Example:* the morning wake-up adds
~+2,450 MW/h on average; extremes hit ±6–7 GW/h. *Why here:* grids must *follow* demand, so the ramp
matters as much as the level.

**Intra-day volatility (swing)** — A day's peak minus its trough. *Example:* summer swings ~17 GW vs
~10 GW in winter (A/C). *Why here:* identifies summer as the operationally hard season.

**Autocorrelation / persistence** — How much the series resembles its own past. *Example:* this hour ≈
last hour (corr 0.97), ≈ yesterday (0.89), ≈ last week (0.78). *Why here:* persistence is *why* load
forecasts well — lag features dominate the ML model (Part 8).

**Trend** — The slow long-run movement. *Example:* PJM demand *fell* ~3% over 2003–17 (efficiency &
rooftop solar beating growth). *Why here:* a caution against assuming demand always rises.

---

## 4. Time-series foundations

**Stationarity** — A series whose statistical behaviour (mean, variance) doesn't drift. *Why here:*
many models assume it; we test for it.

**ADF & KPSS tests** — Two stationarity tests with **opposite** null hypotheses (ADF: non-stationary;
KPSS: stationary). *Example:* on raw load they **conflict** — the fingerprint of strong seasonality.
*Why here:* teaches that seasonal series fool stationarity tests; deseasonalise (or seasonally
difference) first.

**Differencing** — Subtracting a lagged value to remove structure. *Example:* `load[t] − load[t−24]`
(seasonal differencing) removes the daily cycle. *Why here:* makes the conflicting tests agree
(stationary).

**ACF / PACF** — The autocorrelation function and its "partial" version (direct effect of each lag).
*Example:* ACF spikes every 24 lags; PACF points to lag-1 and lag-24. *Why here:* the correlation
skeleton that justifies AR / lag-feature models.

**Decomposition (STL / MSTL)** — Splitting a series into **trend + seasonal(s) + remainder**. **STL**
handles one seasonal cycle; **MSTL** handles several at once. *Example:* MSTL peels off the daily and
weekly cycles, leaving a smooth trend. *Why here:* the right tool for *multiple* seasonality; the
"seasonal strength" quantifies which cycle dominates (daily, 0.86).

---

## 5. The frequency domain

**Periodogram / FFT** — Re-expresses a series as a sum of sine waves and shows the **power** at each
frequency; sharp peaks = strong cycles. *Example:* peaks at 24 h, 12 h, 168 h. *Why here:* confirms the
cycles the ACF suggested, precisely.

**Harmonic** — A multiple of a base frequency. *Example:* the 12 h peak is the 2nd harmonic of the
daily cycle — it's what makes the daily shape non-sinusoidal (double-peaked). *Why here:* explains the
winter double peak.

**The ½-year peak** — *Example:* the dominant low-frequency power sits at ~183 days, not 365 — because
the annual cycle is **bimodal** (two peaks/year). *Why here:* an elegant frequency-domain proof of the
bimodality.

**Spectrogram** — A spectrum computed in a sliding window, showing how cycle strength **changes over
time**. *Example:* the daily-cycle power glows brightest each summer. *Why here:* the structure is
non-stationary *in power*.

**Hurst exponent** — Measures long-range dependence (0.5 = random walk, >0.5 = persistent). *Example:*
load scores ~0.84 — high-demand spells persist (heat waves last days). *Why here:* quantifies
long-memory beyond the cycles → very forecastable.

**Spectral entropy** — How spread-out the spectrum is (0 = a pure cycle, 1 = white noise). *Example:*
load ≈ 0.36 → "highly structured". *Why here:* a one-number "forecastability" score.

---

## 6. Multivariate structure

**Panel** — Several time series side by side. *Example:* the 6 PJM zones aligned hourly. *Why here:*
lets us study how regions move together.

**Common factor (via PCA)** — A single latent signal driving many series. *Example:* **PC1 explains
90%** of all zones' variance with equal loadings = *total system demand*, driven by region-wide
weather. *Why here:* shows the zones are near-copies of one weather-driven signal.

**Lead–lag (cross-correlation)** — Whether one series predicts another at a time offset. *Example:* the
zones' cross-correlation peaks at **lag 0** — they move *simultaneously*. *Why here:* weather hits
everywhere at once, so there's no cross-region forecasting shortcut.

---

## 7. Anomalies & change-points

**Anomaly** — A point far from what's expected. *Example:* an off-season heat-wave afternoon, or a
holiday's unexpectedly low demand. *Why here:* grids care intensely about the unusual.

**Climatology baseline** — The expected value from calendar averages (e.g. mean by month × weekend ×
hour). *Example:* residual = actual − expected; large residuals are anomalies. *Why here:* a cheap,
interpretable anomaly detector — it even **rediscovered the holiday calendar** (July 4th as the top
negative anomaly).

**Change-point** — A lasting shift in the series' *level* (vs a one-off anomaly). *Example:* the demand
trend stepping down into the post-recession "efficiency era". *Why here:* detected with `ruptures`;
explains the declining trend as ~2004/2011/2015 structural breaks.

---

## 8. Forecasting

**Forecast horizon** — How far ahead you predict. *Example:* 24 hours (day-ahead, the operational
task). *Why here:* the core forecasting goal; error grows with horizon.

**Baseline (seasonal-naive)** — A trivial forecast a real model must beat. *Example:* "same hour
yesterday" (lag 24) or "last week" (lag 168). *Why here:* a strong bar for load; **MASE** scales error
by it (<1 beats naive).

**MASE / MAPE / RMSE / WAPE** — Accuracy metrics. **MAPE** = mean % error; **RMSE** = typical error
(punishes big misses); **MASE** = error ÷ naive's error. *Why here:* the model scoreboard.

**Harmonic (Fourier) regression** — A linear model using sine/cosine waves for the cycles. *Example:*
captures the daily/weekly shape but has *no memory* of recent demand. *Why here:* shows the limit of
purely deterministic seasonality (it misses weather-driven levels → worst model).

**MSTL + AutoARIMA** — Decompose the seasonalities, ARIMA-forecast the remainder, add seasonals back.
*Why here:* the classical winner; the only Part-7 model to beat seasonal-naive.

**Lag feature / leakage-safe** — A past value used as a predictor; "leakage-safe" means using only lags
≥ the horizon. *Example:* a 24 h-ahead model uses lags ≥ 24 h. *Why here:* the LightGBM model's main
inputs; the discipline that keeps scores honest.

**Gradient boosting (LightGBM)** — Many small trees, each fixing the last's errors. *Example:* with
lag + rolling + calendar + Fourier features it forecasts day-ahead at ~6% MAPE. *Why here:* the overall
winner — it fuses persistence, calendar and nonlinearity.

---

## 9. Probabilistic forecasting

**Prediction interval** — A range expected to contain the actual value with some probability.
*Example:* an 80% interval should cover 80% of actuals. *Why here:* grids hold reserve generation based
on forecast uncertainty.

**Quantile regression** — Predicting a percentile (e.g. 10th, 90th) instead of the mean, to form an
interval. *Example:* LightGBM with a quantile objective. *Why here:* the first attempt at intervals —
which turn out **over-confident** (56% coverage for a nominal 80%).

**Coverage** — The fraction of actuals that fall inside the interval. *Why here:* the calibration
check; raw quantiles under-cover.

**Conformal prediction / CQR** — A calibration step that adjusts intervals to *guarantee* coverage,
using a held-out set. *Example:* CQR widens the band until coverage hits ~85%. *Why here:* fixes the
over-confidence — distribution-free, essential for planning.

**Walk-forward backtest** — Re-training and forecasting forward through time, fold by fold. *Example:*
retrain each month of 2018, forecast that month. *Why here:* proves stability (per-month MAPE ~4–8%),
vs a single flattering split.

---

## 10. Unsupervised profiling

**Clustering (k-means)** — Grouping similar items without labels. *Example:* grouping days by their
24-hour load shape. *Why here:* finds the grid's distinct "day-types".

**Silhouette score** — Measures how well-separated clusters are (picks the best k). *Example:* k=4
scores best. *Why here:* chooses the number of day-shapes.

**Normalisation (shape vs level)** — Z-scoring each day removes its overall size so clustering sees the
*shape*. *Example:* a hot weekday and mild weekday with the same profile cluster together. *Why here:*
we want shape regimes, not just "big vs small" days.

**Regime** — A recurring characteristic state. *Example:* summer-A/C, winter-double-peak,
mild-weekday, weekend. *Why here:* the four day-types, found unsupervised yet matching the calendar.

---

## 11. Power-systems terms

**Base load / peak load** — The minimum demand always present vs the maximum. *Example:* base ~21 GW,
peak ~62 GW. *Why here:* the peak/base ratio (4.3) frames the grid's core economics.

**Air-conditioning vs heating load** — Weather-driven cooling (summer) and heating (winter) demand.
*Why here:* the two causes of the bimodal annual cycle.

**Reserve / capacity planning** — Holding spare generation for peaks and forecast error. *Why here:*
why peaks, ramps and prediction intervals matter operationally.

---

## 12. Part-by-part map

Each notebook is generated by `build_notebooks.py`.

| Part | Notebook | Goal | Key `src` | Headline finding |
|---|---|---|---|---|
| 0 | `00_data_cleaning` | Assemble & clean | `data.py` | DST duplicate/missing hours fixed; clean PJME series |
| 1 | `01_advanced_eda_distributions` | Distribution & calendar | `eda.py` | Triple + **bimodal** seasonality; load-duration peak/base 4.3 |
| 2 | `02_advanced_eda_temporal` | Dynamics | `data.py` | Ramps, summer volatility (17 GW), persistence 0.97 |
| 3 | `03_ts_foundations` | Stationarity, MSTL | `ts.py` | ADF×KPSS **seasonality trap**; MSTL daily strength 0.86 |
| 4 | `04_spectral` | Frequency domain | — | Cycles as peaks; bimodal shows at ½-year; **Hurst 0.84** |
| 5 | `05_multivariate` | Regional panel | `data.py` | **PC1 = 90%** common weather factor; no lead-lag |
| 6 | `06_anomaly_events` | Anomalies & breaks | `data.py` | Detector rediscovers holidays; change-points → efficiency era |
| 7 | `07_forecasting_univariate` | Classical forecasting | `forecasting.py` | MSTL+AutoARIMA beats naive; harmonic worst |
| 8 | `08_forecasting_ml` | ML forecasting | `forecasting.py` | LightGBM day-ahead **6.3% MAPE (MASE 0.94)** |
| 9 | `09_probabilistic_backtest` | Uncertainty | `forecasting.py` | Quantiles under-cover; **CQR** restores ~85%; walk-forward stable |
| 10 | `10_load_profiling` | Day-shape clustering | `data.py` | 4 unsupervised regimes matching the calendar |
| 11 | `11_capstone` | Synthesis | all | Scoreboard + transferable TS lessons |

**The through-line:** *clean for the domain → understand the structure with the full TS toolkit →
forecast point and probabilistic → respect uncertainty and validation.* The recurring lesson is that
**electricity demand is exceptionally structured** (triple seasonality, strong persistence,
one weather factor) — which makes it both a perfect teaching series and genuinely forecastable, *if*
you handle its quirks (DST, multiple seasonality, over-confident intervals) correctly.

> Next: open **[CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md)** for the line-by-line code justification.

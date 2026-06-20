# Deep Dive — Bike-Sharing, Explained from Scratch

This is the in-depth, **beginner-friendly** companion to the `bike-sharing` practice. It assumes
no prior knowledge: every idea is defined in plain language, with a small **example** and a note on
**why it appears here**. Read it alongside the notebooks (`notebooks/00…06`).

Two files make up the documentation:

1. **This file** — a *concept glossary* (what every term means, in order of the story) plus a
   *part-by-part map* of the seven notebooks.
2. **[CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md)** — every function in `src/`, line by line: what each
   line does and *why it was written that way*.

> **The story in one sentence.** We take two years of hourly bike-rental counts and learn to (a)
> *understand* them, (b) *forecast* tomorrow's demand using weather and the calendar, and (c) say
> *how sure* we are — checking honestly at every step.

---

## Table of contents

- [1. The absolute basics](#1-the-absolute-basics)
- [2. Looking at one variable (distributions)](#2-looking-at-one-variable-distributions)
- [3. Relationships between variables](#3-relationships-between-variables)
- [4. Time-series basics](#4-time-series-basics)
- [5. Splitting a series into parts (decomposition)](#5-splitting-a-series-into-parts-decomposition)
- [6. Is it predictable? (stationarity & autocorrelation)](#6-is-it-predictable-stationarity--autocorrelation)
- [7. Forecasting basics](#7-forecasting-basics)
- [8. Forecasting with covariates](#8-forecasting-with-covariates)
- [9. Judging a forecast honestly (evaluation)](#9-judging-a-forecast-honestly-evaluation)
- [10. Saying how sure you are (uncertainty)](#10-saying-how-sure-you-are-uncertainty)
- [11. Opening the black box (interpretability)](#11-opening-the-black-box-interpretability)
- [12. Part-by-part map](#12-part-by-part-map)

---

## 1. The absolute basics

**Dataset / rows / columns** — A table of data. Each **row** is one observation; each **column** is
one measured thing. *Example:* here each row is **one hour** (e.g. "3pm on 5 June 2012") and the
columns are the count of rentals, the temperature, the hour, etc. *Why here:* 17,379 rows × 17
columns of hourly bike rentals.

**Target (label)** — The thing you want to predict. *Example:* `cnt`, the number of bikes rented in
an hour. *Why here:* every model's job is to predict `cnt`.

**Feature (predictor)** — An input the model uses to predict the target. *Example:* temperature,
hour-of-day, whether it's a working day. *Why here:* weather + calendar columns are our features.

**Data type (dtype)** — How a value is stored: a number (`int`/`float`) or text/`category`.
*Example:* a money column stored as text is a red flag. *Why here:* the weather is numeric, but
`season`, `hour`, `weathersit` are stored as **integers** even though they are really categories.

**Categorical vs numeric** — A **numeric** variable has a meaningful scale (20 °C is twice 10 °C). A
**categorical** variable is a label with no arithmetic (`weathersit = 3` isn't "3 of something").
*Why here:* `season`, `weathersit`, `hr`, `weekday` look numeric but are categorical — if you treat
them as numbers, averages and correlations mislead you.

**Data leakage** — Accidentally giving the model information it wouldn't have at prediction time (or
the answer itself). *Example:* `cnt = casual + registered` exactly, so using `casual`/`registered`
as features hands the model the answer. *Why here:* the dataset's #1 trap — we drop those two columns.

**Missing data** — Values that are absent. They can be obvious (`NaN`) or hidden (a blank, a `0`, or
a *missing row*). *Example:* the bike data has no missing *values*, but **165 hours are missing from
the timeline** — a gap you only find by checking the calendar. *Why here:* Part 0/2 fix the gaps.

---

## 2. Looking at one variable (distributions)

**Distribution** — The full picture of what values a variable takes and how often. *Example:* most
hours have moderate rentals, a few have huge spikes. *Why here:* we always look at the shape before
modelling.

**Histogram** — A bar chart that buckets values and counts how many fall in each bucket; shows the
distribution's shape. *Why here:* the first panel of the "four-view battery" on `cnt`.

**Skew (skewness)** — How lopsided a distribution is. **Right-skew** = a long tail of large values.
*Example:* incomes (a few very high) or hourly rentals (a few very busy hours). *Why here:* `cnt` is
right-skewed (skew 1.28) — typical for **counts**.

**Count data** — A target that is a non-negative whole number (0, 1, 2, …). Its spread usually grows
with its average. *Example:* rentals per hour. *Why here:* `cnt` is a count, which later motivates a
**√ transform** and a **Poisson** loss.

**Square-root / log transform** — Replacing `y` with `√y` or `log y` to tame right-skew and make the
spread more even. *Example:* `√cnt` is far more symmetric than `cnt`. *Why here:* the count analogue
of the log transform used on prices in the finance practice.

**The four-view battery** — Four plots of one variable at once: **histogram** (shape), **box plot**
(spread + outliers), **ECDF** (percentiles), **Q–Q plot** (is it bell-shaped?). *Why here:* `eda.four_view`
shows `cnt`'s shape from every angle in one figure.

**Box plot** — A compact summary showing the middle 50% of values (the box), the median, and unusual
points (dots). *Why here:* the battery's outlier view.

**Outlier** — A value far from the rest. It may be an error *or* a real extreme. *Example:* a
perfect-weather rush hour with record rentals (real, keep it) vs a typo. *Why here:* Part 1 flags
extreme hours but treats them as real events.

---

## 3. Relationships between variables

**Correlation** — A number from −1 to +1 saying how two numeric variables move together. +1 = rise
together, −1 = one up/other down, 0 = no linear link. *Example:* warmer weather ↔ more rentals
(positive). *Why here:* we correlate weather with demand.

**Multicollinearity** — When two features carry almost the *same* information, confusing a model
about which to credit. *Example:* `temp` (actual temperature) and `atemp` ("feels-like") are nearly
identical. *Why here:* a model should keep only one.

**VIF (Variance Inflation Factor)** — A score for how redundant a feature is given the others. **>5**
notable, **>10** serious. *Example:* `temp` and `atemp` score **VIF ≈ 44** — almost duplicates. *Why
here:* `eda.vif_table` exposes that redundancy.

**Cramér's V** — A correlation-like number (0 to 1) for two **categorical** variables. *Example:*
`season` and `weathersit` are associated (weather differs by season). *Why here:* correlation only
works for numbers; Cramér's V handles the categorical columns.

**Interaction** — When the effect of one variable *depends on* another. *Example:* the busy hours
differ on workdays (commute peaks) vs weekends (midday hump) — hour's effect **depends on**
`workingday`. *Why here:* the single most important pattern in bike demand; a model must capture it.

**Simpson's-paradox-style trap** — When an overall average hides opposite patterns in subgroups.
*Example:* workdays and weekends have nearly the same *average* hourly demand (193 vs 181) but
**opposite hourly shapes**. *Why here:* the warning to always segment before trusting an average.

---

## 4. Time-series basics

**Time series** — Data recorded over time, where **order matters** and each value is related to its
own past. *Example:* hourly rentals — this hour looks a lot like the last hour. *Why here:* the
whole second half of the project.

**Datetime index** — Labelling each row with its timestamp so the computer knows the time order and
spacing. *Why here:* we build it from the date + hour columns.

**Frequency** — The spacing between observations (hourly, daily, monthly). *Example:* `"h"` = one row
per hour. *Why here:* lags and seasonal cycles only make sense on a regular frequency.

**Gaps & interpolation** — Real timelines have missing slots. **Interpolation** fills them by drawing
a line between known points. *Example:* a missing 3am rental count estimated from 2am and 4am. *Why
here:* the 165 missing hours are filled so the grid is regular before modelling.

**Trend** — The slow, long-run direction of a series. *Example:* bike rentals **grew +63%** from 2011
to 2012 as the service got popular. *Why here:* a component we separate out in decomposition.

**Seasonality** — A pattern that repeats on a **fixed** period. *Example:* demand rises every morning
and evening (a 24-hour cycle). *Why here:* bike demand is strongly seasonal.

**Multiple seasonality** — More than one repeating cycle at once. *Example:* bike demand repeats
**daily** (every 24 h), **weekly** (every 168 h, workweek vs weekend), and **yearly** (summer peak).
*Why here:* the defining feature of this dataset, and why we need special tools (MSTL, Fourier).

**Cycle** — A long swing with *no fixed* period (unlike seasonality). *Example:* a multi-year boom.
*Why here:* distinguished from seasonality; usually folded into the trend.

---

## 5. Splitting a series into parts (decomposition)

**Decomposition** — Breaking a series into **trend + seasonal + leftover (residual)** so each can be
seen on its own. *Example:* separating bike demand's growth from its daily rhythm from random noise.
*Why here:* the core tool of Part 2.

**STL** — "Seasonal-Trend decomposition using Loess": a robust, flexible way to decompose a series —
but it handles **one** seasonal cycle. *Why here:* the building block of MSTL.

**MSTL (Multiple STL)** — STL extended to peel off **several** seasonal cycles in turn (daily *and*
weekly). *Example:* it splits bike demand into trend + a 24-hour component + a 168-hour component +
residual. *Why here:* single STL can't handle multi-seasonality; MSTL can.

**Residual** — What's left after removing trend and seasonality; ideally pure unpredictable noise.
*Example:* the bit of demand a clean model can't explain. *Why here:* a good decomposition leaves a
near-random residual (we check this).

**Amplitude** — How big a cycle's swing is. *Example:* the daily cycle swings by ~851 rentals, the
weekly by ~688 — so the **daily commute cycle is the biggest driver**. *Why here:* ranks the seasonal
effects.

---

## 6. Is it predictable? (stationarity & autocorrelation)

**Stationary** — A series whose behaviour (average, spread) doesn't drift over time. *Example:* the
*leftover* demand after removing trend and seasonality hovers around a constant level. *Why here:*
most classic forecasting tools assume stationarity; the raw demand isn't (it trends), the
deseasonalised residual is.

**ADF test** — A statistical test whose "null" is *"the series is non-stationary (has a unit root)"*.
A small p-value (<0.05) means **reject that → it looks stationary**. *Why here:* one half of the
stationarity check.

**KPSS test** — A test with the **opposite** null: *"the series is stationary"*. Small p-value means
**reject that → non-stationary**. Using both together is more reliable. *Why here:* paired with ADF;
`ts.stationarity_report` combines them into a verdict.

**p-value** — The chance of seeing data this extreme if the "null" idea were true; small (<0.05) =
strong evidence against the null. *Why here:* the decision rule for every test in the project.

**Autocorrelation** — How much a series correlates with an earlier copy of itself. *Example:* this
hour's rentals look like the same hour yesterday (lag 24) and last week (lag 168). *Why here:* it's
the "memory" a forecaster exploits.

**ACF (Autocorrelation Function)** — Autocorrelation at every lag, shown as a chart. *Example:* big
spikes at lag 24 and 168 reveal the daily and weekly cycles. *Why here:* confirms multi-seasonality
and, after MSTL, confirms the residual is clean.

**PACF (Partial ACF)** — Autocorrelation at a lag with the in-between lags removed (the *direct*
link). *Why here:* shown alongside ACF as the standard diagnostic pair.

**White noise** — A series with no autocorrelation — pure randomness with nothing left to predict.
*Why here:* the ideal a good model's residual should resemble.

**Differencing** — Replacing each value with its *change* from a previous point to remove a trend or
season. *Example:* `today − same-hour-yesterday` removes the daily cycle. *Why here:* one way to reach
stationarity; **seasonal differencing** subtracts the value one season ago.

**Over-differencing** — Differencing a series that's already stationary, which adds noise and a
tell-tale strong negative correlation at lag 1. *Why here:* the warning that "more differencing" isn't
"safer" — with two seasons it gets messy, so we prefer MSTL/Fourier.

---

## 7. Forecasting basics

**Forecast** — A prediction of future values. *Example:* tomorrow's hourly rentals. *Why here:* Parts
3–6.

**Forecast horizon** — How far ahead you predict. *Example:* 24 hours ahead. *Why here:* we forecast
each day's next 24 hours.

**Train/test split (time-ordered)** — Fit the model on earlier data (**train**) and check it on later
data it never saw (**test**). For time series you **never shuffle** — the test must be in the future.
*Example:* train on 2011–most of 2012, test on the final weeks. *Why here:* the basis of every honest
score.

**Baseline** — A trivially simple forecast every real model must beat. *Example:* **seasonal-naive** =
"same hour last week" (`yₜ = yₜ₋₁₆₈`). *Why here:* the bar; if a fancy model can't beat it, it's
useless.

**RMSE (Root Mean Squared Error)** — Typical size of the forecast error, in the target's units, with
big misses punished extra. *Example:* "off by ~60 rentals on average." *Why here:* the main accuracy
number.

**MASE (Mean Absolute Scaled Error)** — The error divided by the seasonal-naive's error. **Below 1 =
better than the naive baseline**; it's comparable across datasets. *Example:* MASE 0.6 ≈ "40% better
than doing nothing." *Why here:* the headline model-comparison metric.

---

## 8. Forecasting with covariates

**Covariate / exogenous regressor** — An *outside* variable used to help predict the target.
*Example:* temperature and whether it's a working day help predict rentals. *Why here:* the finance
series had none; bike demand is *driven* by weather and calendar, so this is the payoff.

**Covariate-availability caveat** — Using a covariate as a predictor assumes you'll *know* its future
value. *Example:* using tomorrow's temperature assumes a perfect weather **forecast** — which nobody
has. *Why here:* an honesty point flagged in Part 3 and fully addressed in Part 6.

**Fourier terms** — Pairs of smooth sine/cosine waves used to represent a repeating cycle as ordinary
columns. More pairs (**K**) = a sharper shape. *Example:* ten sine/cosine waves reproduce the daily
commute curve. *Why here:* the trick that lets a normal regression handle multi-seasonality.

**Harmonic regression (dynamic harmonic regression)** — A regression of demand on Fourier seasonal
terms **plus** the covariates. *Example:* `cnt ~ daily-waves + weekly-waves + temp + humidity + workingday`.
*Why here:* the compact, interpretable classical forecaster of Part 3.

**Why not SARIMA here** — SARIMA (the classic seasonal forecaster) handles **one** seasonal period.
Hourly demand has two strong ones (24 and 168), and a seasonal term at lag 168 over 17,000 points is
computationally hopeless. *Why here:* the reason we reach for Fourier instead.

**Gradient boosting / LightGBM** — A machine-learning model that builds many small decision trees,
each fixing the last one's mistakes. Fast and great with tabular features and interactions. *Why
here:* our strongest forecaster.

**Decision tree** — A model that asks yes/no questions to split data into groups and predicts a
constant per group. It **cannot predict values larger than it saw in training** (can't extrapolate a
trend). *Why here:* explains why we model with lags/known features rather than raw level.

**Lag feature** — A past value of the target used as an input. *Example:* `lag24` = rentals the same
hour yesterday, `lag168` = same hour last week. *Why here:* the single most powerful predictor of
demand.

**Rolling feature** — A moving summary of recent values, e.g. the average of the last 24 hours. *Why
here:* a "recent level" signal — but it must be lagged to avoid leakage.

**Leakage & the horizon rule** — A feature is only safe if it's **knowable at the moment you forecast**.
At a **24-hour-ahead** horizon (forecasting tomorrow from tonight), any value ≥24 h old is known — so
`lag24`/`lag168` are fine, but a rolling average ending an hour ago is **not**. *Why here:* the rule
that decides which features are honest; Part 4 fixes a subtle slip from Part 3.

**Recursive vs direct forecasting** — To predict many steps ahead with a one-step model, **recursive**
feeds each prediction back as the next input (errors pile up); **direct** trains a separate model per
horizon. *Why here:* discussed for multi-step; the 24h-ahead framing sidesteps recursion.

**Stress test** — Deliberately checking a model on a hard case. *Example:* the test window is the
**Christmas/New-Year holidays**, when demand *halves* — calendar models over-predict, lag models
adapt. *Why here:* reveals each model's hidden assumptions.

---

## 9. Judging a forecast honestly (evaluation)

**Walk-forward backtest** — Re-run the model the way it would run live: each day, train on everything
so far and forecast the next day; roll forward across many days. *Example:* 56 daily forecasts across
eight weeks. *Why here:* a single test window is luck; the average over many is the honest verdict.

**Per-day error distribution** — Looking at the error for *each* day, not just the overall average.
*Example:* most days are easy (RMSE ~48) but holidays spike (Thanksgiving RMSE 172). *Why here:* the
mean hides *which* days are hard.

**Prediction interval** — A range (not a single number) expected to contain the truth a stated % of
the time. *Example:* "90% sure tomorrow's 5pm demand is between 300 and 480." *Why here:* honest
forecasts come with uncertainty.

**Coverage** — The fraction of real values that actually land inside the interval; should match the
promise. *Example:* a "90%" interval that really catches 90% is well-calibrated. *Why here:* the test
of an interval's honesty.

**Quantile regression** — Training a model to predict a chosen percentile (e.g. the 5th or 95th)
instead of the average, using the **pinball loss**. *Example:* a 5th- and 95th-percentile model give
a 90% band. *Why here:* turns a point forecast into a range.

**Pinball (quantile) loss** — The scoring rule that makes a model aim for a specific percentile,
penalising being on the wrong side asymmetrically. *Why here:* the objective for quantile regression
and the proper score for interval forecasts.

**Conformal prediction** — A method that makes an interval hit its promised coverage **with a
guarantee**, by measuring how wrong it was on held-out data and adjusting. *Why here:* used in Part 3,
and as **CQR** in Part 4.

**CQR (Conformalized Quantile Regression)** — Conformal prediction applied to quantile-regression
intervals: it **widens** them just enough to reach the promised coverage. *Example:* a raw 90% band
that only covered 76% is fixed to 90%. *Why here:* quantile regression usually under-covers; CQR
repairs it.

**Calibration / reliability** — Whether predicted probabilities/quantiles match reality (a "90%"
should be right 90% of the time). A reliability plot shows this. *Why here:* exposes over-confidence
a single coverage number might hide.

**Poisson loss** — A training objective designed for **counts** (where variance grows with the mean)
that also keeps predictions non-negative. *Why here:* the count-aware alternative to squared error;
here it ties with squared error because the lags already carry the signal.

---

## 10. Saying how sure you are (uncertainty)

**Perfect foresight (assumption)** — Pretending you know the future inputs exactly. *Example:* feeding
the model the *real* future weather. *Why here:* Parts 3–5 did this; Part 6 removes the pretence.

**Monte-Carlo simulation** — Answering "what if?" by trying **many random scenarios** and looking at
the spread of outcomes. *Example:* generate 200 plausible weather forecasts, run the demand model on
each, and see how much the demand forecast wobbles. *Why here:* how we turn weather-forecast error
into demand uncertainty.

**Ensemble** — A collection of forecasts (here, one per simulated scenario) whose spread measures
uncertainty. *Why here:* the 200 Monte-Carlo runs form an ensemble; their per-hour spread is the
input-uncertainty band.

**AR(1) noise** — Random "wobble" that is **correlated over time** (a high value tends to be followed
by another high value), unlike independent noise. *Example:* a weather forecast that's too warm at
9am is usually still too warm at 10am. *Why here:* makes the simulated forecast error realistic.

**Input vs model uncertainty** — Two separate reasons a forecast can be wrong: **input** uncertainty
(we got the weather forecast wrong) and **model** uncertainty (demand is just noisy/unpredictable).
*Why here:* Part 6 measures each — for the lag model, weather is only ~4% of the total.

**Variance decomposition** — Splitting total uncertainty into its sources to see which dominates.
*Example:* model (demand noise) ±106 rentals vs weather ±21 → weather is a small slice. *Why here:*
shows *better demand modelling beats better weather forecasts*.

---

## 11. Opening the black box (interpretability)

**Interpretability** — Understanding *why* a model makes its predictions (not just how accurate it
is). *Why here:* a forecast you can't explain is hard to trust or deploy; Part 5.

**Feature importance** — A ranking of which inputs matter most to the model. Three flavours:
- **Gain** — how much each feature improved the model's fit during training (can be biased).
- **Permutation** — shuffle a feature and see how much accuracy drops (honest, model-agnostic).
- **mean |SHAP|** — average size of each feature's contribution per prediction.
*Why here:* the three disagree (gain buries `hr`, permutation/SHAP rank it high) — never trust one.

**SHAP** — A principled method that splits each individual prediction's gap from the **base value**
(average prediction) fairly among the features, and they **add up exactly**. *Example:* a busy 6pm
prediction = average + (last week was busy) + (rush hour) + (warm) + … *Why here:* gives both a global
picture and a per-prediction explanation you can hand to a stakeholder.

**Base value** — The model's average prediction; SHAP explains each prediction as base value + the
feature contributions. *Example:* base ≈ 191 rentals here. *Why here:* the starting point of every
SHAP explanation.

**Beeswarm plot** — A SHAP chart showing, for every feature, the spread of its impact across many
predictions, coloured by the feature's value (so you read direction + magnitude at once). *Why here:*
the global SHAP summary.

**Partial dependence (PDP)** — The average predicted demand as one feature sweeps its range, holding
others fixed; it reveals the *shape* of an effect. *Example:* warmth raises demand up to ~29 °C, then
**saturates** (too hot to ride). *Why here:* shows nonlinearities the linear model couldn't.

**ICE / dependence plot** — Like PDP but for individual cases or showing how one feature's effect
changes with another (an interaction). *Example:* temperature matters *more* at busy hours. *Why
here:* makes interactions visible.

**Predictor vs cause** — A strong predictor is **not** necessarily a cause. *Example:* `lag168` best
predicts demand, but it doesn't *cause* it. *Why here:* the key caveat — interpretability explains the
**model**, not the world.

---

## 12. Part-by-part map

Each notebook is generated by `build_notebooks.py`. For each: the **goal**, the `src` it uses, the
**concepts**, and the **headline finding**.

| Part | Notebook | Goal | Key `src` | Headline finding |
|---|---|---|---|---|
| 0 | `00_data_cleaning` | Load + clean | `data.py` | `cnt = casual + registered` leakage; integer categoricals; normalized weather; 165 missing hours |
| 1 | `01_advanced_eda` | Understand demand | `eda.py`, `data.py` | Working-day vs weekend have equal *averages* but opposite hourly *shapes*; temp/atemp VIF ≈ 44 |
| 2 | `02_ts_foundations` | Time-series structure | `ts.py`, `data.py` | Multi-seasonal (daily 24 + weekly 168); MSTL → stationary residual |
| 3 | `03_forecasting_covariates` | Forecast with weather/calendar | `featurize.py`, `forecasting.py` | Lags win (MASE 0.61 vs naive 1.66); calendar models fail the holiday stress test |
| 4 | `04_backtesting_probabilistic` | Evaluate rigorously | `backtest.py` | 56-day backtest MASE 0.78; quantile intervals under-cover (76%) → CQR fixes to 90% |
| 5 | `05_interpretability` | Explain the model | `featurize.py`, `backtest.py` | Importance lenses disagree on `hr`; temp saturates ~29 °C; hour×workingday interaction |
| 6 | `06_weather_uncertainty` | Honest input uncertainty | `uncertainty.py` | Lag model degrades only ~4% under weather-forecast error; weather = 4% of total uncertainty |

**The through-line:** *understand → make stationary → forecast with the right tools → evaluate
honestly → quantify uncertainty → explain.* The recurring, quantified lesson of Parts 3–6 is that
**recent-demand "memory" (lags) carries demand forecasting** — it dominates importance, adapts through
holidays, and shrugs off imperfect weather forecasts.

> Next: open **[CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md)** for the line-by-line code justification.

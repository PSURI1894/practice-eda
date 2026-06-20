# Deep Dive — Concepts, Definitions & Project Map

This is the in-depth companion to the **sp500-shiller** project. It has two parts:

1. **This file** — a *concept glossary* (every statistical / ML idea used in the project,
   each with a plain **definition**, a small **example**, and **why it's relevant here**),
   followed by a *part-by-part map* of the 11 notebooks.
2. **[CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md)** — a line-by-line justification of every
   function in `src/`: what each line does and *why it was written that way*.

> How to read it: skim the glossary for any term you hit in a notebook, then open the
> CODE_WALKTHROUGH for the exact implementation. Terms are grouped by theme, roughly in the
> order the course introduces them.

---

## Table of contents

- [1. Concept glossary](#1-concept-glossary)
  - [1.1 EDA & distributions](#11-eda--distributions)
  - [1.2 Outliers & robust statistics](#12-outliers--robust-statistics)
  - [1.3 Relationships between variables](#13-relationships-between-variables)
  - [1.4 Missing data](#14-missing-data)
  - [1.5 Dimensionality reduction](#15-dimensionality-reduction)
  - [1.6 Time-series foundations](#16-time-series-foundations)
  - [1.7 Forecasting models](#17-forecasting-models)
  - [1.8 Forecast evaluation](#18-forecast-evaluation)
  - [1.9 Multivariate time series](#19-multivariate-time-series)
  - [1.10 Volatility modelling](#110-volatility-modelling)
  - [1.11 Machine learning & deep learning](#111-machine-learning--deep-learning)
- [2. Part-by-part map](#2-part-by-part-map)

---

## 1. Concept glossary

### 1.1 EDA & distributions

**Exploratory Data Analysis (EDA)** — Looking at data with summaries and plots *before*
modelling, to understand its shape, quality, and relationships. *Example:* checking that a
"money" column is numeric and right-skewed before fitting a regression. *Why here:* Parts 0–1
are pure EDA; the whole project's stance is "look before you model."

**Structural pass** — The very first look: row/column counts, data types (`dtypes`), missing
counts, and basic ranges. *Example:* spotting that Telco `TotalCharges` is stored as text
(`object`) not a number. *Why here:* it caught both datasets' disguised-missingness traps in Part 0.

**dtype** — The storage type pandas assigns a column (`int64`, `float64`, `object`/text,
`category`). *Example:* a money column showing `object` is a red flag — numbers got stored as
strings. *Why here:* interrogating dtype (not just `isna()`) revealed Telco's 11 blank charges.

**The four moments** — Four numbers that summarise a distribution's shape: **mean** (centre),
**variance/standard deviation** (spread), **skewness** (asymmetry), **kurtosis** (tail
heaviness). *Example:* income has a high mean *and* large right skew. *Why here:* `eda.moments()`
reports all four; the four-view battery visualises them.

**Skewness** — Asymmetry of a distribution. Positive (right) skew = a long right tail.
*Example:* house prices (a few mansions pull the mean above the median). *Why here:* S&P prices,
Telco charges, and returns are all right-skewed → log-transform candidates.

**Kurtosis / excess kurtosis** — Tail heaviness. *Excess* kurtosis subtracts 3 so a normal
distribution scores **0**; positive means fatter tails than normal. *Example:* daily stock
returns have excess kurtosis far above 0. *Why here:* S&P monthly returns have **excess
kurtosis ≈ 16.7** — the headline "fat tails" fact of Part 1.

**Fat tails** — Extreme values occur far more often than a bell curve predicts. *Example:* a
"once in 10,000 years" market move happening every few years. *Why here:* it's why we never
model returns as normal and why risk models (Part 7) use Student-t.

**Histogram** — A bar chart of how many observations fall in each value bin; shows the
distribution's shape. *Example:* a two-humped histogram reveals two subpopulations. *Why here:*
the four-view battery's first panel; revealed `MonthlyCharges` is bimodal.

**KDE (Kernel Density Estimate)** — A smooth curve estimating the probability density, like a
continuous histogram. *Example:* overlaying a KDE to see modes without bin artefacts. *Why
here:* drawn on top of histograms in `four_view`.

**Box plot** — A summary showing the median, the interquartile range (the box), and outliers
beyond the "whiskers". *Example:* quickly seeing a handful of extreme values as dots. *Why
here:* the battery's spread/outlier panel.

**ECDF (Empirical Cumulative Distribution Function)** — For each value `x`, the fraction of
data ≤ `x`; answers "what percentile is this?". *Example:* "today's CAPE exceeds 95% of all
history" is an ECDF lookup. *Why here:* the battery's percentile panel; used to frame the
valuation question.

**Q–Q plot (quantile–quantile)** — Plots data quantiles against a reference distribution's
quantiles; points on the diagonal = good match. *Example:* points curving off the line at the
ends = fatter tails than normal. *Why here:* the battery's normality panel; shows returns'
fat tails visually.

**Normality tests (Jarque–Bera, Shapiro–Wilk, D'Agostino)** — Hypothesis tests with H₀ = "the
data is normal"; a small p-value rejects normality. JB uses skew+kurtosis, Shapiro is powerful
for small samples (n ≤ 5000), D'Agostino combines skew and kurtosis. *Example:* JB p ≈ 0 →
not normal. *Why here:* `eda.normality_battery()` runs all three on returns; all reject.

**p-value** — The probability of seeing data this extreme if the null hypothesis were true; small
(< 0.05) ⇒ evidence against H₀. *Example:* p = 0.001 means "very unlikely under H₀". *Why here:*
the decision rule for every test in the project (normality, ADF, KPSS, Granger, Ljung-Box, Kupiec).

**Log transform / log-normal** — Taking `log(x)` of a positive, right-skewed, multiplicative
quantity to make it more symmetric/normal; a variable whose log is normal is *log-normal*.
*Example:* `log(price)` turns exponential growth into a straight line. *Why here:* Part 1 shows
`log(S&P price)` is far more symmetric; logs convert multiplicative series to additive (Part 2).

**Box–Cox** — A family of power transforms (log is a special case) chosen to best stabilise
variance / normalise. *Why here:* mentioned as the general variance-stabilising tool alongside log.

### 1.2 Outliers & robust statistics

**Outlier** — An observation far from the rest. May be an error *or* a real extreme. *Example:* a
59 mm "diamond" (error) vs the Oct-2008 return (real). *Why here:* Part 1's rule — *investigate,
don't auto-delete*; the worst S&P months are real history.

**IQR / Tukey fences** — The interquartile range `IQR = Q3 − Q1`; points outside
`[Q1 − 1.5·IQR, Q3 + 1.5·IQR]` are flagged. *Example:* a value above Q3 + 1.5·IQR is a mild
outlier. *Why here:* one of two robust flags in `eda.outlier_flags()`.

**MAD (Median Absolute Deviation) & modified z-score** — A robust spread: the median of
`|x − median|`; the modified z-score `0.6745·(x − median)/MAD` flags `|z| > 3.5`. *Example:*
MAD ignores the very outliers that would inflate an ordinary standard deviation. *Why here:* the
second, more robust flag in `eda.outlier_flags()`.

**Robust statistics** — Methods that aren't thrown off by a few extreme values (median, MAD,
Spearman). *Example:* the median income vs the mean income in an unequal society. *Why here:*
used throughout because financial data is full of extremes.

### 1.3 Relationships between variables

**Correlation** — A number in [−1, 1] measuring how two variables move together. *Example:*
height and weight ≈ +0.7. *Why here:* the bread-and-butter of bivariate EDA and the panel
analysis in Part 4.

**Pearson correlation** — Measures the strength of a *linear* relationship. *Example:* perfectly
on a straight line ⇒ ±1. *Why here:* the default, but misleading for non-linear/outlier-driven
links — hence the trio.

**Spearman correlation** — Pearson on the *ranks*; measures any *monotonic* relationship.
*Example:* `y = x³` gives Spearman 1 but Pearson < 1. *Why here:* a big Pearson–Spearman gap flags
non-linearity; computed in `eda.correlation_trio()`.

**Kendall's tau** — Based on the fraction of *concordant* vs discordant pairs; another rank
correlation, more robust for small samples. *Why here:* the third member of the correlation trio.

**Mutual information (MI)** — Measures *any* statistical dependence (linear or not); 0 only if the
variables are independent. *Example:* a U-shaped relation has ~0 correlation but positive MI.
*Why here:* Part 1 ranks Telco features' MI with churn to catch non-monotonic signal.

**Point-biserial correlation** — Pearson correlation between a continuous variable and a binary
one. *Example:* tenure vs churn (0/1). *Why here:* the linear contrast against MI for churn drivers.

**Multicollinearity** — When predictors are strongly correlated with *each other*, making model
coefficients unstable. *Example:* including both "height in cm" and "height in inches". *Why here:*
Telco `TotalCharges ≈ tenure × MonthlyCharges`; caught with VIF.

**VIF (Variance Inflation Factor)** — Per-feature score of how much its variance is inflated by
collinearity; **> 5** notable, **> 10** serious. *Example:* VIF = 10 means that feature is 90%
explained by the others. *Why here:* `eda.vif_table()` flags `TotalCharges` (VIF ≈ 9.5).

**Chi-square test of independence** — Tests whether two *categorical* variables are associated by
comparing observed vs expected cell counts. *Example:* is contract type related to churn? *Why
here:* the engine inside Cramér's V.

**Contingency table (crosstab)** — A table of counts for combinations of two categoricals.
*Example:* rows = contract type, columns = churned/not. *Why here:* input to chi-square / Cramér's V.

**Cramér's V** — Association between two categoricals, scaled to [0, 1] (bias-corrected here).
*Example:* V = 0.4 is a moderately strong categorical association. *Why here:* `eda.cramers_v()`
identifies `Contract` as the dominant churn driver (V ≈ 0.41).

### 1.4 Missing data

**MCAR (Missing Completely At Random)** — Missingness unrelated to anything; safe to drop (rare).
*Example:* a sensor randomly drops readings. *Why here:* the benign baseline we contrast against.

**MAR (Missing At Random)** — Missingness depends on *observed* data; impute using that
relationship. *Example:* income missing more often for younger respondents (age is observed).
*Why here:* Telco `TotalCharges` is missing exactly when `tenure = 0` → MAR, fixed by setting 0.

**MNAR (Missing Not At Random)** — Missingness depends on the *unobserved* value itself; hardest
case. *Example:* high earners refusing to report income. *Why here:* defined for completeness;
the dangerous case to rule out before imputing.

**Disguised missingness** — Missing values hidden as something else (blank text, `0`, `-999`) so
`isna()` reports nothing. *Example:* Shiller uses `0` for "not yet reported". *Why here:* both
datasets hide it; Part 0 checks dtype and `(df == 0).sum()`, not just `isna()`.

**Imputation** — Filling missing values with estimates (mean, median, model-based, or a
domain-correct constant). *Example:* setting tenure-0 customers' total charges to 0. *Why here:*
Part 0 imputes by mechanism, not blindly.

### 1.5 Dimensionality reduction

**PCA (Principal Component Analysis)** — Finds new orthogonal axes (principal components) ordered
by how much variance they capture; compresses correlated variables into a few factors. *Example:*
summarising 12 correlated stocks by 2–3 components. *Why here:* Part 4 extracts the "market factor".

**Explained variance ratio** — The share of total variance each component captures. *Example:*
PC1 = 39% means one direction holds 39% of all the movement. *Why here:* PC1's share measures how
"one-directional" the market is.

**Loadings** — How strongly each original variable contributes to a component. *Example:* PC1 with
all-positive loadings = every stock loads the same way = the market. *Why here:* used to interpret
PC1 as market and PC2 as sector tilt.

**Scree plot** — Bar/line plot of explained variance per component; the "elbow" suggests how many
to keep. *Why here:* Part 4's PCA diagnostic.

**Standardization (z-scoring)** — Rescaling each variable to mean 0, standard deviation 1.
*Example:* `(x − mean)/sd`. *Why here:* required before PCA (so big-variance columns don't
dominate) and before neural-net training (Part 9).

**Factor model / market factor** — Modelling many assets as driven by a few common factors;
the first/biggest is the "market". *Example:* CAPM's single market factor. *Why here:* PCA's PC1
is the empirical market factor — the seed of CAPM/Fama-French.

### 1.6 Time-series foundations

**Time series** — Data indexed by time, where *order matters* and observations are correlated
with their own past. *Example:* monthly S&P prices. *Why here:* the whole second half of the
project; breaks the i.i.d. assumptions of cross-sectional stats.

**DatetimeIndex / frequency** — A pandas index of timestamps with a declared spacing (`MS` =
month-start, `D` = daily). *Example:* a gapless monthly grid. *Why here:* Part 2's "index
hygiene"; lags, rolling windows, and seasonality all depend on a correct frequency.

**Resampling** — Changing a series' frequency: *down* (aggregate, e.g. monthly→annual) or *up*
(interpolate). *Example:* year-end price via `resample("YE").last()`. *Why here:* Part 2 shows
returns *compound*, not add, across resampled periods.

**Trend** — The long-run direction of a series. *Example:* the upward drift of stock prices.
*Why here:* a component of decomposition; the thing that makes price non-stationary.

**Seasonality** — A pattern that repeats on a *fixed* period (month-of-year, day-of-week).
*Example:* CO₂ rising and falling every 12 months. *Why here:* CO₂ has strong seasonality; the
S&P 500 has essentially none (seasonal strength ≈ 0).

**Cycle** — Longer swings with *no fixed* period (business cycles, bull/bear markets). *Example:*
multi-year expansions and recessions. *Why here:* distinguished from seasonality; usually folded
into the trend.

**Additive vs multiplicative** — Whether components combine as `T + S + R` (constant seasonal
size) or `T × S × R` (seasonal size grows with the level). *Example:* retail sales whose holiday
spike grows as the business grows = multiplicative. *Why here:* logging a multiplicative series
makes it additive — why we log financial data before decomposing.

**Decomposition** — Splitting a series into trend + seasonal + residual. *Example:* separating
CO₂'s climb from its yearly breathing. *Why here:* Part 2 uses classical and STL decomposition.

**STL (Seasonal-Trend decomposition using Loess)** — A robust, flexible decomposition that lets
seasonality evolve and resists outliers. *Example:* handling a seasonal pattern that slowly
changes shape. *Why here:* the preferred decomposition; applied to CO₂ and S&P.

**Seasonal/trend strength** — A [0,1] score (Hyndman) of how much of the variation a component
explains: `1 − Var(resid)/Var(component+resid)`. *Example:* seasonal strength 0.98 = very
seasonal. *Why here:* rigorously shows CO₂ seasonal strength 0.98 vs S&P 0.00.

**Stationarity** — A series whose statistical properties (mean, variance, autocovariance) don't
change over time. *Example:* returns hover around a constant mean; prices don't. *Why here:* the
central requirement of ARIMA/VAR; "model returns, not prices".

**Unit root** — A property of a series that makes it a random walk / non-stationary; differencing
removes it. *Example:* a price that wanders without reverting. *Why here:* what the ADF test
checks for.

**ADF test (Augmented Dickey–Fuller)** — H₀: *the series has a unit root* (non-stationary). Small
p ⇒ reject ⇒ stationary. *Example:* ADF p = 0.00 on returns ⇒ stationary. *Why here:* half of the
stationarity decision; `ts.adf_test()`.

**KPSS test** — H₀: *the series is stationary* (the **opposite** null to ADF). Small p ⇒ reject ⇒
non-stationary. *Example:* KPSS p = 0.01 on price ⇒ non-stationary. *Why here:* paired with ADF to
resolve ambiguity; `ts.kpss_test()`.

**ADF × KPSS decision table** — Combining the two opposite-null tests into four verdicts:
stationary / non-stationary (difference) / difference-stationary / trend-stationary (detrend).
*Why here:* `ts.stationarity_report()` returns the verdict; the canonical tool of Part 2.

**Autocorrelation** — Correlation of a series with a lagged copy of itself. *Example:* if warm
days follow warm days, temperature has positive lag-1 autocorrelation. *Why here:* the structure
ARIMA models; measured by ACF/PACF.

**ACF (Autocorrelation Function)** — Autocorrelation at each lag, *including* indirect effects.
*Example:* a slowly decaying ACF signals a trend. *Why here:* identifies MA order and detects
leftover structure in residuals.

**PACF (Partial Autocorrelation Function)** — Autocorrelation at lag *k* with the intermediate
lags removed (the *direct* effect). *Example:* a single PACF spike at lag 1 then nothing ⇒ AR(1).
*Why here:* identifies AR order (the Box-Jenkins table).

**White noise** — A series with no autocorrelation — pure unpredictable noise. *Example:* ideal
model residuals. *Why here:* the target for a good model's residuals (Box-Jenkins adequacy).

**Ljung–Box test** — H₀: *no autocorrelation up to lag k* (i.e. white noise). *Example:* large p
on residuals ⇒ the model captured the structure. *Why here:* `ts.ljung_box()`; used on returns,
squared returns, and model residuals.

**Differencing** — Replacing the level with its change `Δyₜ = yₜ − yₜ₋₁` to remove a trend and
reach stationarity. *Example:* `Δ log(price) = log return`. *Why here:* the "I" in ARIMA, and the
fix that lets trees/nets handle trends (Parts 5, 9).

**Order of integration I(d)** — How many differences are needed to make a series stationary.
*Example:* prices are I(1) (one difference ⇒ stationary returns). *Why here:* sets the ARIMA `d`.

**Seasonal differencing** — Subtracting the value one season ago `yₜ − yₜ₋ₘ` to remove
seasonality. *Example:* CO₂ minus 12-months-ago. *Why here:* the seasonal `D` in SARIMA.

**Over-differencing** — Differencing a series that's *already* stationary, which inflates variance
and stamps a strong **negative** lag-1 autocorrelation. *Example:* differencing returns again.
*Why here:* Part 2's cautionary demo (lag-1 ACF flips +0.27 → −0.31).

**Volatility clustering / ARCH effect** — Large changes tend to follow large changes: the
*variance* has memory even when the level doesn't. *Example:* turbulent and calm market regimes.
*Why here:* squared returns are strongly autocorrelated (Part 2); modelled explicitly in Part 7.

### 1.7 Forecasting models

**Forecast horizon (h)** — How many steps into the future you predict. *Example:* h = 24 months.
*Why here:* fixed at 24 for the CO₂ comparisons.

**Train/test split (time-ordered)** — Holding out the *last* part of the series for testing;
never shuffled (that would leak the future). *Example:* train on 1958–1999, test on 2000–2001.
*Why here:* `forecasting.ts_train_test()`; the discipline behind every honest score.

**Baseline models** — Trivial forecasts every real model must beat: **naive** (repeat last),
**seasonal-naive** (repeat one season ago), **drift** (extend the line), **mean** (historical
average). *Example:* tomorrow's price ≈ today's (naive). *Why here:* `forecasting.baseline_forecasts()`;
the bar (via MASE) for all models.

**Exponential smoothing / ETS** — Forecasts as weights that decay geometrically into the past;
ETS = (Error, Trend, Seasonal) components. *Example:* recent months matter more than old ones.
*Why here:* Part 3's first "real" models.

**SES (Simple Exponential Smoothing)** — ETS with level only; a flat forecast. *Use when:* no
trend, no seasonality. *Why here:* the base of the ETS build-up.

**Holt's linear** — SES + a trend component; a sloped forecast. *Why here:* the second step of the
ETS build-up.

**Holt–Winters** — Holt + a seasonal component (additive or multiplicative). *Example:* tracks
both CO₂'s climb and its yearly cycle. *Why here:* the strongest classical model in the project
(MASE 0.18 on CO₂).

**ARIMA(p, d, q)** — AutoRegressive Integrated Moving Average: AR(`p`) past *values*, I(`d`)
differences, MA(`q`) past *errors*. *Example:* ARIMA(1,1,1). *Why here:* Part 3's core model; `d`
from stationarity, `p,q` from PACF/ACF.

**AR (autoregressive)** — Predicts from past *values* of the series. *Why here:* the `p` in ARIMA.

**MA (moving average)** — Predicts from past *forecast errors*. *Why here:* the `q` in ARIMA.

**SARIMA(p,d,q)(P,D,Q)ₘ** — ARIMA plus a seasonal copy of itself with period `m`. *Example:*
SARIMA(1,1,1)(1,1,1)₁₂ for monthly data. *Why here:* needed for CO₂'s seasonality (plain ARIMA
fails, MASE 1.6).

**Box–Jenkins methodology** — The loop *identify → estimate → check → repeat*; a model is adequate
when residuals are white noise. *Why here:* Part 3's residual diagnostics + Ljung-Box.

**AIC (Akaike Information Criterion)** — Goodness-of-fit penalised by parameter count; lower is
better. *Example:* choosing between ARIMA orders. *Why here:* the score `auto_arima` minimises and
how GJR-GARCH is shown to beat plain GARCH (Part 7).

**auto_arima** — Automatic ARIMA order selection by searching and minimising AIC. *Why here:* used
via **statsforecast** (pmdarima is NumPy-2-incompatible); picks ARIMA(1,1,1)(2,1,1)₁₂ on CO₂.

**Recursive vs direct multi-step** — *Recursive*: predict one step, feed it back, repeat (errors
compound). *Direct*: train a separate model per horizon. *Why here:* `ml_forecast.recursive_forecast()`
implements recursive; both are discussed in Parts 5/9.

**Prophet** — Meta's additive model `trend + seasonality + holidays + noise`; robust, low-effort.
*Why here:* a strong baseline in Part 5 (it actually wins CO₂, MASE 0.17).

### 1.8 Forecast evaluation

**MAE (Mean Absolute Error)** — Average absolute miss, in the series' units. *Example:* "off by
0.24 ppm on average". *Why here:* robust, interpretable; in `forecast_metrics()`.

**RMSE (Root Mean Squared Error)** — Square-root of average squared error; punishes big misses
more. *Why here:* reported alongside MAE; outlier-sensitive.

**MAPE (Mean Absolute Percentage Error)** — Average `|error/actual|` as a %; **explodes when
actuals are near 0** and penalises over- vs under-prediction asymmetrically. *Example:* MAPE
1650% on returns (which cross 0). *Why here:* the cautionary metric — *don't* use it for returns.

**sMAPE (symmetric MAPE)** — A bounded, symmetric percentage variant; still odd near 0. *Why
here:* shown for contrast in `forecast_metrics()`.

**WAPE (Weighted Absolute Percentage Error)** — `Σ|error| / Σ|actual|`; aggregates first so a
single near-zero actual can't hijack it. *Example:* MAPE 56.6% vs WAPE 9.5% on the same series.
*Why here:* the robust %-metric, added in Part 6.

**MASE (Mean Absolute Scaled Error)** — MAE divided by the in-sample naive MAE; **< 1 beats
naive**, and it's comparable across series of different scale. *Example:* MASE 0.18 = 5× better
than naive. *Why here:* the project's default model-selection metric.

**Pinball / quantile loss** — The loss for *probabilistic* (quantile) forecasts; CRPS is its
distribution-wide cousin. *Why here:* named as the right metric when forecasting intervals, not points.

**Time-series cross-validation** — Evaluating over multiple time-ordered folds (never shuffled):
*expanding* (training set grows) or *sliding* (fixed-width window rolls). *Example:* five folds
each testing a later block. *Why here:* `backtest.cv_folds()` / `plot_cv()`; a single split is one
lucky window.

**Gap / embargo** — A buffer between train and test folds so engineered lag/rolling features can't
straddle the boundary. *Why here:* mentioned as the extra safety with lag features.

**Walk-forward backtesting** — Refit-and-roll: forecast, reveal truth, refit, step forward —
mimicking production. *Example:* 126 one-step refits on CO₂. *Why here:* `backtest.walk_forward()`;
the gold-standard evaluation in Part 6.

**Prediction interval** — A range expected to contain the truth (1−α)% of the time, not just a
point. *Example:* a 90% interval. *Why here:* SARIMA's model-based interval (Part 3) and conformal
ones (Part 6/10).

**Coverage** — The fraction of actuals that actually fall inside the interval; should match the
nominal level. *Example:* a 90% interval covering 93.7%. *Why here:* `backtest.coverage()`.

**Conformal prediction** — A *distribution-free* way to build intervals with a coverage guarantee,
calibrated from residual quantiles. *Example:* `point ± (1−α) quantile of |calibration residuals|`.
*Why here:* `backtest.conformal_q()` (Part 6) and neuralforecast's built-in version (Part 10).

**Kupiec POF test** — A likelihood-ratio test that a VaR model's breach rate equals its target.
*Example:* 13 breaches / 1258 days vs a 1% target (p = 0.91 ⇒ well-calibrated). *Why here:*
`volatility.var_backtest()` validates the VaR.

### 1.9 Multivariate time series

**VAR (Vector AutoRegression)** — Each series regressed on past values of *all* series; models
joint dynamics. *Requires:* stationary inputs (returns). *Why here:* Part 4 fits a VAR on a few
stock returns.

**Lag-order selection** — Choosing the VAR/ARIMA number of lags via an information criterion
(AIC/BIC/HQIC). *Why here:* `VAR.select_order()` in Part 4.

**Granger causality** — Series A "Granger-causes" B if A's past improves the forecast of B beyond
B's own past; it's about *predictability, not true causation*, and needs stationary data.
*Example:* MSFT returns leading AAPL returns by a day. *Why here:* `multivariate.granger_matrix()`
builds the directed p-value matrix.

**Spurious regression** — Two unrelated trending (I(1)) series appearing strongly related.
*Example:* two random walks correlating at 0.9 by chance. *Why here:* the trap cointegration is the
legitimate exception to (AAPL & KO look related but aren't cointegrated).

**Cointegration** — Two non-stationary I(1) series whose *linear combination is stationary* — they
share a long-run equilibrium. *Example:* Coca-Cola and Pepsi prices. *Why here:* the basis of pairs
trading; only KO–PEP of six pairs qualifies.

**Engle–Granger test** — Two-step cointegration test: regress one price on the other (the hedge
ratio), then test the residual *spread* for stationarity. *Why here:* `multivariate.engle_granger()`
/ `cointegration_scan()`.

**Johansen test** — A multivariate, rank-based cointegration test (trace statistic vs critical
values) that finds *how many* cointegrating relations exist. *Example:* rank 1 = one equilibrium.
*Why here:* `multivariate.johansen_summary()` confirms KO–PEP.

**VECM (Vector Error Correction Model)** — A VAR on differences plus an *error-correction* term
that pulls cointegrated series back to equilibrium. *Example:* the speed each leg reverts. *Why
here:* models the KO–PEP mean-reversion (Part 4).

**Hedge ratio (β)** — How many units of asset B to short per unit of A so the combination is
stationary. *Example:* spread = KO − β·PEP. *Why here:* the OLS slope in Engle–Granger.

**Pairs trading** — Trading the *spread* of a cointegrated pair: short it when stretched high, long
when stretched low, betting on reversion. *Why here:* the applied payoff of Part 4.

**z-score (of a spread)** — `(spread − mean)/sd`; standardises how stretched the spread is. A
*rolling* z avoids look-ahead. *Example:* |z| > 2 = entry signal. *Why here:* `multivariate.zscore()`.

**DCC-GARCH (Dynamic Conditional Correlation)** — Makes the whole correlation matrix
*time-varying*: univariate GARCH per asset, then a 2-parameter recursion for the conditional
correlation. *Example:* correlations spiking in a crash. *Why here:* `mgarch.dcc()` (Part 8).

**EWMA correlation (RiskMetrics)** — Exponentially-weighted moving correlation; a simple dynamic
baseline (λ = 0.94). *Why here:* the baseline against DCC in Part 8.

### 1.10 Volatility modelling

**Variance vs volatility** — Variance is the squared spread; volatility is its square root (same
units as returns), usually *annualised* by ×√252 (daily) or ×√12 (monthly). *Why here:* the
quantities GARCH forecasts.

**Conditional variance** — The variance *given* the recent past — i.e. time-varying risk.
*Example:* higher expected variance the day after a crash. *Why here:* the output of GARCH.

**ARCH(q)** — Conditional variance as a weighted sum of the last `q` squared shocks (Engle 1982).
*Why here:* the ancestor of GARCH; the ARCH-LM test checks for the effect.

**ARCH-LM test** — H₀: *no ARCH effect* (constant variance). Small p ⇒ volatility clusters.
*Example:* p ≈ 0 on equity returns. *Why here:* `volatility.arch_lm()`; justifies fitting GARCH.

**GARCH(1,1)** — Conditional variance from *both* the last squared shock and the last variance
(Bollerslev 1986): `σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁`. *Why here:* the workhorse volatility model
(`volatility.fit_garch()`).

**Persistence (α + β)** — How slowly variance shocks decay; near 1 ⇒ regimes last for weeks.
*Example:* 0.93. *Why here:* `volatility.persistence()`.

**Long-run / unconditional variance** — The level variance reverts to: `ω/(1 − α − β)`. *Why
here:* the anchor of volatility forecasts (mean reversion).

**Mean reversion** — A series being pulled back toward a long-run level. *Example:* volatility
forecasts decaying to the long-run vol. *Why here:* the shape of the vol term structure (Part 7)
and the spread dynamics (Part 4).

**Student-t innovations** — Using a fat-tailed t-distribution (with `ν` degrees of freedom) for the
shocks instead of normal. *Example:* ν ≈ 6.8 fits returns far better than ν = ∞ (normal). *Why
here:* captures the residual fat tails GARCH alone misses.

**Leverage effect / GJR-GARCH** — Negative shocks raise future volatility more than equal positive
ones; GJR adds a term (`γ > 0`) that only fires on down moves. *Example:* markets fall faster than
they rise. *Why here:* GJR (`o=1`) improves AIC by 42 points in Part 7.

**EGARCH** — An asymmetric GARCH on *log*-variance (guarantees positivity); the log cousin of GJR.
*Why here:* named as the alternative asymmetry model.

**VaR (Value at Risk)** — The loss exceeded only α of the time (e.g. 1% VaR = ~1 day in 100).
*Example:* "we won't lose more than 3% on 99% of days". *Why here:* `volatility.var_series()`; made
time-varying by GARCH.

**Variance risk premium** — Implied volatility (VIX) systematically exceeding realised/GARCH
volatility — investors pay up for protection. *Example:* VIX mean 14.4% vs GARCH 11.1%. *Why
here:* the GARCH-vs-VIX payoff in Part 7.

**VIX** — The market's option-implied 30-day volatility of the S&P 500 (the "fear gauge"). *Why
here:* compared against GARCH (Part 7) and DCC correlation (Part 8).

### 1.11 Machine learning & deep learning

**Supervised learning** — Learning a mapping from features `X` to a target `y` from labelled
examples. *Why here:* the reframing that turns forecasting into regression (Parts 5, 9, 10).

**Feature engineering** — Building informative inputs from raw data. For series: **lag features**
(past values), **rolling features** (moving mean/std), **calendar features** (month). *Why here:*
`ml_forecast.make_supervised()`.

**Leakage / look-ahead bias** — Letting information from the future (or the target) into the
features, so a model looks great in backtest and fails live. *Example:* a rolling mean that includes
today's value. *Why here:* the cardinal sin; prevented with `.shift(1)` (Part 5).

**Gradient boosting / LightGBM** — An ensemble that builds many small decision trees, each
correcting the last; LightGBM is a fast implementation. *Why here:* the tree-based ML model
(Parts 5, 6, 9, 10).

**Decision tree** — A model that splits the feature space into regions and predicts a constant per
region; *cannot extrapolate* beyond seen target values. *Example:* it can't predict a value higher
than any in training. *Why here:* the trend-extrapolation trap that motivates differencing.

**Feature importance** — How much each feature contributed to a tree model. *Example:* lag-12
ranking high ⇒ the model found the seasonality. *Why here:* interpretability check in Part 5.

**Neural network / MLP (multilayer perceptron)** — Stacked layers of `activation(W·x + b)` that
approximate complex functions. *Why here:* built from scratch in `src/neuralnet.py` (Part 9).

**Layer / weights / bias** — A layer linearly mixes its inputs with a weight matrix `W` and bias
`b`, then applies a nonlinearity. *Why here:* the building blocks of the MLP.

**Activation (ReLU)** — A nonlinearity `max(0, z)` that lets a network bend (without it the net is
just linear). *Why here:* the hidden-layer activation in the from-scratch net.

**Forward pass** — Pushing inputs through the layers to produce a prediction. *Why here:*
`MLPRegressor._forward()`.

**Backpropagation** — Applying the chain rule layer-by-layer to get the loss gradients w.r.t. every
weight. *Why here:* the core of `MLPRegressor.fit()` — implemented by hand.

**Gradient descent** — Iteratively nudging parameters opposite the gradient to reduce the loss.
*Why here:* how the net learns.

**Adam** — A gradient-descent optimizer with per-parameter adaptive step sizes and momentum;
converges much faster than plain SGD. *Why here:* implemented from scratch in the MLP.

**He initialization** — Random initial weights scaled by `√(2/fan_in)`, tuned for ReLU, to keep
signals from vanishing/exploding. *Why here:* used in `_init_params`.

**Loss function (MSE)** — The objective minimised; mean squared error for regression. *Why here:*
the MLP's training loss.

**Epoch / batch / learning rate** — One pass over the data (epoch); a small subset per update
(batch / mini-batch); the step size (learning rate). *Why here:* training hyperparameters of the MLP.

**Regularization (L2)** — Penalising large weights to curb overfitting. *Why here:* the `l2` term
in the MLP's gradient.

**Overfitting** — Fitting noise rather than signal, so test performance is poor despite low train
error. *Example:* 2.5M parameters on 480 points. *Why here:* why NHITS fails on the small CO₂
series (Part 10).

**LSTM (Long Short-Term Memory)** — A recurrent network that carries a *hidden state* through time,
giving it sequence memory an MLP lacks. *Why here:* the competitive deep model in Part 10.

**NHITS** — Neural Hierarchical Interpolation for Time Series; multi-rate pooling for long-horizon
forecasting, built for *scale*. *Why here:* over-parameterized and weak on the small CO₂ series — the
honest lesson of Part 10.

**Recurrent network / hidden state** — A network that feeds its own output/state back as input,
maintaining memory across steps. *Why here:* what makes the LSTM more than a windowed MLP.

**Global model** — One model trained across *many* series at once (vs a local per-series model).
*Why here:* the regime where NHITS/LSTM shine — which a single CO₂ series is *not*.

**Foundation model (TimeGPT, Chronos, TimesFM)** — A model pretrained on millions of series that
forecasts *zero-shot* (no training data needed). *Why here:* named as the frontier beyond the
project's scope.

---

## 2. Part-by-part map

Each notebook is generated by `build_notebooks.py` and lives in `notebooks/`. For every part:
the **goal**, the **src** it leans on, the **concepts** exercised, and the **headline finding**.

| Part | Notebook | Goal | Key `src` | Headline finding |
|---|---|---|---|---|
| 0 | `00_data_cleaning` | Acquire + clean both datasets | `data.py` | Disguised missingness: Telco text/MAR, Shiller zero-placeholders |
| 1 | `01_advanced_eda` | Distributions, outliers, association | `eda.py` | Returns' excess kurtosis ≈ 16.7; `Contract` drives churn (V≈0.41) |
| 2 | `02_ts_foundations` | Stationarity, decomposition, ACF/PACF | `ts.py` | Price is I(1), returns stationary; CO₂ seasonal 0.98 vs S&P 0.00 |
| 3 | `03_univariate_forecasting` | Baselines → ETS → ARIMA/SARIMA | `forecasting.py`, `ts.py` | Holt-Winters wins CO₂; nothing beats naive on S&P returns |
| 4 | `04_multivariate` | PCA, VAR, Granger, cointegration, pairs | `multivariate.py` | PC1 = 39% market factor; only KO–PEP cointegrated |
| 5 | `05_ml_forecasting` | Supervised reframing, LightGBM, Prophet | `ml_forecast.py`, `forecasting.py` | Trees can't extrapolate trend (MASE 1.0 → 0.23 after differencing) |
| 6 | `06_evaluation_backtesting` | Metrics, CV, walk-forward, conformal | `backtest.py`, `forecasting.py` | 126-fold backtest confirms HW > LightGBM; conformal 93.7% coverage |
| 7 | `07_volatility_garch` | ARCH/GARCH, leverage, VaR | `volatility.py` | Persistence 0.93; GARCH vs VIX corr 0.75 (variance risk premium) |
| 8 | `08_multivariate_volatility_dcc` | DCC-GARCH dynamic correlation | `mgarch.py`, `volatility.py` | Correlation 0.25→0.46 in Aug-2015; corr 0.70 with VIX |
| 9 | `09_deep_learning_numpy` | Neural net from scratch | `neuralnet.py`, `ml_forecast.py` | Hand-built MLP learns (R²=1) but loses to classical on small data |
| 10 | `10_deep_learning_neuralforecast` | Real NHITS/LSTM | (neuralforecast) | LSTM competitive; NHITS over-parameterized fails; HW still wins |

**The single through-line:** *look before modelling → make it stationary → forecast → evaluate
honestly → only add complexity (multivariate, volatility, ML, DL) when the data justifies it.*
Every model in Parts 3/5/9/10 is scored on the **same CO₂ holdout** with the **same MASE**, so the
comparisons are real.

> Next: open **[CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md)** for the line-by-line justification of
> every `src/` module referenced above.

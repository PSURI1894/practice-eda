# Bike-Sharing — Advanced EDA & Demand Forecasting

A second practice in the `practice-eda` collection, on a very different data regime from
`sp500-shiller`: **hourly bike-rental demand** with rich **exogenous drivers** (weather, calendar)
and **multiple seasonalities** (daily / weekly / yearly) — the things a financial return series
lacks.

> **New to this?** Start with the **beginner-friendly in-depth docs**:
> [`docs/DEEP_DIVE.md`](docs/DEEP_DIVE.md) defines every concept (plain definition + example + why
> it's relevant) and maps each notebook; [`docs/CODE_WALKTHROUGH.md`](docs/CODE_WALKTHROUGH.md)
> justifies every `src/` function line by line.

## Dataset (`data/raw/bike_hour.csv`)
UCI **Bike-Sharing** hourly data, Washington D.C., 2011–2012 (**17,379 rows × 17 cols**;
reassembled from the dotnet/machinelearning-samples train+test mirror). Target = **`cnt`**
(rentals per hour). Features: weather (`temp`, `atemp`, `hum`, `windspeed`, `weathersit`) and
calendar (`season`, `yr`, `mnth`, `hr`, `holiday`, `weekday`, `workingday`).

## Layout
```
data/raw/  data/processed/   source + cleaned CSVs
src/       config · data (bike load/clean) · eda (shared stats & plots)
notebooks/ 00_data_cleaning · 01_advanced_eda   (more parts to come)
reports/figures/             saved PNGs (p1_*)
build_notebooks.py           regenerates notebooks from source
```

## Setup
```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
.venv\Scripts\python.exe -m ipykernel install --user --name bike-sharing --display-name "Python (bike-sharing)"
```
Open the notebooks with the **Python (bike-sharing)** kernel, or run headless:
```powershell
.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks\*.ipynb
```

## Progress
- [x] **Part 0** — Acquire & clean: the `cnt = casual + registered` leakage trap, integer-coded
  categoricals, de-normalized weather (real °C/%/km·h⁻¹), datetime index
- [x] **Part 1** — Advanced EDA: count-target distribution + √ transform, demand rhythms
  (commute vs leisure profiles), weather drivers, temp/atemp collinearity, Cramér's V, outliers
- [x] **Part 2** — TS foundations: hourly index hygiene (gap interpolation), **MSTL** multi-seasonal
  decomposition (daily/weekly), stationarity (ADF×KPSS), ACF/PACF, seasonal differencing
- [x] **Part 3** — Forecasting with covariates: Fourier/dynamic-harmonic regression, exogenous
  weather/calendar regressors, LightGBM (calendar+weather vs +lags), 24h-ahead leakage framing,
  the Christmas/New-Year **holiday stress test**, feature importance, conformal intervals
- [x] **Part 4** — Backtesting & probabilistic forecasting: strict 24h-ahead features, a 56-day
  **walk-forward backtest**, per-day error distribution (holidays as outliers), **quantile
  regression** + pinball loss, **CQR** interval calibration, Poisson loss, reliability/calibration
- [x] **Part 5** — Interpretability: gain vs permutation vs **SHAP** importance, **partial
  dependence** (nonlinear temp/hour effects), the hour×workingday **interaction**, SHAP
  beeswarm/local/dependence plots, and the caveats (PDP independence, predictor ≠ cause)
- [x] **Part 6** — Weather-forecast uncertainty: simulate a day-ahead weather error, **Monte-Carlo**
  propagation into a demand-forecast fan, decompose **input vs model** uncertainty, combined interval

**Complete study (Parts 0–6).**

## Headline findings so far
- **Leakage:** `cnt = casual + registered` exactly → those two must never be predictors.
- **Two rider populations:** working-day and weekend demand have near-identical *averages*
  (193 vs 181) but opposite hourly *shapes* (commute double-peak vs midday hump) — segment before
  trusting an average.
- **Strong trend:** rentals grew **+63%** from 2011 to 2012; plus clear seasonal + weather effects.
- **Multicollinearity:** `temp` and `atemp` are near-duplicates (VIF ≈ 44) — keep one.
- **Index hygiene:** no missing *values*, but **165 hourly slots are absent** from the 2-year grid.
- **Multi-seasonal:** demand has daily (period 24) and weekly (168) cycles — ACF spikes at lag 24 (0.82) and 168 (0.87); **MSTL** ranks the drivers daily (swing 851) > weekly (688) > trend (277). Deseasonalizing leaves a **stationary** residual (its ACF spikes collapse to ≈0).
- **Forecasting:** plain SARIMA can't handle two seasonal periods → use **Fourier/harmonic regression**. Weather is a real signal (+1 °C → +6.9 rentals/hr; +10 % humidity → −12 rentals/hr), but **autoregressive lags win** at 24h-ahead: LGBM+lags MASE **0.61** vs seasonal-naive 1.66. The test window is the **Christmas/NY holidays** (demand halves) — calendar/seasonal models over-predict (Dec 25: harmonic 118 vs actual 42), the lag model adapts. Conformal 90 % interval covers 90 %.
- **Evaluation (Part 4):** a single window is an anecdote — a **56-day walk-forward** backtest gives the honest verdict (MASE **0.78** vs naive 1.47), and the worst days are **holidays** (Thanksgiving RMSE 172 vs median 48). Raw **quantile-regression** intervals **under-cover (76 %)**; **CQR** restores 90 %. Poisson ≈ L2 (lags carry the signal); a reliability plot exposes the over-confidence.
- **Interpretability (Part 5):** the three importance lenses disagree — **gain buries `hr` (3.5 %)** while **permutation/SHAP rank it near the top** (never trust one). Partial dependence shows the nonlinear effects (temperature **saturates ~29 °C**, humidity hurts); the model rediscovered Part 1's **hour×workingday** interaction (workday peak 8am, weekend 5pm); SHAP gives additive per-prediction explanations (base 191 rentals).
- **Input uncertainty (Part 6):** every model used the *realized* weather — perfect foresight. Simulating a day-ahead forecast error (temp RMSE 1.5 °C) and propagating it via Monte-Carlo, the **lag model degrades only ~4 %** vs **~6 %** for the calendar-only model, and weather is just **4 % of total predictive variance** (vs 10 %). **Autoregressive memory buys robustness to input-forecast error** — better demand modelling beats better weather forecasts.

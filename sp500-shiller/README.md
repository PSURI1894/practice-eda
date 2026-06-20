# Advanced EDA + Time Series — hands-on learning project

A runnable, section-by-section walk through **Advanced EDA → Time Series Analysis →
Forecasting (univariate & multivariate) → ML forecasting → Evaluation**, on real-world data.
Seven notebooks (Parts 0–6); reusable statistics live in `src/`.

## Datasets (`data/raw/`)
| file | source | use |
|---|---|---|
| `telco_churn.csv` | IBM Telco Customer Churn (7,043×21) | cross-sectional Advanced EDA — categoricals, missingness, VIF, PCA |
| `sp500_shiller.csv` | Shiller S&P 500 monthly, 1871→ (1,865×10) | time-series & forecasting half |
| `vix_daily.csv` | CBOE VIX daily OHLC (9,210×5) | volatility companion |
| `stock_panel.csv` | 12 S&P 500 large caps, daily 2013–18 (1,259×12) | multivariate: correlation, PCA, VAR, Granger, cointegration/pairs (Part 4) |

## Layout
```
data/raw/         source CSVs (downloaded from GitHub mirrors)
data/processed/   cleaned + feature-engineered outputs (built by Part 0)
src/              config · data · eda · ts · forecasting · multivariate · ml_forecast · backtest
notebooks/        00_data_cleaning · 01_advanced_eda · 02_ts_foundations · 03_univariate_forecasting
                  04_multivariate · 05_ml_forecasting · 06_evaluation_backtesting
reports/figures/  saved PNGs (p1_* … p6_*)
build_notebooks.py  regenerates all notebooks from source (python build_notebooks.py [N])
```

## Setup (one time)
```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
.venv\Scripts\python.exe -m ipykernel install --user --name advanced-eda --display-name "Python (advanced-eda)"
```
> Built on **Python 3.12** on purpose — the system 3.14 lacks wheels for pmdarima/prophet.
> The forecasting extras (pmdarima, prophet, lightgbm) are installed later, in Parts 3 & 5.

## Run
Open a notebook in VS Code / Jupyter and pick the **Python (advanced-eda)** kernel, or run headless:
```powershell
.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks\00_data_cleaning.ipynb
.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks\01_advanced_eda.ipynb
```
Run `00` first (it writes `data/processed/`), then `01`.

## Progress
- [x] **Part 0** — Acquire & clean (two disguised-missingness traps: Telco text/MAR, Shiller zero-placeholders)
- [x] **Part 1** — Advanced EDA: four-view battery, normality/fat-tails, transforms, outliers, correlation trio + MI, VIF, Cramér's V, missing-data mechanisms
- [x] **Part 2** — TS foundations: index hygiene, decomposition (classical/STL on CO₂ + S&P), stationarity (ADF×KPSS decision table), ACF/PACF, volatility clustering (ARCH), differencing
- [x] **Part 3** — Univariate forecasting: time-split + metrics (MASE vs MAPE), baselines, ETS (SES→Holt→Holt-Winters), ARIMA/SARIMA, Box–Jenkins diagnostics, `auto_arima` (statsforecast), S&P reality check
- [x] **Part 4** — Multivariate: clustered correlation, PCA market factor, VAR, Granger-causality matrix, cointegration (Engle–Granger + Johansen), VECM + pairs-trading spread
- [x] **Part 5** — ML forecasting: supervised reframing, leakage discipline (`.shift(1)`), the tree trend-extrapolation trap + difference fix, LightGBM (recursive/direct), feature importance, Prophet, modern landscape
- [x] **Part 6** — Evaluation & backtesting: the metric zoo (MASE/WAPE vs MAPE), time-series CV (expanding/sliding), walk-forward backtester, conformal prediction intervals + course capstone
- [x] **Part 7** _(stretch)_ — Volatility modelling: ARCH-LM test, GARCH(1,1), Student-t fat tails, GJR leverage effect, vol forecasting, **GARCH vs VIX**, time-varying VaR + Kupiec backtest, 150-year regime coda
- [x] **Part 8** _(stretch)_ — Multivariate volatility: **DCC-GARCH** dynamic correlation, EWMA baseline, correlation-spikes-in-crises (Aug 2015), DCC–VIX link, diversification erosion
- [x] **Part 9** _(stretch)_ — Deep learning **from scratch in NumPy**: an MLP with backprop + Adam, the same supervised reframing, honest scoreboard vs classical/LightGBM, and where deep models actually win
- [x] **Part 10** _(stretch)_ — **Real deep models** via `neuralforecast` (PyTorch): NHITS & LSTM with conformal intervals, the grand scoreboard vs every prior model

**Core course complete (Parts 0–6), plus quant & DL stretches (Parts 7–10).**

## Headline findings so far
- S&P 500 monthly returns: **excess kurtosis ≈ 16.7** (normal = 0) — fat tails are the norm, not the exception.
- The 6 most extreme months are real history (1929, 2008, 2020) — *investigate, don't delete*.
- Telco `TotalCharges`: 11 missing values, **100% at tenure 0** → MAR, not random.
- `Contract` is the strongest single churn driver (Cramér's V ≈ 0.41; month-to-month churns ~43% vs ~3% on two-year).
- S&P 500 price is **I(1)** (ADF p=1.0, KPSS p=0.01 → both say non-stationary); its returns are stationary → *model returns, not prices*.
- Decomposition strengths: CO₂ seasonal **0.98** vs S&P 500 seasonal **0.00** — equities trend hard but have no calendar seasonality.
- Returns ≈ white noise, but **squared** returns stay autocorrelated (Ljung-Box grows 17→324) → volatility clustering / ARCH. The lag-1 return autocorrelation (+0.27) is partly a Shiller monthly-averaging artifact.
- Forecasting CO₂: Holt-Winters/SARIMA/AutoARIMA beat every baseline ~8× (MASE ~0.18–0.26 vs naive 1.86); a non-seasonal ARIMA fails (MASE 1.6) — seasonality must be modelled.
- Forecasting S&P **returns**: nothing beats the mean (naive MASE 0.79, ARIMA 0.84) → near-efficient market; forecast *risk*, not direction. MAPE hits 1650% (returns cross 0) → use **MASE**.
- Tooling note: `pmdarima` is NumPy-2-incompatible; this project uses **statsforecast `AutoARIMA`** instead.
- Multivariate: 12 stocks all positively correlated (avg 0.33); **PCA PC1 = 39% = the market factor** (all-positive loadings); clustering recovers sectors unsupervised.
- Cointegration is **rare** — only **KO–PEP** of 6 same-sector pairs (Engle–Granger p=0.03, Johansen rank 1). AAPL–KO are 0.9-correlated in levels yet *not* cointegrated (spurious-regression trap). That stationary spread is the pairs-trading signal (VECM error-correction).
- ML forecasting: trees **can't extrapolate** — LightGBM on CO₂ *levels* fails (MASE 1.01, worse than naive, can't exceed its training ceiling); modelling the **difference** fixes it (MASE 0.23, 4×). On this clean series the decomposition models win (Prophet 0.17 ≈ Holt-Winters 0.18 > LightGBM) — ML isn't automatically better.
- Leakage discipline: every rolling feature is `.shift(1)` (window ends at t-1); the model's feature importance rediscovered the **lag-12 seasonality** on its own.
- Evaluation: a single split is one lucky window — a **126-fold walk-forward** backtest confirms Holt-Winters (MASE 0.196) robustly beats LightGBM (0.219). MAPE 56.6% vs WAPE 9.5% on a near-zero series shows why metric choice matters. **Conformal** intervals hit 93.7% empirical coverage (90% target) with no distributional assumption.
- Volatility (Part 7): returns' *variance* is highly predictable (ARCH-LM p≈0). GARCH(1,1) persistence **0.93**, Student-t ν≈6.8 (fat tails), GJR leverage γ≈0.36 (downside shocks raise vol more). GARCH conditional vol **correlates 0.75 with the VIX**, which runs higher (the variance risk premium). Time-varying 1%/5% VaR backtests clean (Kupiec p=0.91/0.99).
- Multivariate volatility (Part 8): **DCC-GARCH** makes the whole correlation matrix dynamic. Average pairwise correlation swung from **0.25 (calm) to 0.46** on the Aug-2015 sell-off and **correlates 0.70 with the VIX** — diversification erodes in crises (equal-weight portfolio keeps 56% of single-name vol in calm, 71% at the correlation peak).
- Deep learning (Part 9): a hand-built NumPy MLP (backprop + Adam, 3,457 params) learns arbitrary functions (R²=1.0 sanity) and beats naive on CO₂ (MASE 0.32 vs 1.86), but still **loses to tuned Holt-Winters/LightGBM/SARIMA** — neural nets are data-hungry; their edge is many series + scale, not one tidy line.
- Real deep models (Part 10): `neuralforecast` NHITS/LSTM on PyTorch with conformal intervals. **NHITS has 2.5M params for 480 points → over-parameterized, MASE 1.07 (fails); LSTM competitive at 0.32**; Holt-Winters (0.18) still wins the grand scoreboard. Confirms with real libraries what Parts 5/9 showed: match the model to the data regime. _(Requires PyTorch; Smart App Control was disabled to run it.)_

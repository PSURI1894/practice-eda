# Beijing PM2.5 — Air-Quality EDA, Missing Data & Pollution Forecasting

The third practice in the `practice-eda` collection, chosen to exercise the one thing the finance
and bike-sharing practices lacked: **real, structured missing data**. Five years of hourly air
pollution from the US Embassy in Beijing, with weather covariates.

> **New to this?** Start with the **beginner-friendly in-depth docs**:
> [`docs/DEEP_DIVE.md`](docs/DEEP_DIVE.md) defines every concept (plain definition + example + why
> it's relevant) and maps each notebook; [`docs/CODE_WALKTHROUGH.md`](docs/CODE_WALKTHROUGH.md)
> justifies every `src/` function line by line.

## Dataset (`data/raw/beijing_pm25.csv`)
UCI **Beijing PM2.5** data (via the `jbrownlee/Datasets` mirror): **43,824 hourly rows × 13 cols,
2010–2014**. Target = **`pm25`** (PM2.5 concentration, µg/m³). Weather: `DEWP` (dew point), `TEMP`,
`PRES` (pressure), `Iws` (cumulative wind speed), `Is` (snow), `Ir` (rain), `cbwd` (wind direction).
The timeline is complete, but **`pm25` has 2,067 missing values (4.7%)** in 214 gaps.

## Layout
```
data/raw/  data/processed/   source + cleaned CSVs
src/       config · data (air-quality load/clean) · eda · ts (shared helpers)
notebooks/ 00_data_cleaning · 01_advanced_eda (I) · 02_advanced_eda_2 (II) · 03_imputation · 04_forecasting · 05_extreme_forecasting · 06_spectral_diagnostics
src/       …also impute.py (methods + masked-evaluation harness)
reports/figures/             saved PNGs
build_notebooks.py           regenerates notebooks from source
```

## Setup
```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
.venv\Scripts\python.exe -m ipykernel install --user --name beijing-air --display-name "Python (beijing-air)"
```
Open the notebooks with the **Python (beijing-air)** kernel, or run headless with
`jupyter nbconvert --to notebook --execute --inplace notebooks\*.ipynb`.

## Progress
- [x] **Part 0** — Acquire & clean: datetime index, readable names, structural analysis of the
  **missingness** (complete timeline but gappy `pm25`; gap-run lengths, outages up to 155 h)
- [x] **Part 1** — **Advanced EDA I**: data-quality audit, full univariate analysis (the four
  moments, normality, AQI bands), transformations, and the **anatomy of the missingness** (shape +
  a formal MCAR/MAR test)
- [x] **Part 2** — **Advanced EDA II**: temporal rhythms (diurnal × seasonal heatmaps, trend),
  autocorrelation/persistence, the **meteorology** (wind rose, dispersion, Pearson vs Spearman),
  **extreme episodes** (AQI exceedance, episode duration, Jan-2013), and **PCA + pollution regimes**
- [x] **Part 3** — **Missing-data imputation, evaluated**: mask-and-score protocol, five methods
  (ffill/interp/climatology/KNN/MICE) compared **stratified by gap length**, distribution-preservation
  check, and a **hybrid** (interp short + MICE long) that wins — writes a gap-free `beijing_imputed.csv`
- [x] **Part 4** — **Pollution forecasting**: 24h-ahead PM2.5 (baselines → harmonic+weather →
  LightGBM+lags), the **AQI-exceedance warning** framing (precision/recall/F1 + confusion matrix), the
  **systematic under-prediction of extremes**, and feature importance
- [x] **Part 5** — **Forecasting the extremes**: why L2 under-predicts (and why log-target makes it
  worse), **quantile regression** for the tail, the no-free-lunch tradeoff, and a **probabilistic
  exceedance-warning** system with cost-asymmetric threshold tuning (PR curve)
- [x] **Part 6** — **Advanced diagnostics**: spectral analysis (periodogram), cross-correlation
  **lead-lag**, a wind direction×speed **pollution-source** map, and **change-point detection**

**Full study (Parts 0–6): clean → extensive EDA → imputation → forecasting → extremes → diagnostics.**

## Headline findings (extensive EDA)
- **PM2.5 is severe & heavy-tailed** (skew 1.8 → −0.3 after log1p; mean 99 µg/m³); **59% of hours are
  "Unhealthy" or worse** on the US-EPA scale. `Is`/`Ir` (snow/rain) are **zero-inflated** (~99% zeros).
- **Missingness leans MAR, not MCAR:** 2,067 gaps (mostly single hours, but 20 outages >24 h, longest
  **155 h**). A KS test finds temperature/pressure **differ** on missing vs present hours (p≈0) — small
  in magnitude but real → impute *using* the weather; long runs rule out naive interpolation.
- **Pollution is very persistent** (hour-to-hour autocorrelation **0.97**) with a clear daily cycle and
  a **winter + overnight** worst-case (Feb, 1am). Annual means barely improved over 2010–2014.
- **Weather is the master switch:** strong/**NW** winds disperse pollution (cleanest, 70 µg/m³), calm
  air traps it (126); Pearson vs Spearman gaps reveal **nonlinear** relationships. K-means finds the
  dirtiest regime is **cold + low-wind (stagnant)**.
- **Extremes persist for days:** "very unhealthy" spells last up to **4.6 days**; the Jan-2013
  "airpocalypse" peaked at **886 µg/m³ (~59× the WHO guideline)**.
- **Imputation (Part 3):** evaluated by masking known values that mimic the real gaps. **No single
  method wins** — linear interpolation is best for short gaps (MAE 10 at 1-3h) but fails on long ones
  (82 at 49h+) where **MICE** wins (64); **climatology collapses variance** (std 20 vs the true 96).
  A **hybrid** (interp ≤48h + MICE beyond) beats every method (MAE 45 vs 54), and the long outages are
  flagged in the saved gap-free series.
- **Forecasting (Part 4):** day-ahead PM2.5 is hard for naive methods (persistence/climatology MASE
  >1) but **LightGBM+lags+weather wins (MASE 0.68)**. As a warning system it flags hazardous (>150)
  hours at **F1 0.76**, but it **under-predicts the extremes** (actual >300: ~360 vs ~250 forecast;
  the gap *worsens* with severity) — the key caveat for any pollution model. Drivers: recent pollution
  + dew point/pressure, matching the EDA.
- **Extremes (Part 5):** an L2 model predicts the *mean* so it under-predicts spikes — and modelling
  `log(pm25)` makes it **worse** (targets the median). **Quantile regression** captures the tail (q95
  tracks spikes, **0.98 recall** of hazardous hours) but at higher MAE/lower precision — *no free
  lunch*. The principled warning is a **probabilistic exceedance classifier** (avg-precision 0.83) with
  the alarm threshold tuned to the cost asymmetry (lower it to prioritise recall for health).
- **Diagnostics (Part 6):** the **periodogram** confirms power concentrates at the **daily (24h) +
  annual** cycles; **cross-correlation** shows weather acts on pollution **near-instantly** (peak at
  lag 0 → you need a weather *forecast*, not just history); the direction×speed map is a real
  **source-attribution** result (strong N-winds clean to ≈20, SE winds stay dirty ≈100, calm traps
  ≈130); and **change-point detection** finds **no regime shift** — the air didn't improve 2010–14.

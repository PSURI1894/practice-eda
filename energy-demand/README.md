# PJM Energy Demand — Advanced EDA & Time Series (the heavy one)

The fifth practice in the `practice-eda` collection, and the most **time-series-intensive**: sixteen
years of **hourly electricity demand** for the PJM Interconnection (a large US grid operator). Chosen
to exercise the *entire* TS + advanced-EDA toolkit on a single rich series — triple seasonality,
weather-driven extremes, a multivariate regional panel, and real local-time data-quality traps.

## Dataset (`data/raw/pjm_energy.csv`)
PJM hourly demand in **megawatts**, 2002–2018, 11 regional zones (via the `panambY/Hourly_Energy_
Consumption` mirror of the PJM/Kaggle data). Primary series = **PJME** (PJM East, ~145k hourly rows,
the longest gap-free zone); the other zones form a multivariate panel.

## Layout
```
data/raw/  data/processed/   assembled panel + cleaned series
src/       config · data (panel/primary/calendar/overlap) · eda · ts (shared helpers)
notebooks/ 00 … 11            twelve executed notebooks
reports/figures/             saved PNGs
build_notebooks.py           regenerates notebooks from source
```

## Setup
```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
.venv\Scripts\python.exe -m ipykernel install --user --name energy-demand --display-name "Python (energy-demand)"
```
Open the notebooks with the **Python (energy-demand)** kernel, or run headless with
`jupyter nbconvert --to notebook --execute --inplace notebooks\*.ipynb`.

## Curriculum (Parts 0–11)
- [x] **Part 0** — Acquisition & cleaning: regional panel, the **DST local-time trap** (duplicate +
  missing hours), the clean PJME series
- [x] **Part 1** — Advanced EDA I: distribution, **triple seasonality**, summer-vs-winter daily shapes,
  weekend/holiday effects, **load-duration curve**, the declining long-run trend
- [x] **Part 2** — Advanced EDA II: ramp rates (morning +2450 MW/h, extremes ±7 GW/h), summer volatility (17 GW swing), peak-hour shift, rising summer/winter ratio, persistence (lag-1 0.97)
- [x] **Part 3** — TS foundations: ADF×KPSS conflict (seasonality trap, fixed by seasonal differencing), ACF/PACF (24h spikes), **MSTL** (daily strength 0.86, weekly 0.53), annual STL (bimodal)
- [x] **Part 4** — Spectral: periodogram (24h/12h/weekly peaks), the bimodal cycle showing at **183 days (½-year)**, seasonal spectrogram, **Hurst 0.84** long-memory
- [x] **Part 5** — Multivariate: 6-zone panel, all-pairs corr 0.83–0.94, **PCA PC1 = 90%** common (weather) factor (corr 0.99 w/ total), no lead-lag (zones simultaneous)
- [ ] Part 6 — Anomaly & event detection: holidays, extremes, change-points
- [ ] Part 7 — Univariate forecasting: baselines, MSTL-ETS, harmonic regression, AutoARIMA
- [ ] Part 8 — ML forecasting: lag/calendar/Fourier features, LightGBM, multi-horizon
- [ ] Part 9 — Probabilistic forecasting & walk-forward backtesting, prediction intervals, peak
- [ ] Part 10 — Load profiling & day-shape clustering
- [ ] Part 11 — Capstone & synthesis

## Headline findings so far
- **16 years × hourly** PJME demand (14,544–62,009 MW); local-time **DST artifacts** cleaned
  (duplicates dropped, ~30 missing hours interpolated).
- **Triple seasonality** with a distinctive **bimodal** annual cycle — demand peaks in *both* summer
  (A/C) and winter (heating).
- The **daily shape changes with season**: summer is a single afternoon peak (~17:00), winter is
  double-peaked (morning + evening).
- **Weekends −10%, holidays −6%**; **load-duration peak/base ratio 4.3** (the grid is built for a few
  extreme hours); all-time peak **62,009 MW on 2-Aug-2006, 5pm** (summer heat wave).
- The long-run trend is **flat-to-declining (−3% over 2003–17)** — efficiency outpacing growth.

# Bike-Sharing — Advanced EDA & Demand Forecasting

A second practice in the `practice-eda` collection, on a very different data regime from
`sp500-shiller`: **hourly bike-rental demand** with rich **exogenous drivers** (weather, calendar)
and **multiple seasonalities** (daily / weekly / yearly) — the things a financial return series
lacks.

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
- [ ] Part 2 — TS foundations (multi-seasonal decomposition, stationarity, ACF/PACF)
- [ ] Part 3 — Forecasting with covariates (SARIMAX / ETS / LightGBM with weather + calendar)
- [ ] Part 4+ — evaluation & extensions

## Headline findings so far
- **Leakage:** `cnt = casual + registered` exactly → those two must never be predictors.
- **Two rider populations:** working-day and weekend demand have near-identical *averages*
  (193 vs 181) but opposite hourly *shapes* (commute double-peak vs midday hump) — segment before
  trusting an average.
- **Strong trend:** rentals grew **+63%** from 2011 to 2012; plus clear seasonal + weather effects.
- **Multicollinearity:** `temp` and `atemp` are near-duplicates (VIF ≈ 44) — keep one.
- **Index hygiene:** no missing *values*, but **165 hourly slots are absent** from the 2-year grid.

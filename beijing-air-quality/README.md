# Beijing PM2.5 — Air-Quality EDA, Missing Data & Pollution Forecasting

The third practice in the `practice-eda` collection, chosen to exercise the one thing the finance
and bike-sharing practices lacked: **real, structured missing data**. Five years of hourly air
pollution from the US Embassy in Beijing, with weather covariates.

## Dataset (`data/raw/beijing_pm25.csv`)
UCI **Beijing PM2.5** data (via the `jbrownlee/Datasets` mirror): **43,824 hourly rows × 13 cols,
2010–2014**. Target = **`pm25`** (PM2.5 concentration, µg/m³). Weather: `DEWP` (dew point), `TEMP`,
`PRES` (pressure), `Iws` (cumulative wind speed), `Is` (snow), `Ir` (rain), `cbwd` (wind direction).
The timeline is complete, but **`pm25` has 2,067 missing values (4.7%)** in 214 gaps.

## Layout
```
data/raw/  data/processed/   source + cleaned CSVs
src/       config · data (air-quality load/clean) · eda · ts (shared helpers)
notebooks/ 00_data_cleaning · 01_advanced_eda   (more parts to come)
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
- [x] **Part 0** — Acquire & clean: datetime index, readable names, and a structural analysis of the
  **missingness** (complete timeline but gappy `pm25` values; gap-run lengths, outages up to 155 h)
- [x] **Part 1** — Advanced EDA: pollution distribution + WHO exceedance, the **missing-data
  mechanism** (≈MCAR sensor faults), and the **meteorology of pollution** (wind disperses, humidity
  traps; winter peak)
- [ ] Part 2 — Missing-data imputation (forward-fill / interpolation / KNN / MICE), *evaluated*
- [ ] Part 3 — TS foundations (multi-seasonal: daily + yearly), stationarity, ACF/PACF
- [ ] Part 4+ — pollution forecasting with weather covariates, extensions

## Headline findings so far
- **PM2.5 is heavily right-skewed** (skew 1.8, 0–994 µg/m³, mean 99); **84% of hours exceed the WHO
  guideline** (15 µg/m³) and 7% are "hazardous" (>250) — with real Jan-2013 "airpocalypse" extremes.
- **Structured missingness:** 2,067 gaps — mostly single hours (112) but **20 outages >24 h, longest
  155 h**. Weather is similar on missing vs present hours → plausibly **MCAR**, but the long runs rule
  out naive interpolation.
- **Weather drives pollution:** wind **disperses** it (corr −0.25; clean **NW** winds lowest, calm air
  highest), humidity **traps** it (+0.17) — a genuine multivariate structure.
- Strong **winter peak** (February 126 vs August 80 µg/m³, from coal heating + stagnant air).

# practice-eda

Hands-on EDA, time-series, and forecasting practice — **one folder per dataset/practice**.
Each project is self-contained (its own notebooks, data, helper modules, figures, and written
findings) so a new practice never collides with an old one.

## Projects

| folder | dataset(s) | covers |
|---|---|---|
| [`sp500-shiller/`](sp500-shiller/) | S&P 500 (Shiller, 1871→), Telco Churn, Mauna Loa CO₂, 12-stock panel | Full course: Advanced EDA → TS foundations → forecasting → multivariate → ML/DL → evaluation, plus GARCH/DCC volatility stretches (Parts 0–10) |
| [`bike-sharing/`](bike-sharing/) | UCI Bike-Sharing hourly demand (2011–12) | Complete study (Parts 0–6): Advanced EDA → multi-seasonal TS → covariate forecasting → walk-forward backtesting & probabilistic intervals → SHAP interpretability → weather-forecast uncertainty propagation |
| [`beijing-air-quality/`](beijing-air-quality/) | UCI Beijing PM2.5 hourly (2010–14) | Air-quality EDA with **real structured missing data** → missing-data mechanisms & imputation → multi-seasonal TS → pollution forecasting (Parts 0–1; more in progress) |

_New practice → new sibling folder._

## Running a project

Each project pins its own environment. From inside a project folder:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
.venv\Scripts\python.exe -m ipykernel install --user --name advanced-eda --display-name "Python (practice-eda)"
```

Then open the notebooks under `notebooks/` and select the **Python (practice-eda)** kernel,
or run headless with `jupyter nbconvert --to notebook --execute --inplace notebooks\*.ipynb`.

> Built on **Python 3.12** deliberately — the latest scientific/forecasting stack (statsmodels,
> statsforecast) has clean wheels there, unlike the very newest Python releases.

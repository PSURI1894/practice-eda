# Code Walkthrough — PJM Energy Demand, Line by Line

Every function in `src/`, explained for a beginner: **what each line does** and **why it was written
that way**. **Bold terms** are defined in **[DEEP_DIVE.md](DEEP_DIVE.md)**.

The design: notebooks stay short by *calling* these helpers; the real logic lives here. `config`,
`eda` and `ts` are shared with the other practices (shown briefly); the two modules specific to this
practice — `data` (load + DST cleaning + calendar) and `forecasting` (features + metrics) — are shown
in full.

---

## `config.py` / `eda.py` / `ts.py` — shared helpers

- **`config.py`** resolves project paths relative to the file (so notebooks and scripts agree) and
  creates `data/` and `reports/figures/` on import.
- **`eda.py`** — `set_style()`, `savefig(fig, name)`, `four_view(s, …)` (histogram/box/ECDF/Q–Q
  battery), `moments(s)` (mean/std/**skew**/kurtosis). Used in Part 1.
- **`ts.py`** — `adf_test`/`kpss_test`/`stationarity_report` and `acf_pacf_plot`, the **stationarity**
  and autocorrelation helpers used in Part 3.

---

## `data.py` — load, DST-clean & calendar (this practice)

### Constants
```python
REGIONS = ["PJME", "PJMW", "AEP", "COMED", "DAYTON", "DOM", "DUQ", "FE", "DEOK", "NI", "EKPC"]
PRIMARY = "PJME"                                   # the longest, gap-free zone
PANEL_LONG = ["PJME", "PJMW", "AEP", "DAYTON", "DOM", "DUQ"]   # zones with full 2005-2018 overlap
_US_HOL = holidays.US(years=range(2002, 2019))
```
- `PRIMARY` is the series we model (PJM East — 16 full years, no gaps). `PANEL_LONG` is the subset with
  enough overlap for the multivariate analysis (Part 5). `_US_HOL` is a ready-made set of US federal
  **holiday** dates, used to flag demand-suppressing days.

### `load_panel()`
```python
df = pd.read_csv(DATA_RAW / "pjm_energy.csv", parse_dates=["Datetime"]).set_index("Datetime")
return df.sort_index()
```
- Reads the assembled wide panel (one column per zone), parses the timestamp into a **datetime index**,
  and sorts it. The panel is intentionally *ragged* — `NaN` before a zone began reporting.

### `primary()` — the clean modelling series
```python
s = load_panel()[PRIMARY]
s = s[~s.index.duplicated(keep="first")]
full = pd.date_range(s.index.min(), s.index.max(), freq="h")
return s.reindex(full).interpolate("time").rename("load_mw")
```
- Extracts PJME, then handles the **local-time / DST** mess in three steps:
  1. `~s.index.duplicated(keep="first")` drops the autumn "fall-back" **duplicate** hours;
  2. `reindex(full)` snaps onto a *complete* hourly grid, exposing the spring "spring-forward"
     **missing** hours as `NaN`;
  3. `interpolate("time")` fills those few gaps linearly in time (appropriate for a smooth, strongly
     autocorrelated curve).
- *Why this matters:* skip it and the ~30 missing/duplicate hours silently shift every seasonal
  average. This is the single most important cleaning step for a local-time series.

### `add_calendar(df)` — the demand drivers
```python
df["hour"] = idx.hour;  df["dow"] = idx.dayofweek;  df["month"] = idx.month;  df["year"] = idx.year
df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
df["is_holiday"] = pd.Series(idx.date, index=idx).isin(_US_HOL).astype(int).values
df["season"] = (idx.month % 12 // 3).map({0: "winter", 1: "spring", 2: "summer", 3: "fall"})
```
- Attaches the features that *drive* electricity demand. The `month % 12 // 3` trick maps Dec/Jan/Feb →
  winter, etc. (December's 12 % 12 = 0 groups it with Jan/Feb). `is_holiday` tests each date against the
  federal calendar — these days behave like an extra weekend (Part 1, Part 6).

### `clean_primary()` / `panel_overlap()` / `build_processed()`
- `clean_primary()` = the primary series + calendar, the modelling-ready table most notebooks load.
- `panel_overlap()` returns the six long zones on their **common gap-free window** (`dropna()`) — the
  matrix the PCA in Part 5 needs.
- `build_processed()` writes the clean table to `data/processed/`.

---

## `forecasting.py` — features, baselines & metrics (this practice)

### `fourier_features(index, periods)`
```python
t = (index - index[0]) / pd.Timedelta(hours=1)
for P, K in periods:
    for k in range(1, K + 1):
        out[f"sin_{P}_{k}"] = np.sin(2*np.pi*k*t / P)
        out[f"cos_{P}_{k}"] = np.cos(2*np.pi*k*t / P)
```
- Builds smooth **sine/cosine harmonics** for each `(period_in_hours, n_harmonics)` pair (e.g. daily =
  24 h with 5 harmonics). `t` is hours since the series start. A linear model fed these can represent a
  seasonal cycle *and* its harmonics (which shape the non-sinusoidal daily curve). Used by the harmonic
  regression (Part 7) and as calendar context for the ML model (Part 8).

### `ml_features(s, cal, horizon)` — leakage-safe supervised features
```python
for L in sorted({horizon, horizon+1, horizon+2, horizon+24, 168, 168+horizon}):
    if L >= horizon:
        X[f"lag_{L}"] = s.shift(L)
X["roll24"]  = s.shift(horizon).rolling(24).mean()
X["roll168"] = s.shift(horizon).rolling(168).mean()
for c in ("hour", "dow", "month", "is_holiday"): X[c] = cal[c].values
return pd.concat([X, fourier_features(s.index, periods=((8766, 3),))], axis=1)
```
- The supervised reframing for an **`horizon`-hour-ahead** forecast. The crucial discipline is **`if L
  >= horizon`** and the `.shift(horizon)` on the rolling means: every feature is knowable at least
  `horizon` hours before the target, so **nothing leaks**. The rolling means (`roll24`, `roll168`) turn
  out to be the most important features — they summarise the recent level the model rides (**persistence**).
  Calendar columns + an annual Fourier term supply context.

### `seasonal_naive(train, horizon, m)`
```python
last = train.values[-m:]
return np.resize(last, horizon)
```
- The **baseline**: repeat the last full season of length `m` (e.g. `m=168` → "same hour last week").
  `np.resize` tiles it to the forecast length. Simple, but a strong bar for load.

### `mase(...)` and `metrics(...)`
```python
scale = np.mean(np.abs(train.values[m:] - train.values[:-m]))    # in-sample seasonal-naive MAE
return np.mean(np.abs(y_true - y_pred)) / scale
```
- **MASE** divides the forecast error by the in-sample seasonal-naive error, so **< 1 beats "same hour
  yesterday"** — a scale-free, interpretable score. `metrics(...)` bundles MAE, RMSE, MAPE, WAPE and
  MASE into one row, so every model is judged on the same panel (no cherry-picking a flattering metric).

---

## `build_notebooks.py` — the notebook generator

- The twelve notebooks are generated from readable Python via **nbformat** (one cell at a time), then
  executed with `jupyter nbconvert --execute`. *Why:* the cell text stays diff-able and regenerable
  (`python build_notebooks.py 8` rebuilds just Part 8), and **every committed number is actually
  executed**, never hand-typed. A shared `SETUP` string puts the project root on the path so `from src
  import …` resolves identically in every notebook.

---

*Back to **[DEEP_DIVE.md](DEEP_DIVE.md)** for concept definitions, or the
**[energy-demand README](../README.md)** for setup and the part index.*

# Code Walkthrough — Bike-Sharing, Line by Line

Every function in `src/`, explained for a beginner: **what each line does** and **why it was written
that way**. **Bold terms** are defined in **[DEEP_DIVE.md](DEEP_DIVE.md)**.

The design idea: notebooks stay short and readable by *calling* these helpers; all the real logic
and statistics live here in one place. Four modules (`config`, `eda`, `ts`, `forecasting`) are shared
with the finance practice and shown briefly; four (`data`, `featurize`, `backtest`, `uncertainty`)
are bike-specific and shown in full.

---

## `config.py` — where files live (shared)

```python
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW  = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
FIGS      = ROOT / "reports" / "figures"
for _p in (DATA_RAW, DATA_PROC, FIGS):
    _p.mkdir(parents=True, exist_ok=True)
```
- `Path(__file__).resolve().parents[1]` finds the project folder **relative to this file**, not to
  wherever you launched the notebook. *Why:* so a notebook and a script both find `data/` the same way.
- The constants give every other file one agreed name for each folder. *Why:* change the layout once,
  here, instead of editing paths everywhere.
- `mkdir(..., exist_ok=True)` creates the folders if missing, harmlessly if they exist. *Why:* a fresh
  clone won't crash because `reports/figures/` doesn't exist yet.

---

## `eda.py` — distribution & relationship helpers (shared)

Only the functions this practice uses are highlighted.

**`moments(s)`** — returns the four summary numbers of a distribution.
```python
"skew": s.skew(), "excess_kurtosis": s.kurt()
```
- `s.kurt()` is pandas' **excess** kurtosis (0 for a bell curve). *Why:* so the number reads directly
  as "how much fatter-tailed than normal."

**`four_view(s, name, fname)`** — the four-panel battery (histogram, box, ECDF, Q–Q).
- Draws all four views of one variable in a single figure. *Why:* a mean and standard deviation hide
  shape; the battery shows skew, spread, percentiles, and bell-curve-ness at once.
- `if fname: savefig(...)` saves a PNG *and* shows it inline. *Why:* the notebook embeds it and
  `reports/figures/` keeps a reusable copy.

**`outlier_flags(s)`** — counts outliers two robust ways.
```python
lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr           # Tukey fences
mad = np.median(np.abs(s - med)); mod_z = 0.6745*(s - med)/mad   # |z|>3.5
```
- Uses the **IQR** (middle-50% range) and the **MAD** (median-based spread). *Why:* both are
  **robust** — a few extreme values don't distort the very thresholds meant to catch them.

**`cramers_v(x, y)` / `cramers_v_matrix(...)`** — association between two **categorical** columns,
scaled 0–1 (with a bias correction). *Why:* ordinary correlation needs numbers; this measures whether,
say, `season` and `weathersit` go together.

**`vif_table(df_numeric)`** — the **VIF** redundancy score per numeric feature.
```python
X = df_numeric.dropna().assign(_const=1.0)
```
- Adds a constant column before computing VIF. *Why:* the calculation requires an intercept term to be
  correct; without it the numbers are wrong. *Result here:* `temp` and `atemp` score ~44 (near-duplicates).

---

## `ts.py` — stationarity & autocorrelation (shared)

**`adf_test` / `kpss_test`** — the two **stationarity** tests with **opposite nulls**.
```python
return {... "stationary_5pct": p < 0.05}   # ADF: small p => stationary
return {... "stationary_5pct": p > 0.05}   # KPSS: small p => NON-stationary
```
- The `stationary_5pct` flag **flips** between them because their null hypotheses are opposite. *Why:*
  agreement between two opposite tests is far more convincing than either alone.

**`stationarity_report(s)`** — runs both and returns a plain-English **verdict** (stationary /
difference / detrend). *Why:* the useful output is the *action*, not a raw p-value.

**`acf_pacf_plot(s, lags, ...)`** — draws **ACF** and **PACF** side by side. *Why:* the standard pair
for spotting seasonality (big spikes at lag 24 / 168) and checking residuals.

**`ljung_box(s, lags)`** — the formal **white-noise** test (is there leftover autocorrelation?). *Why:*
to confirm a decomposition or model left only noise behind.

---

## `data.py` — load & clean the bike data (bike-specific)

### Constants
```python
SEASON  = {1: "spring", 2: "summer", 3: "fall", 4: "winter"}
WEATHER = {1: "clear", 2: "mist", 3: "light_precip", 4: "heavy_precip"}
CAT_COLS = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
LEAKAGE  = ["casual", "registered"]
```
- `SEASON`/`WEATHER` translate the dataset's integer codes into readable labels. *Why:* "winter" is
  clearer than "4" in plots and tables.
- `CAT_COLS` lists the columns that are stored as integers but are really **categorical**. *Why:* we
  recast them so statistics don't treat `weathersit = 3` as a quantity.
- `LEAKAGE` names the two columns that sum to the target. *Why:* a single place to remember the
  **leakage** trap so every model drops them.

### `clean(df)`
```python
df["datetime"] = pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
df = df.set_index("datetime").sort_index()
```
- Builds a real **datetime index** by adding the hour to the date, then sorts. *Why:* time-series
  tools need to know each row's exact time and order.
```python
df["temp_C"]  = df["temp"]  * 41      # de-normalize
df["hum_pct"] = df["hum"]   * 100
df["wind_kmh"]= df["windspeed"] * 67
```
- The raw weather is **normalized** to 0–1; multiplying back by the UCI constants restores real units
  (°C, %, km/h). *Why:* so `temp_C = 1.0` reads as "41 °C", and coefficients/plots are interpretable.
```python
df["season_name"]  = df["season"].map(SEASON).astype("category")
for c in CAT_COLS: df[c] = df[c].astype("category")
df = df.drop(columns=["instant", "dteday"])
```
- Adds readable labels, recasts the integer categoricals to `category`, and drops the row-id and now-
  redundant date. *Why:* correct types make `groupby`/correlation behave; `instant` is just a counter.

### `features_target(df)`
```python
X = df.drop(columns=LEAKAGE + ["cnt"], errors="ignore")
return X, df["cnt"]
```
- Returns the inputs **without** the two leakage columns and **without** the target. *Why:* this is the
  honest predictor set — anything that includes `casual`/`registered` would cheat.

### `missing_hours(df)`
```python
full = pd.date_range(df.index.min(), df.index.max(), freq="h")
return len(full.difference(df.index))
```
- Builds the *complete* hourly timeline and counts how many of those hours are absent from the data.
  *Why:* finds the **165 missing hours** that `isna()` can't see (they're missing *rows*, not values).

### `hourly_cnt()`
```python
full = pd.date_range(s.index.min(), s.index.max(), freq="h")
out = s.reindex(full).interpolate("time"); out.index.freq = "h"
```
- Puts `cnt` on the complete hourly grid and **interpolates** the gaps (draws a time-weighted line
  through them). *Why:* lags and seasonal periods assume a **regular** index; gaps would misalign them.

### `regular_grid()`
```python
g = df.reindex(full)
for c in [...weather/target...]: g[c] = g[c].interpolate("time")
g["hr"] = g.index.hour; g["dow"] = g.index.dayofweek; g["mnth"] = g.index.month; g["yr"] = g.index.year - 2011
for c in ["workingday","holiday","season","weathersit"]: g[c] = g[c].astype("float").ffill().bfill()
```
- The full frame on a gapless grid: target + weather **interpolated**, calendar columns **rebuilt
  exactly from the timestamp** (the hour/day/month are never uncertain), and slow flags forward/back-
  filled across the few gaps. *Why:* a clean, complete table that the forecasting code (Parts 3–6) can
  index by position without tripping over holes.

---

## `featurize.py` — turning time into features (bike-specific)

```python
CALENDAR = ["hr", "dow", "mnth", "yr", "workingday", "holiday", "season", "weathersit"]
WEATHER  = ["temp_C", "hum_pct", "wind_kmh"]
```
- The two groups of inputs **known in advance** for any future hour (calendar exactly; weather assumed
  from a forecast). *Why:* a forecast may only use features it will actually have.

### `fourier_terms(index, period, K)`
```python
t = np.arange(len(index))
for k in range(1, K+1):
    cols[f"sin{period}_{k}"] = np.sin(2*np.pi*k*t / period)
    cols[f"cos{period}_{k}"] = np.cos(2*np.pi*k*t / period)
```
- Builds `K` sine/cosine wave-pairs that repeat every `period` steps. *Why:* these smooth columns let
  an ordinary regression represent a **seasonal cycle** (daily=24, weekly=168); more `K` = a sharper
  shape. `t = arange(len)` is a simple position counter so the waves stay aligned when train/test are
  sliced from the same index.

### `add_lags(g, target)`
```python
g["lag24"]  = g[target].shift(24)                       # same hour yesterday
g["lag168"] = g[target].shift(168)                      # same hour last week
g["roll24"] = g[target].shift(1).rolling(24).mean()     # last-24h average, ending at t-1
```
- The autoregressive features used by the tree model in Part 3. `shift(1)` *before* `rolling(24)` keeps
  the window from including the current hour. *Why:* these encode "recent demand," the strongest signal.
  (Part 4 notes `roll24` ending at *t−1* is slightly too optimistic for a 24h-ahead forecast and
  replaces it with stricter lags.)

---

## `backtest.py` — honest evaluation (bike-specific)

### `add_strict_lags(g, target)`
```python
g["lag24"]      = g[target].shift(24)
g["lag168"]     = g[target].shift(168)
g["lag_dayavg"] = g[target].shift(24).rolling(24).mean()   # average of the day ending 24h ago
```
- Lag features that are **all ≥24 hours old**, so they're genuinely known when you forecast 24 hours
  ahead. *Why:* fixes the subtle **leakage** in `roll24` (which peeked at the same morning) — rigorous
  evaluation needs rigorous features.

### `walk_forward_24h(g, y, cols, make_model, n_test_days, refit_every, warmup)`
```python
for d in range(n_test_days):
    ds = start + d*24
    if d % refit_every == 0:
        model = make_model(); model.fit(g[cols].iloc[warmup:ds], y.iloc[warmup:ds])
    pred[d*24:(d+1)*24] = model.predict(g[cols].iloc[ds:ds+24])
```
- Simulates daily operation: walk through the test days; **refit** the model every `refit_every` days
  on everything observed so far, then forecast that day's next 24 hours. *Why:* this is the **walk-
  forward backtest** — it averages performance over many windows, the honest verdict (vs one lucky
  test). `make_model` is a function that builds a fresh model, so the same backtester works for any
  model (point, quantile, Poisson).

### `pinball_loss(y_true, q_pred, alpha)`
```python
e = y_true - q_pred
return float(np.mean(np.maximum(alpha*e, (alpha-1)*e)))
```
- The **pinball loss** — the proper score for a single quantile (e.g. the 95th). It penalises being on
  the wrong side asymmetrically. *Why:* it's what makes **quantile regression** aim at a percentile, and
  the right way to grade an interval's edges.

### `coverage(y_true, lo, hi)`
```python
return float(np.mean((y >= lo) & (y <= hi)))
```
- The fraction of true values inside the interval. *Why:* an interval is only honest if its **coverage**
  matches its promise (a "90%" interval should catch ~90%).

### `cqr_offset(y_cal, lo_cal, hi_cal, alpha)`
```python
E = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
return float(np.quantile(E, 1 - alpha))
```
- **CQR**: on held-out *calibration* data, `E` measures how far the truth fell *outside* the
  `[lo, hi]` band (negative if inside); its `(1−alpha)` quantile is how much to **widen** the band.
  *Why:* raw quantile regression under-covers (here 76%); adding this offset restores the promised 90%
  with a guarantee.

---

## `uncertainty.py` — weather-forecast uncertainty (bike-specific)

```python
FORECAST_SD = {"temp_C": 1.5, "hum_pct": 8.0, "wind_kmh": 3.0}
```
- Plausible day-ahead weather-forecast error sizes (temperature ±1.5 °C, etc.). *Why:* we have no real
  historical forecasts, so we **simulate** error of a realistic magnitude.

### `ar1_noise(n, sd, phi, rng)`
```python
e = rng.normal(0, sd*np.sqrt(1-phi**2), n)
for i in range(1, n): x[i] = phi*x[i-1] + e[i]
```
- Generates random wobble that is **correlated over time** (`phi` controls the carry-over) with overall
  size `sd`. *Why:* forecast errors aren't independent hour to hour — too-warm at 9am is usually still
  too-warm at 10am — so **AR(1)** noise is more realistic than independent noise. The `sqrt(1-phi**2)`
  scaling keeps the long-run spread equal to `sd`.

### `perturb_weather(df, rng, sd, phi)`
```python
for c, s in sd.items():
    if c in d: d[c] = np.clip(d[c].to_numpy() + ar1_noise(len(d), s, phi, rng), 0, None)
```
- Returns a copy of the data with each weather column nudged by simulated forecast error; `np.clip(...,
  0, None)` stops humidity/wind going negative. *Why:* this is "the weather we *thought* we'd get."

### `monte_carlo(model, X, cols, n_scenarios, seed, sd)`
```python
for s in range(n_scenarios):
    P[s] = np.clip(model.predict(perturb_weather(X, rng, sd)[cols]), 0, None)
```
- Runs the **fixed** model on `n_scenarios` different simulated weather forecasts and stacks the
  predictions. *Why:* the spread across rows is the demand uncertainty *caused by* the weather forecast
  (a **Monte-Carlo ensemble**). The model isn't refit — only the weather inputs change.

---

## `forecasting.py` — metrics & baselines (shared)

**`forecast_metrics(y_true, y_pred, y_train, m)`** computes MAE, RMSE, MAPE, sMAPE, WAPE, **MASE** in
one place.
```python
scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))   # in-sample seasonal-naive error
mase = mae / scale
```
- **MASE** divides the error by the seasonal-naive's own error (`m = 168` for weekly). *Why:* a value
  below 1 means "better than the naive baseline," and it's comparable across datasets.

**`baseline_forecasts(train, h, m)`** builds the trivial forecasts (naive, seasonal-naive, drift,
mean). *Why:* the **bar** every real model must clear.

---

## `build_notebooks.py` — the notebook generator

- The seven notebooks are generated from readable Python using **nbformat** (one `markdown` or `code`
  cell at a time) rather than hand-edited JSON. *Why:* the cell text stays diff-able and regenerable;
  `python build_notebooks.py 4` rebuilds just Part 4.
- A shared `SETUP` string adds the project root to the path and imports `src`. *Why:* every notebook
  finds `from src import ...` the same way.
- Each notebook is then run with `jupyter nbconvert --execute`, which embeds the real outputs and
  figures. *Why:* the committed notebooks show genuine, reproduced results — never hand-typed numbers.

---

*Back to **[DEEP_DIVE.md](DEEP_DIVE.md)** for concept definitions, or the
**[bike-sharing README](../README.md)** for setup and the part index.*

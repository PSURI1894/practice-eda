# Code Walkthrough — Beijing PM2.5, Line by Line

Every function in `src/`, explained for a beginner: **what each line does** and **why it was written
that way**. **Bold terms** are defined in **[DEEP_DIVE.md](DEEP_DIVE.md)**.

The design: notebooks stay short by *calling* these helpers; the real logic lives here. Three modules
(`config`, `eda`, `ts`) and `forecasting` are shared with the other practices and shown briefly; the
two that are specific to this dataset (`data`, `impute`) are shown in full.

---

## `config.py` — where files live (shared)

```python
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw";  DATA_PROC = ROOT / "data" / "processed";  FIGS = ROOT / "reports" / "figures"
for _p in (DATA_RAW, DATA_PROC, FIGS): _p.mkdir(parents=True, exist_ok=True)
```
- `Path(__file__).resolve().parents[1]` finds the project folder **relative to this file**, so a
  notebook (run from `notebooks/`) and a script find `data/` identically. *This mattered:* Part 4
  first failed because a notebook used a *relative* path — switching to `DATA_PROC` (absolute) fixed it.
- `mkdir(..., exist_ok=True)` creates the folders harmlessly on import, so a fresh clone never crashes.

---

## `eda.py` — distribution helpers (shared)

The functions this practice leans on:
- **`moments(s)`** — mean, std, **skew**, **excess kurtosis** in one row (`s.kurt()` is already
  *excess*, so 0 = a bell curve).
- **`normality_battery(s)`** — runs several normality tests; small p ⇒ "not bell-shaped".
- **`four_view(s, name, fname)`** — the four-panel battery (histogram / box / ECDF / Q–Q). *Why:* a
  mean and standard deviation hide shape; the battery shows skew, spread, percentiles and
  bell-curve-ness at once. `if fname: savefig(...)` saves a PNG *and* shows it inline.
- **`outlier_flags`, `cramers_v`, `vif_table`** — robust outlier counts, categorical association, and
  redundancy scores, reused from the shared toolkit.

---

## `ts.py` — autocorrelation & stationarity (shared)

- **`adf_test` / `kpss_test`** — two stationarity tests with **opposite nulls**; agreement between
  them is convincing. **`stationarity_report`** returns a plain-English verdict.
- **`acf_pacf_plot`** — draws the **ACF**/PACF (spikes reveal cycles). Used in Part 2 to show the
  daily rhythm and pollution's strong persistence.

---

## `data.py` — load & clean the pollution data (this practice)

### Constants
```python
WEATHER = ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
WHO_24H = 15.0
```
- `WEATHER` names the six (complete) weather columns, so analyses can refer to "the weather" in one
  place. `WHO_24H` is the health yardstick (≈7× below Beijing's mean) — a constant, not a magic number
  buried in a plot.

### `clean(df)`
```python
df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
df = df.set_index("datetime").sort_index()
```
- pandas can build a timestamp directly from a frame of `year/month/day/hour` columns; we then set it
  as a sorted **datetime index**. *Why:* time-series tools need to know each row's exact time and order.
```python
df = df.rename(columns={"pm2.5": "pm25", "cbwd": "wind_dir"})
df = df.drop(columns=["No", "year", "month", "day", "hour"])
df["wind_dir"] = df["wind_dir"].astype("category")
df.index.freq = "h"
```
- Rename to friendlier names (`pm2.5` has an awkward dot), drop the row-counter `No` and the now-
  redundant date parts, make wind direction a **category**, and declare the **hourly frequency**. *Why:*
  clean names + correct types + a stated frequency are what every later step assumes.
- **What we deliberately *don't* do:** fill the `pm25` gaps. They're kept as `NaN` because handling
  them is an *evaluated* step (Part 3), not an afterthought.

### `aqi_category(s)`
```python
edges = [-0.1, 12, 35.4, 55.4, 150.4, 250.4, np.inf]
labels = ["Good", "Moderate", "Unhealthy(sens)", "Unhealthy", "Very unhealthy", "Hazardous"]
return pd.cut(s, bins=edges, labels=labels)
```
- `pd.cut` bins a concentration into the **US-EPA health bands**. *Why:* turns an abstract number into
  a health message ("Hazardous"), which is how the public and Part 1's bar chart read pollution. The
  `-0.1` lower edge ensures an exact 0 lands in "Good".

### `gap_runs(s)`
```python
isna = s.isna().to_numpy(); runs, c = [], 0
for v in isna:
    if v: c += 1
    elif c: runs.append(c); c = 0
if c: runs.append(c)
return pd.Series(runs, dtype=int)
```
- Walks the series counting **consecutive** missing values, emitting one length per gap. *Why:* the
  *shape* of the missingness (many short gaps vs a few 155-hour outages) decides everything in Part 3;
  a single "4.7% missing" number hides it.

### `build_processed()`
- Cleans and writes `data/processed/beijing_clean.csv`. *Why:* later notebooks load a clean file
  instead of re-deriving it.

---

## `impute.py` — fill the gaps, and grade the fill (this practice)

### `gap_length_per_point(s)`
```python
while i < n:
    if isna[i]:
        j = i
        while j < n and isna[j]: j += 1
        out[i:j] = j - i; i = j
    else: i += 1
```
- For *each* missing point, records the **length of the run it belongs to**. *Why:* lets us route each
  gap to the right method (short → interpolate, long → MICE) and score errors **stratified by gap
  length**.

### `inject_gaps(truth, source_lengths, n_gaps, seed)`
```python
for L in rng.choice(source_lengths, size=n_gaps):
    for _ in range(30):
        st = rng.integers(0, n - L); sl = slice(st, st + L)
        if obs[sl].all() and not mask[sl].any():
            mask[sl] = True; glen[sl] = L; break
```
- The heart of honest evaluation: it **hides observed values** in artificial gaps whose lengths are
  drawn from the *real* gap distribution (`source_lengths`). *Why:* you can't grade imputation on truly
  missing data (no answer), so you create gaps that *look like* the real ones, where you *do* know the
  truth. The inner loop retries until it finds an all-observed, unused stretch to mask.

### `impute_all(gappy, weather, climatology)`
```python
out = {"ffill": gappy.ffill().bfill(), "interp": gappy.interpolate("time").bfill().ffill()}
fill = pd.Series([climatology.get((m, h)) for m, h in zip(gappy.index.month, gappy.index.hour)], index=gappy.index)
out["climatology"] = gappy.fillna(fill)
X = _design(gappy, weather)
out["KNN"]  = pd.Series(KNNImputer(n_neighbors=8).fit_transform(X)[:, 0], index=gappy.index)
out["MICE"] = pd.Series(IterativeImputer(random_state=0, max_iter=10).fit_transform(X)[:, 0], index=gappy.index)
return out
```
- Runs all five **imputation** methods on the same gappy series.
- **climatology** looks up the typical value for each row's `(month, hour)` — the seasonal average.
- **KNN/MICE** use `_design(...)`, which stacks the gappy `pm25` with the weather plus **cyclical time
  features** (sine/cosine of hour and month) so the multivariate imputers can exploit the daily and
  seasonal pattern, not just the weather. `[:, 0]` pulls the imputed `pm25` column back out.
- *A practical constraint:* `KNN` builds an all-pairs distance matrix, so it's run on a representative
  window in the notebook, not the full 43k rows (memory).

### `hybrid(gappy, weather, short_max=48)`
```python
interp = gappy.interpolate("time").bfill().ffill()
mice = ... IterativeImputer ...
use_mice = gappy.isna() & (gap_length_per_point(gappy) > short_max)
out = interp.copy(); out[use_mice.to_numpy()] = mice[use_mice.to_numpy()]
return out.clip(lower=0)
```
- The recommendation: **interpolate short gaps, MICE the long ones** (`> short_max` hours). *Why:* the
  stratified scores showed interpolation wins for short gaps, MICE for long ones — so take each method
  where it's best.
- `.clip(lower=0)` — a concentration **can't be negative**, but MICE (a regression) can dip slightly
  below zero, so we floor it. *This was a real bug caught in Part 5:* the saved series had tiny negative
  values that broke the Tweedie model until clipped.

---

## `forecasting.py` — metrics & baselines (shared)

- **`forecast_metrics(...)`** computes MAE / RMSE / **MASE** in one place; MASE divides the error by
  the seasonal-naive's error, so **<1 means "better than doing nothing"**.
- **`baseline_forecasts(...)`** builds the trivial forecasts (naive, seasonal-naive, …) — the **bar**
  every model must clear. In Part 4 the naive baselines *lose* to the seasonal naive (MASE > 1), which
  is itself the finding that day-ahead pollution is hard.

---

## `build_notebooks.py` — the notebook generator

- The seven notebooks are generated from readable Python using **nbformat** (one markdown/code cell at
  a time), then executed with `jupyter nbconvert --execute`. *Why:* the cell text stays diff-able and
  regenerable (`python build_notebooks.py 4` rebuilds just Part 4), and every committed number is
  *actually executed* — never hand-typed. A shared `SETUP` string adds the project root to the path so
  `from src import ...` resolves the same way in every notebook.

---

*Back to **[DEEP_DIVE.md](DEEP_DIVE.md)** for concept definitions, or the
**[beijing-air-quality README](../README.md)** for setup and the part index.*

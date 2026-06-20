# Code Walkthrough — Line-by-Line Justification

Every reusable function in `src/`, explained: **what each line does** and **why it was written
that way**. Terms in **bold** are defined in **[DEEP_DIVE.md](DEEP_DIVE.md)**. Modules appear in
the order the course uses them.

Design rule throughout: notebooks stay thin (call + narrate); all statistics live here in one
tested place, so a fix propagates everywhere and the notebooks read as analysis, not plumbing.

---

## `config.py` — project paths

```python
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW  = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
FIGS      = ROOT / "reports" / "figures"
for _p in (DATA_RAW, DATA_PROC, FIGS):
    _p.mkdir(parents=True, exist_ok=True)
```

- `Path(__file__).resolve().parents[1]` — resolve paths **relative to this file**, not the current
  working directory. *Why:* a notebook in `notebooks/` and a script run from the repo root must
  both find `data/` identically; hard-coded relative paths would break one of them.
- The three `DATA_*`/`FIGS` constants are imported everywhere instead of literals. *Why:* one place
  to change the layout; no stray `"../data/raw"` strings drifting out of sync.
- `mkdir(parents=True, exist_ok=True)` — create the folders on import if missing. *Why:* the first
  run on a fresh clone shouldn't crash because `reports/figures/` doesn't exist yet; `exist_ok`
  makes it idempotent.

---

## `data.py` — load & clean the datasets

### Module constants

```python
SHILLER_ZERO_IS_MISSING = ["Dividend","Earnings","Consumer Price Index","Long Interest Rate",
                           "Real Price","Real Dividend","Real Earnings","PE10"]
```
- A column is listed here only if a literal **0 is impossible** for it (a CAPE ratio or CPI can
  never be 0). *Why:* we replace `0 → NaN` *only* on these — blindly replacing 0 everywhere would
  destroy legitimate zeros elsewhere. This is the fix for **disguised missingness**.

### `clean_telco()`

```python
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
```
- `to_numeric(..., errors="coerce")` — force the text column to numbers, turning the 11
  blank/whitespace strings into `NaN` instead of raising. *Why:* the column is stored as text; this
  *exposes* the hidden missing values that `isna()` on the raw text would miss.
- `fillna(0.0)` — those 11 rows are all `tenure == 0` (brand-new customers with no completed
  billing cycle), so a total of **0 is the correct, domain-honest value**. *Why:* this is **MAR**
  imputation by mechanism, not a blind mean-fill that would invent billing history.

```python
df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"}).astype("category")
df = df.drop(columns=["customerID"])
df["churn_flag"] = (df["Churn"] == "Yes").astype(int)
for c in df.select_dtypes("object").columns:
    df[c] = df[c].astype("category")
```
- `SeniorCitizen` map to Yes/No + `category` — it's stored as 0/1 but is **genuinely
  categorical**. *Why:* left numeric, `describe()` and correlations would treat it as continuous
  and mislead.
- `drop("customerID")` — a pure identifier (7043 unique values). *Why:* it carries no signal and a
  model would just memorise rows.
- `churn_flag` — a numeric 0/1 twin of the Yes/No label. *Why:* needed for **point-biserial
  correlation** and **mutual information** against churn.
- Cast remaining `object` columns to `category`. *Why:* correct semantics + lower memory; makes
  `groupby` and Cramér's V behave.

### `clean_shiller()`

```python
df["Date"] = pd.to_datetime(df["Date"]); df = df.set_index("Date").sort_index()
df.index.freq = "MS"
df[SHILLER_ZERO_IS_MISSING] = df[SHILLER_ZERO_IS_MISSING].replace(0.0, np.nan)
```
- Parse to datetime, set as index, sort. *Why:* **index hygiene** — every lag/rolling/seasonal
  operation downstream needs a sorted **DatetimeIndex**.
- `index.freq = "MS"` — declare month-start **frequency**. *Why:* makes the series a regular grid
  so `.diff()`, `.shift()`, and seasonal period 12 are meaningful.
- `replace(0.0, np.nan)` on the impossible-zero columns — convert the **disguised missing**
  placeholders to real `NaN`. *Why:* so PE10's leading zeros (CAPE needs 10y history) and the stale
  trailing zeros stop polluting statistics.

```python
df["return"]      = df["SP500"].pct_change(fill_method=None)
df["log_return"]  = np.log(df["SP500"]).diff()
df["real_return"] = df["Real Price"].pct_change(fill_method=None)
df["cape"]        = df["PE10"]
df["cape_z"]      = (df["cape"] - df["cape"].mean()) / df["cape"].std()
```
- `pct_change(fill_method=None)` — simple **return**; `fill_method=None` stops pandas from
  forward-filling NaNs before differencing. *Why:* avoids silently fabricating returns across gaps
  (and silences the deprecation warning).
- `log(SP500).diff()` — the **log return**, which is additive and the basis for **differencing**.
- `cape_z` — a **standardized** CAPE. *Why:* lets "is the market expensive?" be read as a z-score.

### Stock panel (Part 4+)

```python
SECTORS = {"AAPL":"Tech","MSFT":"Tech","JPM":"Bank","BAC":"Bank", ...}
def load_stock_panel():
    return pd.read_csv(DATA_RAW/"stock_panel.csv", parse_dates=["date"]).set_index("date").sort_index()
def stock_log_returns(prices=None):
    px = load_stock_panel() if prices is None else prices
    return np.log(px).diff().dropna()
```
- `SECTORS` — six same-sector pairs, chosen deliberately as **cointegration** candidates and to
  let clustering/PCA recover sectors. *Why:* the data is engineered to make Parts 4 & 8 teachable.
- `stock_log_returns` returns **log returns** (stationary). *Why:* correlation/PCA/VAR/Granger all
  require **stationary** inputs; price levels would give **spurious** results.

### `build_processed()`
- Cleans all three datasets and writes them to `data/processed/`. *Why:* Part 0 produces
  model-ready files once; later notebooks can load clean data without re-deriving it. Shiller keeps
  its index (`index=True`); Telco doesn't (rows are exchangeable).

---

## `eda.py` — distribution diagnostics & association

### `moments(s)`
```python
return pd.Series({"n":..., "mean":..., "std":..., "skew": s.skew(),
                  "excess_kurtosis": s.kurt(), ...})
```
- `s.kurt()` is pandas' **excess kurtosis** (normal = 0), not raw kurtosis. *Why:* so "16.7" is
  directly readable as "16.7 above normal" — the fat-tails headline.

### `normality_battery(s)`
```python
jb = stats.jarque_bera(s);  k2 = stats.normaltest(s)
s_shap = s if s.size <= 5000 else rng.choice(s, 5000, replace=False)
sh = stats.shapiro(s_shap)
out["normal_at_5pct"] = out["p_value"] > 0.05
```
- Runs three tests with different strengths (JB, D'Agostino, Shapiro). *Why:* one test can mislead;
  agreement across three is convincing.
- Shapiro is **sampled to 5000** when larger. *Why:* `scipy.stats.shapiro` is only valid/meaningful
  up to ~5000 points — guarding avoids a useless or erroring call.
- `p_value > 0.05` → the `normal_at_5pct` flag. *Why:* turns each **p-value** into a plain
  pass/fail at the conventional 5% level.

### `four_view(s, name, fname)`
- One figure, four panels: **histogram+KDE**, **box**, **ECDF**, **Q–Q**. *Why:* `describe()` hides
  shape — two variables with the same mean/std can look completely different; the battery shows
  **shape, spread, percentiles, and normality at once**.
- `stats.probplot(s, dist="norm", plot=ax)` draws the **Q–Q**; the title carries skew/kurtosis.
  *Why:* tail departures from the diagonal *are* the fat tails.
- `if fname: savefig(...)` — save a standalone PNG *and* show inline. *Why:* notebooks embed it;
  `reports/figures/` keeps a reusable copy for slides/README.

### `outlier_flags(s)`
```python
lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr;  iqr_out = (s < lo) | (s > hi)
mad = np.median(np.abs(s - med));  mod_z = 0.6745*(s - med)/mad;  mad_out = mod_z.abs() > 3.5
```
- Two **robust** flags: **Tukey 1.5·IQR** and the **MAD modified z** (|z|>3.5). *Why:* MAD resists
  the very outliers that would inflate an ordinary z-score's denominator — so it doesn't hide the
  outliers it's trying to find. `0.6745` rescales MAD to be comparable to a standard deviation.

### `correlation_trio(df, a, b)`
```python
rows = [(m, *getattr(stats, fn)(pair[a], pair[b]))
        for m, fn in [("pearson","pearsonr"),("spearman","spearmanr"),("kendall","kendalltau")]]
```
- Computes **Pearson, Spearman, Kendall** side by side. *Why:* a large Pearson–Spearman gap reveals
  **non-linearity or outlier leverage** that any single coefficient would hide.

### `cramers_v(x, y)`
```python
chi2 = stats.chi2_contingency(confusion)[0]
phi2corr = max(0, phi2 - (k-1)*(r-1)/(n-1)); ...
return np.sqrt(phi2corr / denom)
```
- **Cramér's V** with **bias correction** (the `…corr` terms). *Why:* raw V is biased upward for
  tables with many categories; the correction (Bergsma) keeps it honest and in [0, 1] — the right
  measure for **categorical** association where correlation doesn't apply.

### `vif_table(df_numeric)`
```python
X = df_numeric.dropna().assign(_const=1.0)
vifs = [variance_inflation_factor(X.values, X.columns.get_loc(c)) for c in cols]
```
- Adds a constant column, then computes **VIF** per feature. *Why:* VIF needs an intercept in the
  design matrix to be correct; without `_const` the values are wrong. Flags **multicollinearity**
  (the `TotalCharges ≈ tenure × MonthlyCharges` redundancy).

---

## `ts.py` — stationarity & autocorrelation

### `adf_test` / `kpss_test`
```python
adfuller(s, regression=regression, autolag="AIC")   # ADF: H0 = unit root
kpss(s, regression=regression, nlags="auto")        # KPSS: H0 = stationary
return {... "stationary_5pct": p < 0.05}            # ADF
return {... "stationary_5pct": p > 0.05}            # KPSS
```
- The two tests have **opposite nulls**, so the `stationary_5pct` flag flips between them (`p < .05`
  for ADF, `p > .05` for KPSS). *Why:* this is the whole point of using them together — agreement
  resolves the ambiguity either alone would leave.
- `autolag="AIC"` (ADF) and `nlags="auto"` (KPSS) — let the tests pick their own lag length.
- KPSS is wrapped in `warnings.catch_warnings()`. *Why:* statsmodels warns when the p-value is
  outside its lookup table (it's capped to [0.01, 0.10]); the warning is expected, not a bug.

### `stationarity_report(s)`
```python
if a and k: "STATIONARY"
elif not a and not k: "NON-STATIONARY (unit root) -> difference"
elif a and not k: "DIFFERENCE-STATIONARY -> difference"
else: "TREND-STATIONARY -> detrend"
```
- Encodes the **ADF × KPSS decision table** as four verdicts with the **action** attached. *Why:*
  the verdict (difference vs detrend) is the actually-useful output; returning a raw p-value would
  make the reader re-derive it every time.

### `decomposition_strengths(result)`
```python
f_trend = max(0, 1 - r.var()/(t+r).var());  f_seas = max(0, 1 - r.var()/(s+r).var())
```
- **Hyndman seasonal/trend strength** in [0,1]. *Why:* an objective, comparable number — it's what
  rigorously proves "CO₂ seasonal 0.98 vs S&P 0.00" instead of eyeballing a plot. `max(0, …)` clips
  tiny negatives from sampling noise.

### `acf_pacf_plot` / `ljung_box`
- Plots **ACF** and **PACF** together. *Why:* the **Box-Jenkins** identification table reads them
  as a pair (AR cuts in PACF, MA cuts in ACF).
- `ljung_box` wraps `acorr_ljungbox`. *Why:* the formal **white-noise** test, reused on returns,
  squared returns, and model residuals.

---

## `forecasting.py` — splits, metrics, baselines

### `ts_train_test(s, h)`
```python
return s.iloc[:-h], s.iloc[-h:]
```
- Holds out the **last** `h` points — never shuffles. *Why:* the test set must lie strictly in the
  future of training, or the model peeks ahead (**leakage**) and the score is fiction.

### `forecast_metrics(y_true, y_pred, y_train, m)`
```python
mape  = np.mean(np.abs(e/y_true))*100
wape  = np.sum(np.abs(e))/np.sum(np.abs(y_true))*100
scale = np.mean(np.abs(y_train[m:] - y_train[:-m]));  mase = mae/scale
```
- Computes MAE, RMSE, **MAPE**, **sMAPE**, **WAPE**, **MASE** in one place.
- MAPE/sMAPE are wrapped in `np.errstate(divide="ignore")`. *Why:* they legitimately blow up when
  actuals cross 0 (returns) — we *want* to show that, not crash.
- **WAPE** aggregates numerator and denominator separately. *Why:* a single near-zero actual can't
  hijack it the way it hijacks MAPE.
- **MASE** divides by the **in-sample naive MAE** (`m`-step for seasonal). *Why:* makes the score
  **scale-free and baseline-relative** — `< 1` literally means "better than doing nothing".

### `baseline_forecasts(train, h, m)`
```python
naive  = np.repeat(last, h)
season = np.array([y[-m + (i % m)] for i in range(h)])
drift  = last + (y[-1]-y[0])/(n-1) * np.arange(1, h+1)
meanf  = np.repeat(y.mean(), h)
```
- The four **baselines**, each optimal in a different world (naive↔random walk, seasonal↔strong
  seasonality, drift↔linear trend, mean↔flat noise). *Why:* a model is only impressive *relative*
  to the best of these — they set the MASE denominator's spirit.

---

## `multivariate.py` — correlation, PCA, VAR/Granger, cointegration

### `cluster_order(corr)`
```python
dist = squareform(1 - corr.values, checks=False); link = linkage(dist, "average")
return [corr.columns[i] for i in leaves_list(link)]
```
- Hierarchically clusters on `1 − corr` (a distance) and returns the leaf order. *Why:* reordering
  the heatmap by similarity makes the **sector blocks** appear on the diagonal *unsupervised* — a
  sanity check that the structure is real.

### `pca_factors(returns)`
```python
z = (returns - returns.mean())/returns.std();  pca = PCA().fit(z.values)
loadings = pd.DataFrame(pca.components_.T, index=returns.columns, columns=evr.index)
```
- **Standardizes** returns before **PCA**. *Why:* PCA is variance-driven; without standardization a
  single high-variance stock would dominate PC1.
- Returns explained-variance ratio, **loadings**, and scores. *Why:* loadings (all-positive on PC1)
  are what identify PC1 as the **market factor**.

### `granger_matrix(returns, maxlag)`
```python
with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
    res = grangercausalitytests(returns[[effect, cause]], maxlag=maxlag)
    out.loc[effect, cause] = min(res[l][0]["ssr_ftest"][1] for l in range(1, maxlag+1))
```
- Builds the directed **Granger** p-value matrix; `returns[[effect, cause]]` order matters —
  statsmodels tests whether *column 2 causes column 1*. *Why:* the matrix is **asymmetric** (that's
  the point — lead/lag), so the column order is load-bearing.
- `redirect_stdout(io.StringIO())` — `grangercausalitytests` prints a wall of text per call. *Why:*
  132 pairs × 5 lags would flood the notebook; we silence it and keep only the p-value.
- Inputs are **returns** (stationary). *Why:* Granger on non-stationary prices gives spurious results.

### `engle_granger(y, x)`
```python
beta = np.polyfit(xx, yy, 1)[0];  spread = yy - beta*xx
t_stat, p_value, _ = coint(yy, xx);  adf_p = adfuller(spread)[1]
```
- OLS slope = the **hedge ratio β**; `spread = y − β·x`. *Why:* **cointegration** is exactly the
  claim that this spread is **stationary**.
- Runs both the `coint` test and an **ADF** on the spread. *Why:* belt-and-braces — the spread's
  own stationarity is the intuition behind the formal test.

### `johansen_summary(prices)` / `zscore(s, window)`
- `coint_johansen` trace stat vs the 95% critical value per rank → the **cointegration rank**.
  *Why:* a multivariate confirmation of Engle–Granger; rank 1 = one equilibrium.
- `zscore(window=…)` uses a **rolling** mean/std when a window is given. *Why:* the full-sample
  z-score peeks at the future; only the **rolling** z is a tradeable, leak-free signal.

---

## `ml_forecast.py` — supervised reframing (the leakage discipline)

### `make_supervised(y, n_lags, roll_windows, calendar)`
```python
for L in range(1, n_lags+1): df[f"lag{L}"] = y.shift(L)
shifted = y.shift(1)
for w in roll_windows:
    df[f"rmean{w}"] = shifted.rolling(w).mean();  df[f"rstd{w}"] = shifted.rolling(w).std()
if calendar: df["month"] = y.index.month
return df.dropna()
```
- Turns a 1-D series into a supervised `(X, y)` table of **lag / rolling / calendar features**.
- **`y.shift(1)` before `.rolling(w)`** is the single most important line: it makes every rolling
  feature end at `t-1`. *Why:* `y.rolling(w)` would include `y[t]` itself — **leakage** that makes
  the model look brilliant in backtest and fail live.
- `dropna()` drops the warm-up rows where lags don't exist yet. *Why:* the first `n_lags` rows have
  undefined features.

### `recursive_forecast(model, history, steps, cols, …)`
```python
feat = {f"lag{L}": hist.iloc[-L] for L in range(1, n_lags+1)}
for w in roll_windows: feat[f"rmean{w}"] = hist.iloc[-w:].mean(); ...
row = pd.DataFrame([feat])[cols];  yhat = float(model.predict(row)[0]);  hist.loc[t] = yhat
```
- Multi-step **recursive** forecasting: predict one step, append it to `hist`, repeat. *Why:* a
  one-step model can't see its own multi-step future, so we feed predictions back as the newest lag.
- The feature row is built to **mirror `make_supervised` exactly** (`hist.iloc[-w:]` equals
  `shift(1).rolling(w)` at time `t`). *Why:* train-time and predict-time features must match or the
  model gets garbage inputs. `[cols]` re-orders columns to the trained order.
- Works with *any* model exposing `.predict` — LightGBM (Part 5) and the from-scratch net (Part 9)
  plug in unchanged. *Why:* one driver, many models.

### `reconstruct_from_diff(diff_preds, last_level)`
```python
return last_level + diff_preds.cumsum()
```
- Turns forecasts of the **difference** back into levels. *Why:* trees/nets can't **extrapolate a
  trend**, so we model the stationary `Δy` and rebuild `y` by cumulative sum — the fix demonstrated
  in Parts 5 and 9.

---

## `backtest.py` — honest evaluation

### `cv_folds(n, n_splits, horizon, mode)`
```python
test_starts = [n - (n_splits - k)*horizon for k in range(n_splits)]
train_idx = np.arange(0, start) if mode=="expanding" else np.arange(max(0,start-width), start)
```
- Yields time-ordered `(train, test)` folds, **expanding** or **sliding**. *Why:* ordinary k-fold
  shuffles and leaks the future; these keep the test block strictly after its training block.

### `walk_forward(y, forecaster, initial, horizon, step)`
```python
while start + horizon <= len(y):
    fc = np.asarray(forecaster(y.iloc[:start], horizon)).ravel()
    preds.append(pd.Series(fc[:len(idx)], index=y.index[start:start+horizon])); start += step
```
- Refit-and-roll: re-train on everything up to `start`, forecast, step forward. *Why:* this mimics
  **production** exactly and yields a whole out-of-sample track record, not one lucky window.
- `forecaster(train, h)` is a plain callable. *Why:* model-agnostic — seasonal-naive, Holt-Winters,
  LightGBM, ARIMA all slot in as one-line adapters.

### `conformal_q(calib_residuals, alpha)` / `coverage(...)`
```python
return float(np.quantile(np.abs(calib_residuals), 1 - alpha))
return ((y_true >= lower) & (y_true <= upper)).mean()
```
- **Conformal** radius = the `(1−α)` quantile of `|calibration residuals|`; `point ± q` then covers
  ≈ `(1−α)` of future points **with no distributional assumption**. *Why:* model-based intervals
  assume the error shape is right; conformal earns coverage from data alone.
- `coverage` just checks the empirical hit rate. *Why:* an interval is only honest if its coverage
  matches its nominal level.

---

## `volatility.py` — ARCH/GARCH & VaR

### `arch_lm(returns, lags)`
```python
stat, p, _, _ = het_arch(r - r.mean(), nlags=lags)
```
- Engle's **ARCH-LM** test (H₀: no ARCH effect). *Why:* formally confirms **volatility clustering**
  before fitting GARCH — don't model what isn't there.

### `fit_garch(returns_pct, p, o, q, dist, mean)`
```python
am = arch_model(returns_pct.dropna(), mean=mean, vol="GARCH", p=p, o=o, q=q, dist=dist)
return am.fit(disp="off")
```
- One wrapper covers GARCH and its variants: `o>0` adds the **GJR leverage** term, `dist="t"` uses
  **Student-t** innovations. *Why:* lets the notebook compare normal vs t and symmetric vs
  asymmetric by flipping one argument.
- Inputs are returns **×100 (percent)**. *Why:* the `arch` library is numerically happier with
  percent-scale returns and warns otherwise.

### `persistence(res)`
```python
return float(a + b + g/2)   # sum of alpha + beta + gamma/2
```
- **α + β** (plus half γ for GJR). *Why:* the single number for how slowly variance shocks decay;
  the `g/2` accounts for the leverage term contributing on average to half the shocks.

### `var_series(res, returns_pct, alpha)`
```python
std_resid = (returns_pct - mu)/cv;  q = np.quantile(std_resid.dropna(), alpha)
return -(mu + cv*q)
```
- One-day **VaR** from the conditional volatility `cv` and the **empirical** α-quantile of
  standardized residuals. *Why:* using the empirical quantile (not a normal one) respects the **fat
  tails**; multiplying by the time-varying `cv` makes VaR widen in turbulent regimes.

### `var_backtest(returns_pct, var_pct, alpha)`
```python
viol = r < -v;  lr = -2*((n-x)*log(1-alpha)+x*log(alpha) - ((n-x)*log(1-pi)+x*log(pi)))
return {... "kupiec_p": 1 - chi2.cdf(lr, 1)}
```
- Counts breaches and runs the **Kupiec POF** likelihood-ratio test (H₀: breach rate = α). *Why:*
  a VaR number is worthless until validated; Kupiec checks that 1% VaR really breaches ~1% of days.

---

## `mgarch.py` — DCC-GARCH (dynamic correlation)

### `standardized_residuals(returns_pct, dist)`
```python
for col in returns_pct.columns:
    z[col] = fit_garch(returns_pct[col].dropna(), p=1, q=1, dist=dist).std_resid
```
- **Step 1 of DCC:** a univariate **GARCH** per asset → its **standardized residuals** (returns
  with their own volatility divided out). *Why:* DCC separates volatility (per-asset GARCH) from
  correlation (the next step); the residuals carry the cross-asset structure to model.

### `dcc(z_df, pairs)`
```python
Q = (1 - a - b)*Qbar + a*np.outer(zt, zt) + b*Q          # the DCC recursion
R = Q / np.outer(d, d)                                    # normalise Q -> correlation
ll += -0.5*(logdet + zt @ np.linalg.solve(R, zt) - zt @ zt)
opt = minimize(neg_loglik, [0.02, 0.95], bounds=[(1e-4,0.5),(1e-4,0.997)])
```
- The **DCC(1,1)** recursion: today's quasi-correlation `Q_t` blends the long-run `Q̄`, yesterday's
  shock outer-product, and yesterday's `Q`. *Why:* just **two parameters** (`a`, `b`) make the
  whole correlation matrix time-varying.
- `R = Q / outer(d, d)` normalises `Q` to a proper **correlation** matrix (unit diagonal). *Why:*
  `Q` isn't a correlation matrix until rescaled by its own diagonal.
- `np.linalg.solve(R, zt)` instead of `inv(R) @ zt`. *Why:* solving is faster and more numerically
  stable than forming the inverse — and this runs inside the per-timestep likelihood loop.
- `minimize(..., bounds=[(1e-4,0.5),(1e-4,0.997)])` with the `a+b<0.999` guard in `neg_loglik`.
  *Why:* estimates `(a,b)` by maximum likelihood while keeping the process **mean-reverting**
  (`a+b<1`) and the matrices positive-definite.
- Replays the recursion after fitting to extract the average and per-pair correlation paths. *Why:*
  the headline (avg correlation spiking in Aug-2015) and the per-pair plot come from `R_t` over time.

---

## `neuralnet.py` — a neural network from scratch

### `_init_params(n_in)`
```python
self.W.append(rng.normal(0, np.sqrt(2/sizes[i]), (sizes[i], sizes[i+1])))
self.b.append(np.zeros(sizes[i+1]))
```
- **He initialization** (`√(2/fan_in)`) for the weights, zeros for biases. *Why:* tuned for
  **ReLU** so signal variance is preserved across layers — bad init makes deep nets fail to train.

### `_forward(X)`
```python
a = z if i == len(self.W)-1 else _relu(z)   # ReLU hidden, LINEAR output
```
- **ReLU** on hidden layers, **linear** on the output. *Why:* ReLU gives nonlinearity; a regression
  output must be linear (a ReLU output couldn't predict negative or large values). Caches
  activations/pre-activations for backprop.

### `fit(X, y)`
```python
self.x_mu, self.x_sd = X.mean(0), X.std(0)+1e-8;  Xs = (X - self.x_mu)/self.x_sd
ys = ((y - self.y_mu)/self.y_sd).reshape(-1,1)
dz = 2*(yhat - yb)/xb.shape[0]                         # dMSE/dpred
dW = acts[i].T @ dz + self.l2*self.W[i]                # grad + L2
if i > 0: dz = (dz @ self.W[i].T) * _relu_grad(pre[i-1])   # backprop through ReLU
m[...] = b1*m + (1-b1)*g;  v[...] = b2*v + (1-b2)*g*g       # Adam moments
p -= self.lr * mhat / (np.sqrt(vhat) + eps)                 # Adam step
```
- **Standardize** X and y internally (`+1e-8` guards divide-by-zero). *Why:* neural nets train
  badly on unscaled inputs — lopsided gradients stall learning. Storing the scalers lets `predict`
  invert them.
- `dz = 2*(yhat - yb)/N` — the **MSE** gradient w.r.t. the output.
- `dW = acts.T @ dz + l2*W` — the weight gradient plus the **L2 regularization** term. *Why:* L2
  penalises large weights to curb **overfitting**.
- `dz = (dz @ W.T) * relu_grad(pre)` — **backpropagation**: push the gradient to the previous layer
  and gate it by ReLU's derivative (1 where the unit was active, 0 otherwise).
- The Adam block — bias-corrected first/second moment estimates and the adaptive step. *Why:*
  **Adam** adapts the learning rate per parameter and converges far faster than plain gradient
  descent, which matters when training by hand in NumPy.

### `predict(X)`
```python
ys = self._forward((X - self.x_mu)/self.x_sd)[0][-1].ravel();  return ys*self.y_sd + self.y_mu
```
- Apply the **same** stored standardization, forward-pass, then **invert** the target scaling.
  *Why:* predictions must come back in the original units; using the train-time scalers (not
  re-fitting on new data) avoids leakage and keeps train/predict consistent. The interface matches
  LightGBM's `.predict`, so `recursive_forecast` drives it with zero changes.

---

## `build_notebooks.py` — the notebook generator

- Notebooks are the deliverable, but hand-authoring `.ipynb` JSON is error-prone, so each is built
  from readable Python via **`nbformat`** (`new_markdown_cell` / `new_code_cell`). *Why:* the cell
  source stays diff-able and regenerable; `python build_notebooks.py 7` rebuilds just Part 7.
- A shared `SETUP` string adds the repo root to `sys.path` and imports `src`. *Why:* every notebook
  resolves `from src import …` identically whether run from the repo root or `notebooks/`.
- Each notebook is then executed with `jupyter nbconvert --execute`, which embeds outputs and
  figures. *Why:* the committed `.ipynb` shows real, reproduced results — never hand-faked numbers.

---

*Back to **[DEEP_DIVE.md](DEEP_DIVE.md)** for concept definitions, or the project
**[README](../README.md)** for setup and the part index.*

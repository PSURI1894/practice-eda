# Code Walkthrough — Wine Quality, Line by Line

Every function in `src/`, explained for a beginner: **what each line does** and **why it was written
that way**. **Bold terms** are defined in **[DEEP_DIVE.md](DEEP_DIVE.md)**.

The design: notebooks stay short by *calling* these helpers; the real logic lives here. `config` and
`eda` are shared with the other practices (shown briefly); the two modules specific to this
classification practice — `data` and `modeling` — are shown in full.

---

## `config.py` — where files live (shared)

```python
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw";  DATA_PROC = ROOT / "data" / "processed";  FIGS = ROOT / "reports" / "figures"
for _p in (DATA_RAW, DATA_PROC, FIGS): _p.mkdir(parents=True, exist_ok=True)
```
- Resolves paths **relative to this file**, so a notebook (run from `notebooks/`) and a script find
  `data/` identically. `mkdir(..., exist_ok=True)` makes a fresh clone work without manual setup.

---

## `eda.py` — distribution & association helpers (shared)

The functions this practice leans on:
- **`moments(s)`** — mean, std, **skew**, excess **kurtosis** in one row (used for the skew audit).
- **`four_view(s, …)`** — histogram / box / ECDF / Q–Q battery for one feature.
- **`vif_table(df)`** — the **VIF** multicollinearity score per feature (Part 1 found `density` ≈ 16).
- **`savefig(fig, name)`** — persists a PNG to `reports/figures/`; in Part 9 we pass it `plt.gcf()` to
  capture SHAP's auto-generated plots.

---

## `data.py` — load, clean & engineer (this practice)

### Constants
```python
NUMERIC = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
           "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"]
```
- The 11 physicochemical features named once, so every notebook refers to "the features" the same way.

### `clean(df=None)`
```python
df = load_raw() if df is None else df.copy()
df["wine_type"] = df["wine_type"].astype("category")
return df
```
- Loads the combined red+white CSV (assembled once during setup) and types the `wine_type` flag.
- **What it deliberately does *not* do:** drop the **duplicate rows**. They're kept here because whether
  they're genuine repeats or artefacts is a *judgement call* surfaced in Part 0 — and they're part of
  the real distribution for EDA. Removing them is a separate, modelling-only step (`dedup`).

### `dedup(df)`
```python
return df.drop_duplicates().reset_index(drop=True)
```
- Drops exact-duplicate rows. *Why separate from `clean`:* this runs **before every train/test split**
  to prevent **leakage** — a wine appearing in both train and test would let a model memorise the
  answer (Part 4 measures the 14-point illusion this causes).

### `quality_band(q)`
```python
return pd.cut(q, [2, 4, 6, 9], labels=["low (<=4)", "mid (5-6)", "high (>=7)"])
```
- Collapses the 7-grade ordinal target into 3 coarse, **ordered** bands — a simpler, imbalance-friendly
  framing used in Part 0 to show the imbalance (mid ≈ 77%).

### `engineer(df)` — the engineered chemistry
```python
d["bound_so2"] = d.total_sulfur_dioxide - d.free_sulfur_dioxide          # the bound SO2 fraction
d["so2_ratio"] = d.free_sulfur_dioxide / (d.total_sulfur_dioxide + 1)    # free share of total SO2
d["total_acidity"] = d.fixed_acidity + d.volatile_acidity + d.citric_acid
d["alcohol_density"] = d.alcohol / d.density
```
- Builds chemically-meaningful **engineered features**. The `+ 1` in `so2_ratio` avoids divide-by-zero.
- **`alcohol_density`** (alcohol ÷ density, a "normalised strength") is the payoff: it correlates +0.48
  with quality — *better than any raw feature* — and becomes the model's #1 driver (Parts 3, 9). A small
  demonstration that domain knowledge beats raw columns.

### `features_target` / `build_processed`
- `features_target` returns the model inputs and the `quality` target; `build_processed` writes the
  cleaned CSV so later notebooks load instead of re-deriving.

---

## `modeling.py` — the evaluation harness (this practice)

This module encodes the three decisions that make every result trustworthy (see **DEEP_DIVE §5**):
dedup-then-stratify, and score with **QWK**.

### `prep(df, engineered=True)`
```python
d = data.engineer(df) if engineered else df
cols = data.NUMERIC + (data.ENGINEERED if engineered else [])
X = d[cols].copy()
X["is_red"] = (d["wine_type"] == "red").astype(int)
return X, d["quality"].astype(int)
```
- Builds the feature matrix (the 11 chemistry features + the 4 engineered ones + a binary `is_red`
  flag) and the ordinal target. One-hot-encoding the two-value `wine_type` as a single `is_red` column
  is all that's needed (no high-cardinality categoricals here).

### `split(df, …)`
```python
d = data.dedup(df) if dedup else df.copy()
X, y = prep(d, engineered=engineered)
return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
```
- The honest split in one call: **dedup → engineer → stratified** train/test. `stratify=y` keeps each
  grade's proportion in both halves — vital because grade 9 has only 5 wines.

### `qwk(y_true, y_pred)` — the metric
```python
yp = np.clip(np.round(np.asarray(y_pred)), 3, 9).astype(int)
return cohen_kappa_score(yt, yp, weights="quadratic", labels=LABELS)
```
- **Quadratic Weighted Kappa.** `weights="quadratic"` is what makes far-off errors hurt more than near
  ones (penalty ∝ squared distance). Regressor outputs are **rounded and clipped** to the valid 3–9
  range first, so the same metric scores classifiers and regressors alike. `labels=LABELS` (3…9) forces
  the confusion matrix to span every grade even if a fold is missing one.

### `report(...)` and `cv_qwk(...)`
- `report` returns the **four complementary scores** (accuracy, macro-F1, MAE, QWK) as one row — we
  always show them together because *accuracy alone lies* under imbalance.
- `cv_qwk` runs **stratified k-fold**, cloning the estimator per fold, and returns the per-fold QWK
  array so notebooks can show **mean ± std**.

### `OptimizedRounder` — learning the cut points
```python
def _apply(self, p, cuts):
    return self.labels[np.digitize(np.asarray(p), np.sort(cuts))]
def fit(self, p, y):
    loss = lambda c: -qwk(y, self._apply(p, c))
    self.cuts_ = minimize(loss, self.cuts_, method="Nelder-Mead", ...).x
```
- A continuous prediction (e.g. 6.4) must become a grade. Naive rounding cuts at the half-integers;
  this learns the **cut points that maximise QWK** on the training predictions instead.
- `np.digitize(p, sorted cuts)` maps each prediction to a bin index → a grade. `fit` minimises
  *negative* QWK (so it *maximises* QWK) with Nelder–Mead (a derivative-free optimiser — QWK isn't
  differentiable).
- The learned cuts come out **asymmetric** (the 8→9 boundary near 8.6, because grade 9 is so rare you
  should rarely commit to it) — which is exactly why it beats naive rounding (Part 6: QWK 0.525 → 0.563).

---

## `build_notebooks.py` — the notebook generator

- The eleven notebooks are generated from readable Python via **nbformat** (one cell at a time), then
  executed with `jupyter nbconvert --execute`. *Why:* the cell text stays diff-able and regenerable
  (`python build_notebooks.py 6` rebuilds just Part 6), and **every committed number is actually
  executed**, never hand-typed. A shared `SETUP` string puts the project root on the path so `from src
  import …` resolves identically in every notebook.

---

*Back to **[DEEP_DIVE.md](DEEP_DIVE.md)** for concept definitions, or the
**[wine-quality README](../README.md)** for setup and the part index.*

# Wine Quality — Cross-Sectional EDA & Imbalanced Ordinal Classification

The fourth practice in the `practice-eda` collection, and the first **cross-sectional** one (no time
axis) — chosen to exercise the toolkit the three time-series practices never needed: imbalanced
**multi-class / ordinal classification**, multicollinearity, and (coming) calibration, fairness-style
per-class analysis, and k-fold cross-validation.

## Dataset (`data/raw/wine.csv`)
UCI **Wine Quality** (red + white combined, via the `zygmuntz/wine-quality` mirror): **6,497 wines ×
13 cols**. Target = **`quality`** (ordinal taster score 3–9). Eleven physicochemical features
(acidity, sugar, chlorides, sulfur dioxide, density, pH, sulphates, alcohol) + a `wine_type` flag.

## Layout
```
data/raw/  data/processed/   source + cleaned CSVs
src/       config · data (wine load/clean/dedup) · eda (shared stats & plots)
notebooks/ 00_data_cleaning · 01_advanced_eda   (more parts to come)
reports/figures/             saved PNGs
build_notebooks.py           regenerates notebooks from source
```

## Setup
```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
.venv\Scripts\python.exe -m ipykernel install --user --name wine-quality --display-name "Python (wine-quality)"
```
Open the notebooks with the **Python (wine-quality)** kernel, or run headless with
`jupyter nbconvert --to notebook --execute --inplace notebooks\*.ipynb`.

## Progress
- [x] **Part 0** — Acquire & clean: combine red+white, the **duplicate-rows leakage trap** (18%, with
  a `dedup()` for modelling), and the **imbalanced ordinal target**
- [x] **Part 1** — Advanced EDA: imbalance + red/white split, univariate sweep of all 11 features,
  **quality drivers** (Spearman), **multicollinearity (VIF)**, red-vs-white chemistry, the good-wine picture
- [ ] Part 2 — Multivariate structure (PCA, red/white separation, feature relationships)
- [ ] Part 3 — Imbalanced ordinal **classification** (baselines, LightGBM, macro-F1, confusion matrix)
- [ ] Part 4+ — calibration, feature importance / SHAP, cost-sensitive thresholds

## Headline findings so far
- **Cross-sectional, fully numeric, no missing values** — but **1,177 duplicate rows (18%)**; split
  carelessly and copies of a wine leak across train/test, so `dedup()` runs before any modelling.
- **The target is ordinal and imbalanced:** quality 5–6 = 76% of wines; the extremes (3,4,8,9) only
  6.8%, and **quality 9 has just 5 wines** — accuracy alone will mislead.
- **Alcohol is the top quality driver** (Spearman +0.45); volatile acidity (−0.26) and density (−0.32)
  hurt. Higher-rated wines are stronger and less acetic.
- **Serious multicollinearity:** `density` VIF ≈ **16** (it's ~a function of alcohol + sugar + acids);
  free/total SO₂ are linked too — important for any linear model.
- **Red and white are chemically distinct:** white has ~2.5× the residual sugar and ~3× the total SO₂;
  red has ~1.9× the volatile acidity — nearly separable on chemistry alone.

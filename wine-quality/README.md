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
- [x] **Part 2** — Multivariate structure: PCA (PC1=red/white axis, 27%), t-SNE, quality is nonlinear (lives on PC2, weakly)
- [x] **Part 3** — Feature engineering: log transforms, outliers, engineered chemistry (alcohol/density +0.48 beats all)
- [x] **Part 4** — Modeling framework: leakage *demonstrated* (1-NN 0.58→0.43 deduped), stratified split, the classify/regress/ordinal fork, **QWK** deep-dive, baselines (majority = 44% acc but **0 QWK**)
- [x] **Part 5** — Classification: logistic/RF/LightGBM (trees QWK ~0.51 > logistic 0.46); 95% within ±1 grade; rare grades 3/9 collapse to 0% recall
- [x] **Part 6** — Regression & ordinal: regression+round (QWK 0.53) > classification (0.49); optimized rounder &rarr; 0.56; Frank&ndash;Hall ordinal; **framing > model**
- [x] **Part 7** — Class imbalance: class weights / oversample / SMOTE all lift QWK+macro-F1+rare-recall (SMOTE 3x rare recall) at ~1.5pt accuracy cost; hard ceiling at grades 3/9 (n&le;22)
- [x] **Part 8** — Calibration: raw LightGBM ECE 0.088; isotonic &rarr; 0.019 (Brier down); SMOTE inflates probabilities (worse Brier) &rarr; resample+recalibrate
- [ ] Part 9 — Interpretability (gain / permutation / SHAP importance, partial dependence)
- [ ] Part 10 — Capstone: red/white classifier + synthesis

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

# Deep Dive — Wine Quality, Explained from Scratch

The in-depth, **beginner-friendly** companion to the `wine-quality` practice. No prior knowledge
assumed: every idea gets a plain **definition**, a small **example**, and a note on **why it appears
here**. Read it next to the notebooks (`notebooks/00…10`).

Two files make up the documentation:

1. **This file** — a *concept glossary* (in the order the story uses them) + a *part-by-part map* of
   the eleven notebooks.
2. **[CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md)** — every function in `src/`, line by line.

> **The story in one sentence.** We take 6,497 wines described by 11 chemical measurements and learn
> to (a) *understand* them, (b) *predict the taster's quality score* — an **ordinal, imbalanced**
> target full of traps — and (c) discover that *how we frame and evaluate* the problem matters more
> than which model we pick.

This is the **first cross-sectional practice** in the collection (the other three are time series),
chosen to exercise a whole new toolkit: imbalanced classification, the right metrics, leakage,
calibration, and interpretability.

---

## Table of contents

- [1. The basics](#1-the-basics)
- [2. Looking at the data (EDA)](#2-looking-at-the-data-eda)
- [3. Multivariate structure](#3-multivariate-structure)
- [4. Feature engineering](#4-feature-engineering)
- [5. The modeling framework — the heart of the practice](#5-the-modeling-framework--the-heart-of-the-practice)
- [6. Models](#6-models)
- [7. Framing an ordinal target](#7-framing-an-ordinal-target)
- [8. Class imbalance](#8-class-imbalance)
- [9. Probability calibration](#9-probability-calibration)
- [10. Interpretability](#10-interpretability)
- [11. Wine domain terms](#11-wine-domain-terms)
- [12. Part-by-part map](#12-part-by-part-map)

---

## 1. The basics

**Cross-sectional data** — A snapshot where each row is a separate *thing* and there's **no time
axis** (unlike a time series). *Example:* each row is one wine, measured once. *Why here:* the first
non-time-series practice — so no autocorrelation, forecasting, or seasonality; instead the focus is
classification.

**Feature / target** — A **feature** is an input; the **target** is what we predict. *Example:*
features = the 11 chemical measurements; target = `quality`. *Why here:* we predict quality from
chemistry.

**Ordinal target** — A label that is **ordered** but not necessarily evenly spaced. *Example:* wine
quality 3 < 4 < 5 … < 9 — a 7 is better than a 6, but "how much better" isn't fixed. *Why here:* the
central property of `quality`; it sits between a plain category (no order) and a number (equal spacing),
and shapes every modelling choice.

**Class imbalance** — When some target values are far rarer than others. *Example:* quality 5–6 are
76% of wines; quality 9 has just **5**. *Why here:* it makes accuracy misleading and the rare grades
nearly unlearnable — a thread through Parts 4–8.

**Duplicate rows** — Identical rows appearing more than once. *Example:* 1,177 wines (18%) have exactly
repeated measurements. *Why here:* harmless for describing the data but a **leakage** risk for
modelling (see §5).

---

## 2. Looking at the data (EDA)

**Distribution / skew / outliers** — The shape of one variable. **Right-skew** = a long tail of high
values; an **outlier** is a far-from-typical value. *Example:* `chlorides` is very right-skewed (skew
5.4); one wine has 65 g/L of residual sugar. *Why here:* skew and outliers steer the transforms in
Part 3 and matter for linear models.

**Correlation (Pearson / Spearman)** — How two variables move together, in [−1, 1]. **Pearson** =
straight-line; **Spearman** = any consistent up/down (rank-based, right for an *ordinal* target).
*Example:* alcohol vs quality, Spearman +0.45. *Why here:* finding quality's drivers (alcohol up;
volatile acidity, density down).

**Multicollinearity / VIF** — When features carry overlapping information. The **Variance Inflation
Factor** scores it (>5 notable, >10 serious). *Example:* `density` has VIF ≈ 16 — it's basically
determined by alcohol + sugar + acids. *Why here:* collinearity destabilises linear-model coefficients
and confuses "importance" (Part 9).

**WHO of the data — red vs white** — The dataset merges two wine types. *Example:* white wines carry
~2.5× the sugar and ~3× the total SO₂; reds ~1.9× the volatile acidity. *Why here:* the types are
chemically distinct — a structure that dominates the variance (Part 2) and a contrast revisited in the
capstone (Part 10).

---

## 3. Multivariate structure

**PCA (Principal Component Analysis)** — Compresses many correlated features into a few new axes
("components") ordered by how much variation they capture. *Example:* PC1 captures 27% of the chemistry
and turns out to be the **red/white axis**. *Why here:* shows the biggest source of variation is the
wine *type*, not quality.

**t-SNE** — A *nonlinear* 2-D map that preserves local neighbourhoods, often revealing clusters PCA
blurs. *Example:* it shows two clean type-clusters with quality smeared *inside* them. *Why here:*
visual proof that quality is a **subtle, nonlinear** signal (so tree models will help).

**Loadings** — The recipe of original features inside a component. *Example:* PC1 loads on SO₂ + sugar
(white markers) vs volatile acidity + chlorides (red markers). *Why here:* tells us what each PCA axis
*means*.

---

## 4. Feature engineering

**Transformation (log1p)** — Replacing `x` with `log(1+x)` to pull in a long right tail. *Example:*
residual sugar skew 1.7 → 0.5. *Why here:* helps *linear* models; irrelevant to trees — a "preprocessing
is model-dependent" lesson.

**Engineered feature** — A new feature built from existing ones using domain knowledge. *Example:*
`alcohol_density` = alcohol ÷ density correlates +0.48 with quality — **better than any raw feature**.
*Why here:* shows good features can out-predict raw columns; this one becomes the model's #1 driver.

**Standardization (scaling)** — Rescaling features to mean 0, sd 1. *Why here:* needed for linear /
distance models (PCA, logistic, KNN), not for trees.

---

## 5. The modeling framework — the heart of the practice

**Data leakage** — When information that wouldn't be available (or that trivially gives away the answer)
sneaks into training, inflating scores. *Example:* a duplicated wine in *both* train and test lets a
model "memorise" the test answer. *Why here:* Part 4 *demonstrates* it — a 1-NN scores 0.576 with
duplicates but only 0.432 deduped (a 14-point illusion). The fix: **`dedup()` before every split**.

**Train/test split** — Fit on one slice, evaluate on an unseen slice. *Why here:* the basis of honest
scores.

**Stratification** — Splitting so each class keeps the **same proportion** in train and test. *Example:*
with 5 grade-9 wines, a random split could put them all in training; stratifying spreads them. *Why
here:* keeps every grade present everywhere — essential under imbalance.

**Cross-validation (stratified k-fold)** — Rotate the test set through k disjoint, class-balanced
slices and average — a more stable estimate than one split. *Why here:* every model is reported as
**mean ± std** QWK.

**Accuracy (and why it lies)** — Fraction correct. *Example:* always guessing "6" scores 44% accuracy
here — but has *zero* real skill. *Why here:* the cautionary baseline; under imbalance, accuracy
flatters useless models.

**Macro-F1** — The F1 score (precision/recall balance) averaged **equally over classes**, so ignoring
rare grades is punished. *Why here:* exposes the rare-grade collapse that accuracy hides.

**MAE (Mean Absolute Error)** — Average absolute distance between predicted and true grade. *Why here:*
a simple ordinal-aware error (predicting 5 for a 7 costs 2).

**Quadratic Weighted Kappa (QWK)** — *The* metric here. Read it as **agreement corrected for chance,
where being *far* off is penalised more than being *close***: the penalty for predicting grade *i* when
the truth is *j* grows with **(i−j)²**. Scale: 1 = perfect, 0 = chance, <0 = worse than chance.
*Example:* true 5 → predict 6 scores far higher than true 5 → predict 8. *Why here:* it's the standard
score for ordinal ratings and the headline number throughout.

**Baseline** — A trivial model a real one must beat (majority class, random). *Why here:* the bar —
and the proof that accuracy lies (majority: 44% accuracy, **0** QWK).

---

## 6. Models

**Logistic regression** — A linear classifier. *Why here:* the interpretable baseline; loses to trees
because quality is nonlinear.

**Random forest / LightGBM** — Tree ensembles that capture nonlinearity and interactions; **LightGBM**
is gradient-boosted (each tree fixes the last's errors). *Why here:* the workhorses (QWK ≈ 0.51–0.52),
beating logistic (0.46).

**Confusion matrix** — A table of predicted vs actual class. *Example:* shows most errors are
**off-by-one** grades, and that grades 3 and 9 are essentially never predicted. *Why here:* reveals
*how* the model is wrong (ordinally close), and the rare-grade collapse.

**Recall (per class)** — Of the wines truly of grade *g*, what fraction did we catch? *Example:* ~0%
for grades 3 and 9. *Why here:* names the imbalance problem that Part 7 attacks.

---

## 7. Framing an ordinal target

**The framing fork** — Three ways to model an ordinal target:
- **Classification** — predict the grade as an unordered category (ignores order).
- **Regression** — predict a number and round (respects order, assumes equal spacing).
- **Ordinal regression** — the principled middle (order without equal-spacing).
*Why here:* Part 6 shows the framing choice is worth *more* than the model choice.

**Optimized rounding** — Learning the **cut points** that turn a continuous prediction into grades so
as to maximise QWK, instead of rounding at .5. *Example:* the learned 8→9 boundary is pushed to ~8.6
because grade 9 is so rare. *Why here:* lifts QWK "for free" (0.525 → 0.563).

**Frank–Hall ordinal regression** — Decompose the K-grade problem into **K−1 binary "is quality > k?"**
models, then reconstruct the grade distribution. *Why here:* the theoretically-right ordinal framing;
also beats plain classification.

---

## 8. Class imbalance

**Class weights** — Telling the loss to count rare-class mistakes more heavily. *Why here:* one cheap
remedy for the rare grades.

**Oversampling / SMOTE** — Rebalancing the training set by **duplicating** minority rows
(oversampling) or **synthesising** new ones by interpolating between neighbours (**SMOTE**). *Why
here:* both lift rare-grade recall; SMOTE most (3×).

**The accuracy-vs-tails tradeoff** — Balancing improves rare-class recall, macro-F1 and QWK but costs
a little accuracy. *Why here:* the honest verdict — under QWK, balancing is a net win.

**The imbalance ceiling** — Resampling redistributes attention but **can't invent information**.
*Example:* grades 3 and 9 (n ≤ 22 in training) stay near 0% recall even after SMOTE. *Why here:* the
sober limit — only *more data* fixes the extreme tails.

---

## 9. Probability calibration

**Calibration** — Whether predicted probabilities match reality: of the wines a model calls "70%
likely good," ~70% should be good. *Why here:* needed before any *decision* (flagging, pricing) built
on a probability.

**Reliability curve** — A plot of predicted probability vs observed frequency; the diagonal is perfect.
*Why here:* the visual calibration check (Part 8).

**Brier score** — Mean squared error of probabilities (lower = better). **ECE (Expected Calibration
Error)** — the average gap between confidence and reality. *Why here:* the two calibration summary
numbers.

**Platt scaling vs isotonic regression** — Post-hoc fixes that re-map a model's scores to honest
probabilities: **Platt** fits a smooth sigmoid (robust, few parameters); **isotonic** fits any
monotonic curve (flexible, needs data). *Example:* isotonic cut ECE from 0.088 to 0.02. *Why here:*
isotonic wins; Platt helps less because the distortion isn't a clean S-shape.

**The SMOTE–calibration link** — Resampling distorts the class balance the model "sees," so its
probabilities inflate. *Why here:* if you SMOTE (Part 7) you must **recalibrate** (Part 8).

---

## 10. Interpretability

**Feature importance — three ways**:
- **Gain** — total loss reduction from a feature's splits (fast, but **biased when features
  correlate** — it scatters credit among look-alikes).
- **Permutation** — shuffle a feature and measure how much error worsens (honest about predictive
  value).
- **SHAP** — game-theoretic, *consistent* attribution that sums to each prediction.
*Example:* gain ranks `alcohol_density` nearly **last**; SHAP and permutation rank it **first**. *Why
here:* the lesson that gain mis-ranks correlated features — prefer SHAP/permutation.

**SHAP beeswarm / dependence / waterfall** — Global view (every wine, every feature), a feature's
effect *across its range*, and a single wine's explanation. *Why here:* confirms the EDA's drivers in
*quality points*, shows shapes (alcohol's diminishing returns), and explains individual wines.

**Associational vs causal** — Importance shows what *predicts* quality, not what *causes* it. *Why
here:* a caveat — "high alcohol predicts higher scores" ≠ "add alcohol to improve a wine."

---

## 11. Wine domain terms

**PM... no — the 11 features.** *Acidity* (`fixed`, `volatile`, `citric`, `pH`): fixed acids are
structural; **volatile acidity** is the vinegar taint that *hurts* quality. **Residual sugar**:
sweetness. **Chlorides**: saltiness. **Free / total sulfur dioxide**: preservative (free is the active
part of total). **Density**: ~determined by alcohol + sugar. **Sulphates**: an additive linked to
quality. **Alcohol**: the single strongest *positive* driver. *Why here:* these are the inputs; the
EDA and SHAP keep returning to alcohol (+) and volatile acidity (−).

**Quality score** — A median of ≥3 blind tasters' 0–10 ratings (here 3–9 in practice). *Why here:* the
target — and **subjective**, which caps how well chemistry alone can predict it (the Part 10 lesson).

---

## 12. Part-by-part map

Each notebook is generated by `build_notebooks.py`. For each: the **goal**, the `src` it uses, the
**concepts**, and the **headline finding**.

| Part | Notebook | Goal | Key `src` | Headline finding |
|---|---|---|---|---|
| 0 | `00_data_cleaning` | Acquire & clean | `data.py` | 18% duplicate-rows leakage trap; imbalanced ordinal target (q9: 5 wines) |
| 1 | `01_advanced_eda` | Cross-sectional EDA | `eda.py`, `data.py` | Alcohol top driver (+0.45); density VIF ≈ 16; red/white chemically distinct |
| 2 | `02_multivariate` | PCA / t-SNE | — | PC1 (27%) is the *type* axis; quality is nonlinear (lives on PC2) |
| 3 | `03_feature_engineering` | Transforms & features | `data.engineer` | `alcohol_density` (+0.48) beats every raw feature; preprocessing is model-dependent |
| 4 | `04_modeling_framework` | Honest setup | `modeling.py` | Leakage *demonstrated* (1-NN 0.58→0.43); QWK defined; majority = 44% acc but **0 QWK** |
| 5 | `05_classification` | Classify quality | `modeling.py` | Trees QWK ≈ 0.51 > logistic 0.46; 95% within ±1 grade; grades 3/9 collapse to 0% recall |
| 6 | `06_regression_ordinal` | Framing the target | `modeling.OptimizedRounder` | Regression + optimized rounder (QWK 0.56) beats classification (0.49); **framing > model** |
| 7 | `07_imbalance` | Rescue the tails | `modeling.py` | SMOTE 3× rare-recall at ~1.5 pt accuracy cost; hard ceiling at grades 3/9 |
| 8 | `08_calibration` | Honest probabilities | `modeling.py` | Isotonic ECE 0.088 → 0.02; SMOTE breaks calibration |
| 9 | `09_interpretability` | Open the box | `modeling.py` | Gain mis-ranks correlated `alcohol_density` (last) vs SHAP (first) |
| 10 | `10_capstone` | Contrast & synthesis | `data.py`, `modeling.py` | Red/white AUC ≈ 1.0 vs quality QWK 0.56 — difficulty lives in the *target* |

**The through-line:** *understand → frame honestly → evaluate with the right metric → respect the
imbalance → calibrate → explain.* The recurring lesson is that **methodology beats model tuning** —
leakage control, metric choice, and problem framing each moved the result more than swapping
algorithms ever did.

> Next: open **[CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md)** for the line-by-line code justification.

"""Generate the wine-quality notebooks from readable source (nbformat).
Re-run:  python build_notebooks.py [N ...]   then execute with jupyter nbconvert.
"""
from __future__ import annotations

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NB_DIR = "notebooks"

SETUP = """\
import sys, pathlib, warnings
warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = pathlib.Path.cwd(); ROOT = ROOT if (ROOT / "src").exists() else ROOT.parent
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src import data, eda
eda.set_style()
pd.set_option("display.width", 120, "display.max_columns", 30)
print("setup ok | numpy", np.__version__, "| pandas", pd.__version__)
"""


def build(cells, name, title):
    nb = new_notebook()
    nb.cells = [new_markdown_cell(title)] + cells
    nb.metadata["kernelspec"] = {"display_name": "Python (wine-quality)",
                                 "language": "python", "name": "wine-quality"}
    with open(f"{NB_DIR}/{name}", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote", name, f"({len(nb.cells)} cells)")


# ===================================================================== Notebook 0
def notebook_0():
    md, co = new_markdown_cell, new_code_cell
    cells = [
        md(
"""## Part 0 — Acquire & Clean (Wine Quality)

A change of pace from the three time-series practices: this is a **cross-sectional** dataset — 6,497
wines (1,599 red + 4,898 white from the UCI Wine Quality data), each described by **11
physicochemical measurements** and rated by tasters on an ordinal **`quality`** score (3–9). The
goal will be to predict and explain quality.

Two features of this dataset make it interesting (and a little dangerous):
- the **target is ordinal and heavily imbalanced** (most wines are average; the extremes are rare),
- there are **1,177 duplicate rows (18%)** — which, if split carelessly, leak test answers into training."""),
        co(SETUP),

        md("### 1. Structure"),
        co("""df = data.clean()
print("shape:", df.shape, "| missing values:", int(df.isna().sum().sum()))
print("wine types:", df.wine_type.value_counts().to_dict())
print("\\n11 features + ordinal quality target:")
print(df.dtypes.to_string())"""),

        md(
"""### 2. The duplicate-rows trap

18% of rows are exact duplicates — wines with identical measurements (plausibly repeat lab readings
or wines from the same batch). They're harmless for *describing* the data, but for *modelling* they
are a leakage risk: a duplicate landing in both train and test lets a model "memorise" a test answer.
So we **keep them for EDA** but expose a `dedup()` to use **before** any train/test split."""),
        co("""print("duplicate rows:", int(df.duplicated().sum()), "(%.0f%%)" % (100*df.duplicated().mean()))
print("after dedup:", data.dedup(df).shape[0], "wines")
print("\\nexample of a repeated wine (first duplicated row appears %d times):"
      % int((df == df[df.duplicated(keep=False)].iloc[0]).all(axis=1).sum()))
df[df.duplicated(keep=False)].sort_values(data.NUMERIC).head(2)[data.NUMERIC[:5] + ["quality"]]"""),

        md(
"""### 3. The target — ordinal and imbalanced

`quality` is an **ordinal** label (6 > 5 is meaningfully ordered, unlike nominal categories) and
badly **imbalanced**: the middle grades dominate, the best and worst wines are rare. Quality 9 has
just **5** wines in the whole dataset — a real challenge for any classifier."""),
        co("""print(df.quality.value_counts().sort_index().to_string())
print("\\ntop class share: %.0f%% | extremes (3,4,8,9): %.1f%% | quality 9: %d wines"
      % (100*df.quality.value_counts(normalize=True).max(), 100*df.quality.isin([3,4,8,9]).mean(), int((df.quality==9).sum())))
print("\\ncoarse 3-band framing (still imbalanced):")
print(data.quality_band(df.quality).value_counts().sort_index().to_string())"""),

        md("### 4. Clean & persist"),
        co("""data.build_processed()
print("wrote data/processed/wine_clean.csv — Part 0 complete (duplicates kept + flagged; dedup() ready for modelling).")"""),
    ]
    build(cells, "00_data_cleaning.ipynb", "# 00 · Wine Quality — Data Acquisition & Cleaning")


# ===================================================================== Notebook 1
def notebook_1():
    md, co = new_markdown_cell, new_code_cell
    cells = [
        md(
"""## Part 1 — Advanced EDA (Wine Quality)

A thorough cross-sectional EDA: the imbalanced ordinal target, the shapes of all 11 chemical
features, **what actually drives quality**, the **multicollinearity** among the chemistry, and how
**red and white wines differ**."""),
        co(SETUP + "\ndf = data.clean()\nprint('wines:', len(df), '| features:', len(data.NUMERIC))"),

        md(
"""### 1. The target — imbalance & the red/white split

Most wines are rated 5–6; the extremes are scarce. White wines are rated slightly higher on average
than reds — a first hint that the two types behave differently."""),
        co("""fig, ax = plt.subplots(1, 2, figsize=(13, 4))
df.quality.value_counts().sort_index().plot(kind="bar", ax=ax[0], color="indianred"); ax[0].set_title("quality distribution (imbalanced)"); ax[0].set_xlabel("quality score")
df.groupby("wine_type", observed=True).quality.mean().plot(kind="bar", ax=ax[1], color=["#7B1E3B","#E8D98A"]); ax[1].set_title("mean quality: white > red"); ax[1].set_ylim(5, 6)
eda.savefig(fig, "p1_target.png"); plt.show()
print("mean quality — red %.2f vs white %.2f" % (df[df.wine_type=='red'].quality.mean(), df[df.wine_type=='white'].quality.mean()))"""),

        md(
"""### 2. Univariate sweep — every chemical feature

The 11 measurements have very different shapes. Several are **strongly right-skewed** (chlorides,
sulphates, residual sugar) with long tails of outliers — a sign that some features may benefit from a
log transform or robust handling."""),
        co("""fig, axes = plt.subplots(3, 4, figsize=(16, 9))
for ax, c in zip(axes.ravel(), data.NUMERIC):
    sns.histplot(df[c], bins=40, ax=ax, color="slategray"); ax.set_title(c, fontsize=9)
axes.ravel()[-1].axis("off")
fig.suptitle("Distribution of every physicochemical feature", y=1.01); fig.tight_layout()
eda.savefig(fig, "p1_univariate.png"); plt.show()
print("most right-skewed features:"); print(df[data.NUMERIC].skew().round(2).sort_values(ascending=False).head(4).to_string())"""),

        md(
"""### 3. What drives quality?

Correlate each feature with the (ordinal) quality using **Spearman** (rank correlation, right for an
ordinal target). **Alcohol** is by far the strongest positive driver; **volatile acidity** (a vinegar
taint) and **density** pull quality down."""),
        co("""corr_q = df[data.NUMERIC].corrwith(df["quality"], method="spearman").sort_values()
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
corr_q.plot.barh(ax=ax[0], color=["tab:red" if v<0 else "tab:green" for v in corr_q]); ax[0].set_title("Spearman corr with quality"); ax[0].axvline(0, color="k", lw=.6)
sns.heatmap(df[data.NUMERIC].corr(), cmap="coolwarm", center=0, ax=ax[1], cbar_kws={"shrink":.7}); ax[1].set_title("feature–feature correlation")
fig.tight_layout(); eda.savefig(fig, "p1_drivers.png"); plt.show()
print("top quality drivers: alcohol %+.2f | volatile_acidity %+.2f | density %+.2f"
      % (corr_q["alcohol"], corr_q["volatile_acidity"], corr_q["density"]))"""),

        md(
"""### 4. Multicollinearity — the chemistry is interlinked

Many features are mechanically related: **density** is essentially determined by alcohol and sugar;
**free** SO₂ is part of **total** SO₂. VIF (>5 notable, >10 serious) flags the redundancy — important
because collinear features make linear-model coefficients unstable and muddy "importance"."""),
        co("""vif = eda.vif_table(df[data.NUMERIC])
print(vif.round(1).to_string(index=False))
print("\\n-> density is the most collinear (it's ~a function of alcohol + sugar + acids); SO2 pair is linked too.")"""),

        md(
"""### 5. Red vs white — two different wines

The `wine_type` flag hides a large chemical gap. White wines carry far more **residual sugar** and
**total SO₂** (preservative), reds far more **volatile acidity** and **chlorides**. They're almost
separable on chemistry alone (which Part 2's PCA will confirm)."""),
        co("""feats = ["residual_sugar", "total_sulfur_dioxide", "volatile_acidity", "chlorides"]
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
for a, c in zip(ax, feats):
    sns.boxplot(x="wine_type", y=c, data=df, ax=a, palette=["#7B1E3B","#E8D98A"], showfliers=False); a.set_title(c)
fig.tight_layout(); eda.savefig(fig, "p1_redwhite.png"); plt.show()
print("white has %.1fx the sugar and %.1fx the total SO2; red has %.1fx the volatile acidity"
      % (df[df.wine_type=='white'].residual_sugar.mean()/df[df.wine_type=='red'].residual_sugar.mean(),
         df[df.wine_type=='white'].total_sulfur_dioxide.mean()/df[df.wine_type=='red'].total_sulfur_dioxide.mean(),
         df[df.wine_type=='red'].volatile_acidity.mean()/df[df.wine_type=='white'].volatile_acidity.mean()))"""),

        md(
"""### 6. The 'good wine' picture

The two strongest drivers, seen directly: average **alcohol rises** steadily with quality, while
**volatile acidity falls** — higher-rated wines are stronger and less acetic."""),
        co("""fig, ax = plt.subplots(1, 2, figsize=(13, 4))
sns.boxplot(x="quality", y="alcohol", data=df, ax=ax[0], color="seagreen", showfliers=False); ax[0].set_title("alcohol rises with quality")
sns.boxplot(x="quality", y="volatile_acidity", data=df, ax=ax[1], color="indianred", showfliers=False); ax[1].set_title("volatile acidity falls with quality")
fig.tight_layout(); eda.savefig(fig, "p1_goodwine.png"); plt.show()"""),

        md(
"""### Takeaways

- **Cross-sectional & numeric**: 11 clean physicochemical features, no missing values — but **18%
  duplicate rows** (a leakage risk handled by `dedup()` before modelling).
- The **ordinal target is imbalanced** (mid grades 77%; quality 9 only 5 wines) — accuracy alone will
  be misleading; later we'll need macro-F1 / per-class metrics and imbalance handling.
- **Alcohol is the top quality driver** (Spearman +0.4); volatile acidity and density hurt. Several
  features are **right-skewed** with outliers.
- The chemistry is **collinear** (density ≈ alcohol+sugar; free/total SO₂) — a VIF story for any linear
  model.
- **Red and white are chemically distinct** (sugar, SO₂, volatile acidity) — almost separable.

**Next — Part 2 (Multivariate structure):** PCA of the chemistry (does it separate red/white and
track quality?), and the relationships that a single correlation misses — before we move to
imbalanced ordinal **classification** of quality."""),
    ]
    build(cells, "01_advanced_eda.ipynb", "# 01 · Wine Quality — Advanced EDA")


# ===================================================================== Notebook 2
def notebook_2():
    md, co = new_markdown_cell, new_code_cell
    cells = [
        md(
"""## Part 2 — Multivariate Structure & Dimensionality

Part 1 looked at features one and two at a time. But 11 chemical measurements form a *joint*
structure that single correlations miss. We compress that structure with **PCA** (linear) and
**t-SNE** (nonlinear), and ask two questions:

1. does the chemistry **separate red from white**?
2. does any direction **track quality** — i.e. is quality a simple combination of the chemistry, or
   something more subtle?

The answers shape how we model: the biggest axis of variation turns out to be the wine *type*, not
the *quality* we actually care about."""),
        co(SETUP + """
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
df = data.clean()
X = StandardScaler().fit_transform(df[data.NUMERIC])     # standardize: PCA is variance-driven
red = (df.wine_type == "red").to_numpy()
print("standardized feature matrix:", X.shape)"""),

        md(
"""### 1. PCA — how much structure, and in how many dimensions?

PCA rotates the 11 features into uncorrelated **components** ordered by variance captured. The scree
plot shows the chemistry is *not* dominated by one direction — it takes several components to explain
most of the variance (no single 'master' chemical axis)."""),
        co("""pca = PCA().fit(X); evr = pca.explained_variance_ratio_
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(range(1, 12), evr*100, color="steelblue"); ax.plot(range(1, 12), evr.cumsum()*100, "o-", color="crimson")
ax.set_xlabel("component"); ax.set_ylabel("% variance"); ax.set_title("Scree: variance explained (bars) & cumulative (line)")
eda.savefig(fig, "p2_scree.png"); plt.show()
print("PC1 %.0f%% | PC2 %.0f%% | PC1+PC2 %.0f%% | components for 90%%: %d"
      % (evr[0]*100, evr[1]*100, evr[:2].sum()*100, (evr.cumsum() < 0.9).sum()+1))"""),

        md(
"""### 2. The PCA map — coloured by type, then by quality

Project every wine onto PC1×PC2. Coloured by **type**, two clouds appear — the chemistry *almost
separates red and white on its own*. Coloured by **quality**, there's only a faint gradient — quality
is **not** a simple linear direction in the chemistry (which is exactly why it's hard to predict)."""),
        co("""S = pca.transform(X)
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
for t, c in [("red", "#7B1E3B"), ("white", "#C9A227")]:
    m = (df.wine_type == t).to_numpy(); ax[0].scatter(S[m,0], S[m,1], s=6, alpha=.3, color=c, label=t)
ax[0].set_title("PCA coloured by wine TYPE — two clouds"); ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2"); ax[0].legend()
sc = ax[1].scatter(S[:,0], S[:,1], s=6, alpha=.4, c=df.quality, cmap="viridis")
ax[1].set_title("PCA coloured by QUALITY — only a faint gradient"); ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2")
plt.colorbar(sc, ax=ax[1], label="quality")
fig.tight_layout(); eda.savefig(fig, "p2_pca_map.png"); plt.show()"""),

        md(
"""### 3. What do the components *mean*? — loadings

The **loadings** say which original features each component is built from. PC1 loads on **SO₂ and
sugar** (the white-wine markers) against **volatile acidity and chlorides** (the red markers) — it
*is* the type axis. The component that best correlates with quality is PC2."""),
        co("""load = pd.DataFrame(pca.components_[:3].T, index=data.NUMERIC, columns=["PC1","PC2","PC3"])
print(load.round(2).to_string())
for k in range(3):
    gap = abs(S[red,k].mean() - S[~red,k].mean()); qc = np.corrcoef(S[:,k], df.quality)[0,1]
    print("PC%d: red/white separation %.2f | corr with quality %+.2f" % (k+1, gap, qc))
print("\\n-> PC1 = the red/white axis (variance 27%%); quality lives mostly on PC2 (corr -0.31), weakly.")"""),

        md(
"""### 4. t-SNE — a nonlinear map

PCA is linear; **t-SNE** preserves *local* neighbourhoods and often reveals clusters PCA blurs. On a
2,000-wine sample it shows the same story more vividly: two well-separated **type** clusters, but
quality smeared *within* them — confirming quality is a subtle, nonlinear signal."""),
        co("""idx = np.random.default_rng(0).choice(len(X), 2000, replace=False)
emb = TSNE(n_components=2, perplexity=30, random_state=0, init="pca").fit_transform(X[idx])
sub = df.iloc[idx]
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
for t, c in [("red", "#7B1E3B"), ("white", "#C9A227")]:
    m = (sub.wine_type == t).to_numpy(); ax[0].scatter(emb[m,0], emb[m,1], s=8, alpha=.5, color=c, label=t)
ax[0].set_title("t-SNE by type — clean clusters"); ax[0].legend()
sc = ax[1].scatter(emb[:,0], emb[:,1], s=8, alpha=.6, c=sub.quality, cmap="viridis"); ax[1].set_title("t-SNE by quality — smeared within clusters")
plt.colorbar(sc, ax=ax[1], label="quality"); fig.tight_layout(); eda.savefig(fig, "p2_tsne.png"); plt.show()"""),

        md(
"""### Takeaways

- The chemistry's **biggest axis of variation (PC1, 27%) is the wine *type*** — not quality. Whites
  and reds nearly separate on chemistry alone (PCA and t-SNE both show two clouds).
- **Quality is not a simple linear direction**: the best component for it (PC2) correlates only −0.31,
  and t-SNE smears quality *within* each type cluster — a clear signal that quality prediction is a
  **subtle, nonlinear** problem (favouring tree/nonlinear models, Part 5).
- A modeling implication: the dominant variance is a **nuisance** (type), so either model red and
  white separately or keep `wine_type` as a feature and let the model factor it out.

**Next — Part 3 (Feature engineering & transformations):** tame the skewed features, handle outliers,
and engineer chemically-meaningful features (acidity ratios, bound SO₂, sugar/alcohol balance) before
modelling."""),
    ]
    build(cells, "02_multivariate.ipynb", "# 02 · Wine Quality — Multivariate Structure & Dimensionality (PCA, t-SNE)")


# ===================================================================== Notebook 3
def notebook_3():
    md, co = new_markdown_cell, new_code_cell
    cells = [
        md(
"""## Part 3 — Feature Engineering & Transformations

Before modelling, prepare the features. The EDA flagged **right-skew** and **outliers** (which hurt
linear and distance-based models, though not trees) and **collinearity**. Here we:

1. **transform** the skewed features (log) and see which improve,
2. **inspect the outliers** (real extreme wines, not errors),
3. **engineer chemically-meaningful features** — and find a ratio that beats every raw feature,
4. ask honestly: **does any of this actually help?** (answer: it depends on the model).

A theme worth stating up front: *feature engineering is model-dependent* — transforms that rescue a
linear model do nothing for a gradient-boosted tree."""),
        co(SETUP + """
df = data.dedup(data.clean())     # dedup FIRST — modelling-oriented from here on
print("modelling rows (deduped):", len(df))"""),

        md(
"""### 1. Taming the skew

Several features have long right tails. A **log1p** transform (`log(1+x)`) pulls them in. It works
well for sugar and the acidities, over-corrects free SO₂, and barely helps **chlorides** — which has a
stubborn outlier the log can't fix."""),
        co("""sk = pd.DataFrame({"raw skew": df[data.SKEWED].skew(),
                   "log1p skew": np.log1p(df[data.SKEWED]).skew()}).round(2).sort_values("raw skew", ascending=False)
print(sk.to_string())
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df.residual_sugar, bins=50, ax=ax[0], color="indianred"); ax[0].set_title(f"residual sugar (skew {df.residual_sugar.skew():.2f})")
sns.histplot(np.log1p(df.residual_sugar), bins=50, ax=ax[1], color="seagreen"); ax[1].set_title(f"log1p (skew {np.log1p(df.residual_sugar).skew():.2f})")
eda.savefig(fig, "p3_skew.png"); plt.show()"""),

        md(
"""### 2. Outliers — real wines, not errors

The extremes are *plausible chemistry*, not data-entry mistakes: a 65 g/L residual-sugar wine is a
genuine dessert white. We **keep** them (they carry real signal) but note that linear models and the
raw-scale distance methods are sensitive to them — another reason trees are attractive here."""),
        co("""ext = df.nlargest(3, "residual_sugar")[["residual_sugar","alcohol","density","quality","wine_type"]]
print("sweetest wines (real dessert whites):"); print(ext.to_string())
print("\\nmax chlorides %.2f, max free SO2 %.0f, max total SO2 %.0f — extreme but physically possible"
      % (df.chlorides.max(), df.free_sulfur_dioxide.max(), df.total_sulfur_dioxide.max()))"""),

        md(
"""### 3. Engineered chemistry — a ratio that beats them all

Domain knowledge suggests combinations: **bound SO₂** (total − free), the **free/total SO₂ ratio**,
**total acidity**, and **alcohol ÷ density** (a normalised "strength"). The last one correlates
**+0.48** with quality — *better than alcohol alone* (+0.45). A good engineered feature can out-predict
every raw measurement."""),
        co("""e = data.engineer(df)
rows = [(c, e[c].corr(e.quality, method="spearman")) for c in data.ENGINEERED + ["alcohol"]]
print(pd.DataFrame(rows, columns=["feature","spearman_vs_quality"]).set_index("feature").round(3).sort_values("spearman_vs_quality").to_string())
print("\\n-> alcohol_density (+0.48) edges out raw alcohol (+0.45); so2_ratio adds a little too.")"""),

        md(
"""### 4. Scaling, and does any of it help?

**Standardization** (mean 0, sd 1) matters for linear/distance models, not for trees. The honest
test: does log-transforming help a logistic-regression model predict "good wine" (quality ≥ 7)? Only
**marginally** — and a tree wouldn't benefit at all. The lesson: *match the preprocessing to the
model; don't transform on autopilot.*"""),
        co("""from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
y = (df.quality >= 7).astype(int)
Xr = df[data.NUMERIC]; Xl = Xr.copy()
for c in data.SKEWED: Xl[c] = np.log1p(Xl[c])
sc = lambda X: cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X, y, cv=5, scoring="roc_auc").mean()
print("LogReg ROC-AUC (good-wine ≥7):  raw %.3f  ->  log-transformed %.3f  (marginal)" % (sc(Xr), sc(Xl)))
Xe = data.engineer(df)[data.NUMERIC + data.ENGINEERED]
print("                                +engineered features %.3f" % sc(Xe))"""),

        md(
"""### Takeaways

- **Log transforms** rescue the skewed features for *linear* models (sugar 1.7 → 0.5) but can't fix a
  hard outlier (chlorides) and are *irrelevant to trees*.
- The **outliers are real** (a 65 g/L dessert wine) — keep them, but prefer models robust to scale.
- **Engineered chemistry pays off**: `alcohol/density` (+0.48) beats every raw feature — domain
  knowledge > raw columns.
- **Preprocessing is model-dependent**: transforms gave a logistic model only a sliver, and would give
  a tree nothing. Don't preprocess on autopilot.

**Next — Part 4 (Modeling framework):** the honest setup — dedup → stratified split, the
classification-vs-regression-vs-**ordinal** fork, the right metric for an imbalanced ordinal target
(**quadratic weighted kappa**), cross-validation, and baselines."""),
    ]
    build(cells, "03_feature_engineering.ipynb", "# 03 · Wine Quality — Feature Engineering & Transformations")


# ===================================================================== Notebook 4
def notebook_4():
    md, co = new_markdown_cell, new_code_cell
    cells = [
        md(
"""## Part 4 — The Modeling Framework

Before fitting a single serious model, we build the **scaffolding that makes every later result
trustworthy**. Three properties of this dataset each set a trap, and each needs a deliberate choice:

| trap | consequence if ignored | the fix |
|---|---|---|
| **18% duplicate rows** | a wine sits in train *and* test → inflated scores | `dedup()` **before** the split |
| **imbalanced target** | rare grades vanish from a fold; accuracy looks good for free | **stratified** split + k-fold |
| **ordinal target** | a 5→8 error is treated like a 5→6 error | score with **quadratic weighted kappa** |

This part demonstrates each trap concretely, defines the metric we'll live by, and establishes the
**baselines** every model in Parts 5–9 must beat. Get this wrong and every later number is a lie —
so it's worth doing slowly."""),
        co(SETUP + """
from src import modeling as M
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
raw = data.clean()
print("raw rows:", len(raw), "| deduped:", len(data.dedup(raw)))"""),

        md(
"""### 1. The leakage trap, *demonstrated*

It's easy to *say* "duplicates cause leakage." Let's **prove it**. We train the same 1-nearest-neighbour
classifier two ways — once on the data with duplicates kept, once deduped — using an identical
stratified split. With duplicates, many test wines have an **exact twin** in the training set, so 1-NN
finds a perfect match and looks far better than it really is.

The gap between the two bars is *pure illusion* — accuracy bought by memorising leaked rows."""),
        co("""def knn_acc(dedup):
    X, y = M.prep(data.dedup(raw) if dedup else raw)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, stratify=y, random_state=0)
    return KNeighborsClassifier(1).fit(Xtr, ytr).score(Xte, yte)
dup, ded = knn_acc(False), knn_acc(True)
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["duplicates kept\\n(leaky)", "deduped\\n(honest)"], [dup, ded], color=["firebrick", "seagreen"])
ax.set_ylabel("1-NN test accuracy"); ax.set_title(f"Leakage illusion: +{100*(dup-ded):.0f} points for free")
for i, v in enumerate([dup, ded]): ax.text(i, v+.01, f"{v:.3f}", ha="center")
eda.savefig(fig, "p4_leakage.png"); plt.show()
print("duplicates inflate 1-NN accuracy %.3f -> deduped %.3f. We dedup before EVERY split from now on." % (dup, ded))"""),

        md(
"""### 2. The stratified split

With only **5 wines at quality 9** in the whole dataset, a random split could easily put *all* of them
in training (or testing), leaving a fold that can neither learn nor be scored on that grade.
**Stratification** forces each grade to appear in train and test in the *same proportion*. The table
below shows the proportions matched almost exactly."""),
        co("""Xtr, Xte, ytr, yte = M.split(raw)   # dedup -> engineer -> stratified 75/25 split
prop = pd.DataFrame({"train %": ytr.value_counts(normalize=True).sort_index()*100,
                     "test %":  yte.value_counts(normalize=True).sort_index()*100}).round(1)
print("train %d wines / test %d wines | features: %d" % (len(ytr), len(yte), Xtr.shape[1]))
print(prop.to_string())
print("\\nq9 -> train %d, test %d (both non-empty thanks to stratification)" % ((ytr==9).sum(), (yte==9).sum()))"""),

        md(
"""### 3. The framing fork — three ways to model an ordinal target

`quality` is **ordinal**: ordered (8 > 7 > 6) but not necessarily evenly spaced. That gives three
modelling stances, each with a different blind spot:

- **Classification** (predict the grade as a *category*). Simple, gives per-class probabilities — but
  *throws away the order*: to a plain classifier, mistaking a 5 for an 8 is no worse than a 5 for a 6.
- **Regression** (predict a *number*, then round). *Respects order* and naturally penalises far misses
  — but assumes the grades are **evenly spaced** (that 5→6 is the same "distance" as 8→9).
- **Ordinal regression** (fit K−1 cumulative "is quality > k?" models). The honest middle: uses the
  order *without* assuming equal spacing — at the cost of complexity.

We'll try **all three** (classification in Part 5, regression & ordinal in Part 6) and let the metric
decide. The metric, therefore, must itself understand order — which rules out plain accuracy."""),

        md(
"""### 4. The metric — Quadratic Weighted Kappa (QWK)

QWK is the standard score for ordinal ratings (it's what the wine and essay-grading literature use).
Read it as **agreement, corrected for chance, where being *far* off is penalised more than being
*close*** — the penalty for predicting grade *i* when the truth is *j* grows with the **square** of the
distance (i−j)². The heatmap below *is* that penalty matrix: the diagonal (correct) is free, and cost
rises sharply as you move away from it. Scale: **1** = perfect, **0** = no better than chance, **<0** =
worse than chance."""),
        co("""K = M.LABELS
W = (K[:, None] - K[None, :])**2 / (len(K)-1)**2
fig, ax = plt.subplots(figsize=(5.5, 4.6))
sns.heatmap(W, annot=True, fmt=".2f", xticklabels=K, yticklabels=K, cmap="Reds", cbar_kws={"label":"penalty"}, ax=ax)
ax.set_xlabel("predicted grade"); ax.set_ylabel("true grade"); ax.set_title("QWK penalty = (true-pred)² / (K-1)²")
eda.savefig(fig, "p4_qwk_weights.png"); plt.show()
print("worked example — true grade 5:")
print("  predict 6 (off by 1): QWK %.3f" % M.qwk([5,6,7,5,6], [6,6,7,5,6]))
print("  predict 8 (off by 3): QWK %.3f  <- far miss punished far harder" % M.qwk([5,6,7,5,6], [8,6,7,5,6]))"""),

        md(
"""### 5. Cross-validation — one number is not enough

A single train/test split is noisy. **Stratified k-fold** rotates the test set through 5 disjoint
slices (each preserving the class balance) and reports the **mean ± spread** — a far more honest
estimate of how a model generalises. `modeling.cv_qwk` wraps this; here it is on the logistic
baseline."""),
        co("""logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
Xall, yall = M.prep(data.dedup(raw))
cv = M.cv_qwk(logit, Xall, yall, n=5)
print("logistic 5-fold QWK: %.3f ± %.3f  (folds: %s)" % (cv.mean(), cv.std(), np.round(cv, 3)))"""),

        md(
"""### 6. Baselines — the bar to beat (and proof that accuracy lies)

Three reference points: **majority** (always predict the most common grade, 6), **stratified random**
(guess in proportion to the class frequencies), and a **logistic** model. Look at the majority row:
**44% accuracy yet QWK = 0 and macro-F1 ≈ 0.09**. It has *zero* real skill — it just exploits the
imbalance. Accuracy alone would have called it a decent model. QWK, macro-F1 and MAE all see through
it; the logistic model is the first to show genuine signal (QWK ≈ 0.46)."""),
        co("""rows = []
for name, mdl in [("majority", DummyClassifier(strategy="most_frequent")),
                  ("stratified-random", DummyClassifier(strategy="stratified", random_state=0)),
                  ("logistic", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)))]:
    mdl.fit(Xtr, ytr); rows.append(M.report(yte, mdl.predict(Xte), name))
board = pd.DataFrame(rows)   # M.report returns named Series -> model names become the index
print(board.to_string())
fig, ax = plt.subplots(figsize=(8, 4))
board[["accuracy", "QWK"]].plot.bar(ax=ax, color=["#bbb", "crimson"], rot=0)
ax.set_title("Accuracy looks fine for the majority baseline — QWK exposes it (0)"); ax.axhline(0, color="k", lw=.6)
eda.savefig(fig, "p4_baselines.png"); plt.show()"""),

        md(
"""### Takeaways

- **Leakage is not theoretical**: duplicates inflated 1-NN accuracy by ~14 points. `dedup()` runs
  before every split.
- **Stratification** keeps all seven grades (even quality 9, n=5) present in train and test.
- An **ordinal** target admits three framings (classify / regress / ordinal) — we test all three and
  let the metric arbitrate.
- That metric is **QWK**, which penalises far misses quadratically; we report **accuracy, macro-F1,
  MAE and QWK** together because *accuracy alone lies under imbalance* (majority baseline: 44% acc, 0
  QWK).
- The **bar to beat**: logistic QWK ≈ 0.46 (5-fold 0.46 ± small). Trees should do better — Part 5.

**Next — Part 5 (Classification models):** logistic vs random forest vs LightGBM, per-class metrics,
the confusion matrix, and the rare-grade problem that imbalance leaves behind."""),
    ]
    build(cells, "04_modeling_framework.ipynb", "# 04 · Wine Quality — The Modeling Framework (leakage, stratification, QWK, baselines)")


# ===================================================================== Notebook 5
def notebook_5():
    md, co = new_markdown_cell, new_code_cell
    cells = [
        md(
"""## Part 5 — Classification Models

Now the models. We treat quality as a **multi-class classification** problem (the classify branch of
Part 4's fork) and pit three estimators against each other:

- **Logistic regression** — a linear baseline (with standardisation),
- **Random forest** — bagged trees, captures nonlinearity & interactions,
- **LightGBM** — gradient-boosted trees, usually the strongest tabular learner.

Part 2 warned that quality is a **nonlinear** signal, so we expect the trees to beat the linear model.
But the more important lesson of this notebook is *where every model fails the same way* — the rare
grades — which motivates the imbalance work in Part 7."""),
        co(SETUP + """
from src import modeling as M
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from lightgbm import LGBMClassifier
raw = data.clean()
Xtr, Xte, ytr, yte = M.split(raw)
models = {
    "logistic":      make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
    "random_forest": RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1),
    "lightgbm":      LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=0, verbose=-1, n_jobs=-1),
}
print("train", len(ytr), "| test", len(yte))"""),

        md(
"""### 1. The scoreboard

Fit all three on the identical split and score with the full metric panel. The trees clearly beat
logistic on QWK; random forest and LightGBM are within noise of each other. Accuracy stalls around
0.55 — *not* because the models are bad, but because exact-grade prediction is genuinely hard (Part 2)."""),
        co("""rows, preds = [], {}
for n, m in models.items():
    m.fit(Xtr, ytr); preds[n] = m.predict(Xte); rows.append(M.report(yte, preds[n], n))
board = pd.DataFrame(rows); print(board.to_string())
print("\\nbest test QWK: %s (%.3f)" % (board.QWK.idxmax(), board.QWK.max()))"""),

        md(
"""### 2. Cross-validated, with error bars

A single split is noisy, so confirm the ranking with **stratified 5-fold QWK** (mean ± std). The
ordering holds: random forest ≈ LightGBM > logistic. The error bars overlap for the two tree models —
honestly, they're a tie, and we'll carry **LightGBM** forward as the workhorse (fast, native SHAP)."""),
        co("""Xall, yall = M.prep(data.dedup(raw))
cvs = {n: M.cv_qwk(m, Xall, yall, n=5) for n, m in models.items()}
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(cvs.keys(), [c.mean() for c in cvs.values()], yerr=[c.std() for c in cvs.values()],
       capsize=5, color=["#bbb", "steelblue", "seagreen"])
ax.set_ylabel("5-fold QWK"); ax.set_title("Trees beat logistic; RF ≈ LightGBM")
eda.savefig(fig, "p5_cv.png"); plt.show()
for n, c in cvs.items(): print("  %-14s QWK %.3f ± %.3f" % (n, c.mean(), c.std()))"""),

        md(
"""### 3. The confusion matrix — *how* it's wrong matters

Row-normalised confusion for LightGBM (each row = what the true grade got predicted as). Two things
jump out: the mass sits **on and next to the diagonal** (errors are mostly off-by-one), and the
columns for grades 3, 4, 8, 9 are nearly **empty** — the model almost never *predicts* an extreme
grade. It has learned that betting on the crowded middle is safest."""),
        co("""cm = confusion_matrix(yte, preds["lightgbm"], labels=range(3, 10))
cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(3,10), yticklabels=range(3,10), ax=ax[0]); ax[0].set_title("counts"); ax[0].set_xlabel("predicted"); ax[0].set_ylabel("true")
sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", xticklabels=range(3,10), yticklabels=range(3,10), ax=ax[1]); ax[1].set_title("row-normalised (recall)"); ax[1].set_xlabel("predicted"); ax[1].set_ylabel("true")
eda.savefig(fig, "p5_confusion.png"); plt.show()
within1 = (np.abs(yte.values - preds["lightgbm"]) <= 1).mean()
print("%.1f%% of predictions are within ±1 grade -> errors are ordinally small (why MAE/QWK are decent despite ~0.55 accuracy)." % (100*within1))"""),

        md(
"""### 4. The rare-grade problem, named

Per-grade **recall** makes the failure explicit: the model catches ~60% of the common 5s and 6s but
**0% of quality-3 and quality-9** wines, and almost none of the 4s and 8s. This is the signature of
**class imbalance** — with 5 examples of grade 9, no loss function has any incentive to predict it.
Macro-F1 (which averages over grades equally) is low *because* of this. Part 7 attacks it directly."""),
        co("""cr = classification_report(yte, preds["lightgbm"], labels=range(3,10), output_dict=True, zero_division=0)
rec = pd.Series({g: cr[str(g)]["recall"] for g in range(3,10)})
sup = pd.Series({g: int(cr[str(g)]["support"]) for g in range(3,10)})
fig, ax = plt.subplots(figsize=(8, 4))
rec.plot.bar(ax=ax, color=["firebrick" if s < 100 else "seagreen" for s in sup], rot=0)
ax.set_xlabel("quality grade"); ax.set_ylabel("recall"); ax.set_title("Rare grades (red, support<100) are essentially never recalled")
for i, g in enumerate(range(3,10)): ax.text(i, rec[g]+.02, f"n={sup[g]}", ha="center", fontsize=8)
eda.savefig(fig, "p5_recall.png"); plt.show()"""),

        md(
"""### 5. What the best model leans on

A quick look at LightGBM's gain importance (full SHAP treatment in Part 9). **Alcohol** and the
engineered **alcohol/density** dominate, with **volatile acidity** and **free SO₂** next — exactly the
drivers the EDA flagged, now confirmed inside a predictive model."""),
        co("""lgbm = models["lightgbm"]
imp = pd.Series(lgbm.feature_importances_, index=Xtr.columns).sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
imp.plot.barh(ax=ax, color="darkgreen"); ax.set_title("LightGBM gain importance"); fig.tight_layout()
eda.savefig(fig, "p5_importance.png"); plt.show()
print("top drivers:", list(imp.tail(4).index[::-1]))"""),

        md(
"""### Takeaways

- **Trees beat the linear model** (RF/LightGBM QWK ≈ 0.51–0.52 vs logistic 0.46) — consistent with
  quality being a nonlinear signal (Part 2). RF and LightGBM are a statistical tie; we carry **LightGBM**
  forward.
- **Accuracy plateaus ~0.55**, but **95% of predictions are within ±1 grade** — the models are
  *ordinally* sensible even when not exact (hence respectable MAE/QWK).
- **The tails collapse**: 0% recall on grades 3 and 9. Plain multi-class loss has no reason to ever
  predict a rare grade — the core problem Part 7 tackles.
- Importance confirms the EDA: **alcohol / alcohol-density, volatile acidity, free SO₂**.

**Next — Part 6 (Regression & ordinal framing):** does treating quality as a *number* (regress + round)
or as a proper *ordinal* model beat classification on QWK? Often it does — because those framings bake
in the order that classification throws away."""),
    ]
    build(cells, "05_classification.ipynb", "# 05 · Wine Quality — Classification Models (logistic vs RF vs LightGBM)")


# ===================================================================== Notebook 6
def notebook_6():
    md, co = new_markdown_cell, new_code_cell
    cells = [
        md(
"""## Part 6 — Regression & Ordinal Framing

Part 5 treated quality as unordered categories — and the metric punished that. Here we take the other
two branches of the framing fork and show they **beat classification on QWK with the very same
LightGBM**, because they use the *order* that classification throws away:

1. **Regression** — predict a continuous score, then round to a grade.
2. **Optimized rounding** — learn the round-off cut points to maximise QWK (naive ".5" rounding is
   wrong under imbalance).
3. **Ordinal regression (Frank–Hall)** — decompose into K−1 "is quality > k?" models, then
   reconstruct the grade.

The headline: *how you frame the target is worth more than which model you pick.*"""),
        co(SETUP + """
from src import modeling as M
from lightgbm import LGBMClassifier, LGBMRegressor
LG = dict(n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=0, verbose=-1, n_jobs=-1)
raw = data.clean(); Xtr, Xte, ytr, yte = M.split(raw)
results = {}"""),

        md(
"""### 1. Classification vs regression

Fit the same LightGBM two ways. The **regressor** predicts a continuous score (note its predictions
pile up *between* grades — it can say "5.4", a genuine half-measure a classifier can't) and, simply
rounded, already beats the classifier on QWK. Predicting a number respects that 5→8 is a bigger error
than 5→6."""),
        co("""clf = LGBMClassifier(**LG).fit(Xtr, ytr); results["classification"] = M.report(yte, clf.predict(Xte), "classification")
reg = LGBMRegressor(**LG).fit(Xtr, ytr); raw_pred = reg.predict(Xte)
results["regression+round"] = M.report(yte, raw_pred, "regression+round")
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.hist(raw_pred, bins=60, color="slateblue"); [ax.axvline(g, color="grey", ls=":", lw=.7) for g in range(4,9)]
ax.set_title("Regressor outputs land BETWEEN integer grades"); ax.set_xlabel("predicted quality (continuous)")
eda.savefig(fig, "p6_regpred.png"); plt.show()
print("classification QWK %.3f  ->  regression+round QWK %.3f" % (results["classification"].QWK, results["regression+round"].QWK))"""),

        md(
"""### 2. The optimized rounder

Rounding at the half-integers (.5) assumes the grades are symmetric and equally likely — but they
aren't. `modeling.OptimizedRounder` searches for the **cut points** that maximise QWK on the training
predictions. The learned cuts come out **asymmetric** — the 8→9 boundary is pushed up near 8.6 because
grade 9 is so rare that you should almost never predict it. This lifts test QWK again, for free."""),
        co("""rnd = M.OptimizedRounder().fit(reg.predict(Xtr), ytr)
results["regression+optimized"] = M.report(yte, rnd.predict(raw_pred), "regression+optimized")
print("learned cut points:", np.round(np.sort(rnd.cuts_), 2), "  (naive would be 3.5 .. 8.5)")
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.hist(raw_pred, bins=60, color="slateblue", alpha=.6)
for c in np.sort(rnd.cuts_): ax.axvline(c, color="crimson", lw=1.4)
ax.set_title("Optimized cut points (red) — note the wide '8' band before the rare '9'"); ax.set_xlabel("predicted quality")
eda.savefig(fig, "p6_optcuts.png"); plt.show()
print("regression+round QWK %.3f  ->  +optimized rounder QWK %.3f" % (results["regression+round"].QWK, results["regression+optimized"].QWK))"""),

        md(
"""### 3. Ordinal regression — Frank & Hall

The proper ordinal approach: instead of one 7-way problem, solve **K−1 = 6 binary problems** — "is
quality > 3?", "> 4?", … "> 8?". Each is a clean, well-balanced classifier. Their cumulative
probabilities reconstruct the full grade distribution P(quality = g), from which we take the
**expected grade**. This uses the order *without* assuming the grades are evenly spaced — the
theoretically "right" framing."""),
        co("""grades = M.LABELS
prob_gt = {k: LGBMClassifier(**LG).fit(Xtr, (ytr > k).astype(int)).predict_proba(Xte)[:, 1] for k in grades[:-1]}
pmf = np.zeros((len(Xte), len(grades))); prev = np.ones(len(Xte))
for i, g in enumerate(grades):
    p_gt = prob_gt.get(g, np.zeros(len(Xte)))   # P(y>g); 0 for the top grade
    pmf[:, i] = prev - p_gt; prev = p_gt
exp_grade = (pmf * grades).sum(1)
results["ordinal (Frank-Hall)"] = M.report(yte, exp_grade, "ordinal (Frank-Hall)")
print("Frank-Hall ordinal QWK %.3f" % results["ordinal (Frank-Hall)"].QWK)
print("example reconstructed P(quality=g) for one wine:", dict(zip(grades, pmf[0].round(2))))"""),

        md(
"""### 4. Head-to-head

All four framings, same model, same split. **Regression with an optimized rounder wins** — and the
spread from worst (classification) to best is ~0.07 QWK, *larger than the gap between LightGBM and
logistic in Part 5*. Framing is a first-class modelling decision, not an afterthought."""),
        co("""board = pd.DataFrame(results.values())
print(board.to_string())
fig, ax = plt.subplots(figsize=(9, 4))
board["QWK"].plot.bar(ax=ax, color=["#bbb","steelblue","seagreen","darkorange"], rot=15)
ax.set_ylabel("test QWK"); ax.set_ylim(0.4, 0.6); ax.set_title("Framing the ordinal target: regression+optimized rounder wins")
for i, v in enumerate(board["QWK"]): ax.text(i, v+.003, f"{v:.3f}", ha="center")
eda.savefig(fig, "p6_framings.png"); plt.show()
print("\\nbest framing: %s (QWK %.3f)" % (board.QWK.idxmax(), board.QWK.max()))"""),

        md(
"""### Takeaways

- **Regression beats classification** for this ordinal target (QWK 0.53 vs 0.49) — predicting a
  *number* respects that far misses are worse, which classification ignores.
- The **optimized rounder** adds more (→ ~0.56) by learning **asymmetric cut points** that account for
  the imbalance (you should rarely commit to a 9).
- **Frank–Hall ordinal** (K−1 binary models) also beats classification and is the most principled
  framing — it encodes order without assuming equal spacing.
- **Framing > model**: the QWK spread across framings exceeds the spread across algorithms. Decide how
  to *pose* an ordinal problem before tuning the learner.

**Next — Part 7 (Class imbalance):** the tails are still being ignored. Can class weights or
resampling (SMOTE) rescue the rare grades — and what does it cost the overall QWK?"""),
    ]
    build(cells, "06_regression_ordinal.ipynb", "# 06 · Wine Quality — Regression & Ordinal Framing (regress, optimized rounding, Frank–Hall)")


# ===================================================================== Notebook 7
def notebook_7():
    md, co = new_markdown_cell, new_code_cell
    cells = [
        md(
"""## Part 7 — The Class-Imbalance Problem

Every model so far has **ignored the tails** — Part 5 measured ~4% recall on the rare grades (3, 4, 8,
9). That's rational under a plain loss: with a handful of examples, never predicting them is the safe
bet. But if we *care* about spotting an exceptional or a faulty wine, we must intervene. Three classic
remedies, and an honest accounting of what each buys and costs:

- **Class weights** — tell the loss to count a rare-grade mistake more heavily.
- **Random oversampling** — duplicate minority rows until classes are balanced.
- **SMOTE** — synthesise *new* minority examples by interpolating between real ones.

The question isn't "do they raise recall" (they do) but **"what do they cost, and is there a
ceiling?"**"""),
        co(SETUP + """
from src import modeling as M
from lightgbm import LGBMClassifier
from sklearn.metrics import recall_score, f1_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
raw = data.clean(); Xtr, Xte, ytr, yte = M.split(raw)
LG = dict(n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=0, verbose=-1, n_jobs=-1)
print("train counts:", ytr.value_counts().sort_index().to_dict())
print("-> grade 9 has just", (ytr==9).sum(), "training examples. Remember this number.")"""),

        md(
"""### 1. Balancing the training set

SMOTE and random oversampling both rebalance every grade to the size of the majority class (here
~1,742 each). The difference: oversampling **copies** rare wines (risking overfitting to those exact
points), while SMOTE **interpolates** new synthetic wines between neighbours (more diverse, but it
invents chemistry that never existed — and can't conjure signal from 4 grade-9 points)."""),
        co("""kmin = ytr.value_counts().min() - 1
Xs, ys = SMOTE(random_state=0, k_neighbors=min(3, max(1, kmin))).fit_resample(Xtr, ytr)
Xo, yo = RandomOverSampler(random_state=0).fit_resample(Xtr, ytr)
print("after SMOTE:           ", pd.Series(ys).value_counts().sort_index().to_dict())
print("after random oversample:", pd.Series(yo).value_counts().sort_index().to_dict())"""),

        md(
"""### 2. The scoreboard — four metrics, four strategies

`rare_recall` averages recall over grades {3,4,8,9}. The pattern here is encouraging *and* sobering:
every remedy lifts rare-recall, macro-F1 **and** QWK — so balancing is a net win under the right
metric — but it shaves a little **accuracy**, and the rare-recall numbers stay low in absolute terms."""),
        co("""def evals(yp, name):
    return pd.Series({"QWK": M.qwk(yte, yp),
                      "macro_F1": f1_score(yte, yp, average="macro", zero_division=0),
                      "rare_recall": recall_score(yte, yp, labels=[3,4,8,9], average="macro", zero_division=0),
                      "accuracy": (yp == yte.values).mean()}, name=name).round(3)
preds = {}
preds["none"]        = LGBMClassifier(**LG).fit(Xtr, ytr).predict(Xte)
preds["class_weight"]= LGBMClassifier(class_weight="balanced", **LG).fit(Xtr, ytr).predict(Xte)
preds["random_over"] = LGBMClassifier(**LG).fit(Xo, yo).predict(Xte)
preds["SMOTE"]       = LGBMClassifier(**LG).fit(Xs, ys).predict(Xte)
board = pd.DataFrame([evals(p, n) for n, p in preds.items()])
print(board.to_string())
fig, ax = plt.subplots(figsize=(10, 4))
board[["QWK","macro_F1","rare_recall","accuracy"]].plot.bar(ax=ax, rot=0, colormap="Set2")
ax.set_title("Imbalance remedies: tails & QWK up, accuracy slightly down"); ax.legend(loc="upper right", ncol=4, fontsize=8)
eda.savefig(fig, "p7_scoreboard.png"); plt.show()"""),

        md(
"""### 3. *Where* the gain happens — per-grade recall

Averages hide the mechanism. Comparing baseline vs SMOTE recall **grade by grade** shows the lift is
concentrated in the **tails** (grades 4 and 8 improve most; the common 5/6 dip slightly as the model
stops over-betting on them). But grades **3 and 9 stay near zero** — no amount of resampling fixes
*4 training examples*. This is the **ceiling**: imbalance methods redistribute attention, they don't
create information."""),
        co("""g = list(range(3,10))
rec_base = recall_score(yte, preds["none"],  labels=g, average=None, zero_division=0)
rec_smote= recall_score(yte, preds["SMOTE"], labels=g, average=None, zero_division=0)
comp = pd.DataFrame({"baseline": rec_base, "SMOTE": rec_smote}, index=g)
fig, ax = plt.subplots(figsize=(9, 4))
comp.plot.bar(ax=ax, color=["#bbb","seagreen"], rot=0); ax.set_xlabel("quality grade"); ax.set_ylabel("recall")
ax.set_title("SMOTE lifts the tails (4, 8) — but grades 3 & 9 (n≤22) stay stuck")
eda.savefig(fig, "p7_pergrade.png"); plt.show()
print(comp.round(2).to_string())"""),

        md(
"""### 4. The honest verdict

So is balancing worth it? **It depends what you optimise for.** If the rare wines matter (flagging
faulty or exceptional bottles), SMOTE's 3× rare-recall and higher QWK justify the ~1.5-point accuracy
hit. If only overall correctness matters, the gain is marginal. Two caveats to carry forward:
*(1)* the absolute rare-recall is still poor — collect more extreme-grade data, don't expect SMOTE to
rescue it; *(2)* resampling **distorts the class proportions**, so the model's predicted
*probabilities* no longer match reality — exactly the problem Part 8 (calibration) examines."""),
        co("""best = board.QWK.idxmax()
print("best QWK strategy: %s (%.3f vs none %.3f)" % (board.loc[best].name, board.QWK.max(), board.loc["none","QWK"]))
print("rare-recall: none %.3f -> SMOTE %.3f  (%.1fx)  | accuracy cost: %.3f" %
      (board.loc["none","rare_recall"], board.loc["SMOTE","rare_recall"],
       board.loc["SMOTE","rare_recall"]/max(board.loc["none","rare_recall"],1e-9),
       board.loc["none","accuracy"] - board.loc["SMOTE","accuracy"]))"""),

        md(
"""### Takeaways

- The rare-grade collapse is a **loss-incentive** problem; class weights / oversampling / SMOTE all
  push the model to attend to minorities.
- Here every remedy **improves QWK, macro-F1 and rare-recall** at a small accuracy cost — under QWK,
  balancing is a net win (SMOTE best). The "no free lunch" shows up as **accuracy vs the tails**.
- There's a hard **ceiling**: grades 3 and 9 (n ≤ 22 in train) stay near 0% recall — resampling
  redistributes attention, it can't invent signal. *More data* is the only real fix for those.
- Resampling **distorts predicted probabilities** → calibrate before trusting them (Part 8).

**Next — Part 8 (Probability calibration):** when the model says "70% chance this is a 6," is it right
70% of the time? Reliability curves, and fixing miscalibration with isotonic / Platt scaling."""),
    ]
    build(cells, "07_imbalance.ipynb", "# 07 · Wine Quality — The Class-Imbalance Problem (class weights, SMOTE, the tradeoff)")


# ===================================================================== Notebook 8
def notebook_8():
    md, co = new_markdown_cell, new_code_cell
    cells = [
        md(
"""## Part 8 — Probability Calibration

A model can rank wines well yet still **lie about its confidence**. If it says *"70% chance this is a
good wine"* for a batch of wines, roughly 70% of them should actually be good — otherwise any
*decision* built on that probability (flag for review, set a price, trigger a warning) is wrong, even
when the *ranking* is fine. This part measures and fixes that.

We use a clean **binary** framing — "good wine" = quality ≥ 7 (base rate ≈ 19%) — because calibration
is clearest with one probability per wine. Two tools: the **reliability curve** (predicted vs actual
frequency) and two summary numbers, **Brier score** (mean squared probability error, lower = better)
and **ECE** (expected calibration error, the average gap between confidence and reality)."""),
        co(SETUP + """
from src import modeling as M
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
d = data.dedup(data.clean()); X, yq = M.prep(d)
y = (yq >= 7).astype(int)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, stratify=y, random_state=0)
LG = dict(n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=0, verbose=-1, n_jobs=-1)
def ece(yt, p, bins=10):
    edges = np.linspace(0, 1, bins+1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i+1])
        if m.sum(): e += abs(yt[m].mean() - p[m].mean()) * m.sum()/len(p)
    return e
print("good-wine base rate: %.3f" % yte.mean())"""),

        md(
"""### 1. Is the raw model calibrated?

Gradient-boosted trees optimise a ranking-friendly loss, not honest probabilities, so they're often
**over-confident** at the extremes. The reliability curve below plots, for each confidence bin, the
*actual* fraction of good wines. Departures from the diagonal are miscalibration; the ECE summarises
the average gap."""),
        co("""base = LGBMClassifier(**LG).fit(Xtr, ytr); pb = base.predict_proba(Xte)[:, 1]
fx, fy = calibration_curve(yte, pb, n_bins=10, strategy="quantile")
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0,1],[0,1],"k--",lw=1,label="perfect")
ax.plot(fy, fx, "o-", color="firebrick", label=f"uncalibrated (ECE {ece(yte.values,pb):.3f})")
ax.set_xlabel("predicted P(good)"); ax.set_ylabel("observed fraction good"); ax.set_title("Reliability curve — raw LightGBM"); ax.legend()
eda.savefig(fig, "p8_reliability_raw.png"); plt.show()
print("uncalibrated:  Brier %.4f  ECE %.3f" % (brier_score_loss(yte, pb), ece(yte.values, pb)))"""),

        md(
"""### 2. Fixing it — isotonic vs Platt

Two post-hoc recalibrators, each fit by cross-validation on the training data:
- **Platt / sigmoid scaling** — fit a logistic curve mapping raw scores → probabilities (good when the
  distortion is a smooth S-shape; few parameters, robust on small data).
- **Isotonic regression** — fit *any* monotonic mapping (more flexible, needs more data).

Isotonic wins clearly here — it bends the curve onto the diagonal and slashes ECE from 0.088 to
~0.02."""),
        co("""iso = CalibratedClassifierCV(LGBMClassifier(**LG), method="isotonic", cv=5).fit(Xtr, ytr); pi = iso.predict_proba(Xte)[:, 1]
sig = CalibratedClassifierCV(LGBMClassifier(**LG), method="sigmoid",  cv=5).fit(Xtr, ytr); ps = sig.predict_proba(Xte)[:, 1]
fig, ax = plt.subplots(figsize=(6, 6)); ax.plot([0,1],[0,1],"k--",lw=1,label="perfect")
for p, c, n in [(pb,"firebrick","uncalibrated"),(ps,"darkorange","Platt/sigmoid"),(pi,"seagreen","isotonic")]:
    fx, fy = calibration_curve(yte, p, n_bins=10, strategy="quantile"); ax.plot(fy, fx, "o-", color=c, label=f"{n} (ECE {ece(yte.values,p):.3f})")
ax.set_xlabel("predicted P(good)"); ax.set_ylabel("observed fraction good"); ax.set_title("Calibration restores honesty"); ax.legend()
eda.savefig(fig, "p8_reliability_fixed.png"); plt.show()
tab = pd.DataFrame({"Brier":[brier_score_loss(yte,p) for p in (pb,ps,pi)], "ECE":[ece(yte.values,p) for p in (pb,ps,pi)]},
                   index=["uncalibrated","Platt","isotonic"]).round(4)
print(tab.to_string())"""),

        md(
"""### 3. The SMOTE distortion (Part 7's loose end)

Part 7 warned that resampling corrupts probabilities. Proof: a model trained on **SMOTE-balanced**
data has seen an artificial 50/50 world, so it **over-states** P(good) relative to the true 19% base
rate — its mean prediction drifts up and its Brier score *worsens*. The lesson is concrete: **if you
resample to fix recall, you must recalibrate** (on data with the real class balance) before trusting
any probability."""),
        co("""Xs, ys = SMOTE(random_state=0).fit_resample(Xtr, ytr)
sm = LGBMClassifier(**LG).fit(Xs, ys); psm = sm.predict_proba(Xte)[:, 1]
print("true base rate            : %.3f" % yte.mean())
print("raw model   mean P(good)  : %.3f | Brier %.4f" % (pb.mean(),  brier_score_loss(yte, pb)))
print("SMOTE model mean P(good)  : %.3f | Brier %.4f  (inflated & worse)" % (psm.mean(), brier_score_loss(yte, psm)))"""),

        md(
"""### 4. Why it matters — a decision

Suppose we **flag wines with P(good) > 0.5** for a premium tasting panel. With *uncalibrated*
probabilities that threshold means something different than we think — we'd flag the wrong number.
Calibrated probabilities make the threshold *mean* what it says, so cost/benefit math (panel time vs
missed good wines) is finally valid. Calibration is what turns a *score* into a *probability you can
act on*."""),
        co("""for name, p in [("uncalibrated", pb), ("isotonic", pi)]:
    flagged = p > 0.5; hit = yte.values[flagged].mean() if flagged.sum() else 0
    print("%-13s: flag %4d wines, of which %.0f%% are truly good (precision)" % (name, flagged.sum(), 100*hit))"""),

        md(
"""### Takeaways

- Raw gradient-boosted probabilities are **mildly miscalibrated** (ECE 0.088); ranking ≠ honest
  probability.
- **Isotonic** recalibration fixes it best here (ECE → ~0.02, Brier down); **Platt** helps less because
  the distortion isn't a clean sigmoid.
- **SMOTE inflates probabilities** (mean P(good) above the true base rate, worse Brier) — resampling
  and calibration must go together.
- Calibration is what makes a **probability threshold** (and the cost/benefit decision behind it)
  trustworthy.

**Next — Part 9 (Interpretability):** open the box — gain vs permutation vs **SHAP** importance,
and partial-dependence curves showing *how* alcohol, volatile acidity and the rest move quality."""),
    ]
    build(cells, "08_calibration.ipynb", "# 08 · Wine Quality — Probability Calibration (reliability, isotonic vs Platt, the SMOTE distortion)")


# ===================================================================== Notebook 9
def notebook_9():
    md, co = new_markdown_cell, new_code_cell
    cells = [
        md(
"""## Part 9 — Interpretability

A model that predicts well but can't be explained is a hard sell to a winemaker. This part opens the
box on the LightGBM quality model with three lenses:

- **Importance — three ways**: gain, permutation, and SHAP. They *disagree*, and *why* they disagree
  is itself a lesson about correlated features.
- **SHAP global (beeswarm)**: every wine, every feature, showing direction and magnitude.
- **Dependence**: *how* a feature moves quality across its range (not just "how much").
- **Local**: why *this one wine* got the score it did.

We interpret the **regression** model (Part 6) so SHAP values are in *quality points* — e.g. "high
alcohol adds +0.3 to predicted quality" — which is far more legible than log-odds."""),
        co(SETUP + """
from src import modeling as M
from lightgbm import LGBMRegressor
from sklearn.inspection import permutation_importance
import shap
raw = data.clean(); Xtr, Xte, ytr, yte = M.split(raw)
reg = LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=0, verbose=-1, n_jobs=-1).fit(Xtr, ytr)
explainer = shap.TreeExplainer(reg); sv = explainer.shap_values(Xte)
print("test QWK %.3f | SHAP base value (mean prediction) %.2f" % (M.qwk(yte, reg.predict(Xte)), explainer.expected_value))"""),

        md(
"""### 1. Three importances that *disagree* — and why

- **Gain**: total loss reduction from a feature's splits. Fast, but **biased when features are
  correlated** — it scatters credit among look-alikes.
- **Permutation**: shuffle a feature, measure how much MAE worsens. Model-agnostic, honest about
  *predictive* value — but splits credit between correlated features too.
- **SHAP**: game-theoretic, *consistent* attribution that sums to each prediction.

Watch **`alcohol_density`**: gain ranks it nearly **last**, yet permutation and SHAP rank it **first**.
Because it's derived from alcohol and density, the tree spreads its splits across the trio and *gain*
under-credits the engineered feature — while SHAP/permutation see its true contribution. **Moral: don't
rank features by gain alone when they're correlated.**"""),
        co("""gain = pd.Series(reg.feature_importances_, index=Xtr.columns)
perm = pd.Series(permutation_importance(reg, Xte, yte, n_repeats=5, random_state=0,
                                        scoring="neg_mean_absolute_error").importances_mean, index=Xte.columns)
shap_imp = pd.Series(np.abs(sv).mean(0), index=Xte.columns)
comp = pd.DataFrame({"gain": gain, "permutation": perm, "SHAP": shap_imp})
norm = comp / comp.max()                       # scale each method to [0,1] to compare shapes
norm = norm.loc[shap_imp.sort_values().index]
fig, ax = plt.subplots(figsize=(10, 6)); norm.plot.barh(ax=ax, color=["#bbb","steelblue","seagreen"])
ax.set_title("Normalised importance — gain disagrees on alcohol_density"); ax.set_xlabel("relative importance")
eda.savefig(fig, "p9_importance_compare.png"); plt.show()
print("alcohol_density ranks -> gain #%d, permutation #%d, SHAP #%d (of %d)" %
      (gain.rank(ascending=False)["alcohol_density"], perm.rank(ascending=False)["alcohol_density"],
       shap_imp.rank(ascending=False)["alcohol_density"], len(gain)))"""),

        md(
"""### 2. SHAP beeswarm — the global picture

Each dot is one wine; position = that feature's SHAP push on its predicted quality; colour = the
feature's value (red high, blue low). Read it directly: **high `alcohol_density` (red) sits on the
right** (pushes quality *up*); **high `volatile_acidity` (red) sits on the left** (pushes quality
*down*). The whole EDA story, confirmed inside the model and quantified in quality points."""),
        co("""shap.summary_plot(sv, Xte, show=False, plot_size=(10, 6))
eda.savefig(plt.gcf(), "p9_beeswarm.png"); plt.show()"""),

        md(
"""### 3. Dependence — *how*, not just *how much*

A dependence plot traces a feature's SHAP value across its range. **`alcohol_density`** rises smoothly
(more "normalised strength" → higher quality, with diminishing returns at the top), while
**`volatile_acidity`** falls then plateaus (a little is fine; past a threshold the vinegar taint
dominates). The colour reveals interactions — the effect shifts with a second feature."""),
        co("""for feat, fname in [("alcohol_density", "p9_dep_alcohol.png"), ("volatile_acidity", "p9_dep_volatile.png")]:
    shap.dependence_plot(feat, sv, Xte, show=False)
    eda.savefig(plt.gcf(), fname); plt.show()"""),

        md(
"""### 4. Local — why *this* wine?

Global importance is the average story; a winemaker asks about **one bottle**. A SHAP waterfall starts
at the base value (mean prediction ≈ 5.8) and adds each feature's contribution to reach this wine's
score — a complete, additive, per-wine explanation."""),
        co("""i = int(np.argmax(reg.predict(Xte)))    # explain the highest-predicted wine
ex = shap.Explanation(values=sv[i], base_values=explainer.expected_value, data=Xte.iloc[i], feature_names=list(Xte.columns))
shap.plots.waterfall(ex, max_display=10, show=False)
eda.savefig(plt.gcf(), "p9_waterfall.png"); plt.show()
print("wine #%d: predicted %.2f (true %d) — top push: %s" %
      (i, reg.predict(Xte)[i], yte.iloc[i], Xte.columns[np.argmax(np.abs(sv[i]))]))"""),

        md(
"""### 5. Caveats — interpret responsibly

- SHAP attributions are **associational, not causal**: "high alcohol *predicts* higher quality" ≠
  "adding alcohol *makes* a wine better" (alcohol co-varies with ripeness, technique, …).
- With **correlated features** (our chemistry), credit-splitting is unavoidable — engineered ratios
  like `alcohol_density` can steal attribution from their parents.
- These explain the *model*, which is only as trustworthy as its data (taster bias, the duplicate
  rows, the rare-grade gaps).

### Takeaways

- **Importance method matters**: gain mis-ranks correlated features (alcohol_density last vs SHAP
  first). Prefer **SHAP / permutation**; use gain only as a rough guide.
- The model's drivers are **alcohol/alcohol-density (+), volatile acidity (−), sulphates (+)** — the
  EDA's signals, now quantified in quality points and validated globally *and* locally.
- Dependence plots show the *shape* (alcohol's diminishing returns; volatile acidity's threshold) —
  richer than a single correlation.
- Explanations are **associational**; don't read them as winemaking advice.

**Next — Part 10 (Capstone):** a contrast that ties the practice together — predict **red vs white**
from chemistry (near-perfect, unlike quality) — plus a synthesis of everything Parts 0–9 taught."""),
    ]
    build(cells, "09_interpretability.ipynb", "# 09 · Wine Quality — Interpretability (gain vs permutation vs SHAP, dependence, local)")


# ===================================================================== Notebook 10
def notebook_10():
    md, co = new_markdown_cell, new_code_cell
    cells = [
        md(
"""## Part 10 — Capstone: Red vs White, and a Synthesis

We close with a **contrast** that crystallises the whole practice. Using the *same chemistry features*,
we now predict a *different target* — is this wine **red or white**? Part 2 hinted it would be easy
(the types nearly separate in PCA). It is **near-perfect**. Holding the data fixed and swapping only
the target — from hard (`quality`, QWK ≈ 0.56) to trivial (`type`, AUC ≈ 1.0) — isolates the real
lesson of all ten notebooks: **difficulty lives in the target's relationship to the features, not in
the model.**"""),
        co(SETUP + """
from src import modeling as M
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
d = data.dedup(data.clean())
X = d[data.NUMERIC]; y = (d.wine_type == "white").astype(int)   # 1 = white
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, stratify=y, random_state=0)
print("predicting red(%d) vs white(%d) from 11 chemistry features" % ((y==0).sum(), (y==1).sum()))"""),

        md(
"""### 1. The near-perfect classifier

LightGBM separates red from white at **~99.5% accuracy / AUC ≈ 0.9996** — only a handful of wines are
misclassified, and those are the genuinely ambiguous bottles (a red with white-like chemistry). Recall
that `quality`, with the identical inputs, topped out around QWK 0.56."""),
        co("""m = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, random_state=0, verbose=-1, n_jobs=-1).fit(Xtr, ytr)
pred = m.predict(Xte); proba = m.predict_proba(Xte)[:, 1]
cm = confusion_matrix(yte, pred)
fig, ax = plt.subplots(figsize=(4.6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="BuPu", xticklabels=["red","white"], yticklabels=["red","white"], ax=ax)
ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Red vs white — almost no errors")
eda.savefig(fig, "p10_redwhite_cm.png"); plt.show()
print("accuracy %.4f | ROC-AUC %.5f | 5-fold acc %.4f | errors: %d/%d" %
      (accuracy_score(yte, pred), roc_auc_score(yte, proba), cross_val_score(m, X, y, cv=5).mean(), (pred != yte).sum(), len(yte)))"""),

        md(
"""### 2. What separates them — the chemistry of colour

The separators are exactly the markers Part 1 flagged: **total SO₂** and **chlorides** (whites are
preserved with more sulfur; reds are saltier), then **density** and **sulphates**. Unlike quality,
red/white is an *objective, chemically-determined* label — so a few features nail it."""),
        co("""imp = pd.Series(m.feature_importances_, index=X.columns).sort_values()
fig, ax = plt.subplots(figsize=(8, 5)); imp.plot.barh(ax=ax, color="purple")
ax.set_title("What separates red from white"); fig.tight_layout()
eda.savefig(fig, "p10_redwhite_imp.png"); plt.show()
print("top separators:", list(imp.tail(4).index[::-1]))"""),

        md(
"""### 3. Why one target is easy and the other is hard

Same wines, same features — opposite difficulty. The reasons are the spine of the whole practice:

| | red vs white | quality |
|---|---|---|
| **target nature** | objective (a chemical fact) | subjective (a taster's score) |
| **separability** | clean clusters (Part 2 PCA, PC1 gap 3.3) | diffuse, nonlinear (lives on PC2, weakly) |
| **signal in features** | high — colour *is* chemistry | modest — taste ≠ just chemistry |
| **class balance** | ~3:1, easy | 7 grades, extreme imbalance (q9: n=5) |
| **best result** | **AUC ≈ 1.00** | **QWK ≈ 0.56** |

No amount of model tuning would close that gap, because the gap is about **how much the target is
*determined* by the inputs** — the ceiling is set by the data, not the algorithm. Recognising which
kind of problem you're in is the most valuable judgement an analyst brings."""),

        md(
"""### 4. Synthesis — the arc of the practice

A cross-sectional counterpart to the three time-series studies, start to finish:

- **0–1 Foundations** — combined red+white (6,497 wines); found the **18% duplicate-row leakage trap**
  and the **imbalanced ordinal** target (q9: 5 wines); alcohol the top driver, serious VIF.
- **2 Structure** — PCA/t-SNE: the biggest axis of variation is *type*, not quality; quality is
  **nonlinear**.
- **3 Features** — log transforms help linear models only; the engineered **alcohol/density** ratio
  beats every raw feature.
- **4 Framework** — *demonstrated* leakage (1-NN +14 pts), stratified split, and the metric that
  doesn't lie under imbalance/ordinality: **QWK**.
- **5–6 Models & framing** — trees beat linear; then **framing the ordinal target as regression +
  optimized rounding** beat classification by more than the model choice did (QWK 0.49 → 0.56).
- **7 Imbalance** — SMOTE/weights trade ~1.5 pts accuracy for 3× rare-recall, but a **hard ceiling**
  remains (no signal to mine from 5 examples).
- **8 Calibration** — isotonic made the probabilities honest (ECE 0.088 → 0.02); SMOTE breaks
  calibration.
- **9 Interpretability** — SHAP > gain when features correlate; alcohol(+)/volatile acidity(−)
  confirmed in quality points.
- **10 Capstone** — the red/white contrast (AUC ≈ 1.0) showing difficulty is a property of the
  *target*."""),
        co("""summary = pd.DataFrame([
    ("best quality model", "LightGBM regression + optimized rounder", "QWK ≈ 0.56"),
    ("red/white classifier", "LightGBM on 11 chemistry features",      "AUC ≈ 1.00"),
    ("top quality driver",  "alcohol / alcohol-density (SHAP #1)",      "+"),
    ("biggest methodology win", "framing the ordinal target",          "+0.07 QWK vs classify"),
    ("the hard limit",      "rare grades (q3, q9: n≤30)",               "~0% recall, needs more data"),
], columns=["item", "what", "result"])
print(summary.to_string(index=False))"""),

        md(
"""### 5. What transfers — the methodology, not the wine

The wine is incidental; these habits move to any tabular project:

1. **Hunt for leakage first** (duplicates, target-derived features) — a clean split is worth more than
   a fancy model.
2. **Pick the metric before the model**, and make it match the target (ordinal → QWK; imbalanced →
   not accuracy).
3. **How you *frame* the target** can beat algorithm choice — try regression *and* classification *and*
   ordinal.
4. **Imbalance has a floor**: methods redistribute attention; only data adds information.
5. **Calibrate** before acting on probabilities; **resampling breaks calibration**.
6. **Explain with SHAP, not gain**, when features correlate — and remember it's *associational*.
7. **Know which problem you're in** — some targets are near-deterministic (red/white), others are
   noisy and capped (quality). Set expectations from the data, not hope.

### The wine-quality practice is complete (Parts 0–10).

See **`docs/`** for the beginner-friendly concept glossary and line-by-line code walkthrough."""),
    ]
    build(cells, "10_capstone.ipynb", "# 10 · Wine Quality — Capstone: Red vs White & Synthesis")


if __name__ == "__main__":
    import sys
    all_nbs = {"0": notebook_0, "1": notebook_1, "2": notebook_2, "3": notebook_3, "4": notebook_4, "5": notebook_5,
               "6": notebook_6, "7": notebook_7, "8": notebook_8, "9": notebook_9, "10": notebook_10}
    for k in (sys.argv[1:] or sorted(all_nbs, key=int)):
        all_nbs[k]()
    print("done.")

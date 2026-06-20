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


if __name__ == "__main__":
    import sys
    all_nbs = {"0": notebook_0, "1": notebook_1, "2": notebook_2}
    for k in (sys.argv[1:] or sorted(all_nbs)):
        all_nbs[k]()
    print("done.")

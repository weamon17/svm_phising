"""Build notebooks/02_experiments.ipynb: every SVM experiment, run cell by cell (English)."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(dedent(text).strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(dedent(text).strip()))


md(r"""
# PHISHING HOSTNAME DETECTION WITH SVM
## Part 2 · Experiments

This notebook runs every experiment directly in its cells, with plots and commentary.

**How to run:** `Run All`. The *Configuration* cell has a `QUICK` switch:

| Mode | Use it for | Estimated time |
|---|---|---|
| `QUICK = True` | Smoke test, checking the notebook | a few minutes |
| `QUICK = False` | Final numbers for the report | about 1 hour |

Each run saves its results to its own folder `outputs/experiments/<timestamp>-quick|full/`
(CSV files + 200-dpi PNG figures); a later run never overwrites an earlier one.

**Evaluation design**
- Train/test split **grouped by registered domain** (20% of registered domains for test):
  the model must recognise phishing on platforms it has never seen.
- Hyperparameters are selected with **k-fold cross-validation, also grouped by registered
  domain**, on the training set only. The test set is used once, for final scoring, and never
  for any selection.
- `StandardScaler` sits inside the pipeline, so each fold is standardised with its own statistics.
- The positive class is **phishing**. The main metric is F1, together with precision, recall,
  ROC-AUC and PR-AUC.

### Contents
1. Configuration
2. Data and split
3. Shared helpers
4. Hyperparameter tuning and kernel comparison (full-hostname features)
5. Feature transformation for skewed counts (log1p / binarize)
6. Subdomain ablation
7. LinearSVC on the full training set
8. Training time versus sample size n
9. Error analysis and borderline support vectors
10. Conclusions
""")

md(r"## 1 · Configuration")

code(r"""
QUICK = True      # True: smoke test in a few minutes · False: full numbers for the report (~1 hour)
SEED = 42

if QUICK:
    N_KERNEL, CV_FOLDS = 2_000, 3
    SIZES, SMO_MAX_N = [500, 1_000, 2_000, 4_000], 4_000
    C_SMALL = [0.1, 1, 10]
    RBF_GRID = {"svc__C": [0.1, 1, 10], "svc__gamma": [0.03, 0.3, 3]}
else:
    N_KERNEL, CV_FOLDS = 8_000, 5
    SIZES, SMO_MAX_N = [500, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000], 8_000
    C_SMALL = [0.01, 0.1, 1, 10]
    RBF_GRID = {"svc__C": [0.1, 0.3, 1, 3, 10, 30, 100],
                "svc__gamma": [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]}

KERNELS = {   # fixed parameters of each kernel + the grid to search
    "linear": (dict(kernel="linear"), {"svc__C": C_SMALL}),
    "poly2": (dict(kernel="poly", degree=2, gamma="scale", coef0=1.0), {"svc__C": C_SMALL}),
    "poly3": (dict(kernel="poly", degree=3, gamma="scale", coef0=1.0), {"svc__C": C_SMALL}),
    "rbf": (dict(kernel="rbf"), RBF_GRID),
}
LINEAR_FULL_C = [0.001, 0.01, 0.1, 1, 10]
""")

code(r"""
import json, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "data.py").exists())
sys.path.insert(0, str(ROOT))
from src.data import PROCESSED_CSV
from src.features import extract_features
from src.preprocessing import VARIANTS, compare_preprocessing, one_standard_error_choice
from src.smo import SMOSVC
from src.splits import group_split, stratified_sample

RUN = ROOT / "outputs" / "experiments" / f"{datetime.now():%Y%m%d-%H%M%S}-{'quick' if QUICK else 'full'}"
RUN.mkdir(parents=True)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})
pd.set_option("display.max_colwidth", 70)

def save(fig, name):
    fig.savefig(RUN / f"{name}.png", dpi=200, bbox_inches="tight")

def log(message):
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)

print(f"Mode: {'QUICK (smoke test)' if QUICK else 'FULL (report)'} · results saved to {RUN.relative_to(ROOT).as_posix()}")
""")

md(r"""
## 2 · Data and split

Uses `data/processed/hosts.csv` from Part 1. If it is missing, run
`python -m scripts.make_clean_data` first.
""")

code(r"""
assert PROCESSED_CSV.exists(), "hosts.csv not found: run `python -m scripts.make_clean_data` first."
hosts = pd.read_csv(PROCESSED_CSV, keep_default_na=False)
y = hosts["phishing"]

train_all, test = group_split(hosts, seed=SEED)
train_k = stratified_sample(train_all, y, N_KERNEL, SEED)    # subsample for kernel SVMs
y_test = y.loc[test].to_numpy()
shared = set(hosts.loc[train_all, "registered_domain"]) & set(hosts.loc[test, "registered_domain"])
assert not shared, "leakage: a registered domain appears in both train and test"

t0 = time.perf_counter()
features = {scope: extract_features(hosts, scope) for scope in ("host", "registered")}
log(f"Feature extraction finished in {time.perf_counter() - t0:.1f} s")

pd.DataFrame({
    "Hostnames": [f"{len(idx):,}" for idx in (train_all, train_k, test)],
    "% phishing": [f"{y.loc[idx].mean():.1%}" for idx in (train_all, train_k, test)],
    "Registered domains": [f"{hosts.loc[idx, 'registered_domain'].nunique():,}" for idx in (train_all, train_k, test)],
}, index=["Train (full)", "Train for kernel SVMs (subsample)", "Test"])
""")

code(r"""
display(Markdown(f"**Scope `host` features** ({features['host'].shape[1]} columns) and "
                 f"**scope `registered` features** ({features['registered'].shape[1]} columns, subdomain not visible):"))
features["host"].loc[train_k].describe().T[["mean", "std", "min", "max"]].round(3)
""")

md(r"""
## 3 · Shared helpers

- `tune`: grid search with cross-validation grouped by registered domain, selected on mean F1.
- `evaluate_scope`: for one feature scope, tunes and scores the 4 kernels, then adds our
  from-scratch SMO (same best RBF hyperparameters), Logistic Regression and a
  majority-class baseline.
""")

code(r"""
def test_scores(decision, fit_seconds, n_train, n_sv=None):
    y_pred = (decision > 0).astype(int)
    return {"n_train": n_train, "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred), "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, decision), "pr_auc": average_precision_score(y_test, decision),
            "n_sv": n_sv, "fit_seconds": fit_seconds}

def timed_fit(model, X, y_):
    start = time.perf_counter()
    model.fit(X, y_)
    return time.perf_counter() - start

def tune(kernel, X, y_, groups):
    base, grid = KERNELS[kernel]
    search = GridSearchCV(make_pipeline(StandardScaler(), SVC(cache_size=1000, **base)), grid,
                          scoring={"f1": "f1", "roc_auc": "roc_auc"}, refit="f1", n_jobs=-1,
                          cv=StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED))
    search.fit(X, y_, groups=groups)
    best = {k.removeprefix("svc__"): v for k, v in search.best_params_.items()}
    return search, {**base, **best}, best

def evaluate_scope(scope):
    F = features[scope]
    X_k, X_test = F.loc[train_k].to_numpy(), F.loc[test].to_numpy()
    y_k, groups = y.loc[train_k].to_numpy(), hosts.loc[train_k, "registered_domain"].to_numpy()
    rows, best_params, grids = [], {}, {}
    for kernel in KERNELS:
        start = time.perf_counter()
        search, params, chosen = tune(kernel, X_k, y_k, groups)
        grids[kernel] = pd.DataFrame(search.cv_results_)
        grids[kernel].to_csv(RUN / f"grid_{scope}_{kernel}.csv", index=False)
        model = make_pipeline(StandardScaler(), SVC(cache_size=1000, **params))
        fit_s = timed_fit(model, X_k, y_k)
        rows.append({"scope": scope, "model": f"SVC {kernel}", "params": json.dumps(chosen),
                     "cv_f1": search.best_score_,
                     **test_scores(model.decision_function(X_test), fit_s, len(train_k), int(model[-1].n_support_.sum()))})
        best_params[kernel] = params
        log(f"[{scope}] {kernel:<6} {json.dumps(chosen):<28} CV F1 {search.best_score_:.3f} · "
            f"test F1 {rows[-1]['f1']:.3f} · {time.perf_counter() - start:.0f}s")

    rbf = best_params["rbf"]
    scaler = StandardScaler().fit(X_k)
    smo = SMOSVC(C=rbf["C"], kernel="rbf", gamma=rbf["gamma"])
    fit_s = timed_fit(smo, scaler.transform(X_k), y_k)
    rows.append({"scope": scope, "model": "SMO from scratch (rbf)", "params": json.dumps({"C": rbf["C"], "gamma": rbf["gamma"]}),
                 "cv_f1": np.nan, **test_scores(smo.decision_function(scaler.transform(X_test)), fit_s,
                                                len(train_k), len(smo.support_))})
    log(f"[{scope}] SMO from scratch: {smo.n_iter_:,} iterations, {fit_s:.1f}s · test F1 {rows[-1]['f1']:.3f}")

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    fit_s = timed_fit(lr, X_k, y_k)
    rows.append({"scope": scope, "model": "Logistic Regression", "params": "{}", "cv_f1": np.nan,
                 **test_scores(lr.decision_function(X_test), fit_s, len(train_k))})
    majority = DummyClassifier(strategy="most_frequent").fit(X_k, y_k).predict(X_test[:1])[0]
    rows.append({"scope": scope, "model": "Majority class", "params": "{}", "cv_f1": np.nan,
                 **test_scores(np.full(len(test), 1.0 if majority == 1 else -1.0), 0.0, len(train_k))})
    return pd.DataFrame(rows), best_params, grids
""")

md(r"""
## 4 · Hyperparameter tuning and kernel comparison (full-hostname features)

Each kernel is tuned separately on the **same** training subsample. The cell below is the
heaviest one in the notebook (FULL mode: about 25 minutes).

- **C** is the penalty for margin violations: a small C gives a wide margin and tolerates
  errors (risk of underfitting); a large C forces every point to be classified correctly
  (risk of overfitting).
- **γ** (RBF kernel $K(x,z) = \exp(-\gamma\lVert x - z\rVert^2)$) is the radius of influence of
  each support vector: a small γ gives a smooth boundary, a large γ a boundary that wraps
  around individual points.
""")

code(r"""
host_rows, best_host, grids_host = evaluate_scope("host")
""")

md(r"### 4.1 · F1 heatmap over C and γ (RBF kernel)")

code(r"""
table = grids_host["rbf"].pivot(index="param_svc__C", columns="param_svc__gamma", values="mean_test_f1")
values = table.to_numpy(dtype=float)
best_cell = np.unravel_index(np.nanargmax(values), values.shape)
fig, ax = plt.subplots(figsize=(7.5, 5))
im = ax.imshow(values, cmap="viridis", aspect="auto", origin="lower")
ax.set_xticks(range(len(table.columns)), [f"{g:g}" for g in table.columns])
ax.set_yticks(range(len(table.index)), [f"{c:g}" for c in table.index])
ax.set_xlabel("γ"); ax.set_ylabel("C"); ax.grid(False)
for (r, c), v in np.ndenumerate(values):
    ax.text(c, r, f"{v:.3f}", ha="center", va="center", fontsize=7,
            color="black" if v > np.nanmean(values) else "white")
ax.add_patch(plt.Rectangle((best_cell[1] - 0.5, best_cell[0] - 0.5), 1, 1, fill=False, ec="red", lw=2))
fig.colorbar(im, ax=ax, label=f"Mean F1 over {CV_FOLDS}-fold CV")
ax.set_title("RBF kernel: F1 over C and γ (red cell = best)")
save(fig, "01_rbf_grid"); plt.show()
display(Markdown(f"**Best:** C = {best_host['rbf']['C']}, γ = {best_host['rbf']['gamma']} · "
                 f"CV F1 across the grid ranges from {np.nanmin(values):.3f} to {np.nanmax(values):.3f}"))
""")

md(r"""
**Reading the plot:** along a row (fixed C), a γ that is too small makes the boundary almost
linear, while a γ that is too large lets each support vector "see" only itself, so the model
memorises the training set. The bright region in the middle is the bias–variance sweet spot.

### 4.2 · Kernel comparison and reference models

- **SMO from scratch** uses exactly the best RBF hyperparameters: it checks that the team's
  own solver reaches the same quality as the library on the real problem.
- **Logistic Regression**: the classic linear model, to answer "is the SVM actually better?".
- **Majority class**: the lowest baseline (always predicts "legitimate", so the phishing F1 = 0).
""")

code(r"""
cols = ["model", "params", "cv_f1", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "n_sv", "fit_seconds"]
display(host_rows[cols].set_index("model").round(3))

plot = host_rows.set_index("model").drop(index="Majority class")
fig, ax = plt.subplots(figsize=(8.5, 4))
x = np.arange(len(plot))
for k, (metric, color) in enumerate([("f1", "#c05621"), ("roc_auc", "#2b6cb0"), ("pr_auc", "#6b46c1")]):
    bars = ax.bar(x + (k - 1) * 0.27, plot[metric], width=0.27, color=color, label=metric.upper().replace("_", "-"))
    ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
ax.set_xticks(x, plot.index, rotation=20, ha="right")
ax.set_ylim(0.5, 1.0); ax.legend(ncol=3, loc="upper left")
ax.set_title("Test-set results (registered domains never seen in training)")
save(fig, "02_kernel_comparison"); plt.show()
""")

md(r"""
## 5 · Feature transformation for skewed counts (log1p / binarize)

Count features are strongly right-skewed (skewness 2.7–6.9) and `n_sensitive_words` is very
sparse (≈2.4% non-zero, σ ≈ 0.16). After plain standardisation a hostname with 3 sensitive
words gets z ≈ 18.7. In the RBF kernel $K(x,z) = \exp(-\gamma\lVert x - z\rVert^2)$ such a point is
far from **every** support vector, so $K \approx 0$, every term $\alpha_i y_i K(x_i, x)$ vanishes and
$f(x)$ collapses to the bias $b$: the model has no evidence exactly for the most suspicious hostnames.

Three variants are compared, each placed **inside** the pipeline (fitted per CV fold), with the
RBF grid re-tuned from scratch for each one:

| Variant | Transformation before StandardScaler |
|---|---|
| `standard` | none (current pipeline, used in every other section) |
| `log1p` | $x \mapsto \log(1 + x)$ on the 10 count features |
| `log1p+binarize` | $\log(1 + x)$ on counts, $\mathbb{1}[x > 0]$ on `n_sensitive_words` |

Besides the usual metrics, two columns target the problem directly:
- `keyword_recall` / `keyword_f1`: scored only on test hostnames containing at least one sensitive word;
- `bias_dominated_test`: share of test hostnames where $\sum_i |\alpha_i y_i K(x_i, x)| < |b|$, i.e. where the
  prediction is driven by the bias rather than by support vectors (RBF only).
""")

code(r"""
prep_results, prep_grids = compare_preprocessing(
    features["host"], y, hosts["registered_domain"], train_k, train_all, test,
    RBF_GRID, LINEAR_FULL_C, CV_FOLDS, SEED)
prep_results.to_csv(RUN / "preprocessing_results.csv", index=False)
show = ["preprocessing", "model", "cv_f1", "precision", "recall", "f1", "roc_auc", "pr_auc",
        "keyword_recall", "keyword_f1", "bias_dominated_test", "n_sv"]
display(prep_results[show].set_index(["model", "preprocessing"]).round(3))
display(prep_results[["model", "preprocessing", "params"]].set_index(["model", "preprocessing"]))
""")

code(r"""
rbf_prep = prep_results[prep_results["model"] == "SVC rbf"].set_index("preprocessing")
metrics = [("f1", "Test F1"), ("keyword_recall", "Recall on keyword hosts"), ("bias_dominated_test", "Bias-dominated share")]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, (metric, title) in zip(axes, metrics):
    bars = ax.bar(rbf_prep.index, rbf_prep[metric], color=["#a0aec0", "#2b6cb0", "#c05621"])
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_title(title); ax.set_ylim(0, max(1.0, rbf_prep[metric].max() * 1.15))
    ax.tick_params(axis="x", labelrotation=15)
fig.suptitle("RBF SVC: effect of transforming count features before standardisation", y=1.03)
save(fig, "07_preprocessing"); plt.show()
best_prep, threshold = one_standard_error_choice(prep_results, CV_FOLDS)
display(Markdown(
    f"**Selected (one-standard-error rule):** `{best_prep}`. Highest CV F1 is "
    f"{rbf_prep['cv_f1'].max():.3f} (`{rbf_prep['cv_f1'].idxmax()}`); any variant scoring at least {threshold:.3f} "
    f"(best minus one standard error of the {CV_FOLDS} folds) is statistically indistinguishable, so the simplest one is kept. "
    f"The choice uses CV only; the test columns are for analysis."))""")

code(r"""
display(rbf_prep[["cv_f1", "cv_f1_std"]].assign(
    standard_error=rbf_prep["cv_f1_std"] / np.sqrt(CV_FOLDS)).round(4))
""")

md(r"""
**Reading the results:** `log1p` compresses the long tail and binarising removes the extreme keyword
z-scores, but grid search re-tunes γ for each variant: a larger γ makes the kernel more local again, so
the bias-dominated share does not necessarily fall. Look at three things together: CV F1 (with its standard
error), the bias-dominated share, and the threshold-free metrics ROC-AUC / PR-AUC. The bias-dominated
hostnames are mostly long, digit-heavy, multi-label hostnames; on this dataset nearly all of them are
phishing, so the positive bias $b$ classifies them correctly by default, which is a property of how the
legitimate list was collected rather than learned evidence. The other sections keep the `standard`
pipeline so their numbers stay comparable with earlier runs.

## 6 · Subdomain ablation

The data audit showed that the legitimate list has almost **no subdomains** (2.9% versus
55.2% in the phishing class), partly because of how it was collected. Here every feature is
recomputed **on the registered domain only** (the model cannot see the subdomain at all),
hyperparameters are tuned again from scratch on **the same rows**, and the results are compared.
""")

code(r"""
reg_rows, best_reg, grids_reg = evaluate_scope("registered")
""")

code(r"""
both = pd.concat([host_rows, reg_rows])
ablation = (both.pivot_table(index="model", columns="scope", values="f1")
            .rename(columns={"host": "Full hostname", "registered": "Registered domain only"})
            .drop(index="Majority class"))
ablation["F1 change"] = ablation["Registered domain only"] - ablation["Full hostname"]
ablation = ablation.sort_values("Full hostname", ascending=False)
display(ablation.round(3))

fig, ax = plt.subplots(figsize=(8, 3.8))
pos = np.arange(len(ablation))
ax.barh(pos - 0.2, ablation["Full hostname"], height=0.4, color="#c05621", label="Full hostname")
ax.barh(pos + 0.2, ablation["Registered domain only"], height=0.4, color="#a0aec0", label="Registered domain only")
ax.set_yticks(pos, ablation.index); ax.invert_yaxis()
ax.set_xlabel("Test F1"); ax.set_xlim(0, 1); ax.legend(loc="lower right")
ax.set_title("How much does the model lose without the subdomain?")
save(fig, "03_subdomain_ablation"); plt.show()
""")

md(r"""
## 7 · LinearSVC on the full training set

A kernel SVM cannot be trained on all ~174 thousand hostnames (the kernel matrix would need
hundreds of GB). `LinearSVC` solves the primal problem directly in $w$, so it learns from the
whole training set in seconds. C is selected with cross-validation grouped by registered
domain on the **whole** training set.

With a linear kernel, $f(x) = w^\top x + b$. Features are standardised, so $|w_k|$ is
comparable across features: $w_k > 0$ pushes towards **phishing**, $w_k < 0$ towards **legitimate**.
""")

code(r"""
linear_rows, linear_models = [], {}
for scope, F in features.items():
    X_all, y_all = F.loc[train_all].to_numpy(), y.loc[train_all].to_numpy()
    search = GridSearchCV(make_pipeline(StandardScaler(), LinearSVC(dual=False)), {"linearsvc__C": LINEAR_FULL_C},
                          scoring="f1", n_jobs=-1, cv=StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED))
    search.fit(X_all, y_all, groups=hosts.loc[train_all, "registered_domain"].to_numpy())
    model = make_pipeline(StandardScaler(), LinearSVC(dual=False, C=search.best_params_["linearsvc__C"]))
    fit_s = timed_fit(model, X_all, y_all)
    linear_models[scope] = model
    linear_rows.append({"scope": scope, "model": "LinearSVC (full train)",
                        "params": json.dumps({"C": search.best_params_["linearsvc__C"]}), "cv_f1": search.best_score_,
                        **test_scores(model.decision_function(F.loc[test].to_numpy()), fit_s, len(train_all))})
    log(f"[{scope}] LinearSVC: C = {search.best_params_['linearsvc__C']} · test F1 {linear_rows[-1]['f1']:.3f} · fit {fit_s:.1f}s")
linear_rows = pd.DataFrame(linear_rows)
display(linear_rows[["scope"] + cols[1:-2] + ["fit_seconds"]].set_index("scope").round(3))
""")

code(r"""
weights = pd.Series(linear_models["host"][-1].coef_[0], index=features["host"].columns).sort_values()
weights.to_csv(RUN / "linear_full_weights.csv", header=["weight"])
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.barh(weights.index, weights, color=np.where(weights > 0, "#c05621", "#2b6cb0"))
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Weight w (standardised features)")
ax.set_title("LinearSVC: orange = pushes to phishing, blue = pushes to legitimate")
save(fig, "04_linear_weights"); plt.show()
""")

md(r"""
## 8 · Training time versus sample size n

A kernel SVM works with an n×n matrix, so its training time grows faster than linearly. We
time it with the best RBF hyperparameters from Section 4, doubling n at each step. The
**slope on the log–log plot** gives the growth order: 1 is linear, 2 is quadratic. The right
plot also answers: does *more data with a simpler model* beat *a more complex model with less data*?
""")

code(r"""
F, X_test = features["host"], features["host"].loc[test].to_numpy()
rbf, lin_C = best_host["rbf"], linear_models["host"][-1].C
timing = []
for n in SIZES:
    rows = stratified_sample(train_all, y, n, SEED)
    X_n, y_n = F.loc[rows].to_numpy(), y.loc[rows].to_numpy()
    candidates = {"SVC rbf (LIBSVM, C++)": make_pipeline(StandardScaler(), SVC(cache_size=1000, **rbf)),
                  "LinearSVC": make_pipeline(StandardScaler(), LinearSVC(dual=False, C=lin_C))}
    if n <= SMO_MAX_N:
        candidates["SMO from scratch (rbf, Python)"] = "smo"
    for name, model in candidates.items():
        if model == "smo":
            scaler = StandardScaler().fit(X_n)
            smo = SMOSVC(C=rbf["C"], kernel="rbf", gamma=rbf["gamma"])
            fit_s = timed_fit(smo, scaler.transform(X_n), y_n)
            decision, n_sv = smo.decision_function(scaler.transform(X_test)), len(smo.support_)
        else:
            fit_s = timed_fit(model, X_n, y_n)
            decision = model.decision_function(X_test)
            n_sv = int(model[-1].n_support_.sum()) if hasattr(model[-1], "n_support_") else None
        timing.append({"model": name, "n": n, "fit_seconds": fit_s, "n_sv": n_sv,
                       "test_f1": f1_score(y_test, (decision > 0).astype(int))})
        log(f"n = {n:>6,}  {name:<30} fit {fit_s:7.2f}s · test F1 {timing[-1]['test_f1']:.3f}")
full_row = linear_rows.set_index("scope").loc["host"]
timing.append({"model": "LinearSVC", "n": len(train_all), "fit_seconds": full_row["fit_seconds"],
               "n_sv": None, "test_f1": full_row["f1"]})
timing = pd.DataFrame(timing)
timing.to_csv(RUN / "timing.csv", index=False)
""")

code(r"""
colors = {"SVC rbf (LIBSVM, C++)": "#2b6cb0", "SMO from scratch (rbf, Python)": "#c05621", "LinearSVC": "#2f855a"}
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
slopes = {}
for name, part in timing.groupby("model", sort=False):
    part = part.sort_values("n")
    axes[0].plot(part["n"], part["fit_seconds"], "o-", color=colors[name], label=name)
    axes[1].plot(part["n"], part["test_f1"], "o-", color=colors[name], label=name)
    fit_part = part[(part["n"] >= 1_000) & (part["n"] <= max(SIZES))]
    slopes[name] = np.polyfit(np.log(fit_part["n"]), np.log(fit_part["fit_seconds"]), 1)[0]
axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[1].set_xscale("log")
axes[0].set_xlabel("n (training hostnames)"); axes[0].set_ylabel("Training time (seconds)")
axes[1].set_xlabel("n (training hostnames)"); axes[1].set_ylabel("Test F1")
axes[0].set_title("Training time (log–log)"); axes[1].set_title("Does more data help?")
axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
save(fig, "05_time_vs_n"); plt.show()
display(pd.Series(slopes, name="Log–log slope").round(2).to_frame())
display(timing.pivot_table(index="n", columns="model", values=["fit_seconds", "test_f1"]).round(3))
""")

md(r"""
**Reading the slopes:** for small n, training time is dominated by fixed overhead, so the slope
is lower than the theory predicts. LinearSVC is close to linear in n in practice; the kernel
SVM grows at roughly quadratic order or more, because the number of support vectors also
grows with n.

## 9 · Error analysis and borderline support vectors

Model analysed: RBF SVC with the best hyperparameters, full-hostname features.
""")

code(r"""
X_k, y_k = F.loc[train_k].to_numpy(), y.loc[train_k].to_numpy()
best_model = make_pipeline(StandardScaler(), SVC(cache_size=1000, **rbf)).fit(X_k, y_k)
decision = best_model.decision_function(X_test)
cm = confusion_matrix(y_test, (decision > 0).astype(int))

fig, ax = plt.subplots(figsize=(4.2, 3.6))
ax.imshow(cm, cmap="Blues"); ax.grid(False)
labels = ["Legitimate", "Phishing"]
ax.set_xticks([0, 1], labels); ax.set_yticks([0, 1], labels)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
for (r, c), v in np.ndenumerate(cm):
    ax.text(c, r, f"{v:,}\n({v / cm[r].sum():.1%})", ha="center", va="center",
            color="white" if v > cm.max() / 2 else "black")
ax.set_title("Confusion matrix (RBF)")
save(fig, "06_confusion_matrix"); plt.show()
""")

code(r"""
analysis = hosts.loc[test, ["host", "phishing"]].assign(decision=decision)
missed = analysis[(analysis["phishing"] == 1) & (analysis["decision"] < 0)].nsmallest(10, "decision")
false_alarm = analysis[(analysis["phishing"] == 0) & (analysis["decision"] > 0)].nlargest(10, "decision")
display(Markdown("**Missed phishing the model is most confident is legitimate** (most negative f(x))"))
display(missed[["host", "decision"]].round(2))
display(Markdown("**Legitimate hostnames flagged as phishing** (most positive f(x))"))
display(false_alarm[["host", "decision"]].round(2))

sv_rows = train_k[best_model[-1].support_]
sv = hosts.loc[sv_rows, ["host", "phishing"]].assign(decision=best_model.decision_function(F.loc[sv_rows].to_numpy()))
borderline = sv.loc[sv["decision"].abs().sort_values().index[:12]]
borderline["label"] = borderline["phishing"].map({0: "legitimate", 1: "phishing"})
display(Markdown("**Borderline support vectors** (smallest |f(x)|: the hardest training cases)"))
display(borderline[["host", "label", "decision"]].round(3))
pd.concat([missed.assign(error="missed phishing"), false_alarm.assign(error="false alarm")]).to_csv(RUN / "errors.csv", index=False)
borderline.to_csv(RUN / "borderline_support_vectors.csv", index=False)
""")

md(r"## 10 · Conclusions")

code(r"""
results = pd.DataFrame([*host_rows.to_dict("records"), *reg_rows.to_dict("records"),
                        *linear_rows.to_dict("records")])
results.to_csv(RUN / "test_results.csv", index=False)
(RUN / "config.json").write_text(json.dumps({
    "quick": QUICK, "seed": SEED, "n_train_all": len(train_all), "n_kernel": len(train_k), "n_test": len(test),
    "cv_folds": CV_FOLDS, "sizes": SIZES, "smo_max_n": SMO_MAX_N,
    "grids": {k: g for k, (_, g) in KERNELS.items()}, "linear_full_C": LINEAR_FULL_C,
    "best_params": {"host": best_host, "registered": best_reg}, "confusion_matrix_rbf_host": cm.tolist(),
}, indent=2, default=str), encoding="utf-8")

h, r = host_rows.set_index("model"), reg_rows.set_index("model")
kernels = h.loc[[m for m in h.index if m.startswith("SVC ")]]
best_kernel = kernels["f1"].idxmax()
smo, svc_rbf, lin = h.loc["SMO from scratch (rbf)"], h.loc["SVC rbf"], linear_rows.set_index("scope").loc["host"]
largest = timing[timing["model"] == "SVC rbf (LIBSVM, C++)"].sort_values("n").iloc[-1]
display(Markdown(f'''
1. **Best kernel:** {best_kernel}, F1 = {kernels.loc[best_kernel, "f1"]:.3f}, ROC-AUC = {kernels.loc[best_kernel, "roc_auc"]:.3f}
   on registered domains never seen in training. The four kernels are only {kernels["f1"].max() - kernels["f1"].min():.3f} F1 apart;
   Logistic Regression reaches {h.loc["Logistic Regression", "f1"]:.3f}.
2. **SMO from scratch** with the same RBF hyperparameters: F1 = {smo["f1"]:.3f} (library {svc_rbf["f1"]:.3f}),
   {smo["n_sv"]:.0f} support vectors (library {svc_rbf["n_sv"]:.0f}); training time {smo["fit_seconds"]:.1f}s versus {svc_rbf["fit_seconds"]:.1f}s.
3. **Without the subdomain:** RBF F1 goes from {svc_rbf["f1"]:.3f} to {r.loc["SVC rbf", "f1"]:.3f} ({r.loc["SVC rbf", "f1"] - svc_rbf["f1"]:+.3f}).
4. **Scale:** LinearSVC learns all {int(lin["n_train"]):,} hostnames in {lin["fit_seconds"]:.1f}s, F1 = {lin["f1"]:.3f};
   RBF SVC on {int(largest["n"]):,} samples takes {largest["fit_seconds"]:.1f}s, F1 = {largest["test_f1"]:.3f}.
   Training-time slopes: RBF SVC ≈ {slopes["SVC rbf (LIBSVM, C++)"]:.1f}, SMO from scratch ≈ {slopes["SMO from scratch (rbf, Python)"]:.1f}, LinearSVC ≈ {slopes["LinearSVC"]:.1f}.
5. **Count transformation (one-standard-error rule):** `{best_prep}` is kept. RBF test F1:
   {", ".join(f"`{k}` {rbf_prep.loc[k, 'f1']:.3f}" for k in rbf_prep.index)};
   bias-dominated test share: {", ".join(f"`{k}` {rbf_prep.loc[k, 'bias_dominated_test']:.1%}" for k in rbf_prep.index)}.

Results saved to `{RUN.relative_to(ROOT).as_posix()}`.
'''))
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
path = ROOT / "notebooks" / "02_experiments.ipynb"
nbf.write(nb, path)
print(f"Wrote {path}")

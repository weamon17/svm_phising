"""Feature transformations before StandardScaler, and the experiment that compares them.

Count features are right-skewed (skewness 2.7–6.9) and `n_sensitive_words` is very sparse,
so after plain standardisation a host with 3 keywords gets z ≈ 18.7. In an RBF kernel
exp(−γ‖x − z‖²) such a point is far from every support vector, K ≈ 0 and f(x) collapses
to the bias b. The variants below compress (log1p) or binarise those columns first.
Column order is preserved so feature names still line up with model weights.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC, LinearSVC

from src.smo import kernel_matrix


COUNT_FEATURES = ["host_len", "n_labels", "n_hyphens", "n_digits", "max_label_len", "domain_label_len",
                  "max_consonant_run", "n_sensitive_words", "subdomain_labels", "subdomain_len"]
SPARSE_COUNTS = ["n_sensitive_words"]
VARIANTS = {
    "standard": "StandardScaler only (current)",
    "log1p": "log1p(count features) → StandardScaler",
    "log1p+binarize": "log1p(counts), 1[n_sensitive_words > 0] → StandardScaler",
}


def transform_counts(X, log_idx=(), binarize_idx=()):
    """Return a copy of X with log1p on `log_idx` columns and x > 0 → 1 on `binarize_idx` columns."""
    X = np.array(X, dtype=np.float64, copy=True)
    if len(log_idx):
        X[:, list(log_idx)] = np.log1p(X[:, list(log_idx)])
    if len(binarize_idx):
        X[:, list(binarize_idx)] = (X[:, list(binarize_idx)] > 0).astype(np.float64)
    return X


def transform_indices(kind, feature_names):
    """(log_idx, binarize_idx) of a variant for this feature order."""
    if kind not in VARIANTS:
        raise ValueError(f"unknown preprocessing {kind!r}; choose from {list(VARIANTS)}")
    names = list(feature_names)
    binarize = SPARSE_COUNTS if kind == "log1p+binarize" else []
    log = [] if kind == "standard" else [c for c in COUNT_FEATURES if c not in binarize]
    return [names.index(c) for c in log if c in names], [names.index(c) for c in binarize if c in names]


def make_preprocessor(kind, feature_names):
    """sklearn steps to put in front of the estimator (fitted inside every CV fold)."""
    log_idx, binarize_idx = transform_indices(kind, feature_names)
    if not log_idx and not binarize_idx:
        return [StandardScaler()]
    counts = FunctionTransformer(transform_counts, kw_args={"log_idx": log_idx, "binarize_idx": binarize_idx})
    return [counts, StandardScaler()]


def bias_dominated_rate(pipeline, X, batch=4_000):
    """Share of rows where Σ|α_i y_i K(x_i, x)| < |b|: the RBF blind spot, f(x) ≈ b."""
    *steps, svc = pipeline
    Z = X
    for step in steps:
        Z = step.transform(Z)
    coef, bias = svc.dual_coef_[0], abs(svc.intercept_[0])
    flagged = 0
    for start in range(0, len(Z), batch):
        K = kernel_matrix(Z[start:start + batch], svc.support_vectors_, "rbf", svc._gamma)
        flagged += int((np.abs(K * coef).sum(axis=1) < bias).sum())
    return flagged / len(Z)


def one_standard_error_choice(results, n_folds, model="SVC rbf"):
    """Simplest variant (VARIANTS order) whose CV F1 is within one standard error of the best.

    Differences smaller than the fold-to-fold noise are not evidence that a more complex
    preprocessing helps, so the simpler pipeline is kept (Hastie, Tibshirani & Friedman, ESL §7.10).
    """
    rows = results[results["model"] == model].set_index("preprocessing").loc[list(VARIANTS)]
    best = rows["cv_f1"].idxmax()
    threshold = rows.loc[best, "cv_f1"] - rows.loc[best, "cv_f1_std"] / np.sqrt(n_folds)
    return next(kind for kind in rows.index if rows.loc[kind, "cv_f1"] >= threshold), threshold


def _scores(y_true, decision):
    y_pred = (decision > 0).astype(int)
    return {"precision": precision_score(y_true, y_pred, zero_division=0), "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred), "roc_auc": roc_auc_score(y_true, decision),
            "pr_auc": average_precision_score(y_true, decision)}


def compare_preprocessing(features, y, groups, train_k, train_all, test, rbf_grid, linear_C, cv_folds, seed):
    """Tune and score an RBF SVC (kernel subsample) and a LinearSVC (all training rows) per variant.

    `features` is a DataFrame indexed like `y` and `groups`. Returns (results, rbf grids per variant).
    `keyword_*` columns score only the test hosts with at least one sensitive word, the rows the
    transformation is meant to fix.
    """
    names = list(features.columns)
    X_test, y_test = features.loc[test].to_numpy(), y.loc[test].to_numpy()
    keyword = features.loc[test, "n_sensitive_words"].to_numpy() > 0
    cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    rows, grids = [], {}
    for kind in VARIANTS:
        search = GridSearchCV(make_pipeline(*make_preprocessor(kind, names), SVC(kernel="rbf", cache_size=1000)),
                              rbf_grid, scoring="f1", n_jobs=-1, cv=cv)
        search.fit(features.loc[train_k].to_numpy(), y.loc[train_k].to_numpy(), groups=groups.loc[train_k].to_numpy())
        grids[kind] = pd.DataFrame(search.cv_results_)
        model = search.best_estimator_
        decision = model.decision_function(X_test)
        rows.append({"preprocessing": kind, "model": "SVC rbf", "params": search.best_params_,
                     "cv_f1": search.best_score_,
                     "cv_f1_std": float(search.cv_results_["std_test_score"][search.best_index_]),
                     **_scores(y_test, decision),
                     "keyword_recall": recall_score(y_test[keyword], decision[keyword] > 0),
                     "keyword_f1": f1_score(y_test[keyword], decision[keyword] > 0),
                     "bias_dominated_test": bias_dominated_rate(model, X_test),
                     "n_sv": int(model[-1].n_support_.sum())})

        search = GridSearchCV(make_pipeline(*make_preprocessor(kind, names), LinearSVC(dual=False)),
                              {"linearsvc__C": linear_C}, scoring="f1", n_jobs=-1, cv=cv)
        search.fit(features.loc[train_all].to_numpy(), y.loc[train_all].to_numpy(),
                   groups=groups.loc[train_all].to_numpy())
        decision = search.best_estimator_.decision_function(X_test)
        rows.append({"preprocessing": kind, "model": "LinearSVC (full train)", "params": search.best_params_,
                     "cv_f1": search.best_score_,
                     "cv_f1_std": float(search.cv_results_["std_test_score"][search.best_index_]),
                     **_scores(y_test, decision),
                     "keyword_recall": recall_score(y_test[keyword], decision[keyword] > 0),
                     "keyword_f1": f1_score(y_test[keyword], decision[keyword] > 0),
                     "bias_dominated_test": np.nan, "n_sv": np.nan})
    return pd.DataFrame(rows), grids

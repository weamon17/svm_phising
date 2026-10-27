"""Verify the from-scratch SMO against sklearn.svm.SVC on real hostname features.

Same data, same hyper-parameters, same tolerance: the two solvers must agree on the
support vectors, the decision values and the predictions. Results go to a new
outputs/smo_check/<timestamp>/ folder; earlier runs are never overwritten.
"""

import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data import PROCESSED_CSV, ROOT
from src.features import extract_features
from src.smo import SMOSVC, kernel_matrix
from src.splits import group_split, stratified_sample


N_TRAIN, N_TEST, TOL, SEED = 2_000, 10_000, 1e-3, 42
CONFIGS = {
    "linear (C=1)": dict(kernel="linear", C=1.0),
    "rbf (C=1, gamma=scale)": dict(kernel="rbf", C=1.0, gamma="scale"),
    "poly d=3 (C=1, gamma=scale, coef0=1)": dict(kernel="poly", C=1.0, gamma="scale", degree=3, coef0=1.0),
}


def sklearn_dual_objective(model, X_train, gamma, params):
    """½ αᵀQα − Σα from sklearn's dual_coef_ (= α_i y_i), for an apples-to-apples comparison."""
    coef = model.dual_coef_[0]
    sv = X_train[model.support_]
    K = kernel_matrix(sv, sv, params["kernel"], gamma, params.get("degree", 3), params.get("coef0", 0.0))
    return float(0.5 * coef @ K @ coef - np.abs(coef).sum())


def scores(y_true, y_pred):
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred), "f1": f1_score(y_true, y_pred)}


def main():
    hosts = pd.read_csv(PROCESSED_CSV, keep_default_na=False)
    train_all, test_all = group_split(hosts, seed=SEED)
    train = stratified_sample(train_all, hosts["phishing"], N_TRAIN, SEED)
    test = stratified_sample(test_all, hosts["phishing"], N_TEST, SEED)
    shared = set(hosts.loc[train, "registered_domain"]) & set(hosts.loc[test, "registered_domain"])
    assert not shared, "group split leaked registered domains"

    features = extract_features(hosts.loc[train.union(test)])
    scaler = StandardScaler().fit(features.loc[train])         # fit on train only
    X_train, X_test = scaler.transform(features.loc[train]), scaler.transform(features.loc[test])
    y_train, y_test = hosts.loc[train, "phishing"].to_numpy(), hosts.loc[test, "phishing"].to_numpy()

    rows = []
    for name, params in CONFIGS.items():
        start = time.perf_counter()
        ours = SMOSVC(tol=TOL, **params).fit(X_train, y_train)
        t_ours = time.perf_counter() - start
        start = time.perf_counter()
        ref = SVC(tol=TOL, **params).fit(X_train, y_train)
        t_ref = time.perf_counter() - start

        d_ours, d_ref = ours.decision_function(X_test), ref.decision_function(X_test)
        sv_ours, sv_ref = set(ours.support_), set(ref.support_)
        # Duplicate feature rows make α non-unique (sv_jaccard can be low); the dual
        # objective and the decision function are unique, so they are the real check.
        row = {
            "model": name, "converged": ours.converged_, "smo_iterations": ours.n_iter_,
            "dual_objective_smo": ours.dual_objective_,
            "dual_objective_sklearn": sklearn_dual_objective(ref, X_train, ours.gamma_, params),
            "n_sv_smo": len(sv_ours), "n_sv_sklearn": len(sv_ref),
            "sv_jaccard": len(sv_ours & sv_ref) / len(sv_ours | sv_ref),
            "intercept_smo": ours.intercept_, "intercept_sklearn": float(ref.intercept_[0]),
            "decision_max_abs_diff": float(np.abs(d_ours - d_ref).max()),
            "decision_max_rel_diff": float(np.abs(d_ours - d_ref).max() / np.abs(d_ref).max()),
            "decision_mean_abs_diff": float(np.abs(d_ours - d_ref).mean()),
            "prediction_agreement": float((ours.predict(X_test) == ref.predict(X_test)).mean()),
            "fit_seconds_smo": t_ours, "fit_seconds_sklearn": t_ref,
        }
        row.update({f"{k}_smo": v for k, v in scores(y_test, ours.predict(X_test)).items()})
        row.update({f"{k}_sklearn": v for k, v in scores(y_test, ref.predict(X_test)).items()})
        rows.append(row)
        print(f"{name:<38} dual {row['dual_objective_smo']:.4f}/{row['dual_objective_sklearn']:.4f}  "
              f"SV {len(sv_ours)}/{len(sv_ref)}  rel|df|={row['decision_max_rel_diff']:.1e}  "
              f"mean|df|={row['decision_mean_abs_diff']:.1e}  agree={row['prediction_agreement']:.4f}  "
              f"F1 {row['f1_smo']:.3f}/{row['f1_sklearn']:.3f}  "
              f"time {t_ours:.1f}s/{t_ref:.1f}s", flush=True)

    out = ROOT / "outputs" / "smo_check" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    table = pd.DataFrame(rows).set_index("model")
    table.to_csv(out / "results.csv")
    (out / "config.json").write_text(json.dumps({
        "n_train": len(train), "n_test": len(test), "tol": TOL, "seed": SEED,
        "train_phishing_share": float(y_train.mean()), "test_phishing_share": float(y_test.mean()),
        "features": list(features.columns), "configs": CONFIGS,
        "split": "GroupShuffleSplit by registered_domain, test_size=0.2, then stratified subsample",
    }, indent=2), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

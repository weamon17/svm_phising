"""Train the demo model with our from-scratch SMO and export it to models/ (npz + json, no pickle).

Same data recipe as notebooks/02_experiments.ipynb (FULL): group split by registered domain,
stratified 8,000-host training subsample, optional log1p/binarize of count features (--prep),
StandardScaler fitted on it, RBF kernel with hyperparameters selected by grouped CV.
"""

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.data import PROCESSED_CSV
from src.demo import MODEL_JSON, MODEL_NPZ
from src.features import extract_features
from src.preprocessing import VARIANTS, transform_counts, transform_indices
from src.smo import SMOSVC, kernel_matrix
from src.splits import group_split, stratified_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--n-train", type=int, default=8_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prep", choices=list(VARIANTS), default="standard",
                        help="feature transformation before StandardScaler (see src/preprocessing.py)")
    args = parser.parse_args()

    hosts = pd.read_csv(PROCESSED_CSV, keep_default_na=False)
    train_all, test = group_split(hosts, seed=args.seed)
    train = stratified_sample(train_all, hosts["phishing"], args.n_train, args.seed)
    features = extract_features(hosts.loc[train.union(test)])
    log_idx, binarize_idx = transform_indices(args.prep, features.columns)
    raw_train = transform_counts(features.loc[train].to_numpy(), log_idx, binarize_idx)
    raw_test = transform_counts(features.loc[test].to_numpy(), log_idx, binarize_idx)
    scaler = StandardScaler().fit(raw_train)
    X_train, X_test = scaler.transform(raw_train), scaler.transform(raw_test)
    y_train, y_test = hosts.loc[train, "phishing"].to_numpy(), hosts.loc[test, "phishing"].to_numpy()

    start = time.perf_counter()
    model = SMOSVC(C=args.C, kernel="rbf", gamma=args.gamma).fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start
    decision = model.decision_function(X_test)
    y_pred = (decision > 0).astype(int)
    # Test rows where Σ|α_i y_i K(x_i, x)| < |b|: the verdict comes from the bias, not support vectors.
    evidence = np.concatenate([
        np.abs(kernel_matrix(X_test[s:s + 4_000], model.support_vectors_, "rbf", model.gamma_) * model.dual_coef_).sum(1)
        for s in range(0, len(X_test), 4_000)])
    bias_dominated = evidence < abs(model.intercept_)

    MODEL_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(MODEL_NPZ, mean=scaler.mean_, scale=scaler.scale_,
             support_vectors=model.support_vectors_, dual_coef=model.dual_coef_)
    support_rows = train[model.support_]
    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "solver": "SMOSVC (from scratch, src/smo.py)", "kernel": "rbf", "preprocessing": args.prep,
        "C": args.C, "gamma": model.gamma_, "intercept": float(model.intercept_),
        "converged": bool(model.converged_), "smo_iterations": int(model.n_iter_),
        "fit_seconds": fit_seconds, "n_train": len(train), "n_test": len(test), "seed": args.seed,
        "features": list(features.columns),
        "support_hosts": hosts.loc[support_rows, "host"].tolist(),
        "support_labels": hosts.loc[support_rows, "phishing"].astype(int).tolist(),
        "test_metrics": {"precision": precision_score(y_test, y_pred), "recall": recall_score(y_test, y_pred),
                         "f1": f1_score(y_test, y_pred), "roc_auc": roc_auc_score(y_test, decision)},
        "bias_dominated_test": {"share": float(bias_dominated.mean()),
                                "phishing_share": float(y_test[bias_dominated].mean()),
                                "accuracy": float((y_pred[bias_dominated] == y_test[bias_dominated]).mean())},
    }
    MODEL_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    m = meta["test_metrics"]
    print(f"SMO: {model.n_iter_:,} iterations in {fit_seconds:.1f}s, {len(model.support_):,} support vectors")
    print(f"Test: F1 {m['f1']:.3f} | precision {m['precision']:.3f} | recall {m['recall']:.3f} | ROC-AUC {m['roc_auc']:.3f}")
    b = meta["bias_dominated_test"]
    print(f"Bias-dominated test rows: {b['share']:.1%} ({b['phishing_share']:.1%} phishing, accuracy {b['accuracy']:.3f})")
    print(f"Saved {MODEL_NPZ} and {MODEL_JSON}")


if __name__ == "__main__":
    main()
